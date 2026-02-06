# cutile-lsp: on


# cutile-lsp:    start
import cuda.tile as ct

ConstInt = ct.Constant[int]

n_stream = 4  # consistent with deepseek paper
num_iter_cg = n_stream * 2

EPS = 1e-10


tilesize = 32


@ct.function(host=False, tile=True)
def matvec_A(R, x):
    """
    R: (tilesize, n_stream, n_stream)
    x: (tilesize, n_stream*2, 1)
    """
    x1 = ct.extract(x, index=(0, 0, 0), shape=(tilesize, n_stream, 1))
    x2 = ct.extract(x, index=(0, 1, 0), shape=(tilesize, n_stream, 1))
    ax1 = x1 + ct.matmul(R, x2)
    ax2 = ct.matmul(R.transpose(-2, -1), x1) + x2
    return ct.cat((ax1, ax2), axis=-2)  # (tilesize, n_stream*2, 1)


@ct.function(host=False, tile=True)
def dot(a, b):  # a/b: (..., dim, 1)
    return ct.matmul(a.transpose(-2, -1), b)


# deliberately buggy: not <typecheck> annotated
@ct.kernel
def sinkhorn_knopp_bwd_implicit_cg(out, dout, res):
    """
    Notes:
    1. Number of CG iterations is typically num_streams*2.
        This is derived from the theoretical properties of CG method.
    2. Matrix R is theoretically singular (not full-rank) and numerically near-singular,
        so the solution of x_sol can be very different from the real solution x_real.
        However, the tensor sum of the first half and the second half of x_sol is same with the result of x_real, which **is what we need**.
        This means the solution set has some mathematical property that applies to every element in it.
        We shall make use of that property.
    """

    i_seq = ct.bid(0)

    R = ct.load(
        out,
        index=(i_seq, 0, 0),
        shape=(tilesize, n_stream, n_stream),
    )
    dR = ct.load(
        dout,
        index=(i_seq, 0, 0),
        shape=(tilesize, n_stream, n_stream),
    )

    RdR = R * dR
    # row sum
    b1 = ct.sum(RdR, axis=-1).reshape((tilesize, n_stream, 1))
    # col sum
    b2 = ct.sum(RdR, axis=-2).reshape((tilesize, n_stream, 1))

    b = ct.cat((b1, b2), axis=-2)

    # Solve: Ax=b =========================================
    R = R.reshape((tilesize, n_stream, n_stream))
    # Conjugate Gradients: init
    x = ct.zeros((tilesize, n_stream * 2, 1), dtype=ct.float32)
    r = b - matvec_A(R, x)
    p = r
    r_normsq = dot(r, r)

    # Conjugate Gradients: iter
    for _ in range(num_iter_cg):
        Ap = matvec_A(R, p)
        pAp = dot(p, Ap)
        # VERY important to avoid divide by zero
        alpha = r_normsq / (pAp + EPS)
        x += alpha * p
        r -= alpha * Ap
        r_new_normsq = dot(r, r)
        # not very important to avoid divide by zero, but it's good to have it
        beta = r_new_normsq / (r_normsq + EPS)
        p = r + beta * p
        r_normsq = r_new_normsq
    # End solve: Ax=b =========================================

    x1 = ct.extract(x, index=(0, 0, 0), shape=(tilesize, n_stream, 1))
    x2 = ct.extract(x, index=(0, 1, 0), shape=(tilesize, n_stream, 1))

    x1_expanded = x1.reshape((tilesize, n_stream, 1))
    x2_expanded = x2.reshape((tilesize, 1, n_stream))

    res_tile = dR - x1_expanded - x2_expanded
    res_tile = res_tile * R

    ct.store(
        res,
        index=(i_seq, 0, 0),
        tile=res_tile,
    )


PAD_ZERO = ct.PaddingMode.ZERO


M = 1024  # Batch size (flattened)
N = 2048  # Feature dimension


TILE_N = 1024  # Tile size along N dimension
GROUP_SIZE_M = 64  # Group size for backward pass
TILE_M = 32  # Tile size for final reduction in backward pass


# deliberately buggy: has invalid cuTile code
@ct.kernel
def layer_norm_fwd(X, W, B, Y, Mean, Rstd, eps, TILE_N: ConstInt):
    """
    <typecheck>
    Tensor((1024, 2048), dtype="float16")
    Tensor((2048,), dtype="float16")
    Tensor((2048,), dtype="float16")
    Tensor((1024, 2048), dtype="float16")
    Tensor((1024,), dtype="float32")
    Tensor((1024,), dtype="float32")
    1e-5
    1024
    </typecheck>
    Forward pass: computes mean/var, normalizes input, and applies affine transform.

    Args:
        X: Input tensor (M, N).
        W: Weight tensor (N,).
        B: Bias tensor (N,).
        Y: Output tensor (M, N).
        Mean: Output mean tensor (M,).
        Rstd: Output reciprocal standard deviation tensor (M,).
        eps: Epsilon for numerical stability.
        TILE_N: Tile size along N dimension.
    """
    bid_m = ct.bid(0)
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_N))
    N = X.shape[1]

    # mean = ct.full((1, TILE_N), 0, dtype=ct.float32) # This is correct
    mean = ct.full((1, TILE_N), 0, dtype=ct.float32).reshape((TILE_N, 1))

    for j in range(num_tiles):
        # Compute mean
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        mean += tx
    mean = ct.sum(mean, axis=1) / N
    ct.store(Mean, index=(bid_m,), tile=mean)

    var = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        # Compute variance
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        mask = (j * TILE_N + ct.arange(TILE_N, dtype=ct.int32)) < N
        centered_tx = ct.where(mask, tx - mean, 0)
        var += centered_tx**2
    var = ct.sum(var, axis=1) / N
    rstd = 1 / ct.sqrt(var + eps)
    ct.store(Rstd, index=(bid_m,), tile=rstd)

    for j in range(num_tiles):
        # Normalize and apply affine transformation
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        tw = ct.load(W, index=(j,), shape=(TILE_N,), padding_mode=PAD_ZERO)
        tb = ct.load(B, index=(j,), shape=(TILE_N,), padding_mode=PAD_ZERO)
        ty = (tx - mean) * rstd
        ty = ty * tw + tb
        ct.store(Y, index=(bid_m, j), tile=ty.astype(Y.dtype))


from math import ceil
from types import SimpleNamespace

import cuda.tile as ct
import torch


@ct.function
def silu_and_mul(
    x,
    y,
    approx: bool,
):
    """
    SiLU(x) * y
    SiLU(x) = x / (1 + exp(-x))
    approx: whether to use approximate exp
    """
    denom = ct.add(1, ct.exp(-x), flush_to_zero=True)
    rounding_mode = ct.RoundingMode.APPROX if approx else None
    sigmoid_x = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=rounding_mode)
    silu = ct.mul(x, sigmoid_x, flush_to_zero=True)
    return ct.mul(silu, y, flush_to_zero=True)


# Should work and provide inlay hints
@ct.kernel(occupancy=ct.ByTarget(sm_120=8))
def gemv_silu_mul_split_k_kernel(
    A,
    B1,
    B2,
    C,
    f32acc,
    COUNTS,
    tn: ConstInt,
    tk: ConstInt,
    SPLIT_K: ConstInt,
    approx: ct.Constant[bool],
):
    """
    <typecheck>
    Tensor((1, 1536), dtype="float16")
    Tensor((8960, 1536), dtype="float16")
    Tensor((8960, 1536), dtype="float16")
    Tensor((1, 8960), dtype="float16")
    Tensor((2, 8960), dtype="float32")
    Tensor((280,), dtype="int32")
    32
    64
    8
    True
    </typecheck>
    """
    GROUP_SIZE_M = 1
    M = 1
    N = B1.shape[1]
    bidx, bidy = 0, ct.bid(0)
    bidz = ct.bid(1)
    # pad tile A to fake_tm rows, to enable tensorcore
    fake_tm = 16
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(1, tk))
    num_tiles_n = ct.num_tiles(B1, axis=0, shape=(tn, tk))
    sum1 = ct.full((fake_tm, tn), 0, dtype=ct.float32)
    sum2 = ct.full((fake_tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
    split_size = ct.cdiv(num_tiles_k, SPLIT_K)
    for k in range(bidz * split_size, bidz * split_size + split_size, 1):
        # tile a has only one effective row, of shape tk. It is not efficient to use TMA to load it.
        a = ct.load(
            A,
            index=(bidx, k),
            shape=(fake_tm, tk),
            padding_mode=zero_pad,
            allow_tma=False,
        ).astype(dtype)
        b1 = ct.load(B1, index=(bidy, k), shape=(tn, tk), padding_mode=zero_pad).astype(dtype)
        b1 = ct.transpose(b1)
        sum1 = ct.mma(a, b1, sum1)
        b2 = ct.load(B2, index=(bidy, k), shape=(tn, tk), padding_mode=zero_pad).astype(dtype)
        b2 = ct.transpose(b2)
        sum2 = ct.mma(a, b2, sum2)
    # only the first row of sum is needed
    sum1 = ct.extract(sum1, index=(0, 0), shape=(1, tn))
    sum1 = ct.reshape(sum1, (tn,))
    sum2 = ct.extract(sum2, index=(0, 0), shape=(1, tn))
    sum2 = ct.reshape(sum2, (tn,))

    count_offset = ct.bid(0)
    C_offset = ct.arange(tn, dtype=ct.int32) + bidy * tn
    ct.atomic_add(f32acc, (0, C_offset), sum1)
    ct.atomic_add(f32acc, (1, C_offset), sum2)
    new_count = ct.atomic_add(COUNTS, count_offset, 1)
    if (new_count + 1) % SPLIT_K == 0:
        result1 = ct.gather(f32acc, (0, C_offset))
        result2 = ct.gather(f32acc, (1, C_offset))
        result = silu_and_mul(result1, result2, approx=approx).astype(C.dtype)

        ct.scatter(C, (0, C_offset), result.astype(C.dtype))
        ct.scatter(f32acc, (0, C_offset), 0)
        ct.scatter(f32acc, (1, C_offset), 0)
        ct.scatter(COUNTS, count_offset, 0)


#       cutile-lsp     :end


BATCH_DIM = 4
M_DIM = 512
K_DIM = 256
N_DIM = 1024


@ct.kernel
def batch_matmul_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    """
    <typecheck>
    Tensor((BATCH_DIM, M_DIM, K_DIM), dtype="bfloat16")
    Tensor((BATCH_DIM, K_DIM, N_DIM), dtype="bfloat16")
    Tensor((BATCH_DIM, M_DIM, N_DIM), dtype="bfloat16")
    32
    64
    128
    </typecheck>
    CuTile kernel for batch matrix multiplication
    A has shape (Batch, M, K), B has shape (Batch, K, N) and C has shape (Batch, M, N)
    Each thread block computes one (tm x tn) tile for a specific batch item.
    The grid is 3D: (Batch_idx, M_tile_idx, N_tile_idx).
    """
    pid_batch = ct.bid(0)  # Batch dimension
    pidx = ct.bid(1)  # M dimension
    pidy = ct.bid(2)  # N dimension

    # Calculate number of K tiles
    # A is (Batch, M, K), so K is axis 2
    # Use A.shape[2] for the total K dimension and ct.cdiv for ceiling division
    num_k_tiles = ct.cdiv(A.shape[2], tk)

    # Initialize accumulator
    accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    # K-dimension loop
    for k in range(num_k_tiles):
        # Load tiles with 3D index and 3D shape
        # A is (Batch, M, K), load (1, tm, tk) tile
        a = ct.load(
            A,
            index=(pid_batch, pidx, k),
            shape=(1, tm, tk),
            padding_mode=zero_pad,
        )
        a = ct.reshape(a, (tm, tk))  # Reshape to 2D for ct.mma

        # B is (Batch, K, N), load (1, tk, tn) tile
        b = ct.load(
            B,
            index=(pid_batch, k, pidy),
            shape=(1, tk, tn),
            padding_mode=zero_pad,
        )
        b = ct.reshape(b, (tk, tn))  # Reshape to 2D for ct.mma

        accumulator = ct.mma(a, b, acc=accumulator)

    # Convert to output dtype and store
    result = ct.astype(accumulator, C.dtype)
    # Store with 3D index and 3D shape, C is (Batch, M, N)
    result_3d = ct.reshape(result, (1, tm, tn))
    ct.store(C, index=(pid_batch, pidx, pidy), tile=result_3d)


# print(type(sinkhorn_knopp_bwd_implicit_cg))
