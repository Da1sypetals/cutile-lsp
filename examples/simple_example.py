# cutile-lsp:on
import cuda.tile as ct


@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    """
    <typecheck>
    Tensor((1024,), "float16")
    Tensor((1024,), "float16")
    Tensor((1024,), "float16")
    32
    </typecheck>
    """
    # Get the 1D pid
    pid = ct.bid(0)

    # Load input tiles
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    # Perform elementwise addition
    result = a_tile + b_tile

    # Store result
    ct.store(c, index=(pid,), tile=result)
