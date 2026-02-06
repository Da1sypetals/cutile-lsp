from typing import Sequence

import cuda.tile as ct
from cuda.tile._context import TileContextConfig
from cuda.tile._ir import hir, ir
from cuda.tile._ir.ir import KernelArgument
from cuda.tile._ir.type import ArrayTy, SizeTy, TupleTy
from cuda.tile._passes.ast2hir import get_function_hir
from cuda.tile._passes.hir2ir import hir2ir

from kernel import sinkhorn_knopp_bwd_implicit_cg


def _check_cutile_semantics(
    pyfunc,
    args: Sequence[ir.KernelArgument],
    config: TileContextConfig,
) -> ir.Block:
    """
    Check cuTile semantics by running the first two steps of the compilation pipeline:
    1. get_function_hir() - AST parsing and syntax validation
    2. hir2ir() - Type checking and type inference

    If a TileError is raised during these steps, it will be re-raised.
    Otherwise, returns the IR block after type checking (before optimization passes).

    Args:
        pyfunc: The Python function to check
        args: Sequence of KernelArgument specifying argument types
        config: TileContextConfig for the compilation context

    Returns:
        ir.Block: The IR block after hir2ir conversion (type-checked IR)

    Raises:
        TileError: If syntax or type errors are found in the cuTile code
    """
    func_hir: hir.Function = get_function_hir(pyfunc, entry_point=True)

    ir_ctx = ir.IRContext(config)
    func_body = hir2ir(func_hir, args, ir_ctx)

    return func_body


def check_semantics_and_type(self, args) -> None:
    """Check cuTile semantics for this kernel with the given arguments.

    This method runs the first two steps of the compilation pipeline:
    1. get_function_hir() - AST parsing and syntax validation
    2. hir2ir() - Type checking and type inference

    Args:
        *args: Kernel arguments. Can be actual arrays/tensors or KernelArgument objects.

    Raises:
        TileError: If syntax or type errors are found in the cuTile code.

    Examples::

        @ct.kernel
        def my_kernel(out, a, b):
            pass

        # Check semantics with actual tensors
        my_kernel.check_semantics_and_type(out_tensor, a_tensor, b_tensor)

        # Or with KernelArgument objects
        from cuda.tile._ir.ir import KernelArgument
        my_kernel.check_semantics_and_type(
            KernelArgument(type=array_ty, is_const=False),
            KernelArgument(type=array_ty, is_const=False),
            KernelArgument(type=array_ty, is_const=False),
        )
    """
    import inspect
    import tempfile

    from cuda.tile import _compile
    from cuda.tile._context import TileContextConfig
    from cuda.tile._ir import ir

    # Get parameter names from the function
    sig = inspect.signature(self._pyfunc)
    param_names = tuple(sig.parameters.keys())

    # Check if args are already KernelArgument objects
    if args and all(isinstance(arg, ir.KernelArgument) for arg in args):
        ir_args = args
    else:
        # Convert runtime values to KernelArgument using _bind_kernel_arguments
        from cuda.tile._const_utils import get_constant_arg_flags

        constant_args = get_constant_arg_flags(self._pyfunc)
        ir_args = _compile._bind_kernel_arguments(param_names, args, constant_args)

    # Create context config
    config = TileContextConfig(
        temp_dir=tempfile.gettempdir(), log_keys=[], compiler_timeout_sec=None, enable_crash_dump=False
    )

    # Run semantics check
    _check_cutile_semantics(self._pyfunc, ir_args, config)


def Tensor(dtype, shape, strides=None):
    """Create an ArrayTy for testing purposes."""
    ndim = len(shape)
    shape_ty = TupleTy(tuple(SizeTy(None) for _ in shape))
    if strides is None:
        # Compute default strides (row-major)
        strides = []
        stride = 1
        for s in reversed(shape):
            strides.insert(0, stride)
            stride *= s
    strides_ty = TupleTy(tuple(SizeTy(None) for _ in strides))

    array_ty = ArrayTy(
        dtype,
        shape=shape_ty,
        strides=strides_ty,
        elements_disjoint=True,
        base_ptr_div_by=dtype.bitwidth // 8,
        stride_div_by=tuple(1 for _ in range(ndim)),
        shape_div_by=tuple(1 for _ in range(ndim)),
    )

    return KernelArgument(type=array_ty, is_const=False)


def main():
    # Create array types for the kernel arguments
    seq_len = 8192
    n_stream = 4  # consistent with deepseek paper
    array_shape = (seq_len, n_stream, n_stream)

    # Create KernelArgument objects
    args = (
        Tensor(ct.float32, array_shape),
        Tensor(ct.float32, array_shape),
        Tensor(ct.float32, array_shape),
    )

    print("\n" + "#" * 60)
    print("# Testing kernel.check_semantics_and_type() method")
    print(f"# Using kernel: sinkhorn_knopp_bwd_implicit_cg")
    print(f"# Array shape: {array_shape}")
    print("#" * 60)

    # Use the new kernel.check_semantics_and_type() API
    print("\nRunning sinkhorn_knopp_bwd_implicit_cg.check_semantics_and_type(...)...")
    check_semantics_and_type(sinkhorn_knopp_bwd_implicit_cg, args)

    print("\n" + "#" * 60)
    print("# Summary")
    print("#" * 60)
    print("kernel.check_semantics_and_type() completed successfully!")


if __name__ == "__main__":
    main()
