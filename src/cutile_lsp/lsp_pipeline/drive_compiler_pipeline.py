import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cuda.tile as ct
from cuda.tile._context import TileContextConfig
from cuda.tile._ir import hir, ir
from cuda.tile._ir.ir import KernelArgument
from cuda.tile._ir.type import ArrayTy, SizeTy, TupleTy
from cuda.tile._passes.ast2hir import get_function_hir
from cuda.tile._passes.hir2ir import hir2ir

from cutile_lsp.lsp_pipeline.type_check import bind_args, get_kernel_shapes_info

DTYPE_MAP = {
    "bool": ct.bool_,
    "uint8": ct.uint8,
    "uint16": ct.uint16,
    "uint32": ct.uint32,
    "uint64": ct.uint64,
    "int8": ct.int8,
    "int16": ct.int16,
    "int32": ct.int32,
    "int64": ct.int64,
    "float32": ct.float32,
    "float64": ct.float64,
    "float16": ct.float16,
    "bfloat16": ct.bfloat16,
    "tfloat32": ct.tfloat32,
    "float8_e4m3fn": ct.float8_e4m3fn,
    "float8_e5m2": ct.float8_e5m2,
}


@dataclass(frozen=True)
class Tensor:
    """Describes a tensor argument for kernel type checking."""

    shape: tuple[int, ...]
    dtype: str
    strides: tuple[int, ...] | None = None

    def to_kernel_argument(self) -> KernelArgument:
        """Convert this Tensor descriptor into a KernelArgument."""
        ndim = len(self.shape)
        shape_ty = TupleTy(tuple(SizeTy(None) for _ in self.shape))

        if self.strides is not None:
            strides = self.strides
        else:
            # Compute default strides (row-major / C-contiguous)
            strides_list = []
            stride = 1
            for s in reversed(self.shape):
                strides_list.insert(0, stride)
                stride *= s
            strides = tuple(strides_list)
        strides_ty = TupleTy(tuple(SizeTy(None) for _ in strides))

        ct_dtype = DTYPE_MAP[self.dtype]
        array_ty = ArrayTy(
            ct_dtype,
            shape=shape_ty,
            strides=strides_ty,
            elements_disjoint=True,
            base_ptr_div_by=ct_dtype.bitwidth // 8,
            stride_div_by=tuple(1 for _ in range(ndim)),
            shape_div_by=tuple(1 for _ in range(ndim)),
        )
        return KernelArgument(type=array_ty, is_const=False)


def _bind_arguments(kernel_func, args: tuple) -> tuple[KernelArgument, ...]:
    """
    Convert user-supplied args into a tuple of KernelArgument.

    For each arg:
      - If it is a Tensor dataclass, convert via Tensor.to_kernel_argument().
      - Otherwise, delegate to shape_check.bind_args which handles KernelArgument
        pass-through and standard cuTile type conversion with error reporting.
    """

    normalized = tuple(arg.to_kernel_argument() if isinstance(arg, Tensor) else arg for arg in args)
    return bind_args(kernel_func, normalized)


def _check_cutile_semantics(
    pyfunc,
    args: Sequence[ir.KernelArgument],
    config: TileContextConfig,
) -> ir.Block:
    func_hir: hir.Function = get_function_hir(pyfunc, entry_point=True)

    ir_ctx = ir.IRContext(config)
    func_body = hir2ir(func_hir, args, ir_ctx)

    return func_body


def check_semantics_and_type(kernel, args) -> None:
    ir_args = _bind_arguments(kernel, args)

    config = TileContextConfig(
        temp_dir=tempfile.gettempdir(), log_keys=[], compiler_timeout_sec=None, enable_crash_dump=False
    )

    _check_cutile_semantics(kernel._pyfunc, ir_args, config)
    type_info = get_kernel_shapes_info(kernel, ir_args)
    return type_info
