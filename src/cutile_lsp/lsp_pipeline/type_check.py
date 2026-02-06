import ast
import inspect
import json
import textwrap
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Sequence

from cuda.tile._const_utils import get_constant_annotations
from cuda.tile._context import init_context_config_from_env
from cuda.tile._execution import kernel
from cuda.tile._ir import hir, ir
from cuda.tile._ir.ir import KernelArgument
from cuda.tile._ir.ops import Assign
from cuda.tile._passes.ast2hir import get_function_hir
from cuda.tile._passes.hir2ir import hir2ir


def get_kernel_shapes_info(kernel_func: kernel, args: tuple[KernelArgument, ...]) -> list[dict]:
    """
    Main entry point: get shape/type info for all assignments in a kernel.

    Returns a list of dicts, each with keys:
        - lineno: source line number
        - end_lineno: source end line number
        - col_start: column start offset
        - col_end: column end offset
        - ty: string representation of the variable's type
    """
    ops = _get_ops_for_shapes_info(kernel_func, args)
    results = []
    for op in ops:
        results.append(
            {
                "name": op.result_var.get_original_name(),
                "lineno": op.loc.line,
                "end_lineno": op.loc.last_line,
                "col_start": op.loc.col,
                "col_end": op.loc.end_col,
                "ty": str(op.result_var.get_type()),
            }
        )
    return results


def dump_kernel_shapes_info_to_json(
    kernel_func: kernel,
    args: tuple[KernelArgument, ...],
    output_path: str,
) -> None:
    """Get shape/type info and write it to a JSON file."""
    shapes_info = get_kernel_shapes_info(kernel_func, args)
    with open(output_path, "w") as f:
        json.dump(shapes_info, f, indent=2)


class ControlFlowToken(StrEnum):
    If = "if"
    Elif = "elif"
    Else = "else"
    While = "while"
    For = "for"


def _get_ops_for_shapes_info(kernel_func: kernel, args: tuple[KernelArgument, ...]) -> list[ir.Operation]:
    """
    Extract all meaningful Assign operations from the kernel IR,
    excluding temporaries and control-flow lines.
    """
    block = _get_kernel_shapecheck_ir(kernel_func, args)
    control_flow_lines = _get_control_flow_lines_mapping(kernel_func._pyfunc)

    filtered_ops = []
    for op in block.traverse():
        if not isinstance(op, Assign):
            continue
        original_name = op.result_var.get_original_name()
        if original_name.startswith("$"):
            continue
        if op.loc.line in control_flow_lines:
            continue
        filtered_ops.append(op)
    return filtered_ops


def bind_args(kernel_func: kernel, args: tuple) -> tuple[KernelArgument, ...]:
    """
    Convert user-supplied args into a tuple of KernelArgument.

    For each arg:
      - If it is already a KernelArgument, use it directly.
      - Otherwise, fall back to the standard cuTile compiler type conversion
        (typeof_pyval + get_constant_value). If that also fails, raise a TypeError
        indicating which argument (0-based index) caused the problem.
    """
    from cuda.tile._compile import get_constant_value
    from cuda.tile._ir.typing_support import typeof_pyval

    pyfunc = kernel_func._pyfunc
    param_names = tuple(inspect.signature(pyfunc).parameters.keys())
    constant_annotations = get_constant_annotations(pyfunc)

    assert len(args) == len(param_names), f"Expected {len(param_names)} arguments, got {len(args)}"

    ir_args: list[KernelArgument] = []
    for idx, (param_name, arg_value) in enumerate(zip(param_names, args, strict=True)):
        if isinstance(arg_value, KernelArgument):
            ir_args.append(arg_value)
        else:
            is_const = param_name in constant_annotations
            try:
                ty = typeof_pyval(arg_value, kernel_arg=not is_const)
            except (TypeError, Exception) as e:
                raise TypeError(
                    f"Argument {idx} ('{param_name}') = {arg_value!r} "
                    f"of type {type(arg_value).__name__} is not a supported kernel argument: {e}"
                ) from e
            const_val = get_constant_value(arg_value) if is_const else None
            ir_args.append(KernelArgument(type=ty, is_const=is_const, const_value=const_val))
    return tuple(ir_args)


def _get_kernel_shapecheck_ir(kernel_func: kernel, args: tuple) -> ir.Block:
    """
    Compile a kernel to IR with type inference (via hir2ir), returning the typed Block.

    Only runs up to hir2ir -- no further optimization passes are needed for shape checking.
    """
    ir_args = bind_args(kernel_func, args)

    func_hir: hir.Function = get_function_hir(kernel_func._pyfunc, entry_point=True)

    config = init_context_config_from_env()
    ir_ctx = ir.IRContext(config)
    func_body = hir2ir(func_hir, ir_args, ir_ctx)
    return func_body


@dataclass
class KernelSource:
    source_code: str
    starting_line: int
    tree: ast.AST


def _get_kernel_source(func: Callable) -> KernelSource:
    """Get the source code of a Python function and parse it into an AST."""
    source_lines, starting_line = inspect.getsourcelines(func)
    source_code = textwrap.dedent("".join(source_lines))
    tree = ast.parse(source_code)
    return KernelSource(source_code=source_code, starting_line=starting_line, tree=tree)


def _get_control_flow_lines_mapping(func: Callable) -> dict[int, ControlFlowToken]:
    """
    Analyze a function's AST to find all control-flow statement lines.

    Returns a mapping from absolute line number to ControlFlowToken.
    """
    assert callable(func), f"Expected a callable, got {type(func)}"

    kernel_src = _get_kernel_source(func)
    results: dict[str, list[int]] = {token.value: [] for token in ControlFlowToken}

    class ControlFlowVisitor(ast.NodeVisitor):
        def visit_If(self, node: ast.If):
            abs_line = node.lineno + kernel_src.starting_line - 1
            results[ControlFlowToken.If].append(abs_line)

            # Distinguish elif vs else
            if node.orelse:
                if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                    # elif branch -- will be visited recursively
                    elif_node = node.orelse[0]
                    elif_abs_line = elif_node.lineno + kernel_src.starting_line - 1
                    results[ControlFlowToken.Elif].append(elif_abs_line)
                else:
                    # else branch: the line before the first statement in the else block
                    first_else_stmt = node.orelse[0]
                    else_abs_line = first_else_stmt.lineno + kernel_src.starting_line - 1 - 1
                    results[ControlFlowToken.Else].append(else_abs_line)

            self.generic_visit(node)

        def visit_While(self, node: ast.While):
            abs_line = node.lineno + kernel_src.starting_line - 1
            results[ControlFlowToken.While].append(abs_line)
            self.generic_visit(node)

        def visit_For(self, node: ast.For):
            abs_line = node.lineno + kernel_src.starting_line - 1
            results[ControlFlowToken.For].append(abs_line)
            self.generic_visit(node)

    ControlFlowVisitor().visit(kernel_src.tree)

    # Invert: line -> token
    revmap: dict[int, ControlFlowToken] = {}
    for token_str, lines in results.items():
        for line in lines:
            revmap[line] = ControlFlowToken(token_str)
    return revmap
