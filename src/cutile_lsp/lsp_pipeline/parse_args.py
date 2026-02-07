import ast
import inspect
from dataclasses import dataclass

from cutile_lsp.constants import END_OF_LINE

from .data_structures import Kernel

TYPECHECK_START = "<typecheck>"
TYPECHECK_END = "</typecheck>"


class TypecheckError(Exception):
    """Base class for typecheck-related errors with position information."""

    def __init__(self, line: int, col: int, message: str, end_col: int = END_OF_LINE):
        super().__init__()
        self.line = line
        self.col = col
        self.end_col = end_col
        self.message = message

    def __str__(self):
        return f"{self.__class__.__name__} at line {self.line}, column {self.col}: {self.message}"

    def to_diagnostics(self, line_offset):
        # line_offset: the line of #cutile_lsp:start
        return EarlyDiagnostics(
            message=self.message,
            line=self.line + line_offset,
            col=self.col,
            end_col=self.end_col,
        )


class NotAnnotatedError(TypecheckError):
    """When code has no <typecheck> annotations"""


class TypecheckSyntaxError(TypecheckError):
    """当typecheck块中的某一行语法错误时抛出"""


class TypecheckParamCountError(TypecheckError):
    """当typecheck参数个数与函数定义的参数个数不匹配时抛出"""


@dataclass
class EarlyDiagnostics:
    message: str
    line: int
    col: int
    end_col: int


def _parse_typecheck_params(
    docstring: str | None,
    source_lines: list[str] = None,
    func_start_line: int = 0,
):
    """
    Process a string by:
    1. Splitting by '<typecheck>' and taking the last part
    2. Splitting by '</typecheck>' and taking the first part
    3. Trimming whitespace from the resulting string
    4. Splitting into lines, trimming whitespace from each line, and returning as a list
    5. Validate each line's Python syntax

    Args:
        docstring: The docstring to process
        source_lines: Source code lines of the function (for calculating line numbers)
        func_start_line: The starting line number of the function in the source file

    Returns:
        list: List of whitespace-trimmed lines from the extracted content

    Raises:
        ValueError: If <typecheck> is not annotated
        TypecheckSyntaxError: If a line in the typecheck block has invalid Python syntax
    """
    if docstring is None or TYPECHECK_START not in docstring or TYPECHECK_END not in docstring:
        raise TypecheckError(
            line=func_start_line,
            col=0,
            message="<typecheck> is not annotated. Typecheck will not be performed.",
        )
    # Split by <typecheck> and get the last part
    after_typecheck = docstring.split(TYPECHECK_START, 1)[-1]

    # Split by </typecheck> and get the first part
    before_close = after_typecheck.rsplit(TYPECHECK_END, 1)[0]

    # Trim whitespace from the extracted content
    trimmed_content = before_close.strip()

    # Handle empty content
    if not trimmed_content:
        return []

    # Split into lines, trim each line, and filter out empty lines if desired
    # (keeping empty lines that were between non-empty content)
    lines = [line.strip() for line in trimmed_content.splitlines()]

    # 计算typecheck块在源文件中的起始行号
    typecheck_start_line_in_file = 0
    if source_lines is not None:
        # 在源代码中找到 <typecheck> 的位置
        for i, source_line in enumerate(source_lines):
            if TYPECHECK_START in source_line:
                typecheck_start_line_in_file = func_start_line + i
                break

    # 验证每一行的Python语法
    for idx, line in enumerate(lines):
        if len(line.strip()) == 0:
            continue  # 跳过空行

        try:
            # 尝试以eval模式解析（表达式）
            ast.parse(line, mode="eval")
        except SyntaxError as e:
            # 计算实际行号：typecheck起始行 + 当前行偏移 + 1（因为<typecheck>占一行）
            actual_line = typecheck_start_line_in_file + idx + 1
            col = e.offset - 1 if e.offset else 0
            end_col = e.end_offset - 1 if e.end_offset else col + 1
            raise TypecheckSyntaxError(
                line=actual_line,
                col=col + 4,
                end_col=end_col + 4,
                message=f"Invalid Python syntax in typecheck parameter: {e.msg}",
            )

    # Filter out empty lines
    lines = [line for line in lines if len(line) > 0]

    return lines


def parse_typecheck(pyfunc) -> list[str]:
    source_lines, start_line = inspect.getsourcelines(pyfunc)

    func_name = pyfunc.__name__
    docs = pyfunc.__doc__

    args_str = _parse_typecheck_params(docs, source_lines, start_line)

    # 检查参数个数是否与函数定义的参数个数匹配
    sig = inspect.signature(pyfunc)
    func_param_count = len(sig.parameters)
    typecheck_param_count = len(args_str)

    if func_param_count != typecheck_param_count:
        raise TypecheckParamCountError(
            line=start_line,
            col=0,
            message=f"Parameter count mismatch in kernel '{func_name}': "
            f"function has {func_param_count} parameters, but <typecheck> has {typecheck_param_count} args.",
        )

    return Kernel(pyfunc=pyfunc, args_str=args_str)
