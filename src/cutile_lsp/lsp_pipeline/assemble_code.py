import os
import textwrap
from pathlib import Path

from .data_structures import Kernel
from ..loggings import get_logger

logger = get_logger(__name__)


HEAD_PATH = Path(__file__).resolve().parent.joinpath("HEAD.py")
HEAD = HEAD_PATH.read_text()

TEMP_DIR = os.getenv("CUTILE_LSP_TEMP_DIR")
if TEMP_DIR is None:
    TEMP_DIR = Path.home().joinpath(".cutile_lsp").resolve()
    TEMP_DIR.mkdir(exist_ok=True)
else:
    TEMP_DIR = Path(TEMP_DIR).resolve()
    # Check existense and is dir
    if not TEMP_DIR.exists():
        raise FileNotFoundError(
            f"Temporary directory (set via env var CUTILE_LSP_TEMP_DIR) {TEMP_DIR} does not exist!"
        )
    elif not TEMP_DIR.is_dir():
        raise NotADirectoryError(
            f"Temporary directory (set via env var CUTILE_LSP_TEMP_DIR) {TEMP_DIR} is not a directory!"
        )


def space(n: int):
    return " " * n


def main_code(
    kernels: list[Kernel],
    source_file_name: str | os.PathLike,  # Path of the inspected file, NOT the assembled file
    uri_hash: str,
):
    launch_func_def_list = []
    launc_func_call_list = []
    for k in kernels:
        funcdef_code = assemble_launch_func_def_code(k, str(source_file_name))
        launch_func_def_list.append(funcdef_code)

        try_code = assemble_launch_func_code(k)
        launc_func_call_list.append(try_code)

    launch_func_def_code = textwrap.indent("\n".join(launch_func_def_list), prefix=space(4))
    launch_func_call_code = textwrap.indent("\n".join(launc_func_call_list), prefix=space(4))
    save_hints_code = textwrap.indent(save_code(uri_hash), prefix=space(4))

    code = f"""
{HEAD}

if __name__ == "__main__":
    _hints = []
    _diagnostics = []

    # Launch func definitions, code is indented outside
{launch_func_def_code}

    # Launch kernels, code is indented outside
{launch_func_call_code}

    # Save hints, code is indented outside
{save_hints_code}

"""
    return code


def save_code(uri_hash: str):
    code = f"""lsp_code_exec_results = dict(hints=_hints, diagnostics=_diagnostics)
with open("{TEMP_DIR}/{uri_hash}/lsp_code_exec_results.json", "w") as f:
    json.dump(lsp_code_exec_results, f)
"""
    return code


def assemble_launch_func_code(kernel: Kernel):
    code = f"{kernel.launch_func_name}(_hints, _diagnostics)"
    return code


def assemble_launch_func_def_code(
    kernel: Kernel,
    source_file_name: str | os.PathLike,  # Path of the inspected file, NOT the assembled file
):
    args = ", ".join(kernel.args_str)
    code = f"""
def {kernel.launch_func_name}(_hints, _diagnostics):
    try:
        args = ({args})
        type_info = check_semantics_and_type({kernel.name}, args)
        _hints.extend(type_info)
    except TileError as e:
        # 序列化Loc信息，只包含必要的字段
        loc_info = {{
            "message": e.message,
            "line": e.loc.line,
            "col": e.loc.col
        }}
        
        # 添加可选的字段（如果存在）
        if e.loc.last_line is not None:
            loc_info["last_line"] = e.loc.last_line
        if e.loc.end_col is not None:
            loc_info["end_col"] = e.loc.end_col
        if e.loc.filename is not None:
            loc_info["filename"] = e.loc.filename
        
        _diagnostics.append(loc_info)
    except Exception as e:
        # 报告任何其他异常
        error_msg = str(e).replace('"', "'").replace('\\n', ' ')
        loc_info = {{
            "message": f"Unexpected error in kernel '{kernel.name}': {{error_msg}}",
            "line": {kernel.start_line},
            "col": 0,
            "filename": "{source_file_name}"
        }}
        _diagnostics.append(loc_info)
"""
    return code
