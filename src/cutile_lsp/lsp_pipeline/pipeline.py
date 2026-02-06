import os
from pathlib import Path

from .assemble_code import assemble_launch_func_def_code, main_code
from .extract_code import check_cutile_lsp_enabled, extract_code
from .get_kernels import load_kernels_to_pyfunc
from .loggings import get_logger
from .parse_args import EarlyDiagnostics, TypecheckError, parse_typecheck

logger = get_logger(__name__)


class NotEnabledError(Exception):
    """Type checking is not enabled for this file"""


def pipeline(source_path: Path, uri_hash: str) -> tuple[str, list[EarlyDiagnostics], int]:
    source_code = source_path.read_text()

    enabled = check_cutile_lsp_enabled(source_code)

    if not enabled:
        raise NotEnabledError

    early_diagnostics = []

    # Get code between start and end, and the line offset
    active_code, line_offset = extract_code(source_code)

    # Load all cuTile kernels
    pyfuncs = load_kernels_to_pyfunc(active_code)

    # Parse typecheck in docstring for each kernel
    kernels = []
    for pyfunc in pyfuncs:
        # 错误在这个层面处理，不通过注入lsp_code进行处理
        try:
            kernel = parse_typecheck(pyfunc)
            kernels.append(kernel)
        except TypecheckError as e:
            # Adjust line number to account for the offset
            early_diagnostics.append(e.to_diagnostics(line_offset))

    lsp_code = active_code + "\n\n" + main_code(kernels, source_path, uri_hash)

    return lsp_code, early_diagnostics, line_offset


if __name__ == "__main__":
    import argparse
    import dataclasses
    import json

    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument("path", type=str, help="Path to the file")
    args = parser.parse_args()

    path = Path(args.path).resolve()
    uri_hash = "test"
    lsp_code, early_diagnostics = pipeline(path, uri_hash)
    # print(len(lsp_code))

    lsp_code_path = Path("lsp_code.py").resolve()
    lsp_code_path.write_text(lsp_code)
    print(f"lsp_code.py written to {lsp_code_path}")

    print()
    early_diagnostics_dicts = [dataclasses.asdict(d) for d in early_diagnostics]
    print(json.dumps(early_diagnostics_dicts, indent=2))
