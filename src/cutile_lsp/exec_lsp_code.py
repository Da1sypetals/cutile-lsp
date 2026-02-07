"""
Executes LSP code and do post-processing
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from cutile_lsp.lsp_pipeline.assemble_code import TEMP_DIR
from cutile_lsp.lsp_pipeline.loggings import get_logger


def get_uri_hash(uri: str) -> str:
    """Generate a hash from URI for directory naming."""
    return hashlib.sha256(uri.encode()).hexdigest()[:16]


logger = get_logger(__name__)


def execute_lsp_code(lsp_code: str, uri: str) -> tuple[list[dict], list[dict]]:
    """
    Execute the generated lsp_code and return hints and diagnostics from JSON.
    Returns: (hints, diagnostics) where both are lists of dicts.
    """
    uri_hash = get_uri_hash(uri)
    work_dir = Path(TEMP_DIR) / uri_hash
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write lsp_code to a temporary file in the work directory
    temp_file = work_dir / "lsp_code.py"
    temp_file.write_text(lsp_code)

    result = subprocess.run(
        [sys.executable, str(temp_file)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    logger.info(f"Executed lsp_code for {uri_hash}, return code: {result.returncode}")
    if result.stderr:
        logger.debug(f"lsp_code stderr: {result.stderr}")
    if result.stdout:
        logger.debug(f"lsp_code stdout: {result.stdout}")

    # Read the JSON results
    json_path = work_dir / "lsp_code_exec_results.json"
    if not json_path.exists():
        logger.warning(f"JSON results file not found: {json_path}")
        return [], []

    with open(json_path, "r") as f:
        data = json.load(f)
    hints = data.get("hints", [])
    diagnostics = data.get("diagnostics", [])
    return hints, diagnostics


def adjust_hints_line_offset(hints: list[dict], line_offset: int) -> list[dict]:
    """
    Adjust hint line numbers by adding line_offset.

    Hints from JSON have line numbers relative to active_code.
    line_offset is the 0-indexed line number of #cutile-lsp:start in the original file.
    After adjustment, lineno/end_lineno are in original file coordinates (1-indexed).
    """
    adjusted = []
    for hint in hints:
        h = dict(hint)
        h["lineno"] = hint.get("lineno", 1) + line_offset
        if "end_lineno" in hint and hint["end_lineno"] is not None:
            h["end_lineno"] = hint["end_lineno"] + line_offset
        adjusted.append(h)
    return adjusted
