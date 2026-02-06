import re

from .loggings import get_logger

logger = get_logger(__name__)


def check_cutile_lsp_enabled(text: str) -> bool:
    """
    Check if the first line comment (after trim) is '#cutile-lsp:on'.

    After trimming (removing leading/trailing whitespace), if the first
    line is a comment starting with '#cutile-lsp:on' (whitespace agnostic
    between 'cutile-lsp' and ':'), return True.
    Otherwise return False.

    Args:
        text: Input string to check

    Returns:
        True if first line comment matches the pattern, False otherwise
    """
    if not text:
        return False

    trimmed = text.strip()
    if not trimmed:
        return False

    first_line = trimmed.split("\n")[0].strip()

    # Robust parsing: allow arbitrary whitespace between 'cutile-lsp' and ':'
    enabled_pattern = re.compile(r"^#\s*cutile-lsp\s*:\s*on\s*$")
    if enabled_pattern.match(first_line):
        return True

    return False


def extract_code(text: str) -> tuple[str, int]:
    """
    Extract code between # cutile-lsp: start and # cutile-lsp: end comments.

    - Robust parsing, whitespace agnostic
    - If start marker not found, default to beginning
    - If end marker not found, default to end
    - If last start appears after first end, return empty string

    Args:
        text: Input string containing code with markers

    Returns:
        Tuple of (extracted_code, start_line_offset) where start_line_offset
        is the 0-indexed line number of the start marker in the original file
        (or 0 if no start marker found)
    """
    if not text:
        return "", 0

    lines = text.split("\n")

    start_pattern = re.compile(r"^\s*#\s*cutile-lsp\s*:\s*start\s*$")
    end_pattern = re.compile(r"^\s*#\s*cutile-lsp\s*:\s*end\s*$")

    start_line = -1
    end_line = -1

    # Find the last occurrence of start marker
    for i, line in enumerate(lines):
        if start_pattern.match(line):
            start_line = i

    # Find the first occurrence of end marker
    for i, line in enumerate(lines):
        if end_pattern.match(line):
            end_line = i
            break

    # If no start marker found, start from beginning
    if start_line == -1:
        start_line = 0

    # If no end marker found, go to end
    if end_line == -1:
        end_line = len(lines) - 1

    # If start appears after end, return empty string
    if start_line > end_line:
        return "", start_line

    logger.info(f"Start line: {start_line}, End line: {end_line}")

    # Extract lines including start and end markers
    extracted_lines = lines[start_line : end_line + 1]
    return "\n".join(extracted_lines), start_line


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract code between cutile-lsp markers")
    parser.add_argument("path", nargs="?", help="Path to file to process (optional)")
    args = parser.parse_args()

    if args.path:
        with open(args.path, "r") as f:
            code = f.read()
    else:
        code = """
before start
#cutile-lsp:        start
content
#     cutile-lsp   :end
after end
"""

    extracted, offset = extract_code(code)
    print(f"Extracted code (offset: {offset}):")
    print(extracted)
