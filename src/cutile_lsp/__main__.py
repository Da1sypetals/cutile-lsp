import hashlib
import traceback
from pathlib import Path

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from cutile_lsp.constants import END_OF_LINE
from cutile_lsp.exec_lsp_code import adjust_hints_line_offset, execute_lsp_code
from cutile_lsp.lsp_pipeline.loggings import get_logger
from cutile_lsp.lsp_pipeline.parse_args import EarlyDiagnostics
from cutile_lsp.lsp_pipeline.pipeline import NotEnabledError, pipeline

logger = get_logger(__name__)

LSP_NAME = "cutile-lsp"

server = LanguageServer(LSP_NAME, "0.1.0")

# Cache for storing hints and diagnostics per URI
hints_cache: dict[str, list[dict]] = {}
diagnostics_cache: dict[str, list[types.Diagnostic]] = {}


def get_uri_hash(uri: str) -> str:
    """Generate a hash from URI for directory naming."""
    return hashlib.sha256(uri.encode()).hexdigest()[:16]


def create_lsp_diagnostics_from_early_diagnostics(early_diagnostics: EarlyDiagnostics):
    lsp_diag = types.Diagnostic(
        range=types.Range(
            start=types.Position(
                line=max(0, early_diagnostics.line - 1),  # Convert to 0-indexed
                character=early_diagnostics.col,
            ),
            end=types.Position(
                line=max(0, early_diagnostics.line - 1),  # Convert to 0-indexed
                character=early_diagnostics.end_col,
            ),
        ),
        message=early_diagnostics.message,
        severity=types.DiagnosticSeverity.Error,
        source=LSP_NAME,
    )
    return lsp_diag


def create_lsp_diagnostics_from_json_diagnostic(diag: dict, line_offset: int) -> types.Diagnostic:
    """Convert a JSON diagnostic dict to LSP Diagnostic.

    Line numbers in JSON are 1-indexed and relative to active_code.
    line_offset is the 0-indexed line number of #cutile-lsp:start in the original file.
    We need to add line_offset to convert to original file line numbers,
    then subtract 1 to convert to LSP 0-indexed line numbers.
    """
    line = diag.get("line", 1) + line_offset - 1  # active_code 1-indexed -> original 0-indexed
    col = diag.get("col", 0)
    last_line = diag.get("last_line")
    end_col = diag.get("end_col")

    if last_line is not None and end_col is not None:
        end_line = last_line + line_offset - 1  # active_code 1-indexed -> original 0-indexed
        end_character = end_col
    else:
        end_line = line
        end_character = col + 1

    return types.Diagnostic(
        range=types.Range(
            start=types.Position(line=max(0, line), character=col),
            end=types.Position(line=max(0, end_line), character=end_character),
        ),
        message=diag.get("message", ""),
        severity=types.DiagnosticSeverity.Error,
        source=LSP_NAME,
    )


def create_lsp_diagnostics_from_unexpected_error(e: Exception) -> types.Diagnostic:
    diag = types.Diagnostic(
        range=types.Range(
            start=types.Position(line=0, character=0),
            end=types.Position(line=0, character=END_OF_LINE),
        ),
        message=f"{LSP_NAME} internal error: {str(e)}",
        severity=types.DiagnosticSeverity.Error,
        source=LSP_NAME,
    )
    return diag


def create_lsp_inlay_hints_from_json_hints(hints: list[dict]) -> list[types.InlayHint]:
    """
    Convert hint dicts to LSP InlayHint objects.
    Hints should be displayed after the variable name (at col_end).

    Expects hints with lineno already adjusted to original file coordinates (1-indexed).
    """
    result = []
    seen = set()  # Deduplicate hints

    for hint in hints:
        # DO NOT BE ROBUST
        line = hint["lineno"] - 1
        col_end = hint["col_end"]
        ty = hint["ty"]
        name = hint["name"]

        # Create a unique key for deduplication
        key = (line, col_end, ty)
        if key in seen:
            continue
        seen.add(key)

        inlay_hint = types.InlayHint(
            position=types.Position(line=max(0, line), character=col_end),
            label=f": {ty}",
            kind=types.InlayHintKind.Type,
            padding_right=True,
        )
        result.append(inlay_hint)

    return result


@server.feature(types.TEXT_DOCUMENT_INLAY_HINT)
def on_inlay_hint(ls: LanguageServer, params: types.InlayHintParams):
    """Handle inlay hint request from client."""
    uri = params.text_document.uri
    hints = hints_cache.get(uri, [])
    start_line = params.range.start.line
    end_line = params.range.end.line

    # Filter hints within the requested range and convert to LSP InlayHint objects
    filtered = [h for h in hints if start_line <= (h.get("lineno", 1) - 1) <= end_line]
    return create_lsp_inlay_hints_from_json_hints(filtered)


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams):
    """Handle document open notification."""
    _process_document(ls, params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: LanguageServer, params: types.DidChangeTextDocumentParams):
    """Handle document change notification."""
    _process_document(ls, params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: LanguageServer, params: types.DidSaveTextDocumentParams):
    """Handle document save notification."""
    _process_document(ls, params.text_document.uri)


def _process_document(ls: LanguageServer, uri: str):
    """Process document and send early diagnostics and hints to client."""
    # Convert URI to file path
    if not uri.startswith("file://"):
        return

    file_path = Path(uri.replace("file://", ""))
    global hints_cache, diagnostics_cache
    uri_hash = get_uri_hash(uri)

    try:
        lsp_code, early_diagnostics, line_offset = pipeline(file_path, uri_hash)

        # Convert early diagnostics to LSP Diagnostic objects
        diagnostics = []
        for diag in early_diagnostics:
            lsp_diagnostic = create_lsp_diagnostics_from_early_diagnostics(diag)
            diagnostics.append(lsp_diagnostic)

        # Execute lsp_code to get hints and diagnostics from JSON
        hints, json_diagnostics = execute_lsp_code(lsp_code, uri)

        # Adjust hint line numbers by line_offset and store in cache
        hints_cache[uri] = adjust_hints_line_offset(hints, line_offset)

        # Convert JSON diagnostics to LSP diagnostics (with line_offset adjustment)
        for diag in json_diagnostics:
            lsp_diagnostic = create_lsp_diagnostics_from_json_diagnostic(diag, line_offset)
            diagnostics.append(lsp_diagnostic)

        # Store diagnostics in cache
        diagnostics_cache[uri] = diagnostics

        # Send all diagnostics to client
        ls.text_document_publish_diagnostics(types.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics))

        # Notify client that inlay hints have changed
        server.workspace_inlay_hint_refresh(None)

    except NotEnabledError:
        # Not a cutile-lsp enabled file, clear diagnostics and hints
        logger.info("Not a cutile-lsp enabled file, clear diagnostics")
        ls.text_document_publish_diagnostics(types.PublishDiagnosticsParams(uri=uri, diagnostics=[]))
        hints_cache.pop(uri, None)
        diagnostics_cache.pop(uri, None)
        return

    except Exception as e:  # Deal with any unexpected error in LSP
        # log the error stack trace
        logger.error(f"Error processing document: {file_path}\n{traceback.format_exc()}")
        # Send error as diagnostic
        ls.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(
                uri=uri,
                diagnostics=[create_lsp_diagnostics_from_unexpected_error(e)],
            )
        )


def main():
    """Start the LSP server."""
    server.start_io()


if __name__ == "__main__":
    main()
