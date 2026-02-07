from pathlib import Path

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from cutile_lsp import async_processor
from cutile_lsp.loggings import get_logger

logger = get_logger(__name__)

LSP_NAME = "cutile-lsp"

server = LanguageServer(LSP_NAME, "0.1.0")

# Initialize async processor for parallel document analysis
async_processor.init_processor(max_workers=4)
async_processor.set_server(server)


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
        key = (line, col_end, ty, name)
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
    hints = async_processor.get_hints(uri)
    start_line = params.range.start.line
    end_line = params.range.end.line

    # Filter hints within the requested range and convert to LSP InlayHint objects
    filtered = [h for h in hints if start_line <= (h.get("lineno", 1) - 1) <= end_line]
    return create_lsp_inlay_hints_from_json_hints(filtered)


@server.feature(types.TEXT_DOCUMENT_DID_OPEN)
def did_open(ls: LanguageServer, params: types.DidOpenTextDocumentParams):
    """Handle document open notification."""
    _submit_document_async(params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls: LanguageServer, params: types.DidChangeTextDocumentParams):
    """Handle document change notification."""
    _submit_document_async(params.text_document.uri)


@server.feature(types.TEXT_DOCUMENT_DID_SAVE)
def did_save(ls: LanguageServer, params: types.DidSaveTextDocumentParams):
    """Handle document save notification."""
    _submit_document_async(params.text_document.uri)


def _submit_document_async(uri: str):
    """Submit document for async processing."""
    if not uri.startswith("file://"):
        return
    file_path = Path(uri.replace("file://", ""))
    async_processor.submit_document(uri, file_path)


def main():
    """Start the LSP server."""
    try:
        server.start_io()
    finally:
        async_processor.shutdown()


if __name__ == "__main__":
    main()
