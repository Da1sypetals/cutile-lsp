"""
Asynchronous document processor for parallel LSP analysis.

Each document change spawns a new background task. All tasks run in parallel.
Only the latest task's results are cached and pushed to the client.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from lsprotocol import types
from pygls.lsp.server import LanguageServer

from cutile_lsp.exec_lsp_code import adjust_hints_line_offset, execute_lsp_code
from cutile_lsp.loggings import get_logger
from cutile_lsp.lsp_pipeline.parse_args import EarlyDiagnostics
from cutile_lsp.lsp_pipeline.pipeline import NotEnabledError, pipeline

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Manages asynchronous processing of LSP documents.

    - Each document change gets a monotonically increasing version number
    - All versions are processed in parallel via ThreadPoolExecutor
    - Only the latest version's results are cached and published
    """

    def __init__(self, max_workers: int = 4):
        # Thread pool for background tasks
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="lsp_worker")

        # Version tracking: uri -> int (latest version assigned)
        self._uri_versions: dict[str, int] = {}
        self._version_lock = threading.Lock()

        # Cache tracking: uri -> int (version currently in cache)
        self._cache_versions: dict[str, int] = {}
        self._hints_cache: dict[str, list[dict]] = {}
        self._diagnostics_cache: dict[str, list[types.Diagnostic]] = {}
        self._cache_lock = threading.Lock()

        # Reference to LSP server for publishing results
        self._server: LanguageServer | None = None

    def set_server(self, server: LanguageServer) -> None:
        """Set the LSP server instance for publishing results."""
        self._server = server

    def submit_document(self, uri: str, file_path: Path) -> None:
        """
        Submit a document for asynchronous processing.

        Called on every document open/change/save event.
        Spawns a new background task immediately without waiting for previous tasks.
        """
        # Assign new version number under lock
        with self._version_lock:
            version = self._uri_versions.get(uri, 0) + 1
            self._uri_versions[uri] = version

        logger.info(f"Submitting document {uri} with version {version}")

        # Submit to thread pool
        self._executor.submit(self._process_task, uri, file_path, version)

    def _process_task(self, uri: str, file_path: Path, version: int) -> None:
        """
        Background task: run pipeline and publish results.

        This runs in a worker thread. After completion, only publishes results
        if this version is the latest (or newer than cached).
        """
        try:
            # Run the full pipeline
            lsp_code, early_diagnostics, line_offset = pipeline(file_path, self._get_uri_hash(uri))

            # Convert early diagnostics
            diagnostics: list[types.Diagnostic] = []
            for diag in early_diagnostics:
                lsp_diag = self._create_lsp_diagnostic_from_early(diag)
                diagnostics.append(lsp_diag)

            # Execute lsp_code (subprocess call, may take time)
            hints, json_diagnostics = execute_lsp_code(lsp_code, uri)

            # Adjust hint line numbers
            hints = adjust_hints_line_offset(hints, line_offset)

            # Convert JSON diagnostics
            for diag in json_diagnostics:
                lsp_diag = self._create_lsp_diagnostic_from_json(diag, line_offset)
                diagnostics.append(lsp_diag)

            # Try to update cache and publish
            self._try_update_cache(uri, version, hints, diagnostics)

        except NotEnabledError:
            logger.info(f"Not a cutile-lsp enabled file: {file_path}")
            # Clear cache for this URI
            with self._cache_lock:
                self._cache_versions.pop(uri, None)
                self._hints_cache.pop(uri, None)
                self._diagnostics_cache.pop(uri, None)
            # Publish empty diagnostics
            self._publish_empty(uri)

        except Exception as e:
            logger.error(f"Error processing document {uri} v{version}: {e}", exc_info=True)
            # Publish error as diagnostic
            error_diag = self._create_error_diagnostic(str(e))
            self._try_update_cache(uri, version, [], [error_diag])

    def _try_update_cache(
        self,
        uri: str,
        version: int,
        hints: list[dict],
        diagnostics: list[types.Diagnostic],
    ) -> None:
        """
        Update cache only if this version is newer than cached.

        Lock-protected to prevent race conditions where older versions
        overwrite newer results.
        """
        should_publish = False

        with self._cache_lock:
            cached_version = self._cache_versions.get(uri, 0)

            if version >= cached_version:
                # This is the latest (or equally new), update cache
                self._cache_versions[uri] = version
                self._hints_cache[uri] = hints
                self._diagnostics_cache[uri] = diagnostics
                should_publish = True
                logger.info(f"Updated cache for {uri} to version {version}")
            else:
                logger.debug(f"Discarding stale result for {uri} v{version}, cache is v{cached_version}")

        # Publish outside the lock to avoid holding lock during I/O
        if should_publish:
            self._publish_results(uri, hints, diagnostics)

    def _publish_results(
        self,
        uri: str,
        hints: list[dict],
        diagnostics: list[types.Diagnostic],
    ) -> None:
        """Publish diagnostics and notify client to refresh hints."""
        if self._server is None:
            logger.warning("LSP server not set, cannot publish results")
            return

        try:
            # Publish diagnostics
            self._server.text_document_publish_diagnostics(
                types.PublishDiagnosticsParams(uri=uri, diagnostics=diagnostics)
            )

            # Notify client to refresh inlay hints
            self._server.workspace_inlay_hint_refresh(None)

        except Exception as e:
            logger.error(f"Error publishing results for {uri}: {e}")

    def _publish_empty(self, uri: str) -> None:
        """Publish empty diagnostics to clear any existing ones."""
        if self._server is None:
            return

        try:
            self._server.text_document_publish_diagnostics(
                types.PublishDiagnosticsParams(uri=uri, diagnostics=[])
            )
            self._server.workspace_inlay_hint_refresh(None)
        except Exception as e:
            logger.error(f"Error publishing empty results for {uri}: {e}")

    def get_hints(self, uri: str) -> list[dict]:
        """Get cached hints for a URI (called by inlay hint handler)."""
        with self._cache_lock:
            return self._hints_cache.get(uri, [])

    def shutdown(self) -> None:
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=True)

    @staticmethod
    def _get_uri_hash(uri: str) -> str:
        """Generate a hash from URI for directory naming."""
        import hashlib

        return hashlib.sha256(uri.encode()).hexdigest()[:16]

    @staticmethod
    def _create_lsp_diagnostic_from_early(early_diag: EarlyDiagnostics) -> types.Diagnostic:
        """Convert early diagnostics to LSP Diagnostic."""
        return types.Diagnostic(
            range=types.Range(
                start=types.Position(
                    line=max(0, early_diag.line - 1),
                    character=early_diag.col,
                ),
                end=types.Position(
                    line=max(0, early_diag.line - 1),
                    character=early_diag.end_col,
                ),
            ),
            message=early_diag.message,
            severity=types.DiagnosticSeverity.Error,
            source="cutile-lsp",
        )

    @staticmethod
    def _create_lsp_diagnostic_from_json(diag: dict, line_offset: int) -> types.Diagnostic:
        """Convert JSON diagnostic to LSP Diagnostic."""
        line = diag.get("line", 1) + line_offset - 1
        col = diag.get("col", 0)
        last_line = diag.get("last_line")
        end_col = diag.get("end_col")

        if last_line is not None and end_col is not None:
            end_line = last_line + line_offset - 1
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
            source="cutile-lsp",
        )

    @staticmethod
    def _create_error_diagnostic(message: str) -> types.Diagnostic:
        """Create a diagnostic for internal errors."""
        from cutile_lsp.constants import END_OF_LINE

        return types.Diagnostic(
            range=types.Range(
                start=types.Position(line=0, character=0),
                end=types.Position(line=0, character=END_OF_LINE),
            ),
            message=f"cutile-lsp internal error: {message}",
            severity=types.DiagnosticSeverity.Error,
            source="cutile-lsp",
        )


# Global singleton instance
_processor: DocumentProcessor | None = None


def init_processor(max_workers: int = 4) -> DocumentProcessor:
    """Initialize the global document processor."""
    global _processor
    _processor = DocumentProcessor(max_workers=max_workers)
    return _processor


def get_processor() -> DocumentProcessor:
    """Get the global document processor instance."""
    if _processor is None:
        raise RuntimeError("Document processor not initialized. Call init_processor() first.")
    return _processor


def submit_document(uri: str, file_path: Path) -> None:
    """Convenience function to submit a document to the global processor."""
    get_processor().submit_document(uri, file_path)


def get_hints(uri: str) -> list[dict]:
    """Convenience function to get hints from the global processor."""
    return get_processor().get_hints(uri)


def set_server(server: LanguageServer) -> None:
    """Set the LSP server on the global processor."""
    get_processor().set_server(server)


def shutdown() -> None:
    """Shutdown the global processor."""
    if _processor is not None:
        _processor.shutdown()
