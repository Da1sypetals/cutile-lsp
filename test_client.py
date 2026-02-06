"""
Test client that sends LSP requests to the cutile-lsp server using kernel.py as sample input.

Usage:
    python test_client.py
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path


def encode_message(obj: dict) -> bytes:
    """Encode a JSON-RPC message with Content-Length header."""
    body = json.dumps(obj).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def read_message(stream) -> dict | None:
    """Read a JSON-RPC message from a stream."""
    # Read headers
    headers = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        line = line.strip()
        if line == b"":
            break
        if b":" in line:
            key, value = line.split(b":", 1)
            headers[key.strip().lower()] = value.strip()

    content_length = int(headers.get(b"content-length", 0))
    if content_length == 0:
        return None

    body = stream.read(content_length)
    return json.loads(body.decode("utf-8"))


def main():
    kernel_path = Path(__file__).parent / "kernel.py"
    assert kernel_path.exists(), f"kernel.py not found at {kernel_path}"

    kernel_uri = kernel_path.resolve().as_uri()
    kernel_text = kernel_path.read_text()

    print(f"[Client] kernel URI: {kernel_uri}")
    print("[Client] Starting LSP server...")

    proc = subprocess.Popen(
        [sys.executable, "-m", "cutile_lsp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(Path(__file__).parent / "src"),
    )

    responses = []
    notifications = []

    def read_stdout():
        while True:
            msg = read_message(proc.stdout)
            if msg is None:
                break
            if "id" in msg:
                responses.append(msg)
            else:
                notifications.append(msg)

    reader_thread = threading.Thread(target=read_stdout, daemon=True)
    reader_thread.start()

    def read_stderr():
        for line in proc.stderr:
            print(f"[Server STDERR] {line.decode('utf-8', errors='replace').rstrip()}")

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()

    msg_id = 0

    def send_request(method: str, params: dict) -> int:
        nonlocal msg_id
        msg_id += 1
        msg = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        data = encode_message(msg)
        proc.stdin.write(data)
        proc.stdin.flush()
        print(f"[Client] Sent request #{msg_id}: {method}")
        return msg_id

    def send_notification(method: str, params: dict):
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        data = encode_message(msg)
        proc.stdin.write(data)
        proc.stdin.flush()
        print(f"[Client] Sent notification: {method}")

    def wait_for_response(req_id: int, timeout: float = 120.0) -> dict | None:
        start = time.time()
        while time.time() - start < timeout:
            for resp in responses:
                if resp.get("id") == req_id:
                    responses.remove(resp)
                    return resp
            time.sleep(0.1)
        return None

    # Step 1: Initialize
    init_id = send_request(
        "initialize",
        {
            "processId": None,
            "capabilities": {
                "textDocument": {
                    "inlayHint": {
                        "dynamicRegistration": True,
                    },
                    "publishDiagnostics": {
                        "relatedInformation": True,
                    },
                },
                "workspace": {
                    "inlayHint": {
                        "refreshSupport": True,
                    },
                },
            },
            "rootUri": Path(__file__).parent.resolve().as_uri(),
        },
    )

    init_resp = wait_for_response(init_id, timeout=10)
    print(f"[Client] Initialize response: {json.dumps(init_resp, indent=2)}")

    # Step 2: Send initialized notification
    send_notification("initialized", {})

    # Step 3: Open the document
    send_notification(
        "textDocument/didOpen",
        {
            "textDocument": {
                "uri": kernel_uri,
                "languageId": "python",
                "version": 1,
                "text": kernel_text,
            }
        },
    )

    # Step 4: Wait for diagnostics to be published (the server processes the document)
    print("[Client] Waiting for server to process document...")
    time.sleep(5)

    # Check for any published diagnostics notifications
    print(f"\n[Client] Received {len(notifications)} notifications so far:")
    for notif in notifications:
        method = notif.get("method", "unknown")
        if method == "textDocument/publishDiagnostics":
            params = notif.get("params", {})
            diags = params.get("diagnostics", [])
            print(f"  Diagnostics for {params.get('uri', '?')}: {len(diags)} diagnostic(s)")
            for d in diags:
                rng = d.get("range", {})
                start = rng.get("start", {})
                end = rng.get("end", {})
                print(
                    f"    [{start.get('line')}:{start.get('character')}-{end.get('line')}:{end.get('character')}] "
                    f"severity={d.get('severity')} message={d.get('message', '')[:100]}"
                )
        elif method == "workspace/inlayHint/refresh":
            print("  InlayHint refresh request received")
        else:
            print(f"  {method}: {json.dumps(notif.get('params', {}))[:200]}")

    # Step 5: Request inlay hints
    total_lines = len(kernel_text.splitlines())
    hint_id = send_request(
        "textDocument/inlayHint",
        {
            "textDocument": {"uri": kernel_uri},
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": total_lines, "character": 0},
            },
        },
    )

    # Wait longer for hint response since the server may still be processing
    print("[Client] Waiting for inlay hints response (may take a while)...")
    hint_resp = wait_for_response(hint_id, timeout=120)

    if hint_resp:
        if "result" in hint_resp:
            hints = hint_resp["result"]
            if hints is None:
                print("\n[Client] InlayHints response: null (no hints)")
            else:
                print(f"\n[Client] InlayHints response: {len(hints)} hint(s)")
                for h in hints[:30]:  # Print first 30
                    pos = h.get("position", {})
                    print(
                        f"  Line {pos.get('line')}, Col {pos.get('character')}: "
                        f"label={h.get('label', '')} kind={h.get('kind')}"
                    )
        elif "error" in hint_resp:
            print(f"\n[Client] InlayHints error: {json.dumps(hint_resp['error'], indent=2)}")
    else:
        print("\n[Client] No inlay hints response received (timeout)")

    # Wait a bit more for any late notifications
    time.sleep(3)

    # Print final notifications summary
    diag_notifs = [n for n in notifications if n.get("method") == "textDocument/publishDiagnostics"]
    hint_refreshes = [n for n in notifications if n.get("method") == "workspace/inlayHint/refresh"]
    print("\n[Client] Summary:")
    print(f"  Total notifications: {len(notifications)}")
    print(f"  Diagnostic notifications: {len(diag_notifs)}")
    print(f"  InlayHint refresh notifications: {len(hint_refreshes)}")

    # Step 6: Shutdown
    shutdown_id = send_request("shutdown", None)
    shutdown_resp = wait_for_response(shutdown_id, timeout=5)
    print(
        f"[Client] Shutdown response: {json.dumps(shutdown_resp, indent=2) if shutdown_resp else 'timeout'}"
    )

    send_notification("exit", None)
    proc.wait(timeout=5)
    print("[Client] Server exited.")


if __name__ == "__main__":
    main()
