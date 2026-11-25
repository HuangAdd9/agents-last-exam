"""Local transport regressions. No cloud jobs or model calls."""

import asyncio
import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ale_run.base_interface import sandbox as wire
from ale_run.executors import sandbox as executor


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            mode = body.get("mode", "drip")
            if mode == "headers":
                time.sleep(0.5)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if mode in {"result", "result_keepalive"}:
                    self.wfile.write(b': keepalive\ndata: {"success":true,"stdout":"ok"}\n\n')
                    self.wfile.flush()
                    if mode == "result":
                        return
                    for _ in range(200):
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        time.sleep(0.01)
                    return
                self.wfile.write(b'data: {"content_b64":"')
                for _ in range(300):
                    self.wfile.write(b"a" * 512)
                    self.wfile.flush()
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield SimpleNamespace(cmd_url=f"http://127.0.0.1:{httpd.server_port}/cmd")
    httpd.shutdown()
    httpd.server_close()
    thread.join()


@pytest.mark.parametrize("mode", ["drip", "headers"])
def test_total_timeout_stops_active_or_delayed_response(server, mode):
    before = time.monotonic()
    assert wire._post_cmd(server, {"mode": mode}, timeout=0.2) is None
    assert time.monotonic() - before < 0.8


def test_sse_first_result(server):
    assert wire._post_cmd(server, {"mode": "result"}, timeout=1) == {
        "success": True,
        "stdout": "ok",
    }


def test_small_result_does_not_wait_for_stream_to_close(server):
    before = time.monotonic()
    assert wire._post_cmd(server, {"mode": "result_keepalive"}, timeout=1)["success"]
    assert time.monotonic() - before < 0.8


def test_retries_share_one_budget():
    def delayed(*args, timeout, **kwargs):
        time.sleep(timeout)

    before = time.monotonic()
    with patch.object(wire, "_post_cmd", side_effect=delayed) as post:
        result = wire._run_remote_sync(SimpleNamespace(), "true", 0.05)
    assert result.returncode == -1
    assert post.call_count == 1
    assert time.monotonic() - before < 0.15


def test_failed_download_preserves_existing_file(tmp_path):
    target = tmp_path / "transcript.jsonl"
    target.write_bytes(b'{"saved":true}\n')
    with patch.object(wire, "_post_cmd", return_value=None):
        assert not wire._download_to_local_sync(SimpleNamespace(), "/trace", str(target), 0.1)
    assert target.read_bytes() == b'{"saved":true}\n'
    assert not list(tmp_path.glob(".ale-download-*"))


def test_binary_download_uses_bounded_chunks_and_atomic_commit(tmp_path):
    content = b"\x00\xff" * 140000
    lengths = []

    def reply(sandbox, body, **kwargs):
        p = body["params"]
        lengths.append(p["length"])
        block = content[p["offset"] : p["offset"] + p["length"]]
        return {"success": True, "content_b64": base64.b64encode(block).decode()}

    target = tmp_path / "out.bin"
    with patch.object(wire, "_post_cmd", side_effect=reply):
        assert wire._download_to_local_sync(SimpleNamespace(), "/out.bin", str(target), 1)
    assert target.read_bytes() == content
    assert len(lengths) == 2 and max(lengths) == 256 * 1024


def test_gather_deadline_includes_blocked_listing(tmp_path):
    async def listing(path):
        await asyncio.sleep(5)

    handle = SimpleNamespace(list_dir=listing)
    ex = executor.SandboxExecutor(sandbox=handle, work_dir="/work", config=None)
    with patch.object(executor, "_GATHER_DEADLINE_S", 0.03):
        report = asyncio.run(ex.gather_dir(src="/work", dst=tmp_path))
    assert "total deadline" in report.error


def test_gather_retry_uses_remaining_budget(tmp_path):
    observed = []

    async def download(remote, local, timeout):
        observed.append(timeout)
        return False

    deadline = time.monotonic() + 0.02
    assert not asyncio.run(
        executor._download_with_retry(
            SimpleNamespace(download_to_local=download),
            "/trace",
            tmp_path / "trace",
            deadline=deadline,
        )
    )
    assert max(observed) <= 0.02
