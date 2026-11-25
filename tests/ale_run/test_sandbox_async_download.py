import asyncio
import base64
import json
from types import SimpleNamespace
from typing import ClassVar

import pytest

from ale_run.base_interface import sandbox as module


class _Response:
    def __init__(self, payloads):
        self.payloads = payloads

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        for payload in self.payloads:
            await asyncio.sleep(0)
            yield payload


class _Client:
    responses: ClassVar[list[_Response]] = []

    def __init__(self, *args, **kwargs):
        del args, kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, *args, **kwargs):
        del args, kwargs
        return self.responses.pop(0)


def _sandbox():
    return SimpleNamespace(
        is_linux=True,
        endpoint="http://sandbox",
        cmd_url="http://sandbox/cmd",
    )


@pytest.mark.asyncio
async def test_async_download_commits_only_complete_file(tmp_path, monkeypatch):
    content = b"answer"
    event = (
        "data: "
        + json.dumps({"success": True, "content_b64": base64.b64encode(content).decode()})
        + "\n\n"
    ).encode()
    _Client.responses = [_Response([event])]
    monkeypatch.setattr(module.httpx, "AsyncClient", _Client)
    target = tmp_path / "result.bin"
    assert await module.SandboxHandle.download_to_local(_sandbox(), "/out", str(target), timeout=1)
    assert target.read_bytes() == content
    assert not list(tmp_path.glob(".ale-download-*"))


@pytest.mark.asyncio
async def test_async_download_cancellation_removes_temp_and_preserves_target(tmp_path, monkeypatch):
    target = tmp_path / "result.bin"
    target.write_bytes(b"old")

    class _BlockedResponse(_Response):
        async def aiter_bytes(self):
            await asyncio.Event().wait()
            yield b""

    _Client.responses = [_BlockedResponse([])]
    monkeypatch.setattr(module.httpx, "AsyncClient", _Client)
    task = asyncio.create_task(
        module.SandboxHandle.download_to_local(_sandbox(), "/out", str(target), timeout=10),
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert target.read_bytes() == b"old"
    assert not list(tmp_path.glob(".ale-download-*"))


@pytest.mark.asyncio
async def test_real_http_timeout_closes_stream_and_never_publishes(tmp_path):
    connected, disconnected = asyncio.Event(), asyncio.Event()

    async def serve(reader, writer):
        try:
            await reader.readuntil(b'\r\n\r\n')
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n')
            await writer.drain()
            connected.set()
            await reader.read()
            disconnected.set()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, '127.0.0.1', 0)
    sandbox = _sandbox()
    sandbox.cmd_url = f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}/cmd'
    target = tmp_path / 'result.bin'
    target.write_bytes(b'old')
    try:
        assert not await module.SandboxHandle.download_to_local(
            sandbox, '/out', str(target), timeout=.3,
        )
        assert connected.is_set()
        await asyncio.wait_for(disconnected.wait(), 1)
        assert target.read_bytes() == b'old'
        assert not list(tmp_path.glob('.ale-download-*'))
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_chunked_download_failure_preserves_previous_submission(tmp_path, monkeypatch):
    _Client.responses = [
        _Response([b'data: {"success":true,"content_b64":"YWJj"}\n\n']),
        _Response([b'data: {"success":false}\n\n']),
    ]
    monkeypatch.setattr(module.httpx, 'AsyncClient', _Client)
    monkeypatch.setattr(module, '_DOWNLOAD_CHUNK_BYTES', 3)
    target = tmp_path / 'result.bin'
    target.write_bytes(b'old')
    assert not await module.SandboxHandle.download_to_local(
        _sandbox(), '/out', str(target), timeout=1,
    )
    assert target.read_bytes() == b'old'
    assert not list(tmp_path.glob('.ale-download-*'))
