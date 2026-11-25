"""Shared control I/O isolated from the supervisor's process-reaping loop."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from pathlib import Path


class GuestControl:
    def __init__(self, directory: str | None, token: str):
        self.directory = Path(directory) if directory else None
        self.token = token
        self.stop_requested = threading.Event()
        self.finished = threading.Event()
        self.published = threading.Event()
        self.lock = threading.Lock()
        self.snapshot = None
        self.terminal = None
        self.thread = None
        if self.directory is not None:
            self.thread = threading.Thread(target=self._serve, daemon=True, name="ale-control")
            self.thread.start()

    def tick(self, sequence: int, state: str, worker_pid: int | None) -> None:
        with self.lock:
            self.snapshot = {
                "run_token": self.token,
                "sequence": sequence,
                "state": state,
                "supervisor_pid": os.getpid(),
                "worker_pid": worker_pid,
            }

    def finish(self, stopped: dict, result: dict) -> None:
        with self.lock:
            self.terminal = (dict(stopped), {**result, "run_token": self.token})
        self.finished.set()
        if self.thread is not None:
            self.published.wait(2)

    def _write(self, name: str, value: dict) -> None:
        target = self.directory / name
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(value))
        temporary.replace(target)

    def _diagnostics(self) -> dict:
        report = {"run_token": self.token, "observed_at_guest": time.time()}
        for path in (
            "/proc/meminfo",
            "/proc/vmstat",
            "/proc/loadavg",
            "/proc/pressure/cpu",
            "/proc/pressure/memory",
            "/proc/pressure/io",
            "/proc/net/dev",
            "/proc/net/tcp",
        ):
            try:
                with Path(path).open() as stream:
                    report[path] = stream.read(16384)
            except OSError as exc:
                report[path] = {"error": str(exc)}
        # Spawning diagnostic children here would race the supervisor's subreaper.
        for path in ("/var/log/kern.log",):
            try:
                with Path(path).open("rb") as stream:
                    stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, stream.tell() - 16384))
                    report[path] = stream.read(16384).decode(errors="replace")
            except OSError as exc:
                report[path] = {"error": str(exc)}
        try:
            with urllib.request.urlopen("http://127.0.0.1:5000/status", timeout=2) as response:
                report["cua_status"] = {
                    "http_status": response.status,
                    "body": response.read(4096).decode(errors="replace"),
                }
        except OSError as exc:
            report["cua_status"] = {"error": str(exc)}
        return report

    def _serve(self) -> None:
        next_diagnostics = 0.0
        while True:
            try:
                request = self.directory / "stop.request"
                try:
                    if request.read_text() == self.token:
                        self.stop_requested.set()
                except FileNotFoundError:
                    pass
                with self.lock:
                    snapshot, terminal = self.snapshot, self.terminal
                if snapshot is not None:
                    self._write("heartbeat.json", snapshot)
                if terminal is not None:
                    stopped, result = terminal
                    self._write("stopped.json", stopped)
                    self._write("result.json", result)
                    self._write("done.json", {"run_token": self.token})
                    self.published.set()
                    return
                if time.monotonic() >= next_diagnostics:
                    self._write("diagnostics.json", self._diagnostics())
                    next_diagnostics = time.monotonic() + 60
            except OSError:
                # A hung/unavailable share must never block containment or cleanup.
                pass
            if self.finished.wait(1):
                if self.published.is_set():
                    return
                time.sleep(0.2)
