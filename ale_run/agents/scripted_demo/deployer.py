"""Zero-cost scripted GUI smoke agent for ALE demo/hello."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import time
from pathlib import Path
from typing import Any, ClassVar

from ale_run.base_interface import (
    AgentRunResult,
    BaseAgentDeployer,
    TrajectoryBuilder,
)

from .config import ScriptedDemoConfig

logger = logging.getLogger(__name__)

_REPORT_NAME = "scripted_demo_report.json"
_RESULT_PATH_RE = re.compile(r"(?P<output_dir>/[^\s`'\"]+/output)/result\.txt\b")


class ScriptedDemoDeployer(BaseAgentDeployer):
    """Deterministic no-LLM agent for demo/hello."""

    default_executor: ClassVar[str] = "local"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"local"})
    hot_artifacts: ClassVar[tuple[str, ...]] = (_REPORT_NAME,)

    @property
    def version(self) -> str | None:
        return "scripted-demo-0.1.0"

    async def install(self) -> None:
        # Import here so dry-run remains lightweight.
        from cua_bench.computers.remote import RemoteDesktopSession  # noqa: F401

        Path(self.executor.work_dir).mkdir(parents=True, exist_ok=True)

    async def launch(self, prompt: str) -> AgentRunResult:
        from cua_bench.computers.remote import RemoteDesktopSession

        cfg: ScriptedDemoConfig = self.config  # type: ignore[assignment]
        work_dir = Path(self.executor.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()

        report: dict[str, Any] = {
            "status": "starting",
            "order_id": None,
            "output_path": None,
            "generated_code": None,
            "errors": [],
        }

        try:
            matches = list(_RESULT_PATH_RE.finditer(prompt))
            output_dirs = {match.group("output_dir") for match in matches}
            if len(output_dirs) != 1:
                raise RuntimeError(
                    "Expected exactly one demo/hello output/result.txt path in task prompt"
                )

            output_dir = output_dirs.pop()
            base_dir = str(Path(output_dir).parent)

            order_path = f"{base_dir}/input/order.json"
            output_path = f"{output_dir}/result.txt"

            report["output_path"] = output_path

            sb = self.executor.sandbox

            session = RemoteDesktopSession(
                api_url=sb.endpoint,
                os_type=sb.os,
                ephemeral=False,
                headless=True,
            )

            ready = await session.wait_until_ready(timeout=cfg.connect_timeout_s)
            if not ready:
                raise RuntimeError("VM cua-server did not become ready")

            r = await session.run_command(
                f"cat {shlex.quote(order_path)}",
                check=False,
            )
            if r.get("return_code") != 0:
                raise RuntimeError(f"Could not read order.json: {r.get('stderr')}")

            order_data = json.loads(r.get("stdout") or "{}")
            order_id = str(order_data["order_id"])
            report["order_id"] = order_id

            chrome_profile = "/tmp/ale-scripted-demo-chrome"
            app_url = f"file://{base_dir}/input/app.html"

            launch_py = (
                "import os,shutil,subprocess;"
                f"p={chrome_profile!r};"
                "shutil.rmtree(p,ignore_errors=True);"
                "env=os.environ.copy();env['DISPLAY']=':0';"
                "log=open('/tmp/ale-scripted-demo-chrome.log','ab');"
                "subprocess.Popen(["
                "'/usr/bin/google-chrome',"
                f"'--user-data-dir={chrome_profile}',"
                "'--no-first-run',"
                "'--no-default-browser-check',"
                "'--window-position=0,0',"
                "'--window-size=1920,1080',"
                "'--new-window',"
                f"{app_url!r}"
                "],env=env,stdin=subprocess.DEVNULL,"
                "stdout=log,stderr=log,start_new_session=True,close_fds=True)"
            )

            r = await session.run_command(
                "python3 -c " + shlex.quote(launch_py),
                check=False,
            )
            if r.get("return_code") != 0:
                raise RuntimeError(f"Could not launch Chrome: {r.get('stderr')}")

            # Query X11 directly. wmctrl depends on a fully EWMH-compliant
            # window manager and can return an empty list in Qizhi's PRoot +
            # lightweight-XFCE runtime even while Chrome is rendering.
            chrome_window_id = None

            for _ in range(30):
                r = await session.run_command(
                    "DISPLAY=:0 xdotool search --onlyvisible "
                    "--name 'Acme Code Generator' 2>/dev/null | tail -n1",
                    check=False,
                )
                wid = (r.get("stdout") or "").strip()
                if wid:
                    chrome_window_id = wid
                    break
                await asyncio.sleep(1)

            if not chrome_window_id:
                raise RuntimeError("Chrome task window was not found")

            r = await session.run_command(
                "DISPLAY=:0 xdotool "
                f"windowmap {shlex.quote(chrome_window_id)} "
                f"windowsize {shlex.quote(chrome_window_id)} 1920 1080 "
                f"windowmove {shlex.quote(chrome_window_id)} 0 0 "
                f"windowraise {shlex.quote(chrome_window_id)} "
                f"windowfocus --sync {shlex.quote(chrome_window_id)}",
                check=False,
            )
            if r.get("return_code") != 0:
                raise RuntimeError("Could not activate Chrome task window")

            await asyncio.sleep(0.5)

            # These coordinates are the fixed 1920x1080 ALE Ubuntu layout.
            # Mouse interaction avoids Chrome toolbar focus from a fresh profile.
            await session.click(540, 322)
            await session.hotkey(["ctrl", "a"])
            await session.type(order_id)

            await session.click(333, 382)

            await asyncio.sleep(1)

            # Triple-click the rendered token and read X11's primary selection.
            # A horizontal drag can cross DOM nodes when font metrics differ,
            # while Ctrl+C does not populate the clipboard reliably through
            # Qizhi's PRoot/XTest stack. This exact X11 action works in both.
            r = await session.run_command(
                "DISPLAY=:0 xdotool mousemove 500 454 "
                "click --repeat 3 --delay 80 1; "
                "sleep 0.2; "
                "DISPLAY=:0 xclip -selection primary -o 2>/dev/null",
                check=False,
            )
            if r.get("return_code") != 0:
                raise RuntimeError(f"Could not read GUI selection: {r.get('stderr')}")

            selected_text = r.get("stdout") or ""

            match = re.search(r"\bCONF-[^\s]+-\d{4}\b", selected_text)
            if not match:
                raise RuntimeError("Generated confirmation code not found in GUI clipboard")

            code = match.group(0)
            report["generated_code"] = code

            r = await session.run_command(
                f"mkdir -p {shlex.quote(output_dir)}",
                check=False,
            )
            if r.get("return_code") != 0:
                raise RuntimeError(f"Could not create output directory: {r.get('stderr')}")

            await session.write_file(output_path, code + "\n")

            report["status"] = "completed"
            self._write_report(work_dir, report)

            logger.info("scripted_demo: %s -> %s", order_id, code)

            return self._result("completed", t0, work_dir)

        except Exception as exc:
            report["status"] = "failed"
            report["errors"].append(f"{type(exc).__name__}: {exc}")
            self._write_report(work_dir, report)

            logger.exception("scripted_demo failed")

            return self._result("failed", t0, work_dir, error=str(exc))

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: ScriptedDemoConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        report_path = work_dir / _REPORT_NAME

        if not report_path.exists():
            builder.add_step(
                source="system",
                message="scripted_demo: report missing",
                extra={"run_status": run_result.status},
            )
            return

        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            builder.add_step(
                source="system",
                message=f"scripted_demo: unreadable report: {exc}",
            )
            return

        builder.add_step(
            source="system",
            message=(
                "scripted_demo: completed deterministic GUI workflow"
                if report.get("status") == "completed"
                else "scripted_demo: GUI workflow failed"
            ),
            extra=report,
        )

        builder.trajectory.extra.setdefault("scripted_demo", {}).update(report)

    def _result(
        self,
        status: str,
        t0: float,
        work_dir: Path,
        *,
        error: str | None = None,
    ) -> AgentRunResult:
        return AgentRunResult(
            status=status,
            duration_s=time.monotonic() - t0,
            transcript_path=str(work_dir / _REPORT_NAME),
            error=error,
        )

    @staticmethod
    def _write_report(
        work_dir: Path,
        report: dict[str, Any],
    ) -> None:
        (work_dir / _REPORT_NAME).write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
