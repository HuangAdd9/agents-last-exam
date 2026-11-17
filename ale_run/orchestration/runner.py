"""Runner: yaml-described experiment → concurrent run units.

Per-unit isolation:
    - One fresh ``ale.make(task_path)`` per unit (env binds task at ctor)
    - One fresh deployer instance per unit (configs are per-run state)

Concurrency is a resizable limiter. The initial value comes from
``spec.concurrency``; while the run is alive, writing an integer to
``<output_root>/concurrency.control`` raises or lowers the pool. Raising it
wakes queued units, while lowering it never interrupts holders.

Provider is shared across units — real providers (gcloud) acquire
a fresh VM per ``acquire()`` call, so concurrent acquires give concurrent
VMs.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from pathlib import Path
from contextlib import suppress

from .experiment_spec import ExperimentSpec, RunUnit, UnitResult
from .factory import EnvironmentRouter
from .judge_preflight import check_judges
from .concurrency import ResizableLimiter

logger = logging.getLogger(__name__)


class Runner:
    """Owns the provider; produces and executes run units."""

    def __init__(self, spec: ExperimentSpec):
        self._spec = spec
        # Router resolves each unit's snapshot to its provider, building +
        # caching provider instances lazily (keeps --dry-run provider-free).
        self._router = EnvironmentRouter(spec.environment)
        self._output_root = Path(spec.output.root) / spec.name

    @property
    def spec(self) -> ExperimentSpec:
        return self._spec

    @property
    def output_root(self) -> Path:
        return self._output_root

    # ---- enumeration ----

    def enumerate_units(self) -> list[RunUnit]:
        """Cartesian product of agents × tasks × variants."""
        out: list[RunUnit] = []
        for agent in self._spec.agents:
            for task in self._spec.tasks:
                for vi in task.variants:
                    out.append(RunUnit(
                        agent_id=agent.id,
                        agent_spec=agent,
                        task_path=task.path,
                        variant_index=vi,
                    ))
        return out

    # ---- execution ----

    async def run(
        self,
        units: Iterable[RunUnit] | None = None,
        *,
        max_attempts: int = 1,
    ) -> list[UnitResult]:
        """Run all units (or a filtered subset). Returns ``list[UnitResult]``.

        Failed units are requeued until they succeed or reach ``max_attempts``.
        Only the final result for each unit is returned; every attempt retains
        its own run directory on disk.
        """
        from .host_preflight import check_host_dependencies
        from .lifecycle import install_signal_handlers, run_one_unit

        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        install_signal_handlers()
        unit_list = list(units) if units is not None else self.enumerate_units()
        if not unit_list:
            logger.warning("Runner.run: no units to execute")
            return []

        await asyncio.to_thread(check_host_dependencies, (unit.task_path for unit in unit_list))
        await check_judges(unit_list)

        self._output_root.mkdir(parents=True, exist_ok=True)

        n = self._spec.concurrency
        sem = ResizableLimiter(n)
        logger.info("runner: %d units, concurrency=%d", len(unit_list), n)

        control_path = self._output_root / "concurrency.control"
        stop_control = asyncio.Event()

        async def _watch_concurrency() -> None:
            last_value: int | None = None
            while not stop_control.is_set():
                try:
                    raw = control_path.read_text(encoding="utf-8").strip()
                    value = int(raw)
                    if value < 1:
                        raise ValueError("must be >= 1")
                    if value != last_value:
                        if await sem.set_limit(value):
                            logger.warning(
                                "concurrency changed: %d (active=%d)",
                                value, sem.active,
                            )
                        last_value = value
                except FileNotFoundError:
                    pass
                except (OSError, ValueError) as exc:
                    logger.warning("ignoring invalid concurrency.control: %s", exc)
                try:
                    await asyncio.wait_for(stop_control.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass

        control_task = asyncio.create_task(_watch_concurrency())

        async def _drive(u: RunUnit) -> UnitResult:
            for attempt in range(1, max_attempts + 1):
                logger.info(
                    "[%s] queued (attempt %d/%d)", u.slug, attempt, max_attempts,
                )
                result = await run_one_unit(
                    unit=u,
                    router=self._router,
                    output_root=self._output_root,
                    artifacts=self._spec.artifacts,
                    sem=sem,
                    cleanup_mode=self._spec.cleanup_mode,
                    prompt_suffix=self._spec.prompt_suffix,
                    wall_time_s=self._spec.wall_time_s,
                )
                logger.info(
                    "[%s] done: status=%s score=%s duration=%.1fs (attempt %d/%d)",
                    u.slug,
                    result.status,
                    result.score,
                    result.duration_s or 0,
                    attempt,
                    max_attempts,
                )
                if result.status != "failed" or attempt == max_attempts:
                    return result
                logger.warning(
                    "[%s] failed on attempt %d/%d; requeued",
                    u.slug,
                    attempt,
                    max_attempts,
                )

            raise AssertionError("unreachable")

        try:
            results = await asyncio.gather(
                *(_drive(u) for u in unit_list),
                return_exceptions=False,
            )
            return list(results)
        finally:
            stop_control.set()
            control_task.cancel()
            with suppress(asyncio.CancelledError):
                await control_task
