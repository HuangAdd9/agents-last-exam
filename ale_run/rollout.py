"""Single-task rollout configuration and execution.

This module is a convenience layer over the normal ALE orchestration stack.
It deliberately stops at building an :class:`ExperimentSpec`; provisioning,
task-data staging, harness execution, evaluation, artifact collection, and VM
cleanup remain owned by :class:`ale_run.orchestration.Runner`.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import yaml

from .orchestration import AgentSpec, ExperimentSpec, OutputSpec, Runner, TaskSpec
from .orchestration.config_loader import (
    _build_agent_from_path,
    _build_artifacts,
    _build_environment_from_path,
    _load_dotenv,
    _resolve_path,
    _substitute_env,
)
from .orchestration.experiment_spec import UnitResult
from .orchestration.factory import build_config, resolve_agent
from .tasks.loader import TaskLoader

_TOP_KEYS = frozenset(
    {"schema_version", "name", "secret_file", "repo_root", "task", "environment", "agent", "run"}
)
_TASK_KEYS = frozenset({"id", "variant"})
_ENVIRONMENT_KEYS = frozenset({"profile", "task_data_source", "output_path"})
_AGENT_KEYS = frozenset(
    {
        "preset",
        "harness",
        "class",
        "id",
        "executor",
        "model",
        "reasoning_effort",
        "api",
        "config",
    }
)
_API_KEYS = frozenset({"provider", "base_url", "api_key_env", "account", "proxy_url"})
_RUN_KEYS = frozenset(
    {"output_root", "wall_time_s", "cleanup_mode", "max_attempts", "prompt_suffix"}
)
_CLEANUP_MODES = frozenset({"delete", "stop", "keep"})
_CODEX_PROVIDER_ROUTING = {
    "mindwaving": "openrouter",
    "mindwaving_bulk": "openrouter",
    "qizhi": "openrouter",
    "openrouter": "openrouter",
    "openai": "direct",
    "direct": "direct",
    "chatgpt": "chatgpt",
    "codex-account": "chatgpt",
}
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SECRET_KEY_RE = re.compile(r"(?:api[_-]?key|token|secret|password|authorization)", re.IGNORECASE)


@dataclass
class RolloutPlan:
    """Validated single-task rollout plus its secret-free run manifest."""

    source_path: Path
    repo_root: Path
    experiment: ExperimentSpec
    manifest: dict[str, Any]
    auth_env: dict[str, str] = field(default_factory=dict, repr=False)


def load_rollout(path: str | Path) -> RolloutPlan:
    """Load and fully validate a single-task rollout YAML."""
    source_path = Path(path).expanduser().resolve()
    raw_text = source_path.read_text(encoding="utf-8")
    pre = _parse_yaml(raw_text, source_path)
    base_dir = source_path.parent

    explicit_secret = pre.get("secret_file")
    if explicit_secret is not None:
        secret_path = _resolve_path(str(explicit_secret), base_dir)
        if not secret_path.is_file():
            raise FileNotFoundError(f"rollout secret_file not found: {secret_path}")
        _load_dotenv(secret_path, override=True)
    else:
        _load_dotenv(base_dir / "secret" / ".env")
        _load_dotenv(base_dir / ".env")

    default_repo_root = Path(__file__).resolve().parents[1]
    pre_repo_root = pre.get("repo_root")
    if isinstance(pre_repo_root, str) and "${env:" not in pre_repo_root:
        candidate_repo_root = _resolve_path(pre_repo_root, base_dir)
    else:
        candidate_repo_root = default_repo_root
    if explicit_secret is None:
        _load_dotenv(candidate_repo_root / "secret" / ".env")

    raw = _parse_yaml(_substitute_env(raw_text), source_path)
    _reject_unknown(raw, _TOP_KEYS, "rollout")
    schema_version = raw.get("schema_version", 1)
    if schema_version != 1:
        raise ValueError(f"schema_version must be 1, got {schema_version!r}")

    name = _required_string(raw, "name", "rollout")
    if not _NAME_RE.fullmatch(name):
        raise ValueError(
            "rollout.name must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'"
        )

    repo_root = _resolve_path(raw.get("repo_root", default_repo_root), base_dir)
    if not (repo_root / "ale_run").is_dir() or not (repo_root / "tasks").is_dir():
        raise ValueError(f"repo_root is not an ALE checkout: {repo_root}")

    task_raw = _required_mapping(raw, "task", "rollout")
    _reject_unknown(task_raw, _TASK_KEYS, "task")
    task_id = _validate_task_id(_required_string(task_raw, "id", "task"))
    variant = _nonnegative_int(task_raw.get("variant", 0), "task.variant")
    task_dir = (repo_root / "tasks" / task_id).resolve()
    tasks_root = (repo_root / "tasks").resolve()
    if not task_dir.is_relative_to(tasks_root):
        raise ValueError(f"task.id escapes repo tasks directory: {task_id!r}")
    if not (task_dir / "main.py").is_file():
        raise FileNotFoundError(f"task main.py not found: {task_dir / 'main.py'}")
    if not (task_dir / "task_card.json").is_file():
        raise FileNotFoundError(f"task_card.json not found: {task_dir / 'task_card.json'}")

    loader = TaskLoader(str(task_dir))
    task_meta = loader.load(variant)
    snapshot = task_meta.get("image_category") or task_meta.get("snapshot_name")
    if not snapshot:
        raise ValueError(f"task {task_id!r} has no VM snapshot in task_card.json")
    if variant > 0 and loader._load_task_variant(variant) is None:
        raise IndexError(f"task {task_id!r} does not define variant index {variant}")

    environment_raw = _required_mapping(raw, "environment", "rollout")
    _reject_unknown(environment_raw, _ENVIRONMENT_KEYS, "environment")
    environment_profile = _required_string(environment_raw, "profile", "environment")
    environment_profile_path = _resolve_path(environment_profile, base_dir)
    environment, profile_artifacts = _build_environment_from_path(
        str(environment_profile_path), base_dir=base_dir
    )
    environment.kind_for(str(snapshot))

    task_data_source = environment_raw.get("task_data_source", profile_artifacts.task_data_source)
    task_data_source = _resolve_local_source(str(task_data_source), base_dir)
    output_path = environment_raw.get("output_path", "local")
    artifacts = _build_artifacts({"task_data_source": task_data_source, "output_path": output_path})
    provider_kind = environment.kind_for(str(snapshot))
    if artifacts.task_data_source.startswith("local:") and provider_kind not in {
        "docker",
        "proxmox",
        "static",
    }:
        raise ValueError(
            "environment.task_data_source=local is supported only by docker, proxmox, "
            "and static CUA environments, "
            f"not {provider_kind!r}"
        )
    _preflight_task_data(artifacts.task_data_source, task_meta)

    agent_raw = _required_mapping(raw, "agent", "rollout")
    _reject_unknown(agent_raw, _AGENT_KEYS, "agent")
    agent, auth_env, auth_manifest = _build_agent(agent_raw, base_dir=base_dir)

    run_raw = raw.get("run") or {}
    if not isinstance(run_raw, dict):
        raise TypeError(f"run must be a mapping, got {type(run_raw).__name__}")
    _reject_unknown(run_raw, _RUN_KEYS, "run")
    output_root = _resolve_path(run_raw.get("output_root", ".logs/ale-rollout"), base_dir)
    wall_time_s = _optional_positive_int(run_raw.get("wall_time_s"), "run.wall_time_s")
    max_attempts = _positive_int(run_raw.get("max_attempts", 1), "run.max_attempts")
    if max_attempts > 3:
        raise ValueError(f"run.max_attempts must be between 1 and 3, got {max_attempts}")
    cleanup_mode = str(run_raw.get("cleanup_mode") or "delete")
    if cleanup_mode not in _CLEANUP_MODES:
        raise ValueError(
            f"run.cleanup_mode must be one of {sorted(_CLEANUP_MODES)}, got {cleanup_mode!r}"
        )
    prompt_suffix = str(run_raw.get("prompt_suffix") or "")

    experiment = ExperimentSpec(
        name=name,
        output=OutputSpec(root=str(output_root)),
        environment=environment,
        agents=[agent],
        tasks=[TaskSpec(path=task_id, variants=[variant])],
        artifacts=artifacts,
        concurrency=1,
        auto_resume=False,
        max_attempts=max_attempts,
        cleanup_mode=cleanup_mode,
        prompt_suffix=prompt_suffix,
        wall_time_s=wall_time_s,
    )

    task_data = task_meta["task_data"]
    manifest = {
        "schema_version": 1,
        "name": name,
        "source_path": str(source_path),
        "repo_root": str(repo_root),
        "task": {
            "id": task_id,
            "variant": variant,
            "variant_name": task_data.variant_name,
            "snapshot": snapshot,
            "os_type": task_meta.get("os_type"),
        },
        "environment": {
            "profile": str(environment_profile_path),
            "provider": provider_kind,
            "task_data_source": artifacts.task_data_source,
            "output_path": artifacts.output_path,
        },
        "agent": {
            "id": agent.id,
            "harness": agent.class_,
            "executor": agent.executor,
            "model": agent.config.get("model"),
            "reasoning_effort": agent.config.get("reasoning_effort"),
            "api": auth_manifest,
            "config": _redact(
                {
                    key: value
                    for key, value in agent.config.items()
                    if key
                    not in {
                        "model",
                        "reasoning_effort",
                        "provider",
                        "base_url",
                        "proxy_url",
                        "api_key",
                    }
                }
            ),
        },
        "run": {
            "output_root": str(output_root),
            "wall_time_s": wall_time_s,
            "cleanup_mode": cleanup_mode,
            "max_attempts": max_attempts,
            "prompt_suffix": prompt_suffix,
        },
    }
    return RolloutPlan(
        source_path=source_path,
        repo_root=repo_root,
        experiment=experiment,
        manifest=manifest,
        auth_env=auth_env,
    )


async def run_rollout(plan: RolloutPlan) -> tuple[UnitResult, dict[str, Any]]:
    """Run one validated rollout through ALE's native Runner."""
    runner = Runner(plan.experiment)
    units = runner.enumerate_units()
    if len(units) != 1:
        raise AssertionError(f"single-task rollout produced {len(units)} run units")

    with _working_directory(plan.repo_root), _temporary_environ(plan.auth_env):
        results = await runner.run(units, max_attempts=plan.experiment.max_attempts)
    if len(results) != 1:
        raise AssertionError(f"single-task rollout returned {len(results)} results")

    result = results[0]
    summary = _build_summary(result, plan)
    if result.run_dir is not None:
        _write_json(result.run_dir / "rollout_manifest.json", plan.manifest)
        _write_json(result.run_dir / "rollout_summary.json", summary)
    return result, summary


def format_plan(plan: RolloutPlan) -> str:
    """Human-readable validation output."""
    manifest = plan.manifest
    task = manifest["task"]
    environment = manifest["environment"]
    agent = manifest["agent"]
    run = manifest["run"]
    return "\n".join(
        [
            f"rollout:    {manifest['name']}",
            f"task:       {task['id']} v{task['variant']} ({task['variant_name']})",
            f"environment: {environment['provider']} via {environment['profile']}",
            f"task data:  {environment['task_data_source']}",
            f"agent:      {agent['harness']} / {agent['model']} / {agent['reasoning_effort']}",
            f"output:     {run['output_root']}/{manifest['name']}",
            f"cleanup:    {run['cleanup_mode']}",
        ]
    )


def _build_agent(
    raw: dict[str, Any], *, base_dir: Path
) -> tuple[AgentSpec, dict[str, str], dict[str, Any]]:
    preset = raw.get("preset")
    if preset is not None:
        if raw.get("harness") is not None or raw.get("class") is not None:
            raise ValueError("agent.preset cannot be combined with agent.harness or agent.class")
        preset_path = _resolve_path(str(preset), base_dir)
        agent = _build_agent_from_path(str(preset_path), base_dir=base_dir)
    else:
        if raw.get("harness") is not None and raw.get("class") is not None:
            raise ValueError("set either agent.harness or agent.class, not both")
        class_name = raw.get("harness") or raw.get("class")
        if not class_name:
            raise KeyError("agent must declare preset, harness, or class")
        agent = AgentSpec(id=str(class_name), class_=str(class_name))

    if raw.get("id") is not None:
        agent.id = str(raw["id"])
    if raw.get("executor") is not None:
        if not isinstance(raw["executor"], str):
            raise TypeError("agent.executor must be a string")
        agent.executor = raw["executor"]
    config_override = raw.get("config") or {}
    if not isinstance(config_override, dict):
        raise TypeError("agent.config must be a mapping")
    if config_override.get("api_key") is not None:
        raise ValueError("agent.config.api_key is forbidden; use agent.api.api_key_env")
    agent.config = {**agent.config, **config_override}
    if raw.get("model") is not None:
        agent.config["model"] = str(raw["model"])
    if raw.get("reasoning_effort") is not None:
        agent.config["reasoning_effort"] = str(raw["reasoning_effort"])

    api_raw = raw.get("api") or {}
    if not isinstance(api_raw, dict):
        raise TypeError("agent.api must be a mapping")
    _reject_unknown(api_raw, _API_KEYS, "agent.api")
    if api_raw and agent.class_ != "codex":
        raise ValueError("agent.api shorthand currently supports only the codex harness")

    auth_env: dict[str, str] = {}
    auth_manifest: dict[str, Any] | None = None
    if agent.class_ == "codex":
        provider = str(api_raw.get("provider") or agent.config.get("provider") or "openrouter")
        if provider not in _CODEX_PROVIDER_ROUTING:
            raise ValueError(
                "Codex API provider must be one of "
                f"{sorted(_CODEX_PROVIDER_ROUTING)}, got {provider!r}"
            )
        routing = _CODEX_PROVIDER_ROUTING[provider]
        agent.config["provider"] = routing
        proxy_url = api_raw.get("proxy_url")
        if proxy_url is not None:
            if not isinstance(proxy_url, str) or not proxy_url.strip():
                raise TypeError("agent.api.proxy_url must be a non-empty string")
            proxy_url = proxy_url.strip()
            _validate_http_url(proxy_url, "agent.api.proxy_url")
            agent.config["proxy_url"] = proxy_url
        account = api_raw.get("account")
        if routing == "chatgpt":
            if "base_url" in api_raw:
                raise ValueError("agent.api.base_url cannot be used with provider=chatgpt")
            if api_raw.get("api_key_env") is not None:
                raise ValueError("agent.api.api_key_env cannot be used with provider=chatgpt")
            if not isinstance(account, str) or not account.strip():
                raise ValueError("agent.api.account is required with provider=chatgpt")
            account = account.strip()
            if not _NAME_RE.fullmatch(account):
                raise ValueError(
                    "agent.api.account must start with an alphanumeric character and "
                    "contain only letters, numbers, '.', '_' or '-'"
                )
            accounts_root = (Path.home() / ".codex-accounts").resolve()
            auth_path = (accounts_root / account / "auth.json").resolve()
            if not auth_path.is_relative_to(accounts_root):
                raise ValueError("agent.api.account escapes ~/.codex-accounts")
            if not auth_path.is_file():
                raise FileNotFoundError(f"Codex account auth file not found: {auth_path}")
            auth_content = auth_path.read_text(encoding="utf-8")
            try:
                auth_payload = json.loads(auth_content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Codex account auth file is invalid JSON: {auth_path}") from exc
            if not isinstance(auth_payload, dict) or not auth_payload:
                raise ValueError(f"Codex account auth file is empty or invalid: {auth_path}")
            auth_env["CODEX_AUTH_JSON"] = auth_content
            agent.config["base_url"] = None
            agent.config["api_key"] = None
            auth_manifest = {
                "provider": "chatgpt",
                "account": account,
                "configured": True,
            }
        else:
            if account is not None:
                raise ValueError("agent.api.account can be used only with provider=chatgpt")
        if routing != "chatgpt" and "base_url" in api_raw:
            base_url = api_raw["base_url"]
            if base_url is not None:
                _validate_http_url(str(base_url), "agent.api.base_url")
                base_url = str(base_url)
            agent.config["base_url"] = base_url

        if routing != "chatgpt":
            source_env = api_raw.get("api_key_env")
            canonical_env = (
                "OPENAI_API_KEY" if routing == "direct" else "OPENROUTER_API_KEY"
            )
            literal_key = agent.config.get("api_key")
            if source_env is not None:
                if not isinstance(source_env, str) or not source_env.isidentifier():
                    raise ValueError(
                        "agent.api.api_key_env must be an environment variable name"
                    )
                secret = os.environ.get(source_env)
                if secret is None:
                    raise KeyError(
                        f"agent.api.api_key_env references unset variable {source_env!r}"
                    )
                auth_env[canonical_env] = secret
                agent.config["api_key"] = None
            elif literal_key is None and canonical_env not in os.environ:
                raise KeyError(
                    f"Codex provider {provider!r} requires agent.api.api_key_env or "
                    f"{canonical_env}"
                )
            auth_manifest = {
                "provider": provider,
                "base_url": agent.config.get("base_url"),
                "api_key_env": source_env or canonical_env,
                "configured": bool(
                    source_env or literal_key or os.environ.get(canonical_env)
                ),
            }
        if auth_manifest is not None and agent.config.get("proxy_url"):
            auth_manifest["proxy_url"] = agent.config["proxy_url"]
    elif api_raw:
        raise ValueError("agent.api requires the codex harness")

    deployer_class, config_class = resolve_agent(agent)
    valid_config_keys = {item.name for item in fields(config_class)}
    unknown_config = set(agent.config) - valid_config_keys
    if unknown_config:
        raise TypeError(
            f"agent.config has keys unknown to {config_class.__name__}: {sorted(unknown_config)}"
        )
    build_config(config_class, agent.config)
    if not getattr(deployer_class, "supported_executors", None):
        raise ValueError(f"agent {agent.class_!r} declares no supported executors")
    return agent, auth_env, auth_manifest or {}


def _build_summary(result: UnitResult, plan: RolloutPlan) -> dict[str, Any]:
    run_id = None
    if result.run_dir is not None:
        run_json = result.run_dir / "run.json"
        try:
            run_id = json.loads(run_json.read_text(encoding="utf-8")).get("run_id")
        except (OSError, ValueError):
            pass
    return {
        "schema_version": 1,
        "name": plan.experiment.name,
        "run_id": run_id,
        "status": result.status,
        "score": result.score,
        "eval_status": result.eval_status,
        "duration_s": result.duration_s,
        "run_dir": str(result.run_dir.resolve()) if result.run_dir is not None else None,
        "task": result.unit.task_path,
        "variant": result.unit.variant_index,
        "agent": result.unit.agent_id,
        "model": result.unit.agent_spec.config.get("model"),
        "cleanup_mode": plan.experiment.cleanup_mode,
        "error": result.error,
    }


def _parse_yaml(text: str, path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"rollout yaml {path} is not valid yaml: {exc}") from exc
    if not isinstance(raw, dict):
        raise TypeError(f"rollout yaml root must be a mapping, got {type(raw).__name__}")
    return raw


def _reject_unknown(raw: dict[str, Any], allowed: frozenset[str], section: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise TypeError(f"{section} has unknown keys: {sorted(unknown)}")


def _required_mapping(raw: dict[str, Any], key: str, section: str) -> dict[str, Any]:
    if key not in raw:
        raise KeyError(f"{section} is missing required field {key!r}")
    value = raw[key]
    if not isinstance(value, dict):
        raise TypeError(f"{section}.{key} must be a mapping, got {type(value).__name__}")
    return value


def _required_string(raw: dict[str, Any], key: str, section: str) -> str:
    if key not in raw:
        raise KeyError(f"{section} is missing required field {key!r}")
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{section}.{key} must be a non-empty string")
    return value.strip()


def _validate_task_id(task_id: str) -> str:
    path = PurePosixPath(task_id)
    if (
        path.is_absolute()
        or len(path.parts) < 2
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("task.id must be a relative ALE path such as 'engineering/task_name'")
    return path.as_posix()


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{field_name} must be a non-negative integer")
    return value


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


def _validate_http_url(value: str, field_name: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute http or https URL")


def _resolve_local_source(source: str, base_dir: Path) -> str:
    if not source.startswith("local:"):
        return source
    root = source.removeprefix("local:")
    if not root:
        raise ValueError("environment.task_data_source local: path is empty")
    return f"local:{_resolve_path(root, base_dir)}"


def _preflight_task_data(source: str, task_meta: dict[str, Any]) -> None:
    task_data = task_meta["task_data"]
    if not source.startswith("local:") or not task_data.requires_task_data:
        return
    task_dir = Path(source.removeprefix("local:")) / task_data.source_relpath
    if not (task_dir / "input").is_dir():
        raise FileNotFoundError(f"local task data is missing input directory: {task_dir / 'input'}")


def _redact(value: Any, key: str = "") -> Any:
    if key and _SECRET_KEY_RE.search(key):
        return "<redacted>" if value is not None else None
    if isinstance(value, dict):
        return {
            item_key: _redact(item_value, str(item_key)) for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _temporary_environ(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
