"""Report task-card and local task-data resources for an ALE task list."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_parse_gce_machine_type = importlib.import_module(
    "ale_run.environments.providers.gcloud"
)._parse_gce_machine_type


DEFAULT_TASK_LIST = REPO_ROOT / "selected_tasks" / "ale_cli.txt"
DEFAULT_DOCKER_LIST = REPO_ROOT / "selected_tasks" / "docker_support.txt"
DEFAULT_TIMEOUT_S = 7200
LARGE_DATA_BYTES = 100 * 1024**2
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


@dataclass
class DataStats:
    present: bool
    total_bytes: int = 0
    input_bytes: int = 0
    software_bytes: int = 0
    reference_bytes: int = 0
    other_bytes: int = 0
    file_count: int = 0
    lfs_pointer_count: int = 0
    scan_error_count: int = 0


@dataclass
class TaskResource:
    task_id: str
    domain: str
    snapshot: str | None
    machine_type: str | None
    vcpus: int | None
    memory_gb: int | None
    timeout_s: int
    timeout_declared: bool
    docker_supported: bool
    required_system_packages: list[str]
    data_path: str | None
    data: DataStats
    resource_bucket: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit CPU, memory, timeout, Docker support, system packages, and local "
            "task-data size for an ALE task list."
        )
    )
    parser.add_argument(
        "--task-list",
        type=Path,
        default=DEFAULT_TASK_LIST,
        help=f"Task list to audit (default: {DEFAULT_TASK_LIST})",
    )
    parser.add_argument(
        "--docker-list",
        type=Path,
        default=DEFAULT_DOCKER_LIST,
        help=f"Direct-Docker allowlist (default: {DEFAULT_DOCKER_LIST})",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Task-data root containing <domain>/<task>/<variant>. If omitted, the "
            "script checks ALE_TASK_DATA_ROOT and common local checkout paths."
        ),
    )
    parser.add_argument(
        "--variant",
        default="base",
        help="Task-data variant directory to scan (default: base)",
    )
    parser.add_argument(
        "--skip-data-scan",
        action="store_true",
        help="Read task cards without walking local task-data files.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "csv"),
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of largest task-data rows in text output (default: 15)",
    )
    parser.add_argument(
        "--host",
        action="store_true",
        help="Include a point-in-time host CPU/RAM capacity estimate.",
    )
    args = parser.parse_args()
    if args.top < 0:
        parser.error("--top must be non-negative")
    return args


def _read_task_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"task list not found: {path}")
    task_ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    duplicates = sorted(task_id for task_id, count in Counter(task_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate task IDs in {path}: {', '.join(duplicates)}")
    return task_ids


def _resolve_data_root(explicit: Path | None, *, skip: bool) -> Path | None:
    if skip:
        return None
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"task-data root not found: {root}")
        return root

    candidates: list[Path] = []
    env_root = os.environ.get("ALE_TASK_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            REPO_ROOT.parent / "agent_last_exam" / "tasks_data" / "tasks",
            REPO_ROOT / "tasks_data" / "tasks",
        ]
    )
    return next((candidate.resolve() for candidate in candidates if candidate.is_dir()), None)


def _scan_data_dir(path: Path) -> DataStats:
    if not path.is_dir():
        return DataStats(present=False)

    stats = DataStats(present=True)
    byte_fields = {
        "input": "input_bytes",
        "software": "software_bytes",
        "reference": "reference_bytes",
    }
    for current_root, _, filenames in os.walk(path, followlinks=False):
        current = Path(current_root)
        for filename in filenames:
            file_path = current / filename
            try:
                file_size = file_path.stat(follow_symlinks=False).st_size
                relative = file_path.relative_to(path)
            except OSError:
                stats.scan_error_count += 1
                continue

            stats.total_bytes += file_size
            stats.file_count += 1
            field = byte_fields.get(relative.parts[0] if relative.parts else "")
            if field is None:
                stats.other_bytes += file_size
            else:
                setattr(stats, field, getattr(stats, field) + file_size)

            if file_size <= 1024 and not file_path.is_symlink():
                try:
                    if file_path.read_bytes().startswith(LFS_POINTER_PREFIX):
                        stats.lfs_pointer_count += 1
                except OSError:
                    stats.scan_error_count += 1
    return stats


def _load_card(task_id: str) -> dict[str, Any]:
    card_path = REPO_ROOT / "tasks" / task_id / "task_card.json"
    if not card_path.is_file():
        raise FileNotFoundError(f"task card not found: {card_path}")
    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {card_path}: {exc}") from exc
    if not isinstance(card, dict):
        raise TypeError(f"task card must contain a JSON object: {card_path}")
    return card


def _positive_int(raw: Any, *, field: str, task_id: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise TypeError(f"{task_id}: {field} must be a positive integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{task_id}: {field} must be a positive integer") from exc
    if value <= 0 or (isinstance(raw, float) and not raw.is_integer()):
        raise ValueError(f"{task_id}: {field} must be a positive integer")
    return value


def _resource_bucket(
    *,
    docker_supported: bool,
    data: DataStats,
    data_scanned: bool,
    vcpus: int | None,
    memory_gb: int | None,
) -> str:
    if data_scanned and not data.present:
        return "data-unavailable-locally"
    if not docker_supported:
        return "vm-only"
    if (vcpus or 0) > 4 or (memory_gb or 0) > 15:
        return "docker-high-resource"
    if data_scanned and data.total_bytes >= LARGE_DATA_BYTES:
        return "docker-large-data"
    return "docker-standard"


def _collect_records(
    task_ids: list[str],
    docker_ids: set[str],
    data_root: Path | None,
    variant: str,
) -> list[TaskResource]:
    records = []
    for task_id in task_ids:
        card = _load_card(task_id)
        vm = card.get("vm") or {}
        if not vm and (card.get("snapshot") or card.get("vm_category")):
            vm = {
                "snapshot": card.get("snapshot") or card.get("vm_category"),
                "machineType": card.get("machineType"),
                "timeout": card.get("timeout"),
            }
        if not isinstance(vm, dict):
            raise TypeError(f"{task_id}: vm must be a JSON object")

        machine_type = vm.get("machineType")
        shape = _parse_gce_machine_type(machine_type) if machine_type else None
        if machine_type and shape is None:
            raise ValueError(f"{task_id}: unparseable machine type {machine_type!r}")
        vcpus = _positive_int(vm.get("vcpus"), field="vm.vcpus", task_id=task_id)
        memory_gb = _positive_int(
            vm.get("memory_gb"), field="vm.memory_gb", task_id=task_id
        )
        if vcpus is None and shape is not None:
            vcpus = shape.vcpus
        if memory_gb is None and shape is not None:
            memory_gb = shape.memory_gb

        raw_timeout = vm.get("timeout_s", vm.get("timeout"))
        declared_timeout = _positive_int(
            raw_timeout, field="vm.timeout_s", task_id=task_id
        )
        timeout_s = declared_timeout or DEFAULT_TIMEOUT_S

        packages = card.get("requiredSystemPackages") or []
        if not isinstance(packages, list) or not all(isinstance(item, str) for item in packages):
            raise ValueError(f"{task_id}: requiredSystemPackages must be a list of strings")

        data_path = data_root / task_id / variant if data_root is not None else None
        data = _scan_data_dir(data_path) if data_path is not None else DataStats(present=False)
        docker_supported = task_id in docker_ids
        records.append(
            TaskResource(
                task_id=task_id,
                domain=task_id.split("/", 1)[0],
                snapshot=vm.get("snapshot"),
                machine_type=machine_type,
                vcpus=vcpus,
                memory_gb=memory_gb,
                timeout_s=timeout_s,
                timeout_declared=declared_timeout is not None,
                docker_supported=docker_supported,
                required_system_packages=packages,
                data_path=str(data_path) if data_path is not None else None,
                data=data,
                resource_bucket=_resource_bucket(
                    docker_supported=docker_supported,
                    data=data,
                    data_scanned=data_root is not None,
                    vcpus=vcpus,
                    memory_gb=memory_gb,
                ),
            )
        )
    return records


def _percentile(values: list[int], percentile: int) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile / 100)
    return ordered[index]


def _host_snapshot() -> dict[str, int | None]:
    memory: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            memory[key] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return {
        "logical_cpus": os.cpu_count(),
        "memory_total_bytes": memory.get("MemTotal"),
        "memory_available_bytes": memory.get("MemAvailable"),
    }


def _summary(
    records: list[TaskResource], data_root: Path | None, include_host: bool
) -> dict[str, Any]:
    package_counts = Counter(
        package for record in records for package in record.required_system_packages
    )
    data_records = [record for record in records if record.data.present]
    data_sizes = [record.data.total_bytes for record in data_records]
    payload: dict[str, Any] = {
        "task_count": len(records),
        "docker_supported_count": sum(record.docker_supported for record in records),
        "machine_types": dict(Counter(record.machine_type or "<unset>" for record in records)),
        "shapes": dict(
            Counter(f"{record.vcpus or '?'}c/{record.memory_gb or '?'}GiB" for record in records)
        ),
        "timeouts_seconds": dict(Counter(str(record.timeout_s) for record in records)),
        "resource_buckets": dict(Counter(record.resource_bucket for record in records)),
        "system_packages": {
            "task_count": sum(bool(record.required_system_packages) for record in records),
            "unique_count": len(package_counts),
            "counts": dict(package_counts.most_common()),
        },
        "data": {
            "root": str(data_root) if data_root is not None else None,
            "present_count": len(data_records),
            "unavailable_count": len(records) - len(data_records) if data_root else None,
            "total_bytes": sum(data_sizes),
            "input_bytes": sum(record.data.input_bytes for record in data_records),
            "software_bytes": sum(record.data.software_bytes for record in data_records),
            "reference_bytes": sum(record.data.reference_bytes for record in data_records),
            "other_bytes": sum(record.data.other_bytes for record in data_records),
            "file_count": sum(record.data.file_count for record in data_records),
            "lfs_pointer_count": sum(record.data.lfs_pointer_count for record in data_records),
            "scan_error_count": sum(record.data.scan_error_count for record in data_records),
            "percentiles_bytes": {
                str(percentile): _percentile(data_sizes, percentile)
                for percentile in (0, 25, 50, 75, 90, 95, 99, 100)
            },
        },
    }
    if include_host:
        payload["host"] = _host_snapshot()
    return payload


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


def _format_duration(seconds: int) -> str:
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def _print_counter(title: str, values: dict[str, int]) -> None:
    print(title)
    for name, count in sorted(values.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {name:<34} {count:>4}")


def _print_text(
    records: list[TaskResource], summary: dict[str, Any], task_list: Path, top: int
) -> None:
    print("ALE task resource audit")
    print(f"  task list:       {task_list}")
    print(f"  tasks:           {summary['task_count']}")
    print(
        f"  direct Docker:   {summary['docker_supported_count']}/{summary['task_count']}"
    )
    print(f"  task-data root:  {summary['data']['root'] or 'not scanned'}")
    print()

    _print_counter("Requested machine types", summary["machine_types"])
    print()
    _print_counter("Resolved task shapes", summary["shapes"])
    print()
    _print_counter(
        "Effective agent timeouts",
        {_format_duration(int(seconds)): count for seconds, count in summary["timeouts_seconds"].items()},
    )
    print()
    _print_counter("Scheduling buckets", summary["resource_buckets"])

    packages = summary["system_packages"]
    print()
    print(
        "System packages\n"
        f"  tasks declaring packages: {packages['task_count']}/{summary['task_count']}\n"
        f"  unique package IDs:        {packages['unique_count']}"
    )
    for package, count in list(packages["counts"].items())[:15]:
        print(f"  {package:<34} {count:>4}")

    data = summary["data"]
    if data["root"] is not None:
        print()
        print("Local task-data")
        print(f"  available:        {data['present_count']}/{summary['task_count']}")
        print(f"  total:            {_human_bytes(data['total_bytes'])}")
        print(f"  input:            {_human_bytes(data['input_bytes'])}")
        print(f"  software:         {_human_bytes(data['software_bytes'])}")
        print(f"  reference:        {_human_bytes(data['reference_bytes'])}")
        print(f"  files:            {data['file_count']}")
        print(f"  Git LFS pointers: {data['lfs_pointer_count']}")
        print(f"  scan errors:      {data['scan_error_count']}")
        print("  size percentiles:")
        for percentile, value in data["percentiles_bytes"].items():
            print(f"    p{percentile:<3} {_human_bytes(value):>12}")

        if top:
            print()
            print(f"Largest {min(top, len(records))} task-data directories")
            print(f"  {'size':>11} {'CPU':>4} {'RAM':>7} {'time':>5} {'Docker':>7}  task")
            largest = sorted(records, key=lambda record: record.data.total_bytes, reverse=True)
            for record in largest[:top]:
                size = _human_bytes(record.data.total_bytes) if record.data.present else "n/a"
                cpu = record.vcpus or "?"
                memory = f"{record.memory_gb or '?'}G"
                print(
                    f"  {size:>11} {cpu!s:>4} "
                    f"{memory:>7} "
                    f"{_format_duration(record.timeout_s):>5} "
                    f"{('yes' if record.docker_supported else 'no'):>7}  {record.task_id}"
                )

        unavailable = [record.task_id for record in records if not record.data.present]
        if unavailable:
            print()
            print("Task data unavailable locally")
            for task_id in unavailable:
                print(f"  {task_id}")

    vm_only = [record.task_id for record in records if not record.docker_supported]
    if vm_only:
        print()
        print("Not in the direct-Docker allowlist")
        for task_id in vm_only:
            print(f"  {task_id}")

    host = summary.get("host")
    if host:
        print()
        print("Host snapshot")
        print(f"  logical CPUs:     {host['logical_cpus'] or 'unknown'}")
        print(f"  RAM total:        {_human_bytes(host['memory_total_bytes'])}")
        print(f"  RAM available:    {_human_bytes(host['memory_available_bytes'])}")
        common_shape = Counter((record.vcpus, record.memory_gb) for record in records).most_common(1)
        if common_shape:
            (vcpus, memory_gb), _ = common_shape[0]
            if vcpus and memory_gb and host["logical_cpus"] and host["memory_total_bytes"]:
                cpu_slots = host["logical_cpus"] // vcpus
                ram_slots = host["memory_total_bytes"] // (memory_gb * 1024**3)
                print(
                    f"  common {vcpus}c/{memory_gb}GiB theoretical slots: "
                    f"{min(cpu_slots, ram_slots)} before VM/container overhead"
                )
            if memory_gb and host["memory_available_bytes"]:
                available_slots = host["memory_available_bytes"] // (memory_gb * 1024**3)
                print(
                    f"  common-shape slots by currently available RAM only: {available_slots}"
                )


def _print_csv(records: list[TaskResource]) -> None:
    fieldnames = [
        "task_id",
        "domain",
        "resource_bucket",
        "docker_supported",
        "snapshot",
        "machine_type",
        "vcpus",
        "memory_gb",
        "timeout_s",
        "timeout_declared",
        "data_present",
        "data_bytes",
        "input_bytes",
        "software_bytes",
        "reference_bytes",
        "file_count",
        "lfs_pointer_count",
        "required_system_packages",
        "data_path",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                "task_id": record.task_id,
                "domain": record.domain,
                "resource_bucket": record.resource_bucket,
                "docker_supported": record.docker_supported,
                "snapshot": record.snapshot,
                "machine_type": record.machine_type,
                "vcpus": record.vcpus,
                "memory_gb": record.memory_gb,
                "timeout_s": record.timeout_s,
                "timeout_declared": record.timeout_declared,
                "data_present": record.data.present,
                "data_bytes": record.data.total_bytes,
                "input_bytes": record.data.input_bytes,
                "software_bytes": record.data.software_bytes,
                "reference_bytes": record.data.reference_bytes,
                "file_count": record.data.file_count,
                "lfs_pointer_count": record.data.lfs_pointer_count,
                "required_system_packages": ";".join(record.required_system_packages),
                "data_path": record.data_path,
            }
        )


def main() -> int:
    args = _parse_args()
    task_list = args.task_list.expanduser().resolve()
    docker_list = args.docker_list.expanduser().resolve()
    task_ids = _read_task_list(task_list)
    docker_ids = set(_read_task_list(docker_list))
    unknown_docker_ids = sorted(docker_ids - set(task_ids))
    if unknown_docker_ids:
        print(
            "warning: Docker list contains tasks outside the audited list: "
            + ", ".join(unknown_docker_ids),
            file=sys.stderr,
        )

    data_root = _resolve_data_root(args.data_root, skip=args.skip_data_scan)
    records = _collect_records(task_ids, docker_ids, data_root, args.variant)
    summary = _summary(records, data_root, args.host)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "task_list": str(task_list),
                    "variant": args.variant,
                    "summary": summary,
                    "tasks": [asdict(record) for record in records],
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.format == "csv":
        _print_csv(records)
    else:
        _print_text(records, summary, task_list, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
