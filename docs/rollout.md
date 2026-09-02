# Single-task rollouts

`ale rollout` is the reusable entry point for running one ALE task. It is
intended for benchmark evaluation, task-authoring smoke tests, and evaluator
debugging. The command builds a normal one-unit `ExperimentSpec` and hands it
to ALE's existing `Runner`.

The wrapper does not implement its own VM, file transfer, harness, or evaluator
logic. The native lifecycle still performs VM provisioning, input and software
staging, task `start()`, agent execution, output collection, delayed reference
staging, `evaluate()`, trajectory construction, and cleanup.

## Commands

Validate every path and configuration field without provisioning a VM:

```bash
uv run python -m ale_run rollout validate configs/rollouts/codex-proxmox.example.yaml
```

Print the normalized, secret-free manifest:

```bash
uv run python -m ale_run rollout validate rollout.yaml --json
```

Run the task:

```bash
uv run python -m ale_run rollout run rollout.yaml
```

Use `--json` for a machine-readable final summary. `--dry-run` on the `run`
command is equivalent to validation. A completed rollout exits with code 0,
including a completed evaluation whose score is zero. Provisioning, harness,
timeout, or lifecycle failures exit with code 1.

## Configuration

The example at
[`configs/rollouts/codex-proxmox.example.yaml`](../configs/rollouts/codex-proxmox.example.yaml)
contains every supported field.

- `task.id` is the path below the repository's `tasks/` directory.
- `task.variant` is the zero-based variant index exposed by the task's
  `main.py`.
- `environment.profile` is a native ALE environment YAML. It controls snapshot
  routing and VM-provider details.
- `environment.task_data_source` optionally overrides the profile. With
  `local:<root>`, the expected layout is
  `<root>/<domain>/<task>/<variant>/{input,software,reference}`. Local staging
  is supported by the Docker and Proxmox providers.
- `environment.output_path` defaults to `local`, which copies the task output
  into the run directory. Set it to `null` to retain only logs and evaluation
  results.
- `agent` accepts either a native agent `preset` YAML or an inline `harness`.
  Inline Codex configuration provides convenient model, reasoning, gateway,
  and API-key fields.
- `run.cleanup_mode` uses the native provider behavior: `delete`, `stop`, or
  `keep`. Proxmox normally creates a linked clone, so `delete` removes the
  disposable clone but not its template.
- `run.max_attempts` is between 1 and 3. Every retry gets its own normal ALE run
  directory.

`repo_root` defaults to the checkout containing the installed `ale_run`
package. It is explicit in production files so the task-code revision is easy
to identify. Relative paths in a rollout resolve from the rollout YAML's
directory.

## Qizhi Docker jobs

Qizhi's Docker adapter guarantees host networking, but does not guarantee
Docker bridge networks or published ports. Use
[`configs/environments/docker_qizhi.yaml`](../configs/environments/docker_qizhi.yaml),
which sets `docker.network_mode: host`. In this mode ALE launches the sandbox
with `docker run --network host`, does not pass `-p`, and connects to CUA at
`http://127.0.0.1:5000`.

ALE's published image also binds noVNC on fixed port 6901. Consequently only
one ALE sandbox may run in a Qizhi Docker job at a time; keep `concurrency: 1`.
The default Docker profile remains in bridge mode and is unaffected.

The no-credential smoke test uses the scripted demo harness:

```bash
export ALE_REPO_ROOT="$PWD"
export ALE_ROLLOUT_ROOT=/inspire/hdd/project/long-working-agent/ky26299/rollouts
scripts/qizhi_load_ale_ubuntu22.sh
uv run python -m ale_run run configs/experiments/qizhi_demo_hello.yaml
```

Run those commands after entering the interactive instance's
`qz-docker-shell`, or inside a Qizhi job that exposes the same Docker adapter.
The loader consumes the verified registry layer from
`cache/docker/ale-ubuntu22-oci/`, streams it into the job's temporary Docker
daemon, and verifies the resulting image ID. It does not create another
persistent copy of the 39 GB compressed layer.

The Qizhi profile reads task data from the persistent project HDD at
`/inspire/hdd/project/long-working-agent/ky26299/task-data`. Populate that tree
using the canonical `<domain>/<task>/<variant>/{input,software,reference}`
layout before running a task. The published image does not bake every task, so
the profile deliberately does not use `baked_in_sandbox`.

## Secrets

Put credentials in the file named by `secret_file`, or export them in the
shell. For inline Codex configuration, `agent.api.api_key_env` names the source
environment variable. The rollout maps that value to `OPENROUTER_API_KEY` or
`OPENAI_API_KEY` only while the native runner is active.

`agent.api.provider` names the actual API service, for example `mindwaving`.
Codex currently reaches MindWaving through its OpenAI-compatible gateway
routing internally; this implementation detail does not change the provider
recorded in the rollout manifest.

Literal `agent.config.api_key` values are rejected. Neither the normalized
manifest nor `rollout_summary.json` contains the API-key value. Environment
profiles may still use the native `${env:VARIABLE}` form for Proxmox or cloud
credentials.

## Results

The normal layout remains:

```text
<output_root>/<rollout-name>/<agent>/<model>/<task>/v<variant>/<timestamp>/
```

All native files remain available, including `events.jsonl`, `run.json`,
`trajectory.json`, `eval_result.json`, `origin_log/`, screenshots, and the
optional pulled `output/`. The wrapper adds two files:

- `rollout_manifest.json` records the normalized, secret-free inputs.
- `rollout_summary.json` records status, score, evaluation status, duration,
  run directory, model, and cleanup policy.

The task package is never copied into a second task format. Task code continues
to come from `<repo_root>/tasks/<task.id>`, while task data comes from the
environment's `task_data_source`.
