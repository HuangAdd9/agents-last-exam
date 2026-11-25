# Local QEMU/KVM provider

Run the 119 CPU tasks in `selected_tasks/cpu.txt` using the v1.1 profile and
frozen task source with matching image and data pins.
QEMU supports only CPU tasks. Run the three tasks in `selected_tasks/gpu.txt`
with `configs/environments/environment_gcloud.yaml`.
The [website tutorial](https://agents-last-exam.org/docs?p=pages/local.html)
explains the provider; the commands and maintainer details follow here.

Butterfly animation uses 8 vCPU / 32 GiB and accepts any software. Skeletal
animation uses 16 vCPU / 64 GiB with CPU rendering. Its complete evaluator
also passed on 8 vCPU / 32 GiB in under 11 minutes. The remaining GPU tasks
require the current CUDA renderer, Resolve compute backend, or Isaac setup;
increasing CPU or RAM does not remove those software/device requirements.

The `qemu` provider runs a complete Ubuntu or Windows guest with QEMU inside a
Docker container. It provisions one VM for each ALE run and deletes it during
normal cleanup.

## Host requirements

- Complete the [host evaluation setup](../env/README.md#host-evaluation-environment)
- Linux with hardware virtualization or nested virtualization enabled
- Docker daemon available to the current user
- `/dev/kvm` present and passable through the Docker daemon
- Sufficient RAM for the task-card machine shape
- Sufficient disk for the cached base images
- `huggingface-hub` for `hf://` disks, installed with the project dependencies
- `gcloud` or `gsutil` when `qemu.disk_source` is a `gs://` URI

From the repository root, install and check the host evaluator before downloading
VM images. These dependencies run outside the guest, including for Windows tasks:

```bash
bash env/install_host_deps.sh
uv sync --all-packages --locked
bash env/verify_host_env.sh
```

The installer supports Ubuntu 22.04/24.04/26.04 and Debian 12/13. Docker, KVM,
cloud authentication, and guest applications remain provider prerequisites.

The published base images are:

- `hf://agents-last-exam/ale-images-qcow2/ale-win10-v1.1.qcow2`
- `hf://agents-last-exam/ale-images-qcow2/ale-ubuntu22-v1.1.qcow2`

These are logical disk paths. The provider automatically uses a multipart
manifest when the dataset stores the disk as verified 4 GB parts, or downloads
the qcow2 directly when it is stored as one object. This packaging is not part
of the environment configuration.

Use `configs/environments/qemu.yaml` as the starting configuration, preserving
each snapshot's `image_revision` pin from the [release manifest](../releases/v1.1/assets.json).
Hugging Face data packages cover the 122 selected tasks. See the
[release guide](releases/v1.1.md) or [original v1 setup](releases/v1.md).

## Check capacity and run

Resolve and check the configured cache/runtime root before downloading or
launching. For the default root:

```bash
mkdir -p ~/.cache/ale/qemu
readlink -f ~/.cache/ale/qemu
findmnt -T ~/.cache/ale/qemu
df -h ~/.cache/ale/qemu
free -h
docker ps --format 'table {{.Names}}\t{{.Status}}'
ps -eo pid,comm,args | rg '[q]emu-system'
test -r /dev/kvm && test -w /dev/kvm
```

Use a persistent disk-backed `qemu.root`, never host `/tmp`, `/dev/shm`, or
another memory-backed filesystem. Budget for base disks, download headroom,
growing overlays, and all running or paused guests, including retained VMs.

Copy `example_exp.yaml` to `my_experiment.yaml` and set:

```yaml
environment: configs/environments/qemu.yaml
tasks: selected_tasks/cpu.txt
concurrency: 1
cleanup_mode: keep
```

Keep the profile's `output_path: null`. Metadata, trajectories, evaluations,
and telemetry are collected; inspect task files in the retained sandbox and
release it afterward. Start with `selected_tasks/hello_both.txt` for a smoke.

```bash
uv run python -m ale_run run my_experiment.yaml --dry-run
uv run python -m ale_run run my_experiment.yaml
```

## Docker, Dockur, and the guest disk

Docker is the host container engine. Dockur is the upstream project whose
container image packages QEMU and its bridge, NAT, UEFI, and noVNC setup. ALE's
`agentslastexam/ale-qemu` image is a thin, versioned runner built on that
upstream image.

Two independent downloads can occur:

1. Docker resolves `runner_image`. With `runner_pull_policy: missing`, it pulls
   from Docker Hub only when that image is not already present locally.
2. The ALE host provider resolves `disk_source` into
   `~/.cache/ale/qemu/images/`. This happens before `docker run`, because Docker
   bind mounts must point to an existing host file when the container is
   created.

The runner entrypoint is invoked by `docker run` after both the read-only base
disk and writable per-run overlay are mounted.

## Runner image

The provider defaults to `agentslastexam/ale-qemu:0.2.0`. Its complete build
definition is in `ale_run/environments/images/ale_qemu/`. The runner image
contains QEMU, networking, noVNC, and lifecycle supervision, but not either
guest disk.

Build and publish it from the repository root:

```bash
docker build \
  -f ale_run/environments/images/ale_qemu/Dockerfile \
  -t agentslastexam/ale-qemu:0.2.0 \
  -t agentslastexam/ale-qemu:latest \
  .
docker push agentslastexam/ale-qemu:0.2.0
docker push agentslastexam/ale-qemu:latest
```

## Storage model

The first run downloads or reconstructs each base qcow2 under
`~/.cache/ale/qemu/images/`. Downloads use both an in-process lock and a
filesystem lock so concurrent ALE processes do not fetch the same large disk
twice. Multipart reconstruction records durable progress after every part, so
an interrupted download resumes without rebuilding completed parts. Every run
then creates a small qcow2 overlay under `~/.cache/ale/qemu/runtime/slots/`.
The base image is mounted read-only into the QEMU container, so concurrent runs
do not modify it or copy its full contents.

The shipped v1.1 profile pins `image_revision` to an immutable dataset commit.
Preserve that pin when customizing resources. Removing it selects the dataset's
mutable `main` branch and can mix task-code and disk versions. GCS and local
disk sources do not accept `image_revision`; remove that field when switching sources.

For `gs://` sources, the provider records the object generation, size, ETag,
and CRC32C in a sidecar next to the cached disk. It checks the remote generation
before every run and downloads a generation-pinned object when the bucket path
changes, so replacing an object at the same URI does not leave a stale cache.

## Lifecycle

1. Validate Docker and `/dev/kvm`.
2. Resolve `disk_source` to a local qcow2.
3. Run `qemu-img` from the runner image to create a per-run backing overlay.
4. Start the runner with KVM, `NET_ADMIN`, dynamic loopback ports, and task shape.
5. Wait for CUA readiness while also monitoring early container exit.
6. Collect run records; copy task output only if explicitly enabled.
7. Remove the container, overlay, and exchange directory on delete cleanup.

For `output_path: local`, the provider bind-mounts an empty per-run exchange
directory into the runner. Dockur exposes that directory to Windows as
`\\host.lan\Data` and to Ubuntu through a temporary CIFS mount. The guest copies
the complete output tree into the share, after which the host moves it into the
run directory. CUA remains a fallback, but does not carry file contents during
the normal local QEMU path.

For `output_path: gs://...`, set `gcs_sa_key` to a host-side service-account
JSON key. The provider injects the key into the guest after CUA becomes ready,
and the guest uploads its output directory directly with `gsutil`. Output bytes
do not pass through CUA or the local exchange directory.

## Independent health and diagnostics

Sandbox execution prepares a token-scoped control directory in the exchange
share before launch. Linux requires `mount.cifs` and noninteractive sudo; the
mount must pass a bidirectional handshake or launch fails. Windows uses the
existing UNC share. The supervisor publishes a progress sequence every five
seconds, mirrored results, and an explicit process-tree cleanup confirmation.
Share I/O runs in a daemon thread so an unavailable SMB service cannot block
the supervisor's deadline or child reaping.

The host accepts heartbeat freshness only after observing sequence advancement
within 45 seconds on its own monotonic clock. This avoids guest clock skew and
cached old files. A live supervisor permits continued waiting when CUA is
unavailable, within the original wall budget. It does not prove that the agent
is making useful task progress. Stopping uses both CUA and the share; evaluation
still requires a matching run token and `stopped: true` confirmation.

When CUA liveness fails, the host independently queries Docker state and QMP
through `docker exec` and runner-local `nc`. QMP reports QEMU execution state
and KVM availability. QMP `running` alone does not establish guest or solver
health. No QMP port needs to be published on the host.

Before releasing an acquired QEMU VM, the lifecycle saves
`<run_dir>/diagnostics/qemu.json`. Collection is bounded to 25 seconds and
cannot replace the primary run error. The report includes Docker running,
paused and OOM state, QMP replies, bounded runner/QEMU log tails, recent probe
history, and available shared guest telemetry. Guest telemetry includes
memory, pressure, VM counters, network state, CUA `/status`, and a kernel log
tail when readable. Missing or failed probes are recorded explicitly.
Docker `OOMKilled: false` does not rule out an OOM inside the guest. The
collector does not read environment variables, credentials, or agent prompts.

For an existing VM, the read-only diagnostic interface is:

```bash
uv run python -m ale_run.environments.qemu_control CONTAINER_NAME
uv run python -m ale_run.environments.qemu_control CONTAINER_NAME \
  --output ./vm-diagnostics \
  --control-dir /path/to/exchange/control/RUN_TOKEN
```

`--control-dir` is optional. A VM started by older code has no supervisor shared
channel; the diagnostic interface reports guest health as unknown and still
collects Docker/QMP evidence. Existing runner processes must finish or restart
before they use the new supervisor protocol. These changes do not modify the
agent prompt or installed CLI version.

## Optional host proxy for guest downloads

The optional `qemu.network` field joins an existing Docker bridge. It defaults
to the usual Docker network and does not set proxy variables inside the guest.
To use the host's existing unauthenticated SOCKS5 listener at `127.0.0.1:7890`,
run this from the repository root:

```bash
sudo python3 scripts/qemu_proxy.py start --mihomo /absolute/path/to/mihomo
sudo python3 scripts/qemu_proxy.py status
```

The script reuses the mihomo executable in a separate systemd transient service.
It creates `ale-qemu-egress` with the fixed bridge interface `br-ale-egress`,
and redirects only that bridge's public IPv4 TCP traffic. A relay on the bridge
gateway at port 17890 forwards it to the existing SOCKS5 listener. TLS/HTTP
sniffing preserves domain names for routing without decrypting TLS. Host and
private destinations bypass the relay; `maas.yicloud.com.cn` connects directly.
Add `--direct-domain example.org` for additional direct domains. Repeated start
keeps an already healthy service's configuration; stop it before changing it.

In each desired snapshot's existing `qemu:` mapping, add:

```yaml
network: ale-qemu-egress
```

Each run joins the shared network. Docker removes its attachment when the
container is deleted; no per-task gateway or custom lease database is needed.
Retained VMs keep their network. Runner network addresses are included in
`diagnostics/qemu.json`; the following commands inspect the shared gateway:

```bash
sudo journalctl -u ale-qemu-egress.service --no-pager -n 30
sudo python3 scripts/qemu_proxy.py stop
```

Stop refuses to tear down a network with attached containers. The manager uses
a host file lock, and deletes the network before stopping the relay so Docker
can reject cleanup if a new container attached concurrently. Startup failure
rolls back its network and rules when no container has attached. It never adopts
an unrelated network of the same name or changes the host's original proxy.

This intentionally small implementation requires Linux, systemd, iptables,
Docker, `ss`, and a mihomo executable. It uses an IPv4-only Docker network; DNS
and other UDP traffic remain direct. UDP/443 is rejected on this bridge to let
compatible HTTPS clients fall back to TCP. It does not implement UDP proxying
or guest IPv6 egress. Existing VM connections are not migrated. The service and
rules are not installed for boot; run start again after reboot. Host-side runner
image and base-disk downloads do not use this guest network.

## Initial limitations

- GPU tasks are not supported. PCIe passthrough requires IOMMU/VFIO host setup,
  a dedicated GPU, QEMU `vfio-pci` configuration, and matching guest drivers.
- `task_data_source: local:...` is not supported. The initial provider expects
  input, software, and encrypted reference data to be baked into the qcow2.
- The host does not enforce a global CPU or memory admission policy. Set ALE
  `concurrency` conservatively for the available machine resources.
