"""Provider implementations for VM lifecycle.

``Provider`` ABC + ``SandboxSpec`` + ``SandboxHandle`` + ``ReleaseMode`` live in
:mod:`ale_run.base_interface`; this package only holds the backends:

  - :class:`GcloudProvider` (``gcloud.py``): ephemeral GCE VMs.
  - :class:`AwsProvider` (``aws.py``): ephemeral EC2 instances.
  - :class:`AliyunProvider` (``aliyun.py``): ephemeral Alibaba Cloud ECS instances.
  - :class:`StaticProvider` (``static.py``): a pre-existing VM endpoint.
  - :class:`DockerProvider` (``docker.py``): ephemeral Docker containers.
  - :class:`QemuProvider` (``qemu.py``): ephemeral local QEMU VMs in Docker.
  - :class:`ProxmoxProvider` (``proxmox.py``): ephemeral Proxmox template clones.
"""

from ...base_interface import Provider, ReleaseMode, SandboxHandle, SandboxSpec
from .aliyun import AliyunProvider, AliyunProviderConfig
from .aws import AwsProvider, AwsProviderConfig
from .docker import DockerProvider, DockerProviderConfig
from .gcloud import GcloudProvider, GcloudProviderConfig
from .proxmox import ProxmoxProvider, ProxmoxProviderConfig
from .qemu import QemuProvider, QemuProviderConfig
from .static import StaticProvider, StaticProviderConfig

__all__ = [
    "AliyunProvider",
    "AliyunProviderConfig",
    "AwsProvider",
    "AwsProviderConfig",
    "DockerProvider",
    "DockerProviderConfig",
    "GcloudProvider",
    "GcloudProviderConfig",
    "Provider",
    "ProxmoxProvider",
    "ProxmoxProviderConfig",
    "QemuProvider",
    "QemuProviderConfig",
    "ReleaseMode",
    "SandboxHandle",
    "SandboxSpec",
    "StaticProvider",
    "StaticProviderConfig",
]
