"""Deterministic GUI smoke agent for ``demo/hello``."""

from .config import ScriptedDemoConfig
from .deployer import ScriptedDemoDeployer

__all__ = ["ScriptedDemoConfig", "ScriptedDemoDeployer"]
