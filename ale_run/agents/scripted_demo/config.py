"""Configuration for the deterministic ``demo/hello`` GUI smoke agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class ScriptedDemoConfig:
    name: ClassVar[str] = "scripted_demo"

    model: str = "none"
    connect_timeout_s: int = 120
