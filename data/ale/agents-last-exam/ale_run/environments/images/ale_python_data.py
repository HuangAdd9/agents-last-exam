"""Accepted scientific Python Docker image with its baked interpreter and entrypoint."""
from dataclasses import replace

from .ale_ubuntu22_docker import IMAGE as BASE_IMAGE

IMAGE = replace(
    BASE_IMAGE,
    name="ale-python-data",
    docker_image="ale/base-coding-python-data:20260924-1",
    docker_entrypoint="/dockerstartup/science-entrypoint.sh",
    python="/opt/ale-science/venv/bin/python",
)
