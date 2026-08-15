"""Static checks on the container setup, for an environment with no Docker daemon.

This exists because the compose file in this repository has never been built or run. That is
stated plainly in the README rather than glossed over. What can still be checked without a
daemon is checked here, and it catches the errors that are actually common: a service that
depends on one that does not exist, a command pointing at a module that was renamed, a bind
mount whose host path is missing, a port declared twice.

What it cannot tell you is whether the image builds, whether the pinned wheels resolve on
linux/amd64, or whether the containers can reach each other. Run `docker compose config` and
then `docker compose up` on a machine with Docker to find that out.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE = PROJECT_ROOT / "docker-compose.yml"
DOCKERFILE = PROJECT_ROOT / "Dockerfile"

VALID_CONDITIONS = {
    "service_started",
    "service_healthy",
    "service_completed_successfully",
}


def check_compose(problems: List[str]) -> dict:
    with open(COMPOSE, "r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)

    services = spec.get("services") or {}
    if not services:
        problems.append("compose file declares no services")
        return spec

    names = set(services)
    published_ports = {}

    for name, service in services.items():
        if "build" not in service and "image" not in service:
            problems.append(f"{name}: neither build nor image is set")

        depends = service.get("depends_on") or {}
        targets = depends.keys() if isinstance(depends, dict) else depends
        for target in targets:
            if target not in names:
                problems.append(f"{name}: depends_on names '{target}', which is not a service")
            if isinstance(depends, dict):
                condition = (depends.get(target) or {}).get("condition")
                if condition and condition not in VALID_CONDITIONS:
                    problems.append(f"{name}: unknown depends_on condition '{condition}'")

        for mapping in service.get("ports", []):
            host = str(mapping).split(":")[0]
            if host in published_ports:
                problems.append(
                    f"{name}: publishes host port {host}, already used by {published_ports[host]}"
                )
            published_ports[host] = name

        for volume in service.get("volumes", []):
            host = str(volume).split(":")[0]
            if host.startswith("./") and not (PROJECT_ROOT / host[2:]).exists():
                problems.append(f"{name}: bind mount source '{host}' does not exist")

        command = service.get("command")
        if isinstance(command, list) and command[:2] == ["python", "-m"]:
            module = PROJECT_ROOT / (command[2].replace(".", "/") + ".py")
            if not module.exists():
                problems.append(f"{name}: command runs {command[2]}, which does not exist")
        if isinstance(command, list) and command and command[0] == "streamlit":
            script = PROJECT_ROOT / command[2]
            if not script.exists():
                problems.append(f"{name}: streamlit script '{command[2]}' does not exist")

    # A service that depends on another being healthy needs that service to define a check.
    for name, service in services.items():
        depends = service.get("depends_on") or {}
        if not isinstance(depends, dict):
            continue
        for target, rule in depends.items():
            if (rule or {}).get("condition") == "service_healthy":
                if target in services and "healthcheck" not in services[target]:
                    problems.append(
                        f"{name}: waits for '{target}' to be healthy, but '{target}' declares "
                        "no healthcheck, so compose will never consider it healthy"
                    )
    return spec


def check_dockerfile(problems: List[str]) -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    if not re.search(r"^FROM\s+\S+", text, re.MULTILINE):
        problems.append("Dockerfile has no FROM instruction")
    if "COPY requirements.txt" not in text:
        problems.append("Dockerfile does not copy requirements.txt before installing")

    # Everything the compose commands need must actually be copied into the image.
    for required in ["src/", "dashboard/", "config/"]:
        if f"COPY {required}" not in text:
            problems.append(f"Dockerfile never copies {required} into the image")

    if "USER " not in text:
        problems.append("Dockerfile runs as root, no USER instruction")

    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    unpinned = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.startswith("#") and "==" not in line
    ]
    if unpinned:
        problems.append(f"unpinned dependencies, the image would not be reproducible: {unpinned}")


def check_dockerignore(problems: List[str]) -> None:
    path = PROJECT_ROOT / ".dockerignore"
    if not path.exists():
        problems.append("no .dockerignore, the 158 MB raw CSV would be sent as build context")
        return
    entries = {line.strip().rstrip("/") for line in path.read_text(encoding="utf-8").splitlines()}
    for required in ["data", "models"]:
        if required not in entries:
            problems.append(f".dockerignore does not exclude '{required}'")


def main() -> int:
    problems: List[str] = []
    spec = check_compose(problems)
    check_dockerfile(problems)
    check_dockerignore(problems)

    services = list((spec.get("services") or {}).keys())
    print(f"compose services: {', '.join(services)}")
    print(f"checked: {COMPOSE.name}, {DOCKERFILE.name}, .dockerignore, requirements.txt")

    if problems:
        print("\nproblems found:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nno structural problems found.")
    print(
        "This is a static check only. The image has not been built and the stack has not been "
        "run. See README, 'Containerisation, and what is not verified'."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
