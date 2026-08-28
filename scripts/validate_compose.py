"""Static checks on the container setup, runnable without a Docker daemon.

This originally existed because there was no daemon available to build with. The stack has
since been built, run, and deployed, so this is no longer the only evidence that the container
path works. It stays because it is a great deal faster than a build and it catches the errors
that are actually common when the compose file changes: a service that depends on one that does
not exist, a command pointing at a module that was renamed, a bind mount whose host path is
missing, a port declared twice.

What it cannot tell you is whether the image builds, whether the pinned wheels resolve on
linux/amd64, or whether the containers can reach each other. Run `docker compose up --build` for
that.
"""

from __future__ import annotations

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
            # A mapping is "container", "host:container" or "ip:host:container". Taking the
            # first field treats the bind address as the port in the three part form and the
            # container port as the host port in the one part form, so read from the right.
            parts = str(mapping).split(":")
            host = parts[-2] if len(parts) >= 2 else parts[-1]
            if host in published_ports:
                problems.append(
                    f"{name}: publishes host port {host}, already used by {published_ports[host]}"
                )
            published_ports[host] = name

        for volume in service.get("volumes", []):
            host = str(volume).split(":")[0]
            if host.startswith("./") and not (PROJECT_ROOT / host[2:]).exists():
                problems.append(f"{name}: bind mount source '{host}' does not exist")

        # Length is checked before indexing. A checker that raises IndexError on a malformed
        # compose file reports a traceback instead of the problem it exists to name.
        command = service.get("command")
        if isinstance(command, list) and len(command) >= 3:
            if command[:2] == ["python", "-m"]:
                module = PROJECT_ROOT / (command[2].replace(".", "/") + ".py")
                if not module.exists():
                    problems.append(f"{name}: command runs {command[2]}, which does not exist")
            elif command[0] == "streamlit":
                script = PROJECT_ROOT / command[2]
                if not script.exists():
                    problems.append(f"{name}: streamlit script '{command[2]}' does not exist")
        elif isinstance(command, list) and command:
            problems.append(f"{name}: command {command} is too short to name a target")

    # The image declares a healthcheck against the API port, and every service inherits it.
    # A role that does not serve that port then reports unhealthy for its whole life unless it
    # says otherwise, which is how the monitor and the dashboard sat permanently red in
    # `docker compose ps` while working perfectly. Silence is the bug, so silence is what is
    # checked: state a healthcheck or disable it, but do not inherit one by accident.
    for name, service in services.items():
        if "healthcheck" not in service:
            problems.append(
                f"{name}: declares no healthcheck and no `healthcheck: disable: true`, so it "
                "inherits the image's API port check. Any role that does not serve that port "
                "will report unhealthy forever"
            )

    # A service that depends on another being healthy needs that service to define a check.
    for name, service in services.items():
        depends = service.get("depends_on") or {}
        if not isinstance(depends, dict):
            continue
        for target, rule in depends.items():
            if (rule or {}).get("condition") == "service_healthy":
                check = services.get(target, {}).get("healthcheck")
                if target in services and not check:
                    problems.append(
                        f"{name}: waits for '{target}' to be healthy, but '{target}' declares "
                        "no healthcheck, so compose will never consider it healthy"
                    )
                elif isinstance(check, dict) and check.get("disable"):
                    problems.append(
                        f"{name}: waits for '{target}' to be healthy, but '{target}' disables "
                        "its healthcheck, so that condition can never be met"
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
        "This is a static check only. It does not build the image or start the stack, run "
        "`docker compose up --build` for that. See README, 'Containerisation and cloud "
        "deployment'."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
