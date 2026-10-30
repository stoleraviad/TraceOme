"""Owner-host adapter: the runner port over the Docker CLI on the machine running the tool."""

from __future__ import annotations

import json
import os
import subprocess

from .runner import Binding, Invocation, RunResult, pinned_binding
from .table import ToolEntry


class DockerHostRunner:
    """Run one invocation with `docker run`, without network, as the calling user.

    Inputs are read-only bind mounts and the output directory the one writable mount.
    The reported digest is the image's repository digest from `docker image inspect`;
    it is None when Docker has no such image, for example after a failed pull.
    """

    name = "owner-host-docker"
    identity_fields = ("image_digest",)

    def __init__(self, *, docker: str = "docker") -> None:
        self._docker = docker

    def bind(self, entry: ToolEntry) -> Binding:
        return pinned_binding(entry, runner=self.name, fields=self.identity_fields)

    def run(self, invocation: Invocation) -> RunResult:
        pin = invocation.pin
        image = f"{pin.image_repository}@{pin.image_digest}"
        argv = [
            self._docker,
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--workdir",
            invocation.workdir,
        ]
        for mount in invocation.mounts:
            spec = f"type=bind,source={mount.host_path},target={mount.container_path}"
            argv += ["--mount", spec + (",readonly" if mount.read_only else "")]
        argv += [image, *invocation.argv]
        if invocation.stdout_path is None:
            completed = subprocess.run(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
                timeout=invocation.timeout_seconds,
            )
        else:
            with invocation.stdout_path.open("xb") as stdout:
                completed = subprocess.run(
                    argv,
                    stdout=stdout,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=invocation.timeout_seconds,
                )
        return RunResult(
            exit_code=completed.returncode,
            identity={"image_digest": self._repository_digest(image)},
            executed_argv=tuple(argv),
            stderr=completed.stderr,
        )

    def _repository_digest(self, image: str) -> str | None:
        inspected = subprocess.run(
            [
                self._docker,
                "image",
                "inspect",
                "--format",
                "{{json .RepoDigests}}",
                image,
            ],
            capture_output=True,
            check=False,
        )
        if inspected.returncode != 0:
            return None
        repository = image.split("@", 1)[0]
        digests = [
            reference.split("@", 1)[1]
            for reference in json.loads(inspected.stdout)
            if reference.split("@", 1)[0] == repository
        ]
        return digests[0] if len(digests) == 1 else None
