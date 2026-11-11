"""Run untrusted Python in a locked-down, throwaway Docker container.

The executor is synchronous (the Docker SDK is blocking); the MCP server calls it from a
worker thread. Security properties of every container:

* ``network_mode="none"``; no host mounts; code and files are sent over stdin
* read-only root filesystem; small ``tmpfs`` at ``/tmp`` (noexec) and ``/work``
* non-root user, ``cap_drop=["ALL"]``, ``no-new-privileges``
* memory limit with swap equal to it, 1 CPU by default, ``pids_limit``, ``nofile`` ulimit
* wall-clock timeout: the container is killed and always removed
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import secrets
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import docker
import docker.errors
import requests.exceptions
from docker.types import LogConfig, Ulimit

from bioagent.mcp_servers.sandbox.models import LimitsInfo, OutputFile, SandboxResult

audit_log = logging.getLogger("bioagent.sandbox.audit")

MAX_CODE_BYTES = 100_000
MAX_INPUT_FILES = 10
MAX_INPUT_FILE_BYTES = 1_000_000
MAX_STREAM_BYTES = 64 * 1024
MAX_OUTPUT_FILES = 5
MAX_OUTPUT_FILE_BYTES = 512 * 1024
SANDBOX_UID = "10001:10001"
OUTPUT_MARKER = "\n<<<BIOAGENT-SANDBOX-OUTPUT:"
RUNNER_COMMAND = ["python", "-s", "/opt/sandbox/runner.py"]
_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_RESERVED_NAMES = {"main.py", "out"}


class SandboxInputError(ValueError):
    """Code or files rejected before execution."""


class SandboxUnavailableError(RuntimeError):
    """Docker daemon or sandbox image is not available."""


@dataclass(frozen=True)
class SandboxLimits:
    image: str = "bioagent-sandbox:latest"
    memory: str = "512m"
    cpus: float = 1.0
    pids: int = 128
    timeout_max_s: int = 120
    tmpfs_size: str = "64m"


class DockerLike(Protocol):
    """The subset of ``docker.DockerClient`` the executor uses (eases mocking)."""

    @property
    def api(self) -> Any: ...

    @property
    def containers(self) -> Any: ...


def validate_inputs(code: str, files: Mapping[str, str] | None) -> dict[str, str]:
    if not code.strip():
        raise SandboxInputError("code is empty")
    if len(code.encode()) > MAX_CODE_BYTES:
        raise SandboxInputError(f"code exceeds {MAX_CODE_BYTES} bytes")
    files = dict(files or {})
    if len(files) > MAX_INPUT_FILES:
        raise SandboxInputError(f"at most {MAX_INPUT_FILES} input files")
    for name, content in files.items():
        if not _FILENAME_RE.match(name) or name in _RESERVED_NAMES or ".." in name:
            raise SandboxInputError(f"invalid input file name: {name[:60]!r}")
        if len(content.encode()) > MAX_INPUT_FILE_BYTES:
            raise SandboxInputError(f"input file {name!r} exceeds {MAX_INPUT_FILE_BYTES} bytes")
    return files


def truncate(data: bytes, limit: int = MAX_STREAM_BYTES) -> tuple[str, bool]:
    """Decode and cap a stream; returns (text, was_truncated)."""
    if len(data) <= limit:
        return data.decode("utf-8", errors="replace"), False
    head = data[:limit].decode("utf-8", errors="ignore")
    return head + f"\n... [truncated {len(data) - limit} bytes]", True


def split_outputs(stdout: bytes, nonce: str) -> tuple[bytes, list[OutputFile]]:
    """Strip the runner's trailing output-files marker (matching ``nonce``) from stdout."""
    marker = (OUTPUT_MARKER + nonce + ">>>").encode()
    idx = stdout.rfind(marker)
    if idx < 0:
        return stdout, []
    raw = stdout[idx + len(marker) :].strip()
    try:
        items = json.loads(raw)
        files = [OutputFile.model_validate(i) for i in items if isinstance(i, dict) and "name" in i]
    except (ValueError, TypeError):
        files = []
    return stdout[:idx], files


def host_config_kwargs(limits: SandboxLimits) -> dict[str, Any]:
    """Arguments for ``api.create_host_config`` (separate so tests can assert them)."""
    return {
        "network_mode": "none",
        "read_only": True,
        "tmpfs": {
            "/tmp": f"rw,noexec,nosuid,nodev,size={limits.tmpfs_size}",
            "/work": f"rw,nosuid,nodev,size={limits.tmpfs_size},uid=10001,gid=10001,mode=0700",
        },
        "mem_limit": limits.memory,
        "memswap_limit": limits.memory,
        "nano_cpus": int(limits.cpus * 1_000_000_000),
        "pids_limit": limits.pids,
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "ulimits": [Ulimit(name="nofile", soft=256, hard=256)],
        "ipc_mode": "none",
        "log_config": LogConfig(
            type=LogConfig.types.JSON, config={"max-size": "8m", "max-file": "1"}
        ),
    }


def create_kwargs() -> dict[str, Any]:
    """Arguments for ``api.create_container`` other than image and host config."""
    return {
        "command": RUNNER_COMMAND,
        "user": SANDBOX_UID,
        "working_dir": "/work",
        "network_disabled": True,
        "environment": {"PYTHONUNBUFFERED": "1"},
        "labels": {"bioagent.sandbox": "1"},
        # With stdin_open and detach=False the SDK sets StdinOnce: the container's stdin closes
        # when our attach disconnects, so the runner sees EOF after the payload.
        "stdin_open": True,
        "detach": False,
    }


class SandboxExecutor:
    def __init__(
        self,
        limits: SandboxLimits,
        client: DockerLike | None = None,
        docker_host: str | None = None,
    ) -> None:
        self.limits = limits
        self._client = client
        self._docker_host = docker_host

    @property
    def client(self) -> DockerLike:
        if self._client is None:
            try:
                self._client = (
                    docker.DockerClient(base_url=self._docker_host)
                    if self._docker_host
                    else docker.from_env()
                )
            except docker.errors.DockerException as exc:
                raise SandboxUnavailableError(f"Docker daemon not reachable: {exc}") from exc
        return self._client

    def clamp_timeout(self, timeout_s: float) -> float:
        return max(1.0, min(float(timeout_s), float(self.limits.timeout_max_s)))

    def run(
        self, code: str, timeout_s: float = 30, files: Mapping[str, str] | None = None
    ) -> SandboxResult:
        files = validate_inputs(code, files)
        timeout = self.clamp_timeout(timeout_s)
        nonce = secrets.token_hex(16)
        payload = json.dumps(
            {
                "code": code,
                "files": files,
                "nonce": nonce,
                "output": {"max_files": MAX_OUTPUT_FILES, "max_bytes": MAX_OUTPUT_FILE_BYTES},
            }
        ).encode()
        info = LimitsInfo(
            image=self.limits.image,
            timeout_s=timeout,
            memory=self.limits.memory,
            cpus=self.limits.cpus,
            pids=self.limits.pids,
        )

        client = self.client
        try:
            host_config = client.api.create_host_config(**host_config_kwargs(self.limits))
            created = client.api.create_container(
                self.limits.image, host_config=host_config, **create_kwargs()
            )
            container = client.containers.get(created["Id"])
        except docker.errors.ImageNotFound as exc:
            raise SandboxUnavailableError(
                f"sandbox image {self.limits.image!r} not found; run `make sandbox-image`"
            ) from exc
        except (docker.errors.APIError, requests.exceptions.ConnectionError) as exc:
            raise SandboxUnavailableError(f"could not create sandbox container: {exc}") from exc

        start = time.monotonic()
        timed_out = False
        exit_code: int | None = None
        oom_killed = False
        try:
            sock = container.attach_socket(params={"stdin": 1, "stream": 1})
            container.start()
            start = time.monotonic()
            _send_stdin(sock, payload)
            try:
                status = container.wait(timeout=timeout)
                exit_code = int(status.get("StatusCode", -1))
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
                timed_out = True
                with contextlib.suppress(docker.errors.APIError):  # may have just exited
                    container.kill()
            duration_ms = int((time.monotonic() - start) * 1000)
            raw_out = container.logs(stdout=True, stderr=False) or b""
            raw_err = container.logs(stdout=False, stderr=True) or b""
            try:
                container.reload()
                oom_killed = bool(container.attrs.get("State", {}).get("OOMKilled", False))
            except docker.errors.APIError:
                pass
        finally:
            try:
                container.remove(force=True)
            except docker.errors.APIError:
                audit_log.warning("failed to remove sandbox container %s", container.id)

        code_hash = hashlib.sha256(code.encode()).hexdigest()
        body, output_files = split_outputs(raw_out, nonce)
        stdout, out_trunc = truncate(body)
        stderr, err_trunc = truncate(raw_err)
        result = SandboxResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            timed_out=timed_out,
            oom_killed=oom_killed,
            stdout_truncated=out_trunc,
            stderr_truncated=err_trunc,
            output_files=output_files,
            limits=info,
        )
        audit_log.info(
            json.dumps(
                {
                    "event": "sandbox_run",
                    "code_sha256": code_hash,
                    "code_bytes": len(code.encode()),
                    "input_files": len(files),
                    "limits": info.model_dump(),
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                    "oom_killed": oom_killed,
                    "duration_ms": duration_ms,
                    "stdout_bytes": len(body),
                    "stderr_bytes": len(raw_err),
                    "output_files": len(output_files),
                    "container": str(getattr(container, "short_id", "")),
                }
            )
        )
        # The code itself is only logged at DEBUG, never at INFO.
        audit_log.debug("sandbox code (sha256=%s):\n%s", code_hash, code)
        return result


def _send_stdin(sock: Any, payload: bytes) -> None:
    """Write the payload to the container's stdin and close it (stdin_once ends input)."""
    raw = getattr(sock, "_sock", sock)
    try:
        raw.sendall(payload)
        with contextlib.suppress(OSError):
            raw.shutdown(socket.SHUT_WR)
    finally:
        with contextlib.suppress(OSError):
            sock.close()
