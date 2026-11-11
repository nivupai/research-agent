"""Sandbox executor tests with a mocked Docker client (no daemon needed)."""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests.exceptions
from mcp.server.mcpserver.exceptions import ToolError

from bioagent.mcp_servers.sandbox import server
from bioagent.mcp_servers.sandbox.executor import (
    MAX_CODE_BYTES,
    MAX_STREAM_BYTES,
    OUTPUT_MARKER,
    SandboxExecutor,
    SandboxInputError,
    SandboxLimits,
    split_outputs,
    truncate,
    validate_inputs,
)


class FakeSocket:
    def __init__(self) -> None:
        self.sent = b""
        self.closed = False
        self._sock = self

    def sendall(self, data: bytes) -> None:
        self.sent += data

    def shutdown(self, how: int) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FakeContainer:
    def __init__(
        self,
        stdout: bytes = b"",
        stderr: bytes = b"",
        status: int = 0,
        wait_error: Exception | None = None,
        oom: bool = False,
    ) -> None:
        self.id = "abc123"
        self.short_id = "abc123"
        self.sock = FakeSocket()
        self._stdout, self._stderr = stdout, stderr
        self._status, self._wait_error = status, wait_error
        self.attrs: dict[str, Any] = {"State": {"OOMKilled": oom}}
        self.started = self.killed = self.removed = False
        self.wait_timeout: float | None = None

    def attach_socket(self, params: dict[str, Any]) -> FakeSocket:
        return self.sock

    def start(self) -> None:
        self.started = True

    def wait(self, timeout: float) -> dict[str, int]:
        self.wait_timeout = timeout
        if self._wait_error:
            raise self._wait_error
        return {"StatusCode": self._status}

    def kill(self) -> None:
        self.killed = True

    def logs(self, stdout: bool, stderr: bool) -> bytes:
        return self._stdout if stdout else self._stderr

    def reload(self) -> None:
        pass

    def remove(self, force: bool) -> None:
        self.removed = force


class FakeClient:
    def __init__(self, container: FakeContainer) -> None:
        self.container = container
        self.api = MagicMock()
        self.api.create_host_config.side_effect = lambda **kw: {"host_config": kw}
        self.api.create_container.return_value = {"Id": container.id}
        self.containers = MagicMock()
        self.containers.get.return_value = container


def make(container: FakeContainer, **limits: Any) -> tuple[SandboxExecutor, FakeClient]:
    client = FakeClient(container)
    return SandboxExecutor(SandboxLimits(**limits), client=client), client


def payload_nonce(container: FakeContainer) -> str:
    return str(json.loads(container.sock.sent)["nonce"])


def test_container_is_locked_down() -> None:
    container = FakeContainer(stdout=b"hi\n")
    executor, client = make(container, memory="256m", cpus=0.5, pids=64)
    result = executor.run("print('hi')")

    hc = client.api.create_host_config.call_args.kwargs
    assert hc["network_mode"] == "none"
    assert hc["read_only"] is True
    assert hc["cap_drop"] == ["ALL"]
    assert hc["security_opt"] == ["no-new-privileges:true"]
    assert hc["mem_limit"] == "256m" and hc["memswap_limit"] == "256m"
    assert hc["nano_cpus"] == 500_000_000
    assert hc["pids_limit"] == 64
    assert set(hc["tmpfs"]) == {"/tmp", "/work"}
    assert "noexec" in hc["tmpfs"]["/tmp"]
    assert "binds" not in hc and "volumes" not in hc and "privileged" not in hc

    args, kwargs = client.api.create_container.call_args
    assert args == ("bioagent-sandbox:latest",)
    assert kwargs["user"] == "10001:10001"
    assert kwargs["network_disabled"] is True
    assert kwargs["stdin_open"] is True and kwargs["detach"] is False
    assert "volumes" not in kwargs

    sent = json.loads(container.sock.sent)
    assert sent["code"] == "print('hi')" and container.sock.closed
    assert container.started and container.removed
    assert result.exit_code == 0 and result.stdout == "hi\n" and not result.timed_out
    assert result.limits.network == "none"


def test_timeout_kills_and_removes() -> None:
    container = FakeContainer(wait_error=requests.exceptions.ReadTimeout("slow"))
    executor, _ = make(container, timeout_max_s=10)
    result = executor.run("while True: pass", timeout_s=500)
    assert container.wait_timeout == 10.0  # clamped to timeout_max_s
    assert result.timed_out and result.exit_code is None
    assert container.killed and container.removed


def test_timeout_is_at_least_one_second() -> None:
    container = FakeContainer()
    executor, _ = make(container)
    executor.run("pass", timeout_s=0)
    assert container.wait_timeout == 1.0


def test_container_removed_even_if_wait_crashes() -> None:
    container = FakeContainer(wait_error=RuntimeError("boom"))
    executor, _ = make(container)
    with pytest.raises(RuntimeError):
        executor.run("pass")
    assert container.removed


def test_stdout_and_stderr_truncated() -> None:
    big = b"x" * (MAX_STREAM_BYTES * 3)
    container = FakeContainer(stdout=big, stderr=big)
    executor, _ = make(container)
    result = executor.run("print('x' * 10**6)")
    assert result.stdout_truncated and result.stderr_truncated
    assert len(result.stdout) < MAX_STREAM_BYTES + 100
    assert "[truncated" in result.stderr


def test_oom_and_exit_code_reported() -> None:
    container = FakeContainer(status=137, oom=True)
    executor, _ = make(container)
    result = executor.run("x = b'1' * 10**10")
    assert result.exit_code == 137 and result.oom_killed


def test_output_files_parsed_with_nonce() -> None:
    container = FakeContainer()
    executor, _ = make(container)

    # The runner prints the marker with the nonce it was sent; emulate that.
    def logs(stdout: bool, stderr: bool) -> bytes:
        if not stdout:
            return b""
        files = [{"name": "plot.png", "mime": "image/png", "size": 3, "base64": "AAAA"}]
        marker = f"{OUTPUT_MARKER}{payload_nonce(container)}>>>{json.dumps(files)}\n"
        return b"result\n" + marker.encode()

    container.logs = logs  # type: ignore[method-assign]
    result = executor.run("pass")
    assert result.stdout == "result\n"
    assert [f.name for f in result.output_files] == ["plot.png"]


def test_forged_marker_without_nonce_is_ignored() -> None:
    forged = f'{OUTPUT_MARKER}wrong>>>[{{"name": "x.png"}}]\n'.encode()
    body, files = split_outputs(b"out" + forged, "right-nonce")
    assert files == [] and body.startswith(b"out")


def test_audit_log_has_hash_not_code(caplog: pytest.LogCaptureFixture) -> None:
    executor, _ = make(FakeContainer())
    secret_code = "print('very-distinctive-code-42')"
    with caplog.at_level(logging.INFO, logger="bioagent.sandbox.audit"):
        executor.run(secret_code)
    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(info) == 1
    entry = json.loads(info[0].getMessage())
    assert entry["event"] == "sandbox_run" and len(entry["code_sha256"]) == 64
    assert entry["limits"]["network"] == "none"
    assert "very-distinctive-code-42" not in info[0].getMessage()


@pytest.mark.parametrize(
    ("code", "files", "match"),
    [
        ("   ", None, "empty"),
        ("x" * (MAX_CODE_BYTES + 1), None, "exceeds"),
        ("pass", {"../evil.py": "x"}, "invalid input file name"),
        ("pass", {"main.py": "x"}, "invalid input file name"),
        ("pass", {f"f{i}.txt": "" for i in range(11)}, "at most"),
    ],
)
def test_input_validation(code: str, files: dict[str, str] | None, match: str) -> None:
    with pytest.raises(SandboxInputError, match=match):
        validate_inputs(code, files)


def test_truncate_handles_multibyte() -> None:
    text, cut = truncate("é".encode() * 10, limit=5)
    assert cut and text.startswith("éé")


async def test_tool_reports_errors_to_client() -> None:
    executor, _ = make(FakeContainer())
    server.set_executor(executor)
    try:
        with pytest.raises(ToolError, match="empty"):
            await server.run_python("  ")
        names = {t.name for t in await server.mcp.list_tools()}
        assert "run_python" in names
    finally:
        server.set_executor(None)
