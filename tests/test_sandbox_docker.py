"""Integration tests against a real Docker daemon. Run with ``make test-docker``.

Skipped automatically when no daemon or no sandbox image is available.
"""

from __future__ import annotations

import base64
import textwrap

import pytest

from bioagent.config import get_settings
from bioagent.mcp_servers.sandbox.executor import SandboxExecutor
from bioagent.mcp_servers.sandbox.models import SandboxResult
from bioagent.mcp_servers.sandbox.server import executor_from_settings

pytestmark = pytest.mark.docker


@pytest.fixture(scope="module")
def sandbox() -> SandboxExecutor:
    docker = pytest.importorskip("docker")
    settings = get_settings()
    try:
        client = docker.from_env()
        client.ping()
        client.images.get(settings.sandbox_image)
    except Exception as exc:
        pytest.skip(f"Docker sandbox unavailable: {exc}")
    return executor_from_settings(settings)


def run(
    sandbox: SandboxExecutor,
    code: str,
    timeout_s: float = 20,
    files: dict[str, str] | None = None,
) -> SandboxResult:
    return sandbox.run(textwrap.dedent(code), timeout_s=timeout_s, files=files)


def test_print(sandbox: SandboxExecutor) -> None:
    result = run(sandbox, "print('hello sandbox')")
    assert result.exit_code == 0 and result.stdout == "hello sandbox\n" and not result.timed_out


def test_numpy_and_input_files(sandbox: SandboxExecutor) -> None:
    result = run(
        sandbox,
        """
        import numpy as np, pandas as pd
        df = pd.read_csv("data.csv")
        print(int(np.sum(df["x"])))
        """,
        files={"data.csv": "x\n1\n2\n3\n"},
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "6"


def test_matplotlib_png_returned(sandbox: SandboxExecutor) -> None:
    result = run(
        sandbox,
        """
        import matplotlib.pyplot as plt
        plt.plot([1, 2, 3])
        plt.savefig("/work/out/plot.png")
        """,
    )
    assert result.exit_code == 0, result.stderr
    [png] = result.output_files
    assert png.name == "plot.png" and png.base64
    assert base64.b64decode(png.base64)[:8] == b"\x89PNG\r\n\x1a\n"


def test_non_root_without_capabilities(sandbox: SandboxExecutor) -> None:
    result = run(
        sandbox,
        """
        import os
        caps = [l for l in open('/proc/self/status') if l.startswith('CapEff')][0].split()[1]
        print(os.getuid(), os.getgid(), int(caps, 16))
        """,
    )
    assert result.stdout.split() == ["10001", "10001", "0"]


def test_network_is_blocked(sandbox: SandboxExecutor) -> None:
    result = run(
        sandbox,
        """
        import socket
        try:
            socket.create_connection(("1.1.1.1", 443), timeout=3)
            print("CONNECTED")
        except OSError as exc:
            print("BLOCKED", type(exc).__name__)
        """,
    )
    assert result.stdout.startswith("BLOCKED"), result.stdout


def test_writes_outside_tmp_and_work_fail(sandbox: SandboxExecutor) -> None:
    result = run(
        sandbox,
        """
        results = []
        for path in ["/etc/evil", "/usr/evil", "/opt/sandbox/runner.py", "/evil"]:
            try:
                open(path, "w").write("x")
                results.append(f"WROTE {path}")
            except OSError:
                results.append("denied")
        open("/tmp/ok", "w").write("x")
        open("/work/ok", "w").write("x")
        print(" ".join(results))
        """,
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout.split() == ["denied"] * 4


def test_infinite_loop_times_out(sandbox: SandboxExecutor) -> None:
    result = run(sandbox, "while True: pass", timeout_s=2)
    assert result.timed_out and result.exit_code is None
    assert result.duration_ms < 10_000


def test_memory_bomb_is_killed(sandbox: SandboxExecutor) -> None:
    result = run(
        sandbox,
        """
        chunks = []
        while True:
            chunks.append(b"x" * (50 * 1024 * 1024))
        """,
        timeout_s=30,
    )
    assert not result.timed_out
    assert result.exit_code == 137


def test_fork_bomb_contained_by_pids_limit(sandbox: SandboxExecutor) -> None:
    result = run(
        sandbox,
        """
        import subprocess
        procs = []
        try:
            for _ in range(500):
                procs.append(subprocess.Popen(["sleep", "30"]))
        except OSError as exc:
            print("LIMIT", len(procs), type(exc).__name__)
        else:
            print("NO LIMIT", len(procs))
        for p in procs:
            p.kill()
        """,
        timeout_s=30,
    )
    assert result.stdout.startswith("LIMIT"), result.stdout + result.stderr
    assert int(result.stdout.split()[1]) < sandbox.limits.pids


def test_classic_fork_bomb_is_contained(sandbox: SandboxExecutor) -> None:
    result = run(
        sandbox,
        """
        import os
        while True:
            try:
                os.fork()
            except OSError:
                pass
        """,
        timeout_s=5,
    )
    assert result.timed_out or result.exit_code != 0
    # The daemon is still healthy and further runs work.
    assert run(sandbox, "print('still alive')").stdout == "still alive\n"
