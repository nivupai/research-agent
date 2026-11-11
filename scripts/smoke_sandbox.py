"""End-to-end smoke test: launch the sandbox MCP server over stdio and run code through it.

Requires Docker and the sandbox image (``make sandbox-image``).
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

CASES: list[tuple[str, dict[str, Any]]] = [
    ("hello", {"code": "import numpy as np; print('sum =', np.arange(10).sum())"}),
    (
        "plot",
        {
            "code": (
                "import matplotlib.pyplot as plt\n"
                "plt.plot([1, 4, 9]); plt.savefig('/work/out/squares.png')\n"
                "print('saved')"
            )
        },
    ),
    (
        "network",
        {
            "code": (
                "import socket\n"
                "try:\n"
                "    socket.create_connection(('1.1.1.1', 443), timeout=3); print('CONNECTED')\n"
                "except OSError as e:\n"
                "    print('blocked:', e)"
            )
        },
    ),
    ("timeout", {"code": "while True: pass", "timeout_s": 2}),
]


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "bioagent.mcp_servers.sandbox"]
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        print("tools:", [t.name for t in tools.tools])
        for label, args in CASES:
            result = await session.call_tool("run_python", args)
            if result.is_error:
                print(f"{label}: ERROR {result.content[0].text}")  # type: ignore[union-attr]
                continue
            data = result.structured_content or {}
            files = [(f["name"], f["size"]) for f in data.get("output_files", [])]
            print(
                f"{label}: exit={data['exit_code']} timed_out={data['timed_out']} "
                f"{data['duration_ms']}ms stdout={data['stdout'].strip()!r} files={files}"
            )
        bad = await session.call_tool("run_python", {"code": "   "})
        print("empty code error:", bad.content[0].text)  # type: ignore[union-attr]


if __name__ == "__main__":
    asyncio.run(main())
