"""Run the sandbox MCP server over stdio: ``python -m bioagent.mcp_servers.sandbox``."""

import logging

from bioagent.mcp_servers.sandbox.server import mcp


def main() -> None:
    # Audit lines go to stderr; stdout carries the MCP protocol.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
