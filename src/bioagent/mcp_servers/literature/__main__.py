"""Run the literature MCP server over stdio: ``python -m bioagent.mcp_servers.literature``."""

from bioagent.mcp_servers.literature.server import mcp


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
