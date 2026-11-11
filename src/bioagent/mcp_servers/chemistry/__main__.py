"""Run the chemistry MCP server over stdio: ``python -m bioagent.mcp_servers.chemistry``."""

from bioagent.mcp_servers.chemistry.server import mcp


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
