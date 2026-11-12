"""Run the retrieval MCP server over stdio: ``python -m bioagent.mcp_servers.retrieval``."""

from bioagent.mcp_servers.retrieval.server import mcp


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
