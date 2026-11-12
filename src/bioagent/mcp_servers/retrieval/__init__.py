"""Corpus retrieval MCP server (hybrid BM25 + dense + rerank over the local abstract corpus).

Kept separate from the literature server: that one is a thin, network-only PubMed/Europe PMC
client, while this one loads local indexes and torch models (hundreds of MB of RAM).
"""
