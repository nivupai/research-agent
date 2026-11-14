"""Prompt-injection defences and the audit log (Weeks 8-9 part 2).

* ``policy``: per-role tool allowlists and argument validation for tool calls.
* ``output``: answer checks (unknown PMIDs, canary leaks, instruction echoes).
* ``audit``: append-only, hash-chained JSONL (+ optional SQLite) record of every job action.

Untrusted-content spotlighting (per-run random delimiters, sanitising) lives in
``bioagent.agents.untrusted``. ``SECURITY_DEFENSES=off`` disables the policy, output checks
and random delimiters (role scopes stay) so the eval can compare.
"""
