"""In-container runner for the bioagent sandbox (copied into the image at /opt/sandbox).

Reads one JSON payload from stdin: {"code": str, "files": {name: text}, "nonce": str,
"output": {"max_files": int, "max_bytes": int}}. Writes the input files and the code into
/work, runs the code in a child Python process (so it cannot see the nonce), then prints a
single marker line with any small PNG/CSV files found in /work/out, base64-encoded.
Exits with the child's exit code (128 + signal number if it was killed by a signal).
"""

import base64
import json
import pathlib
import subprocess
import sys

MARKER = "\n<<<BIOAGENT-SANDBOX-OUTPUT:"
ALLOWED_SUFFIXES = {".png": "image/png", ".csv": "text/csv"}


def collect_outputs(
    out_dir: pathlib.Path, max_files: int, max_bytes: int
) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    if not out_dir.is_dir():
        return files
    for path in sorted(out_dir.iterdir()):
        if len(files) >= max_files:
            break
        mime = ALLOWED_SUFFIXES.get(path.suffix.lower())
        if mime is None or path.is_symlink() or not path.is_file():
            continue
        size = path.stat().st_size
        if size > max_bytes:
            files.append({"name": path.name, "mime": mime, "size": size, "skipped": "too large"})
            continue
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        files.append({"name": path.name, "mime": mime, "size": size, "base64": data})
    return files


def main() -> int:
    payload = json.loads(sys.stdin.buffer.read() or b"{}")
    work = pathlib.Path("/work")
    for name, content in (payload.get("files") or {}).items():
        (work / name).write_text(content, encoding="utf-8")
    out_dir = work / "out"
    out_dir.mkdir(exist_ok=True)
    script = work / "main.py"
    script.write_text(payload.get("code", ""), encoding="utf-8")

    child = subprocess.run(
        [sys.executable, "-s", str(script)], cwd=work, stdin=subprocess.DEVNULL, check=False
    )
    code = child.returncode if child.returncode >= 0 else 128 - child.returncode

    limits = payload.get("output") or {}
    try:
        outputs = collect_outputs(
            out_dir, int(limits.get("max_files", 5)), int(limits.get("max_bytes", 512 * 1024))
        )
    except OSError as exc:
        outputs = [{"error": f"could not collect outputs: {exc}"}]
    sys.stdout.flush()
    sys.stdout.write(MARKER + str(payload.get("nonce", "")) + ">>>" + json.dumps(outputs) + "\n")
    sys.stdout.flush()
    return code


if __name__ == "__main__":
    sys.exit(main())
