"""Submit the working graph to the broker and wait for the answer.

    submit.py validate [--graph results/graph]
    submit.py rehearse [--graph results/graph] [--timeout 540]
    submit.py wait REQUEST_ID [--timeout 540]

A request is a snapshot of the graph plus a small JSON file, written under
``.requests/``. Editing the graph afterwards does not change a request that
was already submitted.
"""
import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUESTS = ROOT / ".requests"


def submit(kind, graph):
    source = (ROOT / graph).resolve()
    if ROOT not in source.parents or not (source / "workflow.json").is_file():
        raise SystemExit(f"error: {graph} is not a graph directory inside the workspace")
    name = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    staging = REQUESTS / (name + ".tmp")
    shutil.copytree(source, staging / "graph", symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (staging / "request.json").write_text(json.dumps(
        {"id": name, "type": kind, "graph": graph, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    staging.rename(REQUESTS / name)  # the broker only sees complete requests
    return name


def wait(name, timeout):
    response = REQUESTS / name / "response.json"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if response.exists():
            try:
                return json.loads(response.read_text())
            except json.JSONDecodeError:
                pass  # still being written
        time.sleep(1.0)
    return None


def show(name, answer):
    if answer is None:
        print(f"request {name} is still running. Continue with: ./eval/run wait {name}")
        return 0
    print(f"request {name}: {'accepted' if answer.get('ok') else 'REJECTED'}")
    for key in ("message", "round", "results", "visible", "remaining_rehearsals"):
        if answer.get(key) is not None:
            value = answer[key]
            print(f"{key}: {value if isinstance(value, (str, int)) else json.dumps(value)}")
    if answer.get("output"):
        print("output:")
        print(answer["output"].rstrip())
    return 0 if answer.get("ok") else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["validate", "rehearse", "wait"])
    p.add_argument("request", nargs="?", help="request id, for wait")
    p.add_argument("--graph", default="results/graph")
    p.add_argument("--timeout", type=float, default=540.0, help="seconds to wait before returning (default 540)")
    a = p.parse_args()
    if a.command == "wait":
        if not a.request or not (REQUESTS / a.request).is_dir():
            p.error("wait needs the id of a submitted request")
        name = a.request
    else:
        name = submit(a.command, a.graph)
        print(f"submitted {a.command} request {name}")
    sys.exit(show(name, wait(name, a.timeout)))


if __name__ == "__main__":
    main()
