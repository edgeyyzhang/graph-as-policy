#!/usr/bin/env python3
"""Print a readable summary of an agent transcript written by launch_agent.sh.

    examples/rehearse_loop/agent_log.py /path/to/<workspace>.agent.jsonl [--full]
"""
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("transcript")
    p.add_argument("--full", action="store_true", help="do not shorten text and tool results")
    a = p.parse_args()
    cut = (lambda s, n: s) if a.full else (lambda s, n: s if len(s) <= n else s[:n] + " ...")
    for line in open(a.transcript, encoding="utf-8"):
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = m.get("type")
        if kind == "system" and m.get("subtype") == "init":
            print(f"[start] model {m.get('model')}, tools {m.get('tools')}, "
                  f"mcp servers {[s.get('name') for s in m.get('mcp_servers', [])]}")
        elif kind == "assistant":
            for c in m["message"]["content"]:
                if c["type"] == "text" and c["text"].strip():
                    print("[agent] " + cut(c["text"].strip(), 1500))
                elif c["type"] == "tool_use":
                    print(f"[tool]  {c['name']} " + cut(json.dumps(c["input"]), 220))
        elif kind == "user":
            for c in m["message"]["content"]:
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    body = c.get("content")
                    body = body if isinstance(body, str) else json.dumps(body)
                    flag = "ERROR " if c.get("is_error") else ""
                    print("        -> " + flag + cut(" ".join(body.split()), 200))
        elif kind == "result":
            print(f"[end]   {m.get('subtype')}, turns {m.get('num_turns')}, "
                  f"cost ${m.get('total_cost_usd')}, duration {round((m.get('duration_ms') or 0) / 60000, 1)} min")


if __name__ == "__main__":
    main()
