"""Rounds rehearsed so far, the remaining budget, and requests still running."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    path = ROOT / "eval" / "status.json"
    if not path.exists():
        print("no status yet: the broker has not written eval/status.json")
        return
    status = json.loads(path.read_text())
    print(f"task: {status.get('instruction')}")
    print(f"sim: {status.get('sim')}; visible cases: {status.get('visible_cases')}")
    print(f"rehearsals used: {status.get('rehearsals_used')} of {status.get('rehearsal_budget')}")
    rounds = status.get("rounds") or []
    if rounds:
        print("rounds (task success on the visible cases):")
    for r in rounds:
        print(f"  round {r['round']:02d}: {r['successes']} of {r['cases']}  ->  {r['results']}")
    pending = [p.name for p in sorted((ROOT / ".requests").iterdir())
               if p.is_dir() and not p.name.endswith(".tmp") and not (p / "response.json").exists()]
    if pending:
        print("still running: " + ", ".join(pending))


if __name__ == "__main__":
    main()
