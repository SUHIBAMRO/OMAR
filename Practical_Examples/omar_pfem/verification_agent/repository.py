"""Append-only store of verified (problem, method, observables, verdict)
records, the solved-case memory D of GRAFT."""
import datetime
import json
import os
import subprocess

from omar_pfem.verification_agent.spec import jaccard

DEFAULT_PATH = os.environ.get("VERIFICATION_REPO",
                              os.path.join(os.path.dirname(__file__), "repository.jsonl"))


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(__file__),
                                       text=True).strip()
    except Exception:
        return None


def record(spec, method, observables, verdicts, path=DEFAULT_PATH):
    rec = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "problem": spec.to_dict(),
        "fingerprint": sorted(spec.fingerprint()),
        "method": method,
        "observables": observables,
        "verdicts": verdicts,
        "accepted": all(v["verdict"] == "PASS" for v in verdicts.values()),
    }
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def load(path=DEFAULT_PATH):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def nearest(spec, k=3, path=DEFAULT_PATH):
    fp = spec.fingerprint()
    scored = [(jaccard(fp, r["fingerprint"]), r) for r in load(path)]
    return sorted(scored, key=lambda t: -t[0])[:k]
