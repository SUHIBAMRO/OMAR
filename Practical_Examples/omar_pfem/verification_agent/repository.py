"""Append-only store of verified (problem, method, observables, verdict)
records, the solved-case memory D of GRAFT.

The file is never rewritten. A later record with the same problem name
SUPERSEDES the earlier one (e.g. after a gate was added or fixed): load()
and nearest() return only the latest record per name unless
latest_only=False, so re-running a check does not double its weight in the
outcome prior."""
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


def load(path=DEFAULT_PATH, latest_only=True):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if not latest_only:
        return recs
    latest = {}
    for i, r in enumerate(recs):
        latest[r["problem"]["name"]] = i
    return [recs[i] for i in sorted(latest.values())]


def nearest(spec, k=3, path=DEFAULT_PATH):
    fp = spec.fingerprint()
    scored = [(jaccard(fp, r["fingerprint"]), r) for r in load(path)]
    return sorted(scored, key=lambda t: -t[0])[:k]
