"""
A single, uniform record of every run we do, written as JSON next to that
run's own outputs.

Why this exists: results for this project are spread across dozens of Drive
folders produced over weeks, and writing them up has repeatedly meant hunting
for "which checkpoint produced this number, on what date, with which flags?".
A real error already came from that -- a stale pre-fix checkpoint's inference
latency sat in the report's Table 7 for several revisions because nothing
recorded which run a number came from. Two GPU-timing runs with slightly
different numbers also coexist on Drive with no way to tell them apart except
their folder.

So every script should call `write_manifest(...)` when it finishes. The
manifest captures, in one place:
  - when the run started and finished, and how long it took,
  - the exact command line, so it can be reproduced verbatim,
  - the git commit of the code that produced it,
  - every argparse argument,
  - the environment (torch/CUDA/GPU/CPU), since timings are meaningless
    without it,
  - the headline results themselves,
  - every output file the run wrote.

Manifests are append-only per directory (`run_manifest.json` holds a list), so
re-running a script adds a record rather than silently overwriting the history
of what was done.
"""
import os
import sys
import json
import time
import socket
import platform
import subprocess
from datetime import datetime, timezone


def _git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return {"commit": sha or None, "dirty": bool(dirty)}
    except Exception:
        return {"commit": None, "dirty": None}


def _environment():
    env = {"python": platform.python_version(), "host": socket.gethostname(),
           "platform": platform.platform()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            env["gpu_total_mem_mb"] = props.total_memory / 1e6
    except Exception:
        pass
    try:
        env["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    return env


def write_manifest(out_dir, kind, args=None, results=None,
                   started_at=None, outputs=None, notes=None):
    """Append one record to <out_dir>/run_manifest.json.

    out_dir    : directory the run wrote its results into
    kind       : short label, e.g. "zeroshot_train", "physical_quantities"
    args       : argparse.Namespace (or dict) of every flag used
    results    : the headline numbers, so the manifest alone can answer
                 "what did this run produce?" without opening other files
    started_at : time.time() captured at the start, to record duration
    outputs    : list of files this run wrote
    notes      : anything a future reader would otherwise have to reconstruct
    """
    os.makedirs(out_dir, exist_ok=True)
    now = time.time()
    rec = {
        "kind": kind,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "finished_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_s": (now - started_at) if started_at else None,
        "duration_human": _human(now - started_at) if started_at else None,
        "command": " ".join([sys.executable.split("/")[-1], "-m",
                             _module_name()] + sys.argv[1:]),
        "argv": sys.argv,
        "git": _git_commit(),
        "environment": _environment(),
        "args": _as_dict(args),
        "results": results,
        "outputs": outputs or [],
        "notes": notes,
    }
    if started_at:
        rec["started_at_utc"] = datetime.fromtimestamp(
            started_at, timezone.utc).isoformat(timespec="seconds")

    path = os.path.join(out_dir, "run_manifest.json")
    history = []
    if os.path.exists(path):
        try:
            history = json.load(open(path))
            if not isinstance(history, list):
                history = [history]
        except Exception:
            history = []
    history.append(rec)

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2, default=str)
    os.replace(tmp, path)
    print(f"[manifest] run #{len(history)} recorded -> {path}"
          + (f"  ({rec['duration_human']})" if rec["duration_human"] else ""))
    return path


def _module_name():
    mod = sys.modules.get("__main__")
    name = getattr(mod, "__spec__", None)
    return name.name if name else (getattr(mod, "__file__", "?") or "?")


def _as_dict(args):
    if args is None:
        return None
    if isinstance(args, dict):
        return args
    try:
        # `func` and friends are argparse's set_defaults callables -- they carry
        # no information about the run, only a memory address that differs every
        # time and makes two identical runs look different.
        return {k: v for k, v in vars(args).items()
                if not k.startswith("_") and not callable(v)}
    except TypeError:
        return str(args)


def _human(sec):
    if sec is None:
        return None
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")
