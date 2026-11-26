"""Claims check of the combined progress report (24-26 November, 18 pages): the claims of the
two earlier new-results PDFs (round_o_claims, round_r_claims) are recomputed from the stored
files and searched for in the combined report.

Claims intentionally NOT in the combined report (listed, not silently skipped):
  - Cook 3D local code check up to n = 16 (3.41 / 2.14, 2.73 / 2.31) and the 26.8 GB estimate:
    superseded by the cluster result up to n = 32, which is checked instead;
  - "now 0.57 for graded": reworded to "priors were 0.57 for graded" (checked as 0.57).

    python -m omar_pfem.verification_agent.progress_report_claims
"""
import contextlib
import io
import os
import re
import subprocess

from omar_pfem.verification_agent import round_o_claims as O, round_r_claims as R

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.abspath(os.path.join(HERE, "..", "..", "..", "advisor_feedback",
                                   "2026-11-26_Autonomous_Workflow_Progress_Report.pdf"))
SUPERSEDED = {"Cook 3D Q2", "Cook 3D P2", "Cook ref P2 memory"}
REWORDED = {"prior graded": "priors were 0.57 for graded"}


def main():
    flat = re.sub(r"\s+", " ", subprocess.run(["pdftotext", PDF, "-"], capture_output=True,
                                             text=True).stdout).replace("−", "-")
    bad = n = 0
    for name, val, fmt, cond in O.claims():
        s = REWORDED.get(name, fmt.format(val))
        ok = ((s.lstrip("-") in flat) if isinstance(val, (int, float)) else True) and \
            (cond is None or cond(val))
        n += 1
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} [part 1] {name}: {s}")
    for name, computed, printed in R.claims():
        if name in SUPERSEDED:
            print(f"skip [part 2] {name}: superseded by the cluster result")
            continue
        ok = computed == printed and printed in flat
        n += 1
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} [part 2] {name}: {printed}")
    print(f"{n - bad} of {n} claims found and consistent")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
