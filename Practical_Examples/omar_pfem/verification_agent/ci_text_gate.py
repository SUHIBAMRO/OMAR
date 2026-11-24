"""CI gate (2026-11-24): fail if any claim's printed value is not found in the current
report PDF (report_text_check.PDF)."""
import sys

from omar_pfem.verification_agent.report_text_check import main

if __name__ == "__main__":
    summ, rows = main([])
    print(summ)
    missing = [r for r in rows if r["text"] == "NOT FOUND"]
    for r in missing:
        print(f"NOT FOUND [{r['section']}] {r['claim']}: {r['printed']}")
    sys.exit(1 if missing else 0)
