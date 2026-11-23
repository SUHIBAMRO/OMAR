# Billed LLM-vs-rules ablation, round M (2026-11-23, approved by Omar)

Backend: claude-cli; 7 problems x 3 runs = 21 runs; total cost USD 12.16; no run raised an error.
"missed" = a quantity that FAILed in that run but was not predicted to fail; "false alarm" =
predicted to fail but did not. The rules' numbers come from the rule-only path (deterministic).

| problem  | spec correct | verdict agreement (mean) | rules missed / false alarms        | LLM missed (min-max) | LLM false alarms (min-max) | USD  |
|----------|--------------|--------------------------|------------------------------------|----------------------|----------------------------|------|
| cook     | 3/3          | 0.86                     | strain_energy / -                  | 1.67 (0-3)           | 1.00 (0-2)                 | 1.81 |
| plate    | 3/3          | 1.00                     | - / hole_peak_sigma_xx             | 0                    | 3.00 (3-3)                 | 1.76 |
| lamina   | 0/3 (regularity) | 1.00                 | - / -                              | 0                    | 3.00 (2-4)                 | 1.72 |
| lshape   | 3/3          | 1.00                     | - / -                              | 0                    | 3.33 (3-4)                 | 1.72 |
| crack    | 3/3          | 0.86 (2 valid runs)      | - / -                              | 0                    | 3.00 (2-4) over 2 valid runs | 1.29 |
| cylinder | 3/3          | 0.17                     | peak_hoop / -                      | 3.67 (3-4)           | 0                          | 2.22 |
| block    | 3/3          | 1.00                     | root_gk_avg_von_mises / -          | 0                    | 1.67 (1-2)                 | 1.65 |

Findings (read with care; 3 runs per problem):
1. Same pattern as on B1-B3: the LLM recovers the specification from text (18/21 runs fully;
   on the lamina it labels the regularity differently in all 3 runs), and raises 1-3 false
   alarms per run on most problems.
2. It caught the one failure the rules missed on the block (root_gk_avg_von_mises) in 3/3
   runs, and the cook energy in 1/3 runs.
3. Crack run 2 is degenerate: the LLM chose no quantities (7 spec errors, one call, USD 0.08).
   The script counted it as a run with 0 misses and 0 false alarms; it is excluded above.
4. Cylinder (agreement 0.17): the LLM added the strategy-robustness checks. On this problem the
   grading-2.0 strategy is a failed solve (min det F = -1.03, energy NaN; report Section 31.1:
   too few load increments), and the robustness checks turn that into FAIL for 5 quantities
   the convergence checks certify (p = 3.8-4.0). The LLM did not predict those FAILs, hence the
   "misses". This is a weakness of the robustness checks (a failed solve should give
   INCONCLUSIVE, not FAIL), not a real failure of the quantities. Not changed: changing a
   check after seeing results would be post hoc.
