# =====================================================================
#  CELL -- B3: break-even analysis (resolution-matched + accuracy-matched),
#  per Timon's "finish the paper" request (2026-10-01): complete the
#  accuracy-matched FEM/VINO comparison and break-even analysis for the
#  QoIs that are reproduced well (displacement, energy, reaction).
#
#  Resolution-matched (NO and FEM both at B3's own 6,840-element training
#  resolution) needs no new data -- it is pure arithmetic on already-
#  verified numbers. Accuracy-matched (cheapest FEM mesh at least as
#  accurate as the NO, both judged against the SAME independent
#  243,360-element fine reference) uses the qois_finer_resolution.json
#  the previous cell just produced, which now also reports displacement/
#  energy/reaction error vs. that fine reference (added 2026-10-01 to
#  evaluate_B3_qois_finer.py -- previously that script only reported
#  region stress). See break_even_B3.py's own module docstring for the
#  full derivation and the sourced convergence-ladder table it uses.
# =====================================================================
import sys

sys.argv = [
    'break_even_B3.py',
    '--fine_qois_json', OUT_JSON,
    '--out_json', f'{R}/b3_training_normalized/break_even_B3.json',
]
from omar_pfem.break_even_B3 import main
main()

print('\nDone.')
