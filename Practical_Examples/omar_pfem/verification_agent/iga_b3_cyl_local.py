"""Local run of iga_b3_cyl_job.sh (same steps, this machine): re-solves spline levels 1-4
(displacement, then P1) with the cylindrical Gaussian averages; writes iga_b3_cyl_result.json and
checks each level reproduces the stored Cartesian value. python -m omar_pfem.verification_agent.iga_b3_cyl_local"""
import json, os
from omar_pfem.verification_agent import iga_b3 as I
stored = json.load(open(I.OUT))
out = os.path.join(os.path.dirname(I.OUT), "iga_b3_cyl_result.json")
I.OUT = out
cache = json.load(open(out)) if os.path.exists(out) else {}
for form in ("disp", "fbar_p1"):
    for lv in (1, 2, 3, 4):
        row = I.run(lv, 2, cache, form)
        key = f"L{lv}:p2" + ("" if form == "disp" else ":fbar_p1")
        d = abs(row["gk2trunc_sigma_xx_avg"] / stored[key]["gk2trunc_sigma_xx_avg"] - 1)
        print(key, "reproduces stored gk2trunc_sigma_xx to", d, flush=True)
        assert d < 1e-6
