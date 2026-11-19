"""Re-solve every stored B3 verification-agent case after the stress fix.

Until 2026-11-19 b3_global.solve_extended converted torch-fem's returned
stress (already Cauchy in torch-fem 0.14) a second time, sigma F^T / J, so
every stored Gauss-point / region / surface STRESS was off by O(|grad u|)
(a few %); displacements, reaction moment and strain energy were not
affected. This driver keeps the old files under pre_stressfix_20261119/ and
re-solves each case ONCE, writing the row and, where the old data had one,
the Gauss-point dump and surface dump, at the same paths as before.

    python -m omar_pfem.verification_agent.b3_redo_stress_fix [--only small|all]

Resumable: a case whose new output exists is skipped.
"""
import argparse
import json
import os
import shutil
import time

from omar_pfem.verification_agent.b3_global import _family_resolutions, solve_extended

HERE = os.path.dirname(os.path.abspath(__file__))
BACKUP = os.path.join(HERE, "pre_stressfix_20261119")
DONE = os.path.join(HERE, "b3_redo_stress_fix_done.json")
FILES = ["b3_extended.json", "b3_fullint.json", "b3_radial.json", "b3_solver_crosscheck.json",
         "region_study_result.json", "surface_study_result.json"]
DIRS = ["b3_gauss_near_groove", "b3_gauss_near_groove_fullint", "b3_gauss_radial"]


def backup():
    if os.path.exists(BACKUP):
        return
    os.makedirs(BACKUP)
    for f in FILES:
        if os.path.exists(os.path.join(HERE, f)):
            shutil.copy2(os.path.join(HERE, f), BACKUP)
    for d in DIRS:
        if os.path.isdir(os.path.join(HERE, d)):
            shutil.copytree(os.path.join(HERE, d), os.path.join(BACKUP, d))


def cases():
    fam = _family_resolutions()
    fine = [tuple(r) for r in json.load(open(os.path.join(HERE, "b3_fine_levels.json")))]
    g = lambda *p: os.path.join(HERE, *p)
    out = []
    # (case id, solve kwargs, row file, row key fields)
    for r in fam:
        out.append((f"family {r}", dict(Ntheta=r[0], Nr=r[1], Nz=r[2],
                                        dump_path=g("b3_gauss_near_groove",
                                                    f"{r[0]}_{r[1]}_{r[2]}_zbNone.npz")),
                    "b3_extended.json"))
    for zb in (1.0, 2.0, 3.0):
        out.append((f"z_beta {zb}", dict(Ntheta=21, Nr=18, Nz=31, z_beta=zb,
                                         dump_path=g("b3_gauss_near_groove",
                                                     f"21_18_31_zb{zb}.npz")),
                    "b3_extended.json"))
    for r in (fam[0], fam[2]):
        out.append((f"rtol 1e-11 {r}", dict(Ntheta=r[0], Nr=r[1], Nz=r[2], rtol=1e-11,
                                            atol=1e-11), "b3_extended.json"))
        out.append((f"fullint {r}", dict(Ntheta=r[0], Nr=r[1], Nz=r[2], quadrature="full",
                                         dump_path=g("b3_gauss_near_groove_fullint",
                                                     f"{r[0]}_{r[1]}_{r[2]}_full.npz")),
                    "b3_fullint.json"))
    out.append(("direct solver 2376", dict(Ntheta=fam[0][0], Nr=fam[0][1], Nz=fam[0][2],
                                           linear_solver="direct"), "b3_solver_crosscheck.json"))
    for rg in (1.1, 1.2, 1.3):
        out.append((f"radial {rg}", dict(Ntheta=21, Nr=18, Nz=31, r_grading=rg,
                                         linear_solver="direct",
                                         dump_path=g("b3_gauss_radial", f"21_18_31_rg{rg}.npz"),
                                         surface_path=g("b3_gauss_radial",
                                                        f"21_18_31_rg{rg}_surface.npz")),
                    "b3_radial.json"))
    for r in fine:                                   # 59,400 el: ~10 GB, last
        out.append((f"family {r}", dict(Ntheta=r[0], Nr=r[1], Nz=r[2],
                                        dump_path=g("b3_gauss_near_groove",
                                                    f"{r[0]}_{r[1]}_{r[2]}_zbNone.npz")),
                    "b3_extended.json"))
    return out


def _key(r):
    return (r["Ntheta"], r["Nr"], r["Nz"], r.get("z_beta"), r.get("rtol"),
            r.get("quadrature", "reduced"), r.get("r_grading", 1.0), r.get("linear_solver"))


def store(row, fname):
    path = os.path.join(HERE, fname)
    if fname == "b3_solver_crosscheck.json":
        keys = ("n_elem", "linear_solver", "elapsed_s", "reaction_moment_y", "strain_energy",
                "region_avg_sigma_frob", "region_avg_pressure", "region_max_von_mises",
                "min_detF")
        json.dump([{k: row[k] for k in keys}], open(path, "w"), indent=1)
        return
    rows = json.load(open(path)) if os.path.exists(path) else []
    old = [r for r in rows if (r["Ntheta"], r["Nr"], r["Nz"], r.get("z_beta"), r.get("rtol"),
                               r.get("r_grading", 1.0))
           == (row["Ntheta"], row["Nr"], row["Nz"], row.get("z_beta"), row.get("rtol"),
               row.get("r_grading", 1.0))
           and r.get("quadrature", "reduced") == row.get("quadrature", "reduced")]
    for r in old:
        rows.remove(r)
    row["stress_fix_20261119"] = True
    rows.append(row)
    json.dump(rows, open(path, "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=("small", "all"))
    a = ap.parse_args(argv)
    backup()
    done = set(json.load(open(DONE))) if os.path.exists(DONE) else set()
    todo = cases()
    if a.only == "small":
        todo = [c for c in todo if "(41, 28, 56)" not in c[0]]
    for cid, kw, fname in todo:
        if cid in done:
            continue
        t0 = time.time()
        row = solve_extended(**kw)
        store(row, fname)
        done.add(cid)
        json.dump(sorted(done), open(DONE, "w"), indent=1)
        print(f"{cid:<26} {row['n_elem']:>6} el {time.time() - t0:6.0f}s  "
              f"M_y={row['reaction_moment_y']:.6f}  frob_avg={row['region_avg_sigma_frob']:.5f}  "
              f"p_avg={row['region_avg_pressure']:.5f}", flush=True)


if __name__ == "__main__":
    main()
