"""
Independent, code-free sanity check for data_generate_B2.py's FEM ground
truth: compares each sample's radially-averaged displacement profile
against the classical closed-form Lame/Timoshenko solution for a thick-
walled cylinder under internal pressure (Timoshenko & Goodier, "Theory of
Elasticity" -- the thick-cylinder/Lame problem) -- exactly B2's geometry
and BCs (pressure at r=R_in, traction-free at r=R_out) in the small-
strain, homogeneous-material idealization.

This does NOT touch train_B2.py, the network, or convert_B2_quad.py at
all -- it reads directly from the raw H5 dataset data_generate_B2.py
writes (hyperelastic_dataset_physics.h5 + mesh_info.npz) and only uses
numpy, so it validates the FEM solver itself as an independent baseline,
before trusting anything built on top of it (the network, the force fix,
etc.). Since it never looks at inner_force_consistent, node_forces, or
any training code, it runs identically on both pre-fix and post-fix raw
datasets -- the FEM solve itself was never touched by the force-export
fix, only how forces are handed to training was.

Approximation used, and why it's reasonable here: the closed-form
solution assumes a HOMOGENEOUS material and UNIFORM pressure, but B2's
E/nu/pressure fields are heterogeneous (GRF-sampled). Since
p_mean/E_mean ~= 5/1000 = 0.5% (see data_generate_B2.py's
generate_random_sample_ring defaults), strains are small, so (a)
Neo-Hookean's nonlinearity is negligible at this load level (a small-
strain linear-elastic solution should closely match the nonlinear FEM
solve) and (b) using each sample's MEAN E, nu, and pressure as a
homogeneous/uniform stand-in, and comparing against the THETA-AVERAGED
radial displacement at each radius (averaging out the heterogeneity's
theta-dependent perturbation), gives a meaningful trend/magnitude check
without needing a full heterogeneous analytical solution (which has no
closed form). As a second, independent check, it also reports
u_theta_rms/u_r_rms: Lame's solution is purely radial (u_theta=0), so a
small ratio here is itself evidence the FEM solve behaves as expected.

Usage:
  python -m omar_pfem.validate_against_timoshenko \
      --h5_dir /content/drive/MyDrive/pfem_run/B2_force_fix_ablation/force_fixed/raw \
      --n_samples 15 --out_dir /content/drive/MyDrive/pfem_run/timoshenko_check \
      --out_json /content/drive/MyDrive/pfem_run/timoshenko_check/summary.json
(point --h5_dir at the directory data_generate_B2.py wrote to -- the one
containing hyperelastic_dataset_physics.h5 and mesh_info.npz, NOT the
converted training NPZ.)
"""
import os
import json
import argparse

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def lame_radial_displacement(r, R_in, R_out, p, E, nu):
    """Timoshenko & Goodier's closed-form thick-cylinder-under-internal-
    pressure solution (plane strain) -- matches the plane-strain Lame
    parameterization omar_pfem/data/materials.py's E_nu_to_neo_hookean
    uses (mu=E/(2(1+nu)), lam=E*nu/((1+nu)(1-2nu))) in the small-strain
    limit the FEM ground truth reduces to at this dataset's p/E ratio:
      sigma_rr = A - B/r^2, sigma_tt = A + B/r^2
      A = p*R_in^2 / (R_out^2 - R_in^2)
      B = p*R_in^2*R_out^2 / (R_out^2 - R_in^2)
      u_r(r) = (1+nu)/E * [(1-2nu)*A*r + B/r]
    """
    A = p * R_in**2 / (R_out**2 - R_in**2)
    B = p * R_in**2 * R_out**2 / (R_out**2 - R_in**2)
    return (1.0 + nu) / E * ((1.0 - 2.0 * nu) * A * r + B / r)


def main():
    parser = argparse.ArgumentParser(
        "Compare B2's FEM ground truth against the closed-form Timoshenko/"
        "Lame thick-cylinder solution, independent of train_B2.py/the network."
    )
    parser.add_argument("--h5_dir", type=str, required=True,
                         help="Directory produced by data_generate_B2.py (has "
                              "hyperelastic_dataset_physics.h5 and mesh_info.npz)")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default=None,
                         help="If set, saves a radial-profile comparison plot per sample")
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    mesh_info = np.load(os.path.join(args.h5_dir, "mesh_info.npz"))
    nodes = mesh_info["nodes"]
    R_in = float(mesh_info["R_in"])
    R_out = float(mesh_info["R_out"])

    tol = 1e-6
    r_all = np.linalg.norm(nodes, axis=1)
    theta_all = np.arctan2(nodes[:, 1], nodes[:, 0])
    r_layers = np.unique(np.round(r_all, 6))

    h5_path = os.path.join(args.h5_dir, "hyperelastic_dataset_physics.h5")
    with h5py.File(h5_path, "r") as f:
        valid_mask = ~np.isnan(f["displacements"][:, 0, 0])
        valid_indices = np.where(valid_mask)[0]
        n_use = min(args.n_samples, len(valid_indices))
        sample_indices = valid_indices[:n_use]

        E_nodes_all = f["E_nodes"][sample_indices]
        nu_nodes_all = f["nu_nodes"][sample_indices]
        inner_pressure_all = f["inner_pressure"][sample_indices]
        disp_all = f["displacements"][sample_indices]

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    per_sample = []
    for k, sidx in enumerate(sample_indices):
        E_mean = float(E_nodes_all[k].mean())
        nu_mean = float(nu_nodes_all[k].mean())
        p_mean = float(inner_pressure_all[k].mean())

        u = disp_all[k]
        u_r_fem = u[:, 0] * np.cos(theta_all) + u[:, 1] * np.sin(theta_all)
        u_t_fem = -u[:, 0] * np.sin(theta_all) + u[:, 1] * np.cos(theta_all)

        fem_profile, analytical_profile = [], []
        for r_layer in r_layers:
            layer_mask = np.abs(r_all - r_layer) < tol
            fem_profile.append(float(u_r_fem[layer_mask].mean()))
            analytical_profile.append(
                lame_radial_displacement(r_layer, R_in, R_out, p_mean, E_mean, nu_mean)
            )
        fem_profile = np.array(fem_profile)
        analytical_profile = np.array(analytical_profile)

        err = fem_profile - analytical_profile
        rel_l2 = float(np.linalg.norm(err) / (np.linalg.norm(analytical_profile) + 1e-12))

        ut_rms = float(np.sqrt(np.mean(u_t_fem**2)))
        ur_rms = float(np.sqrt(np.mean(u_r_fem**2)))
        ut_over_ur = ut_rms / (ur_rms + 1e-12)

        rec = {
            "sample_idx": int(sidx),
            "E_mean": E_mean, "nu_mean": nu_mean, "p_mean": p_mean,
            "rel_l2_radial_profile_vs_lame": rel_l2,
            "u_theta_rms_over_u_r_rms": ut_over_ur,
            "u_r_fem_profile": fem_profile.tolist(),
            "u_r_lame_profile": analytical_profile.tolist(),
            "r_layers": r_layers.tolist(),
        }
        per_sample.append(rec)
        print(f"[sample {sidx}] E={E_mean:.1f} nu={nu_mean:.3f} p={p_mean:.3f}  "
              f"rel_L2(u_r vs Lame)={rel_l2:.4f}  "
              f"u_theta_rms/u_r_rms={ut_over_ur:.4f} (0=purely radial, matches Lame's assumption)")

        if args.out_dir:
            plt.figure(figsize=(6, 4))
            plt.plot(r_layers, fem_profile, "o-", label="FEM (theta-averaged)")
            plt.plot(r_layers, analytical_profile, "s--", label="Timoshenko/Lame (closed-form)")
            plt.xlabel("r"); plt.ylabel("u_r")
            plt.title(f"sample {sidx}: rel_L2={rel_l2:.3f}")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"timoshenko_check_sample{sidx}.png"), dpi=150)
            plt.close()

    rel_l2s = np.array([r["rel_l2_radial_profile_vs_lame"] for r in per_sample])
    ratios = np.array([r["u_theta_rms_over_u_r_rms"] for r in per_sample])

    print(f"\n{'='*80}\nSUMMARY over {len(per_sample)} samples")
    print(f"rel_L2(u_r vs Lame): mean={rel_l2s.mean():.4f} std={rel_l2s.std():.4f} max={rel_l2s.max():.4f}")
    print(f"u_theta_rms/u_r_rms: mean={ratios.mean():.4f} std={ratios.std():.4f}")
    if rel_l2s.mean() < 0.10:
        print("\n=> FEM ground truth's radial displacement trend matches the closed-form "
              "Timoshenko/Lame solution well (mean rel_L2 < 10%) -- the FEM solver itself "
              "looks physically correct; the B2 error is likely downstream (loss scaling, "
              "network capacity, mesh resolution) rather than in data_generate_B2.py's solve.")
    else:
        print("\n=> FEM ground truth deviates notably from the closed-form solution "
              "(mean rel_L2 >= 10%) -- worth double-checking data_generate_B2.py's solver "
              "itself (BCs, material mapping, sign conventions) before trusting anything "
              "built on top of it.")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"per_sample": per_sample,
                       "mean_rel_l2": float(rel_l2s.mean()),
                       "mean_u_theta_over_u_r": float(ratios.mean())}, f, indent=2)
        print(f"\nWritten to {args.out_json}")


if __name__ == "__main__":
    main()
