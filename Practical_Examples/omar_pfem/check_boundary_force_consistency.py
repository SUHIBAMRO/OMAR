"""
No-retraining sanity check for B2's accuracy-diagnostics finding (see
accuracy_diagnostics.py): error concentrated in the RADIAL direction and at
the INNER (pressure-loaded) boundary, uncorrelated with E/nu/load
magnitude. That pattern is exactly what a systematic mismatch between the
boundary FORCE REPRESENTATION used by the ground-truth FEM solve and the
one fed into the DEM training loss would produce, so this checks that
specific hypothesis directly, on an already-generated raw H5 dataset.

The mismatch: data_generate_B2.py's own FEM solve loads the inner arc with
the FEM-consistent nodal force (assemble_traction_inner_curved: each node's
force is integral(N_a * p * normal) ds, i.e. pressure times the edge's own
chord normal, weighted by 1D Gauss quadrature over the edge's Jacobian
edge_len/2 -- see that function's docstring). That is the Fext actually
used to produce `displacements` (the ground truth this study evaluates
against).

convert_B2_quad.py, however, does NOT reuse that consistent force. It
recomputes a per-node "boundary_forces" as simply
  inner_pressure[node] * (exact radial unit vector at that node)
i.e. the raw nodal pressure VALUE times the EXACT radial direction, with no
edge-length/quadrature weighting at all (see its own docstring, "raw nodal
value, no further quadrature weighting"). That recomputed quantity is what
both (a) the network sees as an input feature (fun_material's fx, fy) and,
more importantly, (b) enters train_B2.py's loss directly as
W = sum(node_forces * uv) / len(inner_edges) -- the external-work term of
the Pi = U - W potential energy the network is trained to minimize.

If W's node_forces don't match the consistent Fext that actually produced
the ground-truth displacement field, the network is being trained toward
the stationary point of a DIFFERENT physical problem than the one that
generated its own evaluation targets -- a sample-independent, boundary-
localized, direction-specific bias, matching all four things
accuracy_diagnostics.py found (radial >> circumferential, inner boundary >
interior > outer, ~zero correlation with E/nu/load magnitude since a
structural mismatch, not a regime-dependent difficulty).

This script recomputes the TRUE consistent nodal force (reusing
data_generate_B2.py's own assemble_traction_inner_curved -- no
reimplementation) for a handful of already-generated samples and compares
it directly against convert_B2_quad.py's recomputed "raw" force, reporting
per-node magnitude ratio and direction agreement so the mismatch (if any)
is quantified before touching any code that would require retraining.

Usage:
  python -m omar_pfem.check_boundary_force_consistency \
      --h5_dir /content/drive/MyDrive/pfem_run/results/datasets/B2_neo_hookean_raw \
      --n_samples 10
(point --h5_dir at the directory data_generate_B2.py wrote to, i.e. the one
containing hyperelastic_dataset_physics.h5 and mesh_info.npz -- NOT the
converted training NPZ.)
"""
import os
import json
import argparse

import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

from omar_pfem.data.data_generate_B2 import assemble_traction_inner_curved


def main():
    parser = argparse.ArgumentParser(
        "Compare convert_B2_quad.py's raw boundary force against the FEM-consistent "
        "nodal force data_generate_B2.py actually solved with, on an existing raw H5 dataset."
    )
    parser.add_argument("--h5_dir", type=str, required=True,
                         help="Directory produced by data_generate_B2.py (has "
                              "hyperelastic_dataset_physics.h5 and mesh_info.npz)")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    mesh_info = np.load(os.path.join(args.h5_dir, "mesh_info.npz"))
    nodes = mesh_info["nodes"]
    elements = mesh_info["elements"]
    R_in = float(mesh_info["R_in"])
    R_out = float(mesh_info["R_out"])

    h5_path = os.path.join(args.h5_dir, "hyperelastic_dataset_physics.h5")
    with h5py.File(h5_path, "r") as f:
        Ntheta = int(f.attrs["Ntheta"])
        Nr = int(f.attrs["Nr"])
        num_samples = int(f.attrs["num_samples"])
        valid_mask = ~np.isnan(f["displacements"][:, 0, 0])
        valid_indices = np.where(valid_mask)[0]
        n_use = min(args.n_samples, len(valid_indices))
        sample_indices = valid_indices[:n_use]

        p_fields = f["p_fields"][sample_indices]           # (n, Ntheta, Nr)
        inner_pressure = f["inner_pressure"][sample_indices]  # (n, n_inner)

    tol = 1e-9
    r_all = np.linalg.norm(nodes, axis=1)
    inner_nodes = np.where(np.abs(r_all - R_in) < tol)[0]
    inner_radial_dirs = nodes[inner_nodes] / R_in

    thetas = np.linspace(0.0, np.pi / 2, Ntheta)
    rs = np.linspace(R_in, R_out, Nr)

    per_sample = []
    for k, sidx in enumerate(sample_indices):
        p_interp = RegularGridInterpolator(
            (thetas, rs), p_fields[k], method="linear", bounds_error=False, fill_value=None
        )
        Fext_full = assemble_traction_inner_curved(nodes, elements, R_in, p_interp)
        consistent_force = Fext_full.reshape(-1, 2)[inner_nodes]  # (n_inner, 2) -- what the FEM solve actually used

        raw_force = inner_pressure[k][:, None] * inner_radial_dirs  # (n_inner, 2) -- what convert_B2_quad.py uses

        cons_mag = np.linalg.norm(consistent_force, axis=1)
        raw_mag = np.linalg.norm(raw_force, axis=1)
        nz = raw_mag > 1e-12

        mag_ratio = cons_mag[nz] / raw_mag[nz]  # consistent / raw, per node
        cos_dir = np.sum(consistent_force[nz] * raw_force[nz], axis=1) / (
            cons_mag[nz] * raw_mag[nz] + 1e-12
        )  # 1.0 = same direction

        rec = {
            "sample_idx": int(sidx),
            "total_consistent_force_mag": float(np.linalg.norm(consistent_force.sum(axis=0))),
            "total_raw_force_mag": float(np.linalg.norm(raw_force.sum(axis=0))),
            "mean_per_node_mag_ratio_consistent_over_raw": float(mag_ratio.mean()),
            "std_per_node_mag_ratio": float(mag_ratio.std()),
            "mean_direction_cosine": float(cos_dir.mean()),
            "min_direction_cosine": float(cos_dir.min()),
        }
        per_sample.append(rec)
        print(f"[sample {sidx}] total|F| consistent={rec['total_consistent_force_mag']:.4f}  "
              f"raw={rec['total_raw_force_mag']:.4f}  "
              f"ratio(consistent/raw)={rec['total_consistent_force_mag']/max(rec['total_raw_force_mag'],1e-12):.4f}  "
              f"mean_per_node_ratio={rec['mean_per_node_mag_ratio_consistent_over_raw']:.4f}  "
              f"mean_dir_cos={rec['mean_direction_cosine']:.5f}")

    ratios = np.array([r["mean_per_node_mag_ratio_consistent_over_raw"] for r in per_sample])
    cosines = np.array([r["mean_direction_cosine"] for r in per_sample])

    print(f"\n{'='*80}\nSUMMARY over {len(per_sample)} samples")
    print(f"mean_per_node_mag_ratio (consistent/raw): mean={ratios.mean():.4f} std={ratios.std():.4f}")
    print(f"mean_direction_cosine: mean={cosines.mean():.5f} std={cosines.std():.5f}")
    if abs(ratios.mean() - 1.0) > 0.05:
        print(f"\n=> CONFIRMED: convert_B2_quad.py's raw node_forces differ from the FEM-consistent "
              f"force the ground truth was solved with by a factor of ~{ratios.mean():.3f}x. "
              f"This directly biases train_B2.py's W = sum(node_forces*uv)/len(inner_edges) term, "
              f"which is a plausible root cause of the radial/inner-boundary-localized error pattern.")
    else:
        print(f"\n=> Magnitude ratio is close to 1.0 -- the force representations agree in magnitude; "
              f"re-examine direction agreement / other hypotheses instead.")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"per_sample": per_sample,
                       "mean_mag_ratio": float(ratios.mean()),
                       "mean_direction_cosine": float(cosines.mean())}, f, indent=2)
        print(f"\nWritten to {args.out_json}")


if __name__ == "__main__":
    main()
