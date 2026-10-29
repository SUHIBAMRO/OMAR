"""FEM-discretization error for B3: how much of the gap between a
coarse FEM solution and the 950,400-element reference is just MESH
error, with no neural network involved at all.

Built 2026-10-30, per Professor Rabczuk's own precise follow-up email:
"Can you please also report, for the same samples and QoIs, the errors
of the 6,840-element and 43,400-element FEM solutions relative to the
950,400-element reference. This will let us clearly separate FEM
discretisation error from neural-operator error."

This compares the SAME held-out dataset samples (same per-node random
E/nu realization, same rocking angle phi) solved at three resolutions:
  - 6,840 elements (21,20,19): the project's own training/eval mesh --
    already solved, loaded directly from dataset.h5, no new FEM solve.
  - 43,400 elements (41,36,32): the paper's own accuracy-matched
    reference -- solved via evaluate_B3_qois_finer.generate_fine_reference
    (same function already used for the neural-operator evaluation).
  - 950,400 elements (101,100,97): the paper's own mesh-convergence
    reference -- solved the same way.

For displacement/H1/regional-stress (spatial FIELDS), the coarser
solution's own displacement field is interpolated onto the 950,400
mesh's own node positions (the SAME RegularGridInterpolator-based
machinery mesh_convergence_B3.py's own `compare_to_reference` already
validated for exactly this purpose, at n_components=3 -- see that
function's docstring and its 2026-09-21 fix history), then the SAME QoI
functions already used for the neural-operator-vs-FEM comparison
(compute_H1_seminorm_rel, compute_region_cauchy_field_rel,
compute_region_sigma_components_rel) are applied with u_pred = the
interpolated coarse field and u_true = the 950,400 mesh's own solution,
combined with the 950,400 mesh's own (interpolated) E/nu field exactly
as the neural-operator evaluation already does for its own prediction --
this is what makes the FEM-vs-FEM number and the NN-vs-FEM number
genuinely comparable: they are computed by the literal same formula.

For energy/reaction/region-average-sigma-xx (SCALARS), each resolution's
own value is computed directly from its own FEM solution with no
interpolation needed, and the two are compared as a plain relative
error -- no new formula, matching mesh_convergence_B3.py's own
`scalar_qoi_rel_errors` convention.

Usage:
  from omar_pfem.evaluate_B3_qois_finer import generate_fine_reference, FINE_RESOLUTION_STRESS
  from omar_pfem.evaluate_B3_fem_discretization_error import (
      load_coarse_reference, compute_fem_discretization_errors)
  coarse_ref = load_coarse_reference(dataset_path, 10, device, dtype)
  mid_ref = generate_fine_reference(dataset_path, (41, 36, 32), 10, device, dtype)
  fine_ref = generate_fine_reference(dataset_path, FINE_RESOLUTION_STRESS, 10, device, dtype)
  result = compute_fem_discretization_errors(coarse_ref, mid_ref, fine_ref, device, dtype)
"""
import h5py
import numpy as np
import torch

from omar_pfem.train_B3 import build_fixed_geometry
from omar_pfem.data.data_generate_B3_dataset import DEFAULT_RESOLUTION, R_GRADING
from omar_pfem.evaluate_B3_qois import (
    build_region_mask, compute_H1_seminorm_rel, compute_region_sigma_full,
    cauchy_field_rel_from_full, sigma_components_rel_from_full, compute_region_sigma_xx,
    compute_energy, compute_reaction, STRESS_COMPONENTS,
)
from omar_pfem.evaluate_B3_qois_finer import interpolate_node_field_to_finer_mesh


def load_coarse_reference(dataset_path, n_samples, device, dtype=torch.float64, verbose=True):
    """Loads the ALREADY-SOLVED 6,840-element (DEFAULT_RESOLUTION)
    FEM solution directly from dataset.h5 -- no new FEM solve, since
    this mesh IS the project's own training/eval mesh. Same dict shape
    as evaluate_B3_qois_finer.generate_fine_reference's own return value
    (minus force_rel_residual/elapsed_s, which dataset.h5 does store per
    sample too, so they're included for completeness) so the same
    downstream code can treat all three resolutions uniformly."""
    Ntheta_c, Nr_c, Nz_c = DEFAULT_RESOLUTION
    region_mask_c_np = build_region_mask(Ntheta_c, Nr_c, Nz_c, R_GRADING)
    region_mask_c_torch = torch.tensor(region_mask_c_np, device=device)
    geom_c = build_fixed_geometry(Ntheta_c, Nr_c, Nz_c, device, dtype=dtype)

    with h5py.File(dataset_path, "r") as h5:
        E_node_all = h5["E_node"][:]
        nu_node_all = h5["nu_node"][:]
        phi_all = h5["phi"][:]
        u_all = h5["displacements"][:]
        frr_all = h5["force_rel_residual"][:] if "force_rel_residual" in h5 else None
        elapsed_all = h5["elapsed_s"][:] if "elapsed_s" in h5 else None
    n_available = E_node_all.shape[0]
    n_samples = min(n_samples, n_available)
    if verbose:
        print(f"Coarse (training) resolution: {(Ntheta_c, Nr_c, Nz_c)} -- loaded directly "
              f"from {dataset_path}, no new FEM solve")

    samples = []
    for idx in range(n_samples):
        E_c = E_node_all[idx].astype(np.float64)
        nu_c = nu_node_all[idx].astype(np.float64)
        phi = float(phi_all[idx])
        u_c = u_all[idx].astype(np.float64)

        samples.append({
            "sample_index": idx, "phi": phi,
            "E_f_t": torch.tensor(E_c, dtype=dtype, device=device).unsqueeze(0),
            "nu_f_t": torch.tensor(nu_c, dtype=dtype, device=device).unsqueeze(0),
            "phi_t": torch.tensor([phi], dtype=dtype, device=device),
            "u_true_f": torch.tensor(u_c, dtype=dtype, device=device).unsqueeze(0),
            "force_rel_residual": float(frr_all[idx]) if frr_all is not None else None,
            "elapsed_s": float(elapsed_all[idx]) if elapsed_all is not None else None,
        })

    return {"geom_fine": geom_c, "region_mask_fine_torch": region_mask_c_torch,
            "fine_resolution": [Ntheta_c, Nr_c, Nz_c], "coarse_resolution": [Ntheta_c, Nr_c, Nz_c],
            "n_region_fine": int(region_mask_c_np.sum()), "samples": samples}


def compute_fem_discretization_errors(coarse_ref, mid_ref, fine_ref, device, dtype=torch.float64, verbose=True):
    """Compares coarse_ref (6,840) and mid_ref (43,400) against fine_ref
    (950,400) for the SAME samples -- pure FEM-vs-FEM discretization
    error, no neural network involved. All three *_ref dicts come from
    load_coarse_reference / generate_fine_reference and must have been
    built from the SAME dataset_path and sample ordering (so sample i in
    each corresponds to the same physical case)."""
    Ntheta_f, Nr_f, Nz_f = fine_ref["fine_resolution"]
    geom_fine = fine_ref["geom_fine"]
    region_mask_fine_torch = fine_ref["region_mask_fine_torch"]
    n_samples = min(len(coarse_ref["samples"]), len(mid_ref["samples"]), len(fine_ref["samples"]))

    def _one_resolution_vs_fine(ref, tag):
        Ntheta_c, Nr_c, Nz_c = ref["fine_resolution"]
        rows = []
        for idx in range(n_samples):
            s_c = ref["samples"][idx]
            s_f = fine_ref["samples"][idx]
            assert s_c["sample_index"] == s_f["sample_index"], "sample ordering mismatch between resolutions"

            u_c_np = s_c["u_true_f"][0].cpu().numpy()  # (n_nodes_c, 3)
            u_c_on_fine_np = interpolate_node_field_to_finer_mesh(
                u_c_np, Ntheta_c, Nr_c, Nz_c, Ntheta_f, Nr_f, Nz_f, R_GRADING, n_components=3)
            u_c_on_fine = torch.tensor(u_c_on_fine_np, dtype=dtype, device=device).unsqueeze(0)
            u_true_f = s_f["u_true_f"]
            E_f_t, nu_f_t = s_f["E_f_t"], s_f["nu_f_t"]

            err = u_c_on_fine - u_true_f
            l2_num = torch.sqrt(torch.mean(torch.sum(err ** 2, dim=2)))
            l2_den = torch.sqrt(torch.mean(torch.sum(u_true_f ** 2, dim=2))) + 1e-12
            disp_rel_l2 = float((l2_num / l2_den).item())

            h1_rel = float(compute_H1_seminorm_rel(u_true_f, u_c_on_fine, geom_fine)[0].item())
            sigma_true_f, w_true_f = compute_region_sigma_full(
                u_true_f, E_f_t, nu_f_t, geom_fine, region_mask_fine_torch)
            sigma_c_on_fine, _ = compute_region_sigma_full(
                u_c_on_fine, E_f_t, nu_f_t, geom_fine, region_mask_fine_torch)
            field_rel = float(cauchy_field_rel_from_full(sigma_true_f, w_true_f, sigma_c_on_fine)[0].item())
            comp_rel = sigma_components_rel_from_full(sigma_true_f, w_true_f, sigma_c_on_fine)

            avg_true_f, _, _ = compute_region_sigma_xx(u_true_f, E_f_t, nu_f_t, geom_fine, region_mask_fine_torch)
            avg_c_on_fine, _, _ = compute_region_sigma_xx(u_c_on_fine, E_f_t, nu_f_t, geom_fine, region_mask_fine_torch)
            region_avg_sigma_xx_rel = float(
                (torch.abs(avg_c_on_fine - avg_true_f) / (torch.abs(avg_true_f) + 1e-12))[0].item())

            # Scalars: each resolution's OWN FEM solution on its OWN mesh/material, no interpolation.
            geom_c, mask_c = ref["geom_fine"], ref["region_mask_fine_torch"]
            u_c_native = s_c["u_true_f"]
            E_c_t, nu_c_t = s_c["E_f_t"], s_c["nu_f_t"]
            energy_c = compute_energy(u_c_native, E_c_t, nu_c_t, geom_c)
            energy_f = compute_energy(u_true_f, E_f_t, nu_f_t, geom_fine)
            energy_rel = float((torch.abs(energy_c - energy_f) / (torch.abs(energy_f) + 1e-12))[0].item())

            _, Mr_c, _ = compute_reaction(u_c_native, E_c_t, nu_c_t, geom_c)
            _, Mr_f, _ = compute_reaction(u_true_f, E_f_t, nu_f_t, geom_fine)
            moment_rel = float((torch.abs(Mr_c - Mr_f) / (torch.abs(Mr_f) + 1e-12))[0].item())

            rows.append({
                "sample_index": s_c["sample_index"],
                "disp_rel_l2_vs_950400": disp_rel_l2,
                "H1_seminorm_rel_vs_950400": h1_rel,
                "region_cauchy_field_rel_vs_950400": field_rel,
                "region_sigma_components_rel_vs_950400": {
                    name: float(comp_rel[name][0].item()) for name, _, _ in STRESS_COMPONENTS},
                "region_avg_sigma_xx_rel_vs_950400": region_avg_sigma_xx_rel,
                "energy_rel_vs_950400": energy_rel,
                "reaction_moment_y_rel_vs_950400": moment_rel,
            })

        n_elem = (Ntheta_c - 1) * (Nr_c - 1) * (Nz_c - 1)
        summary = {"resolution": [Ntheta_c, Nr_c, Nz_c], "n_elements": n_elem, "n_samples": n_samples}
        for key in ("disp_rel_l2_vs_950400", "H1_seminorm_rel_vs_950400", "region_cauchy_field_rel_vs_950400",
                    "region_avg_sigma_xx_rel_vs_950400", "energy_rel_vs_950400", "reaction_moment_y_rel_vs_950400"):
            vals = np.array([row[key] for row in rows])
            vals_valid = vals[~np.isnan(vals)]
            summary[f"mean_{key}"] = float(np.mean(vals_valid)) if len(vals_valid) else float("nan")
            summary[f"median_{key}"] = float(np.median(vals_valid)) if len(vals_valid) else float("nan")
        for name, _, _ in STRESS_COMPONENTS:
            vals = np.array([row["region_sigma_components_rel_vs_950400"][name] for row in rows])
            vals_valid = vals[~np.isnan(vals)]
            summary[f"mean_region_sigma_{name}_rel_vs_950400"] = float(np.mean(vals_valid)) if len(vals_valid) else float("nan")

        if verbose:
            print(f"  [{tag}, {n_elem:,} elements] disp_rel_l2={summary['mean_disp_rel_l2_vs_950400']:.4f}  "
                  f"H1={summary['mean_H1_seminorm_rel_vs_950400']:.4f}  "
                  f"region_cauchy_field={summary['mean_region_cauchy_field_rel_vs_950400']:.4f}  "
                  f"energy={summary['mean_energy_rel_vs_950400']:.4f}  "
                  f"moment={summary['mean_reaction_moment_y_rel_vs_950400']:.4f}")

        return {"summary": summary, "per_sample": rows}

    if verbose:
        print(f"FEM discretization error vs. {Ntheta_f, Nr_f, Nz_f} "
              f"({(Ntheta_f - 1) * (Nr_f - 1) * (Nz_f - 1):,} elements), n_samples={n_samples}")
    result_coarse = _one_resolution_vs_fine(coarse_ref, "6,840-element")
    result_mid = _one_resolution_vs_fine(mid_ref, "43,400-element")
    return {"fine_resolution": [Ntheta_f, Nr_f, Nz_f], "n_samples": n_samples,
            "vs_6840": result_coarse, "vs_43400": result_mid}
