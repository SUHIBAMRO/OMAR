"""Rigorous calibration of mixed_stress_B3's P_OUTPUT_SCALE, the same
methodology that originally found train_B3.py's own OUTPUT_SCALE=0.02
(see that constant's docstring): measure the network's RAW untrained
output magnitude for the stress channels, measure a REALISTIC true P
magnitude from real sampled material fields, and derive the scale that
aligns them -- rather than guessing one.

Run 2026-10-21 (CPU, production model size), after the first real
mixed-stress training run (50,000 iterations) reached combined
displacement/energy/reaction errors ~100x worse than the DEM baseline,
with the "constitutive" loss term showing no clear downward trend
across the whole run: derived scale = 246.6, vs. the hand-picked
default of 500.0 actually used (a real, ~2x miscalibration -- smaller
than OUTPUT_SCALE's original 5-50x mismatch, so very likely a
contributing factor but not the full explanation by itself; the
per-component constitutive residual observed during training (~65) was
comparable in magnitude to the realistic true P's own RMS (~74), i.e.
close to 100% relative error throughout, which a mere 2x initial-scale
correction would not fully resolve on its own).

"Realistic true P" uses the SAME Dirichlet-ansatz construction
train_B3.apply_dirichlet_b3 already uses (the boundary-consistent
"particular" field, network's own raw contribution set to zero) -- not
the full true FEM solution (lives only on Drive, not available for a
quick CPU check), but a physically grounded field built from REAL
sampled (E,nu,phi) via this project's own trusted sample_batch.

    python -m omar_pfem.calibrate_p_output_scale
"""
import torch

from omar_pfem.data.data_generate_B3_dataset import DEFAULT_RESOLUTION
from omar_pfem.mixed_stress_B3 import build_fixed_geometry_mixed, neo_hookean_P_closed_form
from omar_pfem.train_B3 import apply_dirichlet_b3, sample_batch
from omar_pfem.train_B3_mixed_stress import build_model, get_args


def main():
    torch.manual_seed(0)
    dtype = torch.float64
    device = "cpu"
    Ntheta, Nr, Nz = DEFAULT_RESOLUTION
    geom = build_fixed_geometry_mixed(Ntheta, Nr, Nz, device, dtype=dtype)
    print(f"geometry: {geom['n_elements']} elements, {geom['n_nodes']} nodes")

    Batch = 16
    E_node, nu_node, phi = sample_batch(Batch, Ntheta, Nr, Nz, seed_offset=0,
                                         dtype=dtype, device=device)
    u_net_raw_zero = torch.zeros(Batch, geom["n_nodes"], 3, dtype=dtype)
    u_particular = apply_dirichlet_b3(u_net_raw_zero, geom, phi)

    elements, B_op = geom["elements"], geom["B"]
    ue = u_particular[:, elements, :].permute(0, 1, 3, 2)
    E_elem = E_node[:, elements].mean(dim=2)
    nu_elem = nu_node[:, elements].mean(dim=2)
    mu_elem = E_elem / (2 * (1 + nu_elem))
    lam_elem = E_elem * nu_elem / ((1 + nu_elem) * (1 - 2 * nu_elem))
    eye = torch.eye(3, dtype=dtype).view(1, 1, 3, 3)

    all_P = []
    for g in range(B_op.shape[0]):
        H = torch.einsum("bedq,ecq->bedc", ue, B_op[g])
        F = eye + H
        P = neo_hookean_P_closed_form(F, mu_elem, lam_elem)
        all_P.append(P)
    all_P = torch.stack(all_P, dim=0)

    print("\nPart 1 -- realistic true P magnitude (BC-consistent field, real E/nu/phi)")
    print("  mean |P|:", all_P.abs().mean().item())
    print("  max |P|:", all_P.abs().max().item())
    true_rms = (all_P ** 2).mean().sqrt().item()
    print("  RMS |P|:", true_rms)

    model_args = get_args([])
    model = build_model(model_args, device).to(dtype)
    xyz_batch = geom["nodes"].unsqueeze(0).expand(Batch, -1, -1)
    fun_material = torch.stack([E_node, nu_node, phi[:, None].expand(-1, geom["n_nodes"])], dim=2)
    with torch.no_grad():
        raw = model(xyz_batch, fun_material)
    p_raw = raw[..., 3:]

    print("\nPart 2 -- raw UNTRAINED network output, stress channels (production model size)")
    print("  mean |raw|:", p_raw.abs().mean().item())
    raw_rms = (p_raw ** 2).mean().sqrt().item()
    print("  RMS |raw|:", raw_rms)

    needed_scale = true_rms / raw_rms
    print(f"\nPart 3 -- derived scale = {true_rms:.3f} / {raw_rms:.3f} = {needed_scale:.3f}")
    return needed_scale


if __name__ == "__main__":
    main()
