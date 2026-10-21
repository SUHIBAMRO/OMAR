"""Correctness check for equilibrium_augmented_B3.py (Professor
Rabczuk's items 2 and 3, 2026-10-16 email). Direct closed-form
verification isn't available here (B3's free groove/lateral surfaces
mean the divergence-theorem boundary term doesn't vanish for an
arbitrary field), so this checks the strongest available empirical
signal: minimizing total potential energy (train_B3.py's own loss, with
NO equilibrium term in its gradient at all) should, as a side effect,
also drive this INDEPENDENTLY-computed weak-form residual down --
confirming the two are genuinely connected (stationarity of the energy
functional IS the weak form of equilibrium), not checking a
disconnected number.

    python -m omar_pfem.test_equilibrium_augmented_B3
"""
import torch

from omar_pfem.equilibrium_augmented_B3 import weak_form_equilibrium_residual
from omar_pfem.mixed_stress_B3 import build_fixed_geometry_mixed
from omar_pfem.train_B3 import apply_dirichlet_b3, build_model, get_args, total_potential_energy_B3

torch.set_default_dtype(torch.float64)


def test_residual_tracks_energy_minimization():
    torch.manual_seed(0)
    dtype, device = torch.float64, "cpu"
    Ntheta, Nr, Nz = 5, 4, 4
    geom = build_fixed_geometry_mixed(Ntheta, Nr, Nz, device, dtype=dtype)
    n_nodes = geom["n_nodes"]
    Batch = 2

    E_node = (1000.0 + 50.0 * torch.randn(Batch, n_nodes, dtype=dtype)).abs() + 500.0
    nu_node = torch.full((Batch, n_nodes), 0.45, dtype=dtype)
    phi = torch.tensor([0.05, 0.07], dtype=dtype)

    model_args = get_args(["--n_hidden", "32", "--n_layers", "2", "--n_heads", "2",
                           "--mlp_ratio", "1", "--slice_num", "16", "--ref", "4"])
    model = build_model(model_args, device).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    xyz_batch = geom["nodes"].unsqueeze(0).expand(Batch, -1, -1)
    fun_material = torch.stack([E_node, nu_node, phi[:, None].expand(-1, n_nodes)], dim=2)
    gen = torch.Generator(device=device).manual_seed(123)

    residuals, energies = [], []
    for it in range(400):
        u_net = model(xyz_batch, fun_material)
        u = apply_dirichlet_b3(u_net, geom, phi)
        loss = total_potential_energy_B3(u, E_node, nu_node, geom).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            u2 = apply_dirichlet_b3(model(xyz_batch, fun_material), geom, phi)
            resid = weak_form_equilibrium_residual(u2, E_node, nu_node, geom,
                                                   n_virtual=20, generator=gen)
        residuals.append(resid.item())
        energies.append(loss.item())

    first, last = sum(residuals[:20]) / 20, sum(residuals[-20:]) / 20
    assert last < first, (
        f"weak-form residual did not decrease as energy was minimized "
        f"(first20={first:.3e}, last20={last:.3e}) -- not tracking equilibrium")
    print(f"PASS  weak-form residual tracks energy minimization "
          f"(first20={first:.3e} -> last20={last:.3e})")


if __name__ == "__main__":
    test_residual_tracks_energy_minimization()
    print("\nall equilibrium-augmented-B3 checks passed")
