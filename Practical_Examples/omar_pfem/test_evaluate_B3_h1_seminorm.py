"""Correctness checks for evaluate_B3_qois.compute_H1_seminorm_rel,
added 2026-10-26 to close a real gap: Professor Rabczuk's 2026-10-20
email explicitly named "the H1 semi-norm" as a QoI to track, but it had
never been computed for any of the four candidate error-reduction
methods.

    python -m omar_pfem.test_evaluate_B3_h1_seminorm
"""
import torch

from omar_pfem.evaluate_B3_qois import compute_H1_seminorm_rel
from omar_pfem.train_B3 import build_fixed_geometry

torch.set_default_dtype(torch.float64)


def _small_geom():
    return build_fixed_geometry(Ntheta=5, Nr=4, Nz=4, device="cpu", dtype=torch.float64)


def test_zero_for_identical_fields():
    geom = _small_geom()
    Batch = 3
    u = 0.1 * torch.randn(Batch, geom["n_nodes"], 3, dtype=torch.float64)
    rel = compute_H1_seminorm_rel(u, u.clone(), geom)
    err = rel.abs().max().item()
    assert err < 1e-12, f"H1 semi-norm rel error is not exactly 0 for identical fields (err={err})"
    print(f"PASS  H1 semi-norm rel error == 0 for u_pred == u_true (max abs {err:.2e})")


def test_matches_hand_computation_for_affine_field():
    """u_true(X) = G_true @ X, u_pred(X) = G_pred @ X (globally affine,
    exactly representable by trilinear shape functions) -- grad(u) is
    then the SAME constant matrix (G_true or G_pred) at every Gauss
    point everywhere, so the volume-weighted relative H1 semi-norm error
    must reduce to the plain Frobenius-norm ratio ||G_pred-G_true|| /
    ||G_true||, independent of volume weights (same constant at every
    point). Checked against that closed form directly."""
    torch.manual_seed(0)
    geom = _small_geom()
    nodes = geom["nodes"]
    Batch = 2
    G_true = 0.1 * torch.randn(Batch, 3, 3, dtype=torch.float64)
    G_pred = G_true + 0.03 * torch.randn(Batch, 3, 3, dtype=torch.float64)

    u_true = torch.einsum("bij,nj->bni", G_true, nodes)
    u_pred = torch.einsum("bij,nj->bni", G_pred, nodes)

    rel = compute_H1_seminorm_rel(u_true, u_pred, geom)
    expected = torch.linalg.norm((G_pred - G_true).reshape(Batch, -1), dim=1) / \
        torch.linalg.norm(G_true.reshape(Batch, -1), dim=1)
    err = (rel - expected).abs().max().item()
    assert err < 1e-9, f"H1 semi-norm rel error does not match the closed-form affine-field value (err={err})"
    print(f"PASS  H1 semi-norm rel error matches the closed-form affine-field reference (max abs diff {err:.2e})")


def test_nonzero_and_finite_for_random_fields():
    geom = _small_geom()
    Batch = 4
    u_true = 0.1 * torch.randn(Batch, geom["n_nodes"], 3, dtype=torch.float64)
    u_pred = 0.1 * torch.randn(Batch, geom["n_nodes"], 3, dtype=torch.float64)
    rel = compute_H1_seminorm_rel(u_true, u_pred, geom)
    assert torch.isfinite(rel).all() and (rel > 0).all(), "expected finite, strictly positive values"
    print(f"PASS  H1 semi-norm rel error finite and positive for random fields ({rel.tolist()})")


if __name__ == "__main__":
    test_zero_for_identical_fields()
    test_matches_hand_computation_for_affine_field()
    test_nonzero_and_finite_for_random_fields()
    print("\nall evaluate_B3_h1_seminorm checks passed")
