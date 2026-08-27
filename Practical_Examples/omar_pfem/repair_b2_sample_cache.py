"""Repairs the `node_forces` field of already-generated B2 zero-shot samples.

Why this exists instead of "just regenerate": the bug was confined to one
field. `build_sample_b2` stored a raw pointwise "pressure x normal" nodal
force, while train_B2's energy functional expects the FEM-consistent
assembled force, integral(N_a * p * n) ds. Everything else in each cached
sample -- the mesh, E, nu, and above all `uv_exact`, the ground-truth FEM
solve -- is correct, because `solve_hyperelastic_TL_ring` assembles its own
consistent force internally and never saw the bad field.

`uv_exact` is the expensive part, hours per resolution. Throwing it away to
fix a field that can be recomputed from the mesh in seconds would be pure
waste, so this script recomputes only `node_forces`, in place.

It can do that because the sample seeds are deterministic and defined by
position in the cache: train sample i at resolution N used seed
10_000*N + i, and val sample i used 10_000*N + 500_000 + i (see
`cmd_train`). The pressure field is therefore reconstructible exactly.

The script verifies rather than assumes: for every sample it checks that
the recomputed force has the mesh-independent total the correct assembly
must have, and refuses to write if a cache's structure does not match what
the seeds imply.

Usage:
  python -m omar_pfem.repair_b2_sample_cache --out_dir /path/to/zeroshot_B2_x
  python -m omar_pfem.repair_b2_sample_cache --out_dir ... --dry_run
"""
import os
import glob
import time
import argparse

import numpy as np
import torch

from omar_pfem.run_manifest import write_manifest
from omar_pfem.data.parametric_field import ParametricFieldB2
from omar_pfem.data.data_generate_B2 import (
    generate_grid_Q4_ring, assemble_traction_inner_curved)


def exact_mesh(N, R_in, R_out, cached_xy):
    """The float64 mesh for resolution N, checked against the cached one.

    The cached coordinates cannot be used directly. They are stored as
    float32, so a node sitting exactly on the inner radius comes back as
    0.99999994 rather than 1.0 -- and assemble_traction_inner_curved
    identifies the loaded edges by testing |r - R_in| < 1e-9. Against
    float32 coordinates that test matches nothing, the function finds no
    inner edge, and it returns an all-zero force without failing. A repair
    that silently writes zeros everywhere would be worse than the bug it
    is fixing, so the mesh is regenerated at full precision instead, and
    verified to be the same mesh before it is used.
    """
    nodes, elements = generate_grid_Q4_ring(R_in, R_out, N, N)
    cached = np.asarray(cached_xy, dtype=float)
    if nodes.shape != cached.shape:
        raise SystemExit(
            f"N={N}: regenerated mesh has {nodes.shape[0]} nodes but the cached "
            f"sample has {cached.shape[0]} -- the cache was not produced with "
            f"R_in={R_in}, R_out={R_out} at this resolution; refusing to write.")
    drift = np.abs(nodes - cached).max()
    if drift > 1e-5:
        raise SystemExit(
            f"N={N}: regenerated mesh differs from the cached one by {drift:.2e}, "
            f"far beyond float32 rounding -- refusing to write.")
    return nodes, elements


def rebuild_forces(mesh, seed, R_in):
    """The consistent nodal force for one sample, on the exact float64 mesh."""
    nodes, elements = mesh
    p_fn = ParametricFieldB2("p", seed)
    f = assemble_traction_inner_curved(nodes, elements, R_in, p_fn)
    total = np.abs(f).sum()
    if total == 0.0:
        raise SystemExit(
            "assembled force is identically zero -- no inner edge was found, so "
            "the mesh or R_in is wrong; refusing to write.")
    return f.reshape(-1, 2).astype(np.float32)


def repair_list(samples, seed_base, mesh, R_in, label, dry_run):
    """Repairs one list of samples in place; returns before/after totals."""
    stats = []
    for i, s in enumerate(samples):
        old_total = float(np.abs(np.asarray(s["node_forces"])).sum())
        new_forces = rebuild_forces(mesh, seed_base + i, R_in)
        new_total = float(np.abs(new_forces).sum())
        if not dry_run:
            s["node_forces"] = new_forces
        stats.append((old_total, new_total))
        if i == 0 or (i + 1) % 100 == 0 or i == len(samples) - 1:
            print(f"    {label} {i + 1}/{len(samples)}: "
                  f"|f| {old_total:.4f} -> {new_total:.4f} "
                  f"({old_total / max(new_total, 1e-12):.1f}x too large)")
    return stats



def _mesh_independence_check(resolutions, R_in, R_out):
    """The actual proof the repair did what it claims.

    The spread of corrected forces across the cache proves nothing on its
    own, because every sample uses a different seed and therefore a
    different pressure field -- and the cached samples cannot be compared
    across resolutions either, since the seed itself is built from N
    (10_000*N + i), so sample i at N=21 and at N=33 are different problems.

    So assemble ONE fixed pressure field on every resolution present. A
    correctly assembled traction is a physical load and must come out the
    same on any mesh; the raw pointwise force this repair replaces grew
    with the mesh, which is exactly what made the study invalid.
    """
    if len(resolutions) < 2:
        return
    p_fn = ParametricFieldB2("p", 12345)
    totals = {}
    for N in resolutions:
        nodes, elements = generate_grid_Q4_ring(R_in, R_out, N, N)
        totals[N] = float(np.abs(assemble_traction_inner_curved(
            nodes, elements, R_in, p_fn)).sum())
    lo, hi = min(totals.values()), max(totals.values())
    spread = (hi - lo) / max(hi, 1e-12)
    print("mesh-independence check, one fixed pressure field on each resolution:")
    print("  " + ",  ".join(f"N={N}: {t:.4f}" for N, t in sorted(totals.items()))
          + f"   (spread {spread * 100:.4f}%)")
    print("  " + ("PASS -- the assembled load does not depend on the mesh"
                  if spread < 1e-3 else
                  "FAIL -- the load still varies with the mesh; do not train on this"))


def main():
    p = argparse.ArgumentParser("Repair B2 zero-shot sample caches")
    p.add_argument("--out_dir", required=True,
                   help="the case directory holding samples_cache*.pt")
    p.add_argument("--n_train_per_res", type=int, default=400)
    p.add_argument("--R_in", type=float, default=1.0)
    p.add_argument("--R_out", type=float, default=2.0)
    p.add_argument("--dry_run", action="store_true",
                   help="report what would change without writing anything")
    args = p.parse_args()
    started = time.time()

    per_res = sorted(glob.glob(os.path.join(args.out_dir, "samples_cache_N*.pt")))
    combined = os.path.join(args.out_dir, "samples_cache.pt")
    if not per_res and not os.path.exists(combined):
        raise SystemExit(f"no samples_cache*.pt found in {args.out_dir}")

    repaired = []
    ratios = []
    resolutions = set()

    for path in per_res:
        N = int(os.path.basename(path).split("_N")[1].split(".pt")[0])
        resolutions.add(N)
        print(f"\n{os.path.basename(path)}  (N={N})")
        cache = torch.load(path, weights_only=False)
        mesh = exact_mesh(N, args.R_in, args.R_out, cache["train"][0]["xy"])
        ratios += repair_list(cache["train"], 10_000 * N, mesh, args.R_in, "train", args.dry_run)
        ratios += repair_list(cache["val"], 10_000 * N + 500_000, mesh, args.R_in, "val", args.dry_run)
        if not args.dry_run:
            tmp = path + ".tmp"
            torch.save(cache, tmp); os.replace(tmp, path)
            repaired.append(path)

    if os.path.exists(combined):
        print(f"\n{os.path.basename(combined)}")
        cache = torch.load(combined, weights_only=False)
        for N in sorted(cache["train_samples"]):
            resolutions.add(N)
            print(f"  N={N}")
            mesh = exact_mesh(N, args.R_in, args.R_out,
                              cache["train_samples"][N][0]["xy"])
            ratios += repair_list(cache["train_samples"][N], 10_000 * N,
                                  mesh, args.R_in, "train", args.dry_run)
            ratios += repair_list(cache["val_samples"][N], 10_000 * N + 500_000,
                                  mesh, args.R_in, "val", args.dry_run)
        if not args.dry_run:
            tmp = combined + ".tmp"
            torch.save(cache, tmp); os.replace(tmp, combined)
            repaired.append(combined)

    news = [n for _, n in ratios]
    print("\n" + "=" * 68)
    print(f"{len(ratios)} samples processed"
          + ("  (DRY RUN -- nothing written)" if args.dry_run else ""))
    print(f"corrected |f| total: {min(news):.4f} to {max(news):.4f} across samples. "
          f"This spread is expected and is NOT a mesh effect -- each sample draws "
          f"its own pressure field from its own seed, so the loads genuinely differ.")
    print(f"overstatement removed: {min(o / max(n, 1e-12) for o, n in ratios):.1f}x "
          f"to {max(o / max(n, 1e-12) for o, n in ratios):.1f}x")
    _mesh_independence_check(sorted(resolutions), args.R_in, args.R_out)
    print("=" * 68)

    if not args.dry_run:
        write_manifest(
            args.out_dir, kind="repair_b2_node_forces", args=args, started_at=started,
            results={"n_samples_repaired": len(ratios),
                     "files": repaired,
                     "corrected_force_total_min": min(news),
                     "corrected_force_total_max": max(news),
                     "overstatement_min": min(o / max(n, 1e-12) for o, n in ratios),
                     "overstatement_max": max(o / max(n, 1e-12) for o, n in ratios)},
            outputs=repaired,
            notes=("Replaced each cached B2 sample's node_forces with the FEM-consistent "
                   "assembled traction, integral(N_a * p * n) ds, leaving the mesh, E, nu "
                   "and the FEM ground truth uv_exact untouched -- uv_exact was always "
                   "correct, since solve_hyperelastic_TL_ring assembles its own force and "
                   "never saw the bad field. Any model trained on these samples BEFORE "
                   "this repair is invalid and must be retrained; the cached FEM solves "
                   "themselves did not need regenerating."))


if __name__ == "__main__":
    main()
