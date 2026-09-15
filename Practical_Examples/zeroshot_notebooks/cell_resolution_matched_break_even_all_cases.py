# =====================================================================
#  CELL -- resolution-matched break-even (NO vs. torch-fem, BOTH at
#  N=1401) for ALL 6 cases, completing Timon round-11 point 1 (which
#  explicitly asked to KEEP both break-even comparisons) alongside
#  point 2 (extend everything to all 6 cases). Full scope, per Omar's
#  explicit request.
#
#  PREREQUISITE (2026-09-14, already committed, verified before this
#  cell was written): torchfem_comparison.py's build_torchfem_model/
#  solve_theirs were hardcoded to B1xNeo-Hookean only. Generalized to
#  all 3 materials (new mooney_rivlin_psi_3d/arruda_boyce_psi_3d, exact
#  3D reductions of materials_torch.py's own 2D formulas, verified
#  numerically identical before use) and both geometries (constraints
#  built from fixed_dofs' own per-DOF decomposition instead of assuming
#  "every fixed node has both components fixed", which is false for
#  B2's symmetry edges). Two further real bugs found and fixed along
#  the way: torch.linalg.det()'s own second derivative is NaN at F=I
#  (fixed by routing through slogdet), and B2's per-Gauss-point material
#  sampling is incompatible with torch-fem's own per-ELEMENT-only params
#  API (fixed by averaging to one value per element for torch-fem's B2
#  calls specifically -- a small, bounded, clearly-documented
#  approximation, not a new accuracy claim). Verified end-to-end against
#  the slow CPU reference solvers for all 6 cases at N=7 before ever
#  being pointed at N=1401: B1's three materials match to ~1e-11
#  (machine precision), B2's three materials to ~8e-4 (small, expected
#  discretization-level difference from the averaging approximation).
#
#  COST: cheap. torch-fem's own N=1401 solve for B1xNeo-Hookean (round-9's
#  own real measurement) was 133.83s -- the whole point of this
#  comparison is that torch-fem is fast. Six cases, each one solve at
#  N=1401 plus a quick NO inference timing (a few hundred forward
#  passes, seconds) -- expect well under 15 minutes total on a real GPU,
#  not hours.
#
#  WHAT THIS PRODUCES: for each of the 6 cases, torch-fem's real N=1401
#  wall-clock (matched FP64/1e-8 precision, same convention as every
#  other torch-fem number in this project), the NO's own inference time
#  at N=1401 (eager fp32), and the resulting resolution-matched
#  break-even (training cost already known/measured for B1xNeo-Hookean;
#  for the other 5 cases, training cost is read from each checkpoint's
#  own metrics_history.json if present, else left as "unknown" rather
#  than guessed).
# =====================================================================
# Real bug found on a live A100 run (2026-09-14): EVERY case failed to
# converge ("Newton-Raphson did not converge ... after 10 cutbacks"),
# including B1xNeo-Hookean, which had solved cleanly in an earlier run of
# this exact notebook. Traced to omar_pfem.data.materials unconditionally
# importing omar_pfem.data.material_models_jax at module level (needed
# for Mooney-Rivlin/Arruda-Boyce's JAX-autodiff-based E_nu_to_* --
# neither this cell nor any material actually needs it to run on GPU,
# it only converts two scalars, E and nu, to material parameters).
# JAX's own default behavior the moment it first touches a GPU is to
# preallocate ~90% of that GPU's ENTIRE memory for itself and never give
# it back for the life of the process -- and this reservation is
# invisible to torch.cuda.memory_allocated()/memory_reserved() (JAX
# manages its own separate CUDA memory pool), which is exactly why the
# "[gpu mem before this case] allocated=12.51GB reserved=20.06GB" print
# below looked perfectly healthy while torch-fem's own solve immediately
# hit "78.61 GiB memory in use" and OOM'd. Since data.materials is
# imported for every material (including Neo-Hookean, which never
# actually needs JAX itself), this poisoned the WHOLE run, not just the
# Mooney-Rivlin/Arruda-Boyce cases. Fixed by forcing JAX onto CPU only,
# BEFORE any omar_pfem import -- jax's actual work here (two-scalar
# conversions) is instant on CPU regardless, so this costs nothing and
# leaves the entire GPU to PyTorch/torch-fem as intended.
#
# NOTE: omar_pfem.data.material_models_jax already sets this SAME env var
# via os.environ.setdefault(...) at its own import time -- and has for a
# long time, with its own comment describing this exact failure mode. That
# it still happened means either (a) something imports the bare `jax`
# package before material_models_jax.py ever gets a chance to run (a
# setdefault after jax is already imported/initialized has no effect on
# jax's already-resolved backend), or (b) this Colab runtime's environment
# already had JAX_PLATFORMS set to something else beforehand, which
# setdefault would not override. Using a forced assignment here, at the
# absolute top of this script, before any pip install/import, closes both
# gaps regardless of which one it actually was.
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import json
import subprocess
import sys
import time

_started = time.time()


def run(cmd):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


from google.colab import drive
drive.mount('/content/drive')

REPO = '/content/OMAR'
if not os.path.isdir(REPO):
    run(['git', 'clone', '-b', 'claude/claude-code-question-d307wp',
         'https://github.com/SUHIBAMRO/OMAR.git', REPO])
else:
    run(['git', '-C', REPO, 'fetch', 'origin', 'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'checkout', 'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'reset', '--hard', 'origin/claude/claude-code-question-d307wp'])

run([sys.executable, '-m', 'pip', 'install', '-q',
     'einops', 'timm', 'h5py', 'jax', 'tqdm'])
run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-sla'])
run([sys.executable, '-m', 'pip', 'install', '-q', 'nvmath-python[cu12]==0.9.0'])
run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-fem'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)
sys.path.insert(0, f'{WORK}/report_builders')

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import numpy as np
import torch
assert torch.cuda.is_available(), 'this cell needs a real GPU'
print('GPU:', torch.cuda.get_device_name(0))

from omar_pfem.high_dof_convergence_study import build_mesh_and_bcs
from omar_pfem.torchfem_comparison import solve_theirs
from omar_pfem.measure_inference_latency import build_model
import argparse

# Diagnostic: confirm JAX actually resolved to CPU as intended (checked
# directly, not assumed, given a previous run failed despite JAX_PLATFORMS
# being set) -- import it explicitly here since data.materials only
# imports it lazily on first use, which would otherwise happen deep
# inside the first case's own build_mesh_and_bcs call, too late to catch
# a wrong backend before spending real GPU time on a doomed sweep.
import jax
jax_devices = jax.devices()
print(f'JAX devices: {jax_devices}')
assert all(d.platform == 'cpu' for d in jax_devices), (
    f'JAX resolved to a non-CPU backend ({jax_devices}) despite JAX_PLATFORMS=cpu -- '
    f'it will preallocate most of the GPU for itself and starve torch-fem. Refusing to '
    f'proceed; check whether something else imports jax before this cell, or whether '
    f'this Colab runtime pre-sets JAX_PLATFORMS/JAX_PLATFORM_NAME to something else.')
print('JAX confirmed CPU-only -- safe to proceed with the full GPU for PyTorch/torch-fem.')

R = '/content/drive/MyDrive/pfem_run'
device = torch.device('cuda')
N = 1401

# (geometry, material, checkpoint path, model_args)
model_args_b1 = argparse.Namespace(
    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
    use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_out=2.0,
)
model_args_b2 = argparse.Namespace(
    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,
    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,
    use_soft_dirichlet=1, R_in=1.0, R_out=2.0,
)

CASES = [
    ('B1', 'neo_hookean', f'{R}/zeroshot_B1_neo_hookean_multires/model_best.pt', model_args_b1),
    ('B1', 'mooney_rivlin', f'{R}/zeroshot_B1_mooney_rivlin/model_best.pt', model_args_b1),
    ('B1', 'arruda_boyce', f'{R}/zeroshot_B1_arruda_boyce/model_best.pt', model_args_b1),
    ('B2', 'neo_hookean', f'{R}/zeroshot_B2_neo_hookean_fixedsel/model_best.pt', model_args_b2),
    ('B2', 'mooney_rivlin', f'{R}/zeroshot_B2_mooney_rivlin_fixedsel/model_best.pt', model_args_b2),
    ('B2', 'arruda_boyce', f'{R}/zeroshot_B2_arruda_boyce_fixedsel/model_best.pt', model_args_b2),
]

# Already-verified NO@N=1401 timing for B1xNeo-Hookean (multi-res checkpoint,
# 2026-09-14) -- reused, not remeasured, since it is already an independently
# confirmed number.
KNOWN_NO_MS = {('B1', 'neo_hookean'): 2292.1}
KNOWN_TRAINING_S = {('B1', 'neo_hookean'): 41881.28}

results = []
for geometry, material, ckpt, model_args in CASES:
    case_id = f'{geometry}_{material}'
    print('\n' + '#' * 78)
    print(f'# {case_id}')
    print('#' * 78)

    # ---- Start each case with a genuinely clean GPU memory state ----
    # Real OOM hit on a live A100 run (2026-09-14): peak_mem_mb climbed
    # case-to-case within the SAME process (Neo-Hookean 70.8GB ->
    # Mooney-Rivlin 73.6GB -> Arruda-Boyce OOM'd trying to allocate just
    # 1.18GB more with 78.6GB already "in use") even though each case
    # solves the identical-size problem independently -- the classic
    # PyTorch caching-allocator fragmentation pattern the OOM message
    # itself points at ("reserved but unallocated memory is large").
    # torch-fem's own tangent-stiffness Hessian (vmap(jacrev(jacrev(psi))))
    # is memory-hungry at N=1401's ~4M DOF regardless, and Arruda-Boyce's
    # own psi (a 5-term power series, the longest computational graph of
    # the three materials) needs more of it than Neo-Hookean/Mooney-Rivlin's
    # simpler polynomial forms -- but it should not need MORE than its own
    # honest requirement just because two earlier cases ran first in the
    # same kernel. gc.collect() + empty_cache() between cases (not inside
    # solve_theirs itself, to avoid touching that already-verified function)
    # gives each case a fresh allocator state instead of fighting whatever
    # fragmentation the previous case left behind.
    import gc
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        print(f'  [gpu mem before this case] allocated={torch.cuda.memory_allocated(device)/1e9:.2f}GB '
              f'reserved={torch.cuda.memory_reserved(device)/1e9:.2f}GB')

    # ---- torch-fem @ N=1401, matched FP64/1e-8 precision ----
    nodes, elements, free_dofs, fext_full, elem_params = build_mesh_and_bcs(
        geometry, 'Q4', N, material, device, torch.float64)
    fixed_dofs = np.setdiff1d(np.arange(2 * nodes.shape[0]), free_dofs)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t0 = time.time()
    try:
        u_theirs, t_theirs, peak_mb = solve_theirs(
            nodes, elements, *elem_params, fext_full=fext_full, fixed_dofs=fixed_dofs,
            material=material, dtype=torch.float64, device=device, tol=1e-8)
    except RuntimeError as e:
        # torchfem's own Newton-Raphson loop (base.py) wraps every Newton
        # iteration in a bare "except RuntimeError", retries with cutbacks,
        # and after max_cutbacks re-raises a NEW, generic RuntimeError
        # ("Newton-Raphson did not converge ... after N cutbacks") that does
        # NOT include the original error's text in str(e) -- so this catch
        # can never actually confirm whether the underlying cause was OOM,
        # a singular matrix, or something else. Two real fix attempts for
        # Arruda-Boyce specifically (a per-element N=401 pre-check, then a
        # chunked-Hessian rewrite that DID succeed at avoiding OOM inside
        # its own chunk loop down to 390 points/chunk with no crash) both
        # still hit this exact same generic failure -- confirming the real
        # bottleneck is elsewhere in torch-fem's own pipeline (most likely
        # its un-chunked global stiffness assembly) or a genuine numerical
        # non-convergence, not conclusively memory. Omar's own explicit
        # decision (2026-09-15), given two failed fix attempts and this
        # being a secondary comparison baseline (not the neural operator's
        # own accuracy story, which already succeeds for this material
        # independently): accept this as a torch-fem limitation for
        # Arruda-Boyce at this N and move on, rather than keep chasing it.
        is_oom = 'out of memory' in str(e).lower() or 'did not converge' in str(e).lower()
        if not is_oom:
            raise
        # Real candidate cause found 2026-09-15 (CPU-verified, not yet confirmed on a
        # real solve): arruda_boyce_psi_3d clamps I1_bar at the chain-locking limit
        # to avoid NaN/overflow; a direct test of that exact function shows the
        # clamp makes the chain-stretch part of the stress exactly flat once
        # triggered (bit-identical under a +-1e-4 perturbation), which is a
        # suspicious, physically-backwards behavior right where Newton would need
        # a well-conditioned tangent. The chunked material class above now reports
        # (via its own print, see torchfem_comparison.py) whenever a real solve
        # actually reaches this regime -- if that print appeared above, this is
        # very likely the true cause; if it never appeared, this is NOT it and the
        # search continues. torch-fem's own except RuntimeError in base.py still
        # discards the real underlying error before re-raising this generic
        # message, so err.__cause__ is the only way to see it -- surfaced below.
        real_cause = getattr(e, '__cause__', None)
        print(f'  *** {case_id} failed at N={N} -- torch-fem\'s own Newton-Raphson solve did '
              f'not converge ("{e}"). Real underlying error (via __cause__): '
              f'{type(real_cause).__name__ if real_cause else "none captured"}: {real_cause}. '
              f'Check the chunked-hyperelastic print above for chain-locking-clamp hits -- see '
              f'PROJECT_STATUS.md, 2026-09-15 entry for the full investigation. Skipping this '
              f'case rather than crashing the whole sweep. ***')
        # Free whatever partial allocation remains from the failed attempt before continuing.
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        results.append({'geometry': geometry, 'material': material, 'N': N, 'failed': str(e)})
        continue
    # Explicitly drop the large mesh/solution arrays for this case before moving on,
    # rather than letting them survive (referenced by the loop variable) until the
    # next iteration reassigns them.
    del nodes, elements, free_dofs, fext_full, elem_params, u_theirs
    print(f'  torch-fem @ N={N}: wall_clock={t_theirs:.2f}s, peak_mem_mb={peak_mb}')

    # ---- NO @ N=1401 inference (eager fp32) ----
    if (geometry, material) in KNOWN_NO_MS:
        no_ms = KNOWN_NO_MS[(geometry, material)]
        print(f'  NO @ N={N}: {no_ms:.1f} ms/sample (already verified, reused)')
    else:
        assert os.path.exists(ckpt), f'checkpoint not found: {ckpt}'
        model = build_model(model_args, device).to(torch.float32)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        if geometry == 'B1':
            from omar_pfem.resolution_invariance_zeroshot import build_sample_b1 as build_sample
            from omar_pfem.train_B1 import total_potential_energy_Q4_hyperelastic as tpe
            sample, _ = build_sample(N, seed=0, material=material, Lx=model_args.Lx,
                                      Ly=model_args.Ly, solve_fem=False)
            edge_key, node_key = 'top_edges', 'bottom_nodes'
        else:
            from omar_pfem.resolution_invariance_zeroshot import build_sample_b2 as build_sample
            from omar_pfem.train_B2 import total_potential_energy_Q4_hyperelastic as tpe
            sample, _ = build_sample(N, seed=0, material=material, R_in=model_args.R_in,
                                      R_out=model_args.R_out, solve_fem=False)
            edge_key, node_key = None, None  # B2 uses 3 node sets, handled below

        xy = torch.tensor(sample['xy'], device=device, dtype=torch.float32)
        quad = torch.tensor(sample['quad'], device=device, dtype=torch.long)
        E_b = torch.tensor(sample['E_node'][None], device=device, dtype=torch.float32)
        nu_b = torch.tensor(sample['nu_node'][None], device=device, dtype=torch.float32)
        f_b = torch.tensor(sample['node_forces'][None], device=device, dtype=torch.float32)

        def _one_forward():
            with torch.no_grad():
                if geometry == 'B1':
                    top_edges = torch.tensor(sample['top_edges'], device=device, dtype=torch.long)
                    bottom_nodes = torch.tensor(sample['bottom_nodes'], device=device, dtype=torch.long)
                    tpe(xy, quad, top_edges, bottom_nodes, model, E_b, nu_b, f_b,
                        use_soft_dirichlet=True, mode='plane_strain', dtype=torch.float32,
                        fun_dim=model_args.fun_dim, material=material, Ly=model_args.Ly)
                else:
                    inner_edges = torch.tensor(sample['inner_edges'], device=device, dtype=torch.long)
                    theta0_nodes = torch.tensor(sample['theta0_nodes'], device=device, dtype=torch.long)
                    thetahalfpi_nodes = torch.tensor(sample['thetahalfpi_nodes'], device=device, dtype=torch.long)
                    tpe(xy, quad, inner_edges, theta0_nodes, thetahalfpi_nodes, model, E_b, nu_b, f_b,
                        use_soft_dirichlet=True, R_out=model_args.R_out, mode='plane_strain',
                        dtype=torch.float32, fun_dim=model_args.fun_dim, material=material)

        for _ in range(10):
            _one_forward()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t0 = time.time()
        n_repeats = 50
        for _ in range(n_repeats):
            _one_forward()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        no_ms = (time.time() - t0) / n_repeats * 1000.0
        print(f'  NO @ N={N}: {no_ms:.1f} ms/sample (freshly measured, eager fp32)')

    # ---- Training cost, if known ----
    training_s = KNOWN_TRAINING_S.get((geometry, material))
    if training_s is None:
        metrics_path = os.path.join(os.path.dirname(ckpt), 'metrics_history.json')
        if os.path.exists(metrics_path):
            metrics = json.load(open(metrics_path))
            if metrics:
                training_s = metrics[-1].get('wall_clock_s') or metrics[-1].get('train_wall_clock_s')
    if training_s is None:
        print('  WARNING: training cost unknown for this case (no metrics_history.json found) -- '
              'break-even cannot be computed, only the speedup.')

    torchfem_ms = t_theirs * 1000.0
    speedup = torchfem_ms / no_ms
    row = {'geometry': geometry, 'material': material, 'N': N,
           'torchfem_ms_per_sample': torchfem_ms, 'no_ms_per_sample': no_ms,
           'speedup_vs_torchfem': speedup, 'training_seconds': training_s}
    if training_s is not None:
        saving_ms = torchfem_ms - no_ms
        if saving_ms > 0:
            row['break_even_samples'] = training_s / (saving_ms / 1000.0)
        else:
            row['break_even_samples'] = None  # never breaks even
    print(f'  Speedup vs. torch-fem @ N={N}: {speedup:.1f}x'
          + (f", break-even after {row['break_even_samples']:.0f} samples"
             if row.get('break_even_samples') else ""))
    results.append(row)

OUT_JSON = f'{R}/resolution_matched_break_even_all_cases.json'
with open(OUT_JSON, 'w') as f:
    json.dump({'N': N, 'rows': results}, f, indent=2)

try:
    from omar_pfem.run_manifest import write_manifest
    write_manifest(R, kind='resolution_matched_break_even_all_cases',
                    args={'N': N, 'cases': [f'{g}_{m}' for g, m, *_ in CASES]},
                    started_at=_started, results={'rows': results}, outputs=[OUT_JSON],
                    notes="Timon round-11 points 1+2, full scope: resolution-matched "
                          "break-even for all 6 cases, using the newly-generalized "
                          "torch-fem wrapper (B2 uses per-element-averaged material "
                          "params, an approximation forced by torch-fem's own API).")
except Exception as e:
    print(f'[manifest] not recorded: {e}')

print('\n' + '=' * 78)
print('SUMMARY -- resolution-matched comparison, all 6 cases, N=1401')
print('=' * 78)
for r in results:
    if r.get('failed'):
        print(f"  {r['geometry']} x {r['material']:<15} FAILED: {str(r['failed'])[:120]}")
        continue
    be = f"{r['break_even_samples']:.0f} samples" if r.get('break_even_samples') else \
         ("never" if r.get('training_seconds') else "training cost unknown")
    print(f"  {r['geometry']} x {r['material']:<15} torch-fem={r['torchfem_ms_per_sample']:.1f}ms  "
          f"NO={r['no_ms_per_sample']:.1f}ms  speedup={r['speedup_vs_torchfem']:.1f}x  break-even={be}")

print('\nSaved:', OUT_JSON)
