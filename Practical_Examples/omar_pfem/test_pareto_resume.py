"""Exercise pareto_analysis.main()'s file protocol with the physics stubbed.

What is under test is only WHERE results land and WHAT gets skipped:
  1. a run killed part-way leaves NO final JSON, only a .progress file;
  2. restarting resumes and skips what the progress file holds;
  3. the final JSON appears only when every requested resolution is present,
     and the progress file is removed;
  4. a partial file written by the OLD code (rows in out_json) is still
     resumed, not thrown away;
  5. re-running a complete sweep over a SUBSET does not delete the other rows.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

import omar_pfem.pareto_analysis as pa

CALLED = []


def fake_build_sample(N, seed=0, material='', solve_fem=True):
    CALLED.append(N)
    n = 4
    rng = np.random.default_rng(seed + N)
    return {"xy": rng.random((n, 2)), "uv_exact": rng.random((n, 2)),
            "E_node": rng.random(n), "nu_node": rng.random(n),
            "node_forces": rng.random((n, 2))}, None


class Dummy(nn.Module):
    def forward(self, *a, **k):
        return torch.zeros(1, 4, 2)


def install_stubs(kill_at=None):
    CALLED.clear()
    pa.build_sample_b1 = fake_build_sample
    pa.build_model = lambda args, device: Dummy()
    pa.mesh_tensors_of = lambda *a, **k: {}
    pa.interpolate_to_reference = lambda xy, uv, fine_xy: np.asarray(uv)
    pa.write_manifest = lambda *a, **k: None

    def fake_predict(args, mesh_t, model, E, nu, f, dtype):
        if kill_at is not None and CALLED and CALLED[-1] == kill_at:
            raise KeyboardInterrupt(f'simulated Colab death at N={kill_at}')
        return torch.zeros(1, 4, 2)

    pa.predict = fake_predict


def run(out_dir, ckpt, resolutions, kill_at=None):
    install_stubs(kill_at)
    sys.argv = ['pareto_analysis', '--geometry', 'B1', '--material',
                'neo_hookean', '--checkpoint', ckpt,
                '--resolutions', ','.join(str(N) for N in resolutions),
                '--fine_N', '5', '--n_samples', '2', '--batch_size', '1',
                '--out_dir', out_dir, '--cpu']
    try:
        pa.main()
        return 'finished'
    except KeyboardInterrupt as e:
        print(f'  [killed] {e}')
        return 'killed'


def rows_of(path):
    return [r['N'] for r in json.load(open(path))['rows']]


def main():
    tmp = tempfile.mkdtemp()
    ckpt = os.path.join(tmp, 'model_best.pt')
    torch.save(Dummy().state_dict(), ckpt)
    out = os.path.join(tmp, 'case')
    os.makedirs(out)
    final = os.path.join(out, 'pareto_B1_neo_hookean.json')
    prog = final + '.progress'
    ok = True

    def check(label, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        ok = ok and cond

    print('\n1. a run killed at N=17')
    assert run(out, ckpt, [13, 17, 21], kill_at=17) == 'killed'
    check('no final JSON was written', not os.path.exists(final))
    check('the progress file holds N=[13]',
          os.path.exists(prog) and rows_of(prog) == [13])
    check('an os.path.exists(out_json) caller would NOT skip this case',
          not os.path.exists(final))

    print('\n2. restart -- it resumes')
    assert run(out, ckpt, [13, 17, 21]) == 'finished'
    check('N=13 was not recomputed', 13 not in CALLED)
    check('N=17 and N=21 were computed', set(CALLED) == {17, 21})
    check('the final JSON holds all three', rows_of(final) == [13, 17, 21])
    check('the progress file is gone', not os.path.exists(prog))

    print('\n3. re-running a complete sweep over a SUBSET keeps the rest')
    assert run(out, ckpt, [17]) == 'finished'
    check('nothing was recomputed', CALLED == [])
    check('all three rows survive', rows_of(final) == [13, 17, 21])

    print('\n4. a partial file from the OLD code (rows in out_json) resumes')
    shutil.rmtree(out)
    os.makedirs(out)
    prev = {"geometry": "B1", "material": "neo_hookean", "checkpoint": ckpt,
            "checkpoint_fingerprint": pa.hashlib.sha256(
                open(ckpt, 'rb').read()).hexdigest(),
            "fine_N": 5, "n_samples": 2, "batch_size": 1, "device": "cpu",
            "rows": [{"N": 13, "n_nodes": 4, "fem_rel_L2": 1.0,
                      "fem_rel_L2_std": 0.0, "fem_ms_per_sample": 1.0,
                      "operator_rel_L2": 1.0, "operator_rel_L2_std": 0.0,
                      "operator_ms_per_sample": 1.0}]}
    json.dump(prev, open(final, 'w'))
    assert run(out, ckpt, [13, 17]) == 'finished'
    check('the old partial row was reused, not recomputed', 13 not in CALLED)
    check('the sweep completed', rows_of(final) == [13, 17])

    print('\n5. a checkpoint change forces a fresh start')
    torch.save(Dummy().state_dict(), ckpt)          # same bytes -> same hash
    other = os.path.join(tmp, 'other.pt')
    with open(other, 'wb') as fh:
        fh.write(open(ckpt, 'rb').read() + b'\0')
    assert run(out, other, [13, 17]) == 'finished'
    check('both resolutions were recomputed', set(CALLED) == {13, 17})

    shutil.rmtree(tmp)
    print('\n' + ('ALL CHECKS PASSED' if ok else 'SOMETHING FAILED'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
