"""Checks on the input-normalization knob (advisor round-6, point 1).

The one that matters is the first: with the knob off, the network's inputs
must be bit-identical to what they were before the knob existed, because
every result in the report was produced on that path.

    python -m omar_pfem.test_input_norm
"""
import json
import os
import tempfile

import numpy as np
import torch

from omar_pfem import train_B1 as T


def _fake_samples(n=5, nodes=9, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        f = np.zeros((nodes, 2))
        f[-3:, 1] = rng.normal(-5.0, 2.0, 3)   # only the top edge is loaded
        out.append({"E_node": rng.normal(1000.0, 200.0, nodes),
                    "nu_node": rng.normal(0.3, 0.05, nodes),
                    "node_forces": f})
    return out


def test_off_is_identity():
    T.set_input_norm(None)
    x = torch.randn(2, 7, 4, dtype=torch.float64)
    y = T._apply_input_norm(x)
    assert y is x, 'the knob is off but the tensor was replaced'
    print('PASS  off is a true no-op (same object, not merely equal)')


def test_on_standardizes():
    stats = {"mean": [1000.0, 0.3, 0.0, -1.0], "std": [200.0, 0.05, 1.0, 2.0]}
    T.set_input_norm(stats)
    x = torch.tensor([[[1000.0, 0.3, 0.0, -1.0],
                       [1200.0, 0.35, 2.0, 1.0]]], dtype=torch.float64)
    y = T._apply_input_norm(x)
    assert torch.allclose(y[0, 0], torch.zeros(4, dtype=torch.float64)), y[0, 0]
    assert torch.allclose(y[0, 1], torch.tensor([1.0, 1.0, 2.0, 1.0], dtype=torch.float64)), y[0, 1]
    T.set_input_norm(None)
    print('PASS  on maps the mean to 0 and one std to 1, per channel')


def test_stats_are_of_the_training_set():
    s = _fake_samples()
    st = T.compute_input_norm(s, fun_dim=4)
    E = np.concatenate([x["E_node"] for x in s])
    assert abs(st["mean"][0] - E.mean()) < 1e-12
    assert abs(st["std"][0] - E.std()) < 1e-12
    assert st["channels"] == ["E", "nu", "fx", "fy"]
    # fy must be computed over ALL nodes, zeros included, not just loaded ones
    fy = np.concatenate([x["node_forces"][:, 1] for x in s])
    assert abs(st["mean"][3] - fy.mean()) < 1e-12, \
        'fy statistics exclude the unloaded nodes, which the network is still fed'
    st3 = T.compute_input_norm(s, fun_dim=3)
    assert st3["channels"] == ["E", "nu", "fy"]
    print('PASS  statistics are the training set\'s own, over every node')


def test_channel_count_mismatch_is_caught():
    T.set_input_norm({"mean": [0.0] * 3, "std": [1.0] * 3})
    try:
        T._apply_input_norm(torch.zeros(1, 4, 4))
    except AssertionError as e:
        assert 'fun_dim' in str(e)
        print('PASS  a 3-channel norm on a 4-channel model is refused')
    else:
        raise AssertionError('a channel-count mismatch was NOT caught')
    finally:
        T.set_input_norm(None)


def test_constant_channel_passes_through():
    """B1's traction is purely vertical, so with fun_dim=4 the fx channel is
    identically zero everywhere. It must pass through untouched, not divide
    the network's input by zero."""
    s = _fake_samples()
    st = T.compute_input_norm(s, fun_dim=4)
    assert st["constant_channels_passed_through"] == ["fx"], \
        st["constant_channels_passed_through"]
    assert st["mean"][2] == 0.0 and st["std"][2] == 1.0
    T.set_input_norm(st)
    x = torch.zeros(1, 3, 4, dtype=torch.float64)
    y = T._apply_input_norm(x)
    assert torch.isfinite(y).all(), 'a constant channel produced non-finite input'
    assert (y[..., 2] == 0).all(), 'the dead fx channel was altered'
    T.set_input_norm(None)
    print('PASS  a constant channel (B1\'s fx) passes through, no divide-by-zero')


def test_checkpoint_sidecar_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        stats = T.compute_input_norm(_fake_samples(), fun_dim=4)
        with open(os.path.join(d, 'input_norm.json'), 'w') as f:
            json.dump(stats, f)
        open(os.path.join(d, 'model_best.pt'), 'wb').close()
        got = T.install_input_norm_for_checkpoint(os.path.join(d, 'model_best.pt'))
        assert got['mean'] == stats['mean']
        assert T.get_input_norm() is not None
        try:
            T.install_input_norm_for_checkpoint(os.path.join(d, 'model_best.pt'),
                                                required=False)
        except RuntimeError:
            print('PASS  sidecar round-trips, and required=False refuses it')
        else:
            raise AssertionError('required=False did not refuse a normalized ckpt')
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, 'model_best.pt'), 'wb').close()
        assert T.install_input_norm_for_checkpoint(os.path.join(d, 'model_best.pt')) is None
        assert T.get_input_norm() is None
        try:
            T.install_input_norm_for_checkpoint(os.path.join(d, 'model_best.pt'),
                                                required=True)
        except FileNotFoundError:
            print('PASS  a bare checkpoint clears the norm, and required=True refuses it')
        else:
            raise AssertionError('required=True did not refuse a bare checkpoint')


def test_forward_paths_agree():
    """The training path and the inference path must normalize identically --
    a mismatch between them would show up as an accuracy gap between training
    and deployment that no amount of tuning would explain."""
    import inspect
    for fn in (T.total_potential_energy_Q4_hyperelastic,
               T.predict_displacement_Q4_only):
        src = inspect.getsource(getattr(fn, '__wrapped__', fn))
        assert '_apply_input_norm(fun_material)' in src, \
            f'{fn.__name__} does not apply the input normalization'
    print('PASS  both the energy path and the inference path normalize')


def test_default_path_is_bit_identical():
    """The claim that matters: with the knob off, the real energy function
    must produce bit-identical output to the code as it stood before the knob
    existed. Verified by running it against a version whose normalization hook
    is patched out entirely, on the same inputs, and comparing exactly."""
    torch.manual_seed(0)
    n = 3
    xs, ys = torch.meshgrid(torch.linspace(0, 1, n), torch.linspace(0, 1, n),
                            indexing='ij')
    xy = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1).double()
    quad = torch.tensor([[i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1]
                         for i in range(n - 1) for j in range(n - 1)], dtype=torch.long)
    top = torch.tensor([[n * (n - 1) + j, n * (n - 1) + j + 1] for j in range(n - 1)],
                       dtype=torch.long)
    bottom = torch.arange(0, n * n, n)

    class Stub(torch.nn.Module):
        """Depends on fun_material, so any change to the inputs shows up."""
        def forward(self, xy_domain, fun_material):
            return torch.stack([fun_material.sum(-1) * 1e-6,
                                fun_material.mean(-1) * 1e-6], dim=-1)

    E = torch.full((1, n * n), 1000.0, dtype=torch.float64)
    nu = torch.full((1, n * n), 0.3, dtype=torch.float64)
    f = torch.zeros(1, n * n, 2, dtype=torch.float64)
    f[0, -n:, 1] = -5.0
    kw = dict(use_soft_dirichlet=True, Ly=1.0, mode='plane_strain',
              dtype=torch.float64, fun_dim=4, material='neo_hookean')

    T.set_input_norm(None)
    got = T.total_potential_energy_Q4_hyperelastic(
        xy, quad, top, bottom, Stub(), E, nu, f, **kw)

    orig = T._apply_input_norm
    T._apply_input_norm = lambda x: x           # the pre-knob code path
    try:
        ref = T.total_potential_energy_Q4_hyperelastic(
            xy, quad, top, bottom, Stub(), E, nu, f, **kw)
    finally:
        T._apply_input_norm = orig

    for name, a, b in zip(('Pi', 'U', 'W', 'uv', 'Fg'), got, ref):
        assert torch.equal(a, b), f'{name} differs with the knob off'
    print('PASS  knob off is bit-identical to the code without the knob (Pi, U, W, uv, Fg)')

    # and with it ON the result must actually change, or the test above proves nothing
    T.set_input_norm({"mean": [1000.0, 0.3, 0.0, -1.0],
                      "std": [200.0, 0.05, 1.0, 2.0]})
    on = T.total_potential_energy_Q4_hyperelastic(
        xy, quad, top, bottom, Stub(), E, nu, f, **kw)
    T.set_input_norm(None)
    assert not torch.equal(on[0], got[0]), \
        'the knob ON produced the same Pi as OFF -- it is not wired in'
    print('PASS  knob on does change the forward pass, so the check above is meaningful')


if __name__ == '__main__':
    test_off_is_identity()
    test_on_standardizes()
    test_stats_are_of_the_training_set()
    test_channel_count_mismatch_is_caught()
    test_constant_channel_passes_through()
    test_checkpoint_sidecar_roundtrip()
    test_forward_paths_agree()
    test_default_path_is_bit_identical()
    print('\nall input-normalization checks passed')
