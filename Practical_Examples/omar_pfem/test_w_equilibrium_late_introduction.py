"""Correctness checks for train_B3_equilibrium_augmented.w_equilibrium_
late_introduction_at -- the new late-introduction schedule for items 2/3
(2026-10-26, per Omar's explicit instruction to keep trying after the
three prior independent fixes -- recalibration, weight/noise sweep,
decay-out -- were all falsified). Pure scheduling-function logic, no
physics, no GPU needed:

    python -m omar_pfem.test_w_equilibrium_late_introduction
"""
from omar_pfem.train_B3_equilibrium_augmented import w_equilibrium_late_introduction_at


def test_disabled_by_default_returns_zero_always():
    """introduce_start=introduce_end=0 (the default) must return 0.0
    UNCONDITIONALLY -- the opposite convention from w_equilibrium_at's own
    disabled state (which returns w0 unconditionally), since this
    function's whole point is "do not introduce the term at all" until
    told otherwise."""
    for it in (1, 1000, 50000):
        w = w_equilibrium_late_introduction_at(it, w0=0.12, introduce_start=0, introduce_end=0)
        assert w == 0.0, f'disabled schedule must return 0.0 at it={it}, got {w}'
    print('PASS  disabled (0,0) schedule returns 0.0 at every iteration')


def test_zero_before_introduce_start():
    w = w_equilibrium_late_introduction_at(5000, w0=0.12, introduce_start=10000, introduce_end=12000)
    assert w == 0.0, f'expected exactly 0.0 before introduce_start, got {w}'
    w_at_start = w_equilibrium_late_introduction_at(10000, w0=0.12, introduce_start=10000, introduce_end=12000)
    assert w_at_start == 0.0, f'expected exactly 0.0 AT introduce_start, got {w_at_start}'
    print('PASS  w=0.0 before and at introduce_start')


def test_full_weight_at_and_after_introduce_end():
    w = w_equilibrium_late_introduction_at(12000, w0=0.12, introduce_start=10000, introduce_end=12000)
    assert abs(w - 0.12) < 1e-12, f'expected exactly w0 at introduce_end, got {w}'
    w_after = w_equilibrium_late_introduction_at(50000, w0=0.12, introduce_start=10000, introduce_end=12000)
    assert abs(w_after - 0.12) < 1e-12, f'expected exactly w0 after introduce_end, got {w_after}'
    print('PASS  w=w0 at and after introduce_end')


def test_linear_ramp_at_midpoint():
    w = w_equilibrium_late_introduction_at(11000, w0=0.12, introduce_start=10000, introduce_end=12000)
    assert abs(w - 0.06) < 1e-12, f'expected exactly w0/2 at the midpoint, got {w}'
    print(f'PASS  linear ramp: w={w} at the midpoint (expected 0.06)')


if __name__ == '__main__':
    test_disabled_by_default_returns_zero_always()
    test_zero_before_introduce_start()
    test_full_weight_at_and_after_introduce_end()
    test_linear_ramp_at_midpoint()
    print('\nall w_equilibrium_late_introduction_at checks passed')
