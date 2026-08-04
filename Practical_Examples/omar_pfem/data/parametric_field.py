"""
Resolution-independent, randomly-parametrized analytic (E, nu, load) fields
for the true zero-shot resolution-invariance study (per the advisor's
request: "evaluate the SAME trained model on resolutions not used during
training ... All resolutions should be compared against a common fine-mesh
FEM reference").

Why this exists instead of reusing the GRF sampler in data/grf.py: that
sampler is a spectral method whose random phase array is SIZED BY (Nx, Ny)
-- the same seed at two different mesh resolutions does NOT resample the
same continuous field at higher resolution, it draws a different, merely
statistically-similar realization (see mesh_convergence.py's own docstring
for the identical issue in the h-refinement-convergence context, which it
solves the same way: a genuine closed-form function of (x, y) / (theta, r),
evaluable identically at any resolution).

A true zero-shot resolution-invariance test additionally needs an ENSEMBLE
of distinct random instances (not one single fixed field, which is all
mesh_convergence.py needs since it only ever checks solver convergence).
This module generalizes mesh_convergence.py's fixed AnalyticFieldB1/B2 to a
small number of per-sample RANDOM low-order Fourier coefficients, seeded
per-sample -- every mesh resolution that queries the SAME seed sees the
EXACT SAME physical field, just at different (x, y) / (theta, r) points,
while different seeds still give a real ensemble of distinct problem
instances for training and evaluation.

K (Fourier order) and the coefficient scaling are chosen so each field's
mean and standard deviation match the (E_mean=1000, E_std=200) / (nu_mean
=0.3, nu_std=0.05) / (ty or p mean/std) ensemble used by the GRF-based
fields everywhere else in this study, and so the dominant wavelength is
comparable to the GRF's correlation_length=0.5*Lx -- this changes the
sampling MECHANISM (so it can be resolution-consistent), not the physical
regime the network was trained on elsewhere in this study.
"""
import numpy as np


class ParametricFieldB1:
    """Random, resolution-independent (E, nu, ty) field for B1, indexed by
    an integer seed. Calling convention matches RegularGridInterpolator:
    pts is (N, 2) Cartesian (x, y), returns (N,)."""

    def __init__(self, kind, seed, K=2,
                 E_mean=1000.0, E_std=200.0,
                 nu_mean=0.3, nu_std=0.05, nu_clip=(0.2, 0.4),
                 ty_mean=-5.0, ty_std=2.0):
        self.kind = kind
        self.K = K
        self.E_mean, self.E_std = E_mean, E_std
        self.nu_mean, self.nu_std, self.nu_clip = nu_mean, nu_std, nu_clip
        self.ty_mean, self.ty_std = ty_mean, ty_std

        rng = np.random.RandomState(seed)
        # Independent coefficient sets per field so E, nu, ty are not
        # perfectly correlated realizations of the same shape.
        self._a, self._phi, self._psi = self._draw(rng)
        self._a_nu, self._phi_nu, self._psi_nu = self._draw(rng)
        self._a_ty, self._phi_ty, self._psi_ty = self._draw(rng)

    def _draw(self, rng):
        return (rng.normal(0.0, 1.0, size=self.K),
                rng.uniform(0.0, 2 * np.pi, size=self.K),
                rng.uniform(0.0, 2 * np.pi, size=self.K))

    def _series(self, x, y, a, phi, psi):
        s = np.zeros_like(x)
        for k in range(self.K):
            s = s + a[k] * np.sin(np.pi * (k + 1) * x + phi[k]) * np.cos(np.pi * (k + 1) * y + psi[k])
        norm = np.sqrt(np.sum(a ** 2) / 2.0)  # RMS normalizer for unit-variance coefficients
        return s / norm if norm > 1e-12 else s

    def __call__(self, pts):
        x, y = pts[:, 0], pts[:, 1]
        if self.kind == "E":
            return self.E_mean + self.E_std * self._series(x, y, self._a, self._phi, self._psi)
        elif self.kind == "nu":
            val = self.nu_mean + self.nu_std * self._series(x, y, self._a_nu, self._phi_nu, self._psi_nu)
            return np.clip(val, self.nu_clip[0], self.nu_clip[1])
        elif self.kind == "ty":
            return self.ty_mean + self.ty_std * self._series(x, y, self._a_ty, self._phi_ty, self._psi_ty)
        raise ValueError(self.kind)


class ParametricFieldB2:
    """Random, resolution-independent (E, nu, p) field for B2, indexed by
    an integer seed. Calling convention matches RegularGridInterpolator:
    pts is (N, 2) polar (theta, r), returns (N,)."""

    def __init__(self, kind, seed, K=2,
                 E_mean=1000.0, E_std=200.0,
                 nu_mean=0.3, nu_std=0.05, nu_clip=(0.2, 0.4),
                 p_mean=5.0, p_std=2.0):
        self.kind = kind
        self.K = K
        self.E_mean, self.E_std = E_mean, E_std
        self.nu_mean, self.nu_std, self.nu_clip = nu_mean, nu_std, nu_clip
        self.p_mean, self.p_std = p_mean, p_std

        rng = np.random.RandomState(seed)
        self._a, self._phi = self._draw(rng)
        self._a_nu, self._phi_nu = self._draw(rng)
        self._a_p, self._phi_p = self._draw(rng)

    def _draw(self, rng):
        return (rng.normal(0.0, 1.0, size=self.K),
                rng.uniform(0.0, 2 * np.pi, size=self.K))

    def _series(self, theta, a, phi):
        s = np.zeros_like(theta)
        for k in range(self.K):
            s = s + a[k] * np.sin((k + 1) * theta + phi[k])
        norm = np.sqrt(np.sum(a ** 2) / 2.0)
        return s / norm if norm > 1e-12 else s

    def __call__(self, pts):
        theta = pts[:, 0]
        if self.kind == "E":
            return self.E_mean + self.E_std * self._series(theta, self._a, self._phi)
        elif self.kind == "nu":
            val = self.nu_mean + self.nu_std * self._series(theta, self._a_nu, self._phi_nu)
            return np.clip(val, self.nu_clip[0], self.nu_clip[1])
        elif self.kind == "p":
            return self.p_mean + self.p_std * self._series(theta, self._a_p, self._phi_p)
        raise ValueError(self.kind)
