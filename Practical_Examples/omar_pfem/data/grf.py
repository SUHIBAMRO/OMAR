"""
Gaussian-Random-Field samplers shared by data_generate_B1.py and
data_generate_B2.py: a 2D exponential-covariance spectral-method sampler
(used for the E, nu material fields) and a 1D Cholesky-based sampler (used
for the boundary load profile -- traction along B1's top edge, pressure
along B2's inner arc). Unchanged from PFEM-main's
data/hyper/data_generate_beam.py.
"""
import numpy as np


def generate_gaussian_random_field_2d(Lx, Ly, Nx, Ny, mean, std, correlation_length, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=x[1]-x[0])
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=y[1]-y[0])
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)

    L = correlation_length
    power_spectrum = (std**2 * L**2) / (np.pi * (1 + (K * L)**2)**(1.5))
    power_spectrum[0, 0] = 0

    phase = np.random.uniform(0, 2*np.pi, (Nx, Ny))
    amplitudes = np.sqrt(power_spectrum) * np.exp(1j * phase)

    amplitudes[0, 0] = 0
    if Nx > 1:
        amplitudes[Nx//2, 0] = np.real(amplitudes[Nx//2, 0])
    if Ny > 1:
        amplitudes[0, Ny//2] = np.real(amplitudes[0, Ny//2])
    if Nx > 1 and Ny > 1:
        amplitudes[Nx//2, Ny//2] = np.real(amplitudes[Nx//2, Ny//2])

    for i in range(1, Nx//2):
        for j in range(1, Ny//2):
            amplitudes[Nx-i, Ny-j] = np.conj(amplitudes[i, j])

    field = np.fft.ifft2(amplitudes).real
    field_std = np.std(field)
    field = (field / field_std) * std
    field = mean + field

    return field, x, y


def generate_gaussian_random_field_3d(Lx, Ly, Lz, Nx, Ny, Nz, mean, std, correlation_length, seed=None):
    """3D extension of generate_gaussian_random_field_2d, for B3's material
    fields E(theta,r,z)/nu(theta,r,z) over the 3D bushing. Uses a simpler,
    exactly-correct construction than the 2D version's manual phase/
    symmetry bookkeeping: real white noise's own FFT is already exactly
    Hermitian-symmetric, so filtering it by a real, isotropic amplitude
    filter (sqrt of the exponential-covariance power spectrum, generalized
    to 3D: (1+(kL)^2)^-2, vs. the 2D case's (1+(kL)^2)^-1.5) preserves that
    symmetry and its inverse FFT is real up to floating-point roundoff --
    no per-index conjugate-symmetry loop needed. The overall normalization
    constant of the filter does not matter, since the field is rescaled to
    the exact target std/mean afterward either way, same as the 2D version.
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    z = np.linspace(0, Lz, Nz)

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=x[1] - x[0])
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=y[1] - y[0])
    kz = 2 * np.pi * np.fft.fftfreq(Nz, d=z[1] - z[0])
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    L = correlation_length
    amp_filter = 1.0 / (1 + (K * L)**2)**2.0

    amp_filter[0, 0, 0] = 0.0  # remove the DC/mean term, same as the 2D version's amplitudes[0, 0] = 0

    noise = np.random.randn(Nx, Ny, Nz)
    field_hat = np.fft.fftn(noise) * amp_filter
    field = np.fft.ifftn(field_hat).real

    field_std = np.std(field)
    field = (field / field_std) * std
    field = mean + field

    return field, x, y, z


def generate_gaussian_random_field_1d(Lx, Nx, mean, std, correlation_length, seed=None):
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(0, Lx, Nx)
    x_mesh = np.meshgrid(x, x)
    dist = np.abs(x_mesh[0] - x_mesh[1])
    cov = std**2 * np.exp(-dist / correlation_length)

    L = np.linalg.cholesky(cov + 1e-10 * np.eye(Nx))
    field = mean + L @ np.random.randn(Nx)

    return field, x
