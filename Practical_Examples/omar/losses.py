"""
Loss functions and FNO boundary-condition wrappers for the B1 and B2
hyperelasticity benchmarks.

Everything here follows the exact style of utils/fno_utils.py's
DemHyperelasticityLoss (energy-based / Deep-Energy-Method loss, central
finite differences via grad_disp, trapezoidal Wint integration weights) and
of VINO_Hyperelasticity.py's FNO2d_HyperElasticity (BC enforcement + grid
mapping). The FNO architecture (utils/fno_2d.py), training loop and
optimizer (utils/fno_utils.py: train_fno, jax.example_libraries.optimizers)
are reused unmodified.

Only the following are new:
  - ArrudaBoyce2D material model (added in the same style as
    MooneyRivlin2D / NeoHookean2D).
  - B1 geometry: unit square, top-edge traction, fixed bottom edge.
  - B2 geometry: quarter ring mapped to [0, 1] x [0, 1], internal pressure,
    symmetry BCs on the two straight edges.
"""
import jax.numpy as jnp

from utils.fno_2d import FNO2D
from utils.fno_utils import DemHyperelasticityLoss


# ======================================================================
#  Material model: Arruda-Boyce (added in the same style as
#  MooneyRivlin2D / NeoHookean2D in DemHyperelasticityLoss)
# ======================================================================
class OmarHyperelasticityLoss(DemHyperelasticityLoss):
    def __init__(self, model, model_data, normalizers, d=2, p=2, size_average=True, reduction=True):
        super().__init__(model, model_data, normalizers, d=d, p=p, size_average=size_average, reduction=reduction)
        if self.type == 'arrudaboyce':
            self.mu_ab = model_data["beam"]["param_mu_ab"]
            self.N_ab = model_data["beam"]["param_N_ab"]
            self.kappa_ab = model_data["beam"]["param_kappa_ab"]

    def ArrudaBoyce2D(self, y_out):
        dudx, dudy, dvdx, dvdy = self.grad_disp(y_out)
        fxx = dudx + 1
        fxy = dudy + 0
        fyx = dvdx + 0
        fyy = dvdy + 1
        det_f = fxx * fyy - fxy * fyx
        j = det_f
        i1 = fxx * fxx + fxy * fxy + fyx * fyx + fyy * fyy

        # 8-chain Arruda-Boyce series (5 terms), applied to the *isochoric* invariant
        # i1_bar = I1 / J. Using the raw I1 here (like NeoHookean2D's tr_c) is unstable
        # under compression: I1 alone can drop below 2 (its value at F=I) with no lower
        # bound, and the odd-order terms in the I1^k polynomial then make the energy
        # decrease without bound as the material compresses -- confirmed numerically
        # (e.g. -10% uniaxial strain already gives negative, still-decreasing energy).
        # I1_bar = I1 / J is >= 2 for any J > 0 (2D AM-GM: fxx^2+fxy^2+fyx^2+fyy^2 >=
        # 2*|fxx*fyy - fxy*fyx|), so (I1_bar^k - 2^k) >= 0 always and the deviatoric
        # part only responds to shape change; the kappa_ab*(J-1)^2 term still handles
        # volume change separately, same split NeoHookean2D/MooneyRivlin2D achieve via
        # their -mu*ln(J) / -d*ln(J) terms.
        i1_bar = i1 / j
        alpha = [1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750]
        strain_energy = jnp.zeros_like(i1_bar)
        i1_bar_pow = i1_bar
        for k, a in enumerate(alpha, start=1):
            strain_energy = strain_energy + self.mu_ab * a / (self.N_ab ** (k - 1)) * (i1_bar_pow - 2.0 ** k)
            i1_bar_pow = i1_bar_pow * i1_bar
        strain_energy = strain_energy + self.kappa_ab / 2 * (j - 1) ** 2
        return jnp.where(jnp.any(jnp.isnan(strain_energy)), 1e9, strain_energy)

    def getStoredEnergy(self, y_out):
        if self.type == 'arrudaboyce':
            if self.dim == 2:
                return self.ArrudaBoyce2D(y_out)
            raise ValueError("3D setup is not implemented yet.")
        return super().getStoredEnergy(y_out)


# ======================================================================
#  B1 -- unit square, top-edge traction, fixed bottom edge
# ======================================================================
class FNO2d_B1(FNO2D):
    """Same style as FNO2d_HyperElasticity: BC enforcement + grid scaling."""
    model_data: dict = None

    def __call__(self, x_: jnp.ndarray) -> jnp.ndarray:
        x_ = super().__call__(x_)
        x_ = x_.at[:, 0, :, :].set(0)  # fixed bottom edge (y = 0)
        return x_

    def get_grid(self, x_):
        grid = super().get_grid(x_)
        gridy = grid[..., 0:1] * self.model_data['width']
        gridx = grid[..., 1:2] * self.model_data['length']
        return jnp.concatenate((gridy, gridx), -1)


class B1Loss(OmarHyperelasticityLoss):
    """
    Potential energy for B1:
        Pi = int_Omega psi dOmega - int_(y=W) t . u dGamma
    Fixed bottom edge is enforced in FNO2d_B1, not in the loss.
    """

    def loss_fn(self, params, x, y):
        if self.model_data["normalized"]:
            y_out = self.model.apply(params, x)
            y_out = self.normalizer_y.decode(y_out)
            x_real = self.normalizer_x.decode(x)
        else:
            y_out = self.model.apply(params, x)
            x_real = x

        # calculate the interior loss
        stored_energy = self.getStoredEnergy(y_out)
        loss_int = jnp.sum(stored_energy * self.Wint) * (self.L / self.num_x * self.W / self.num_y)

        # calculate the boundary loss (traction on the top edge, y = W)
        disp_x, disp_y = y_out[..., 0:1], y_out[..., 1:2]
        disp_y_top = disp_y[:, -1, :, :]
        ty = x_real[:, 0, :, 0:1]
        loss_bnd = jnp.sum(disp_y_top * ty) * (self.L / self.num_x)

        return jnp.where(jnp.isnan(jnp.min(stored_energy)), 1e6, loss_int - loss_bnd)


# ======================================================================
#  B2 -- quarter ring (R_in=1, R_out=2), mapped to [0, 1] x [0, 1]
#  rows (num_y axis)    -> theta in [0, pi/2]
#  columns (num_x axis) -> r     in [R_in, R_out]
# ======================================================================
class FNO2d_B2(FNO2D):
    """
    BC (symmetry conditions on the two straight edges of the quarter ring):
        u_y = 0 on theta = 0   (row index 0)
        u_x = 0 on theta = pi/2 (row index -1)
    """
    model_data: dict = None

    def __call__(self, x_: jnp.ndarray) -> jnp.ndarray:
        x_ = super().__call__(x_)
        x_ = x_.at[:, 0, :, 1:2].set(0)   # u_y = 0 on theta = 0
        x_ = x_.at[:, -1, :, 0:1].set(0)  # u_x = 0 on theta = pi/2
        return x_

    def get_grid(self, x_):
        grid = super().get_grid(x_)
        r_in = self.model_data["R_inner"]
        r_out = self.model_data["R_outer"]
        theta = grid[..., 0:1] * (jnp.pi / 2.0)
        r = grid[..., 1:2] * (r_out - r_in) + r_in
        gridx = r * jnp.cos(theta)
        gridy = r * jnp.sin(theta)
        return jnp.concatenate((gridx, gridy), -1)


class B2Loss(OmarHyperelasticityLoss):
    """
    Potential energy for B2, with the polar Jacobian |d(x,y)/d(r,theta)| = r:
        Pi = int_Omega psi * r dr dtheta - int_(r=R_in) p * u_r * R_in dtheta
    Symmetry BCs on the two straight edges are enforced in FNO2d_B2.
    """

    def __init__(self, model, model_data, normalizers, d=2, p=2, size_average=True, reduction=True):
        super().__init__(model, model_data, normalizers, d=d, p=p, size_average=size_average, reduction=reduction)
        self.R_in = model_data["beam"]["R_inner"]
        self.R_out = model_data["beam"]["R_outer"]

    def loss_fn(self, params, x, y):
        if self.model_data["normalized"]:
            y_out = self.model.apply(params, x)
            y_out = self.normalizer_y.decode(y_out)
            x_real = self.normalizer_x.decode(x)
        else:
            y_out = self.model.apply(params, x)
            x_real = x

        # calculate the interior loss (with the polar Jacobian r)
        stored_energy = self.getStoredEnergy(y_out)
        dr = (self.R_out - self.R_in) / self.num_x
        dtheta = (jnp.pi / 2.0) / self.num_y
        r_col = jnp.linspace(0, 1, self.num_x) * (self.R_out - self.R_in) + self.R_in
        r_grid = jnp.broadcast_to(r_col[None, None, :, None], stored_energy.shape)
        loss_int = jnp.sum(stored_energy * r_grid * self.Wint) * dr * dtheta

        # calculate the boundary loss (internal pressure on r = R_in)
        disp_x, disp_y = y_out[..., 0:1], y_out[..., 1:2]
        u_x_inner = disp_x[:, :, 0, :]
        u_y_inner = disp_y[:, :, 0, :]
        theta_row = jnp.linspace(0, 1, self.num_y) * (jnp.pi / 2.0)
        theta_col = theta_row[None, :, None]
        u_r_inner = u_x_inner * jnp.cos(theta_col) + u_y_inner * jnp.sin(theta_col)
        pressure = x_real[:, :, 0, :]  # GRF pressure profile, varies with theta (rows)
        loss_bnd = jnp.sum(u_r_inner * pressure) * self.R_in * dtheta

        return jnp.where(jnp.isnan(jnp.min(stored_energy)), 1e6, loss_int - loss_bnd)
