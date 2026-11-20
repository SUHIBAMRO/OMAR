"""Verification-aware multi-fidelity active learning (the third agent of Prof.
Rabczuk's plan: "active learning, which can decide which new expensive
simulations are most informative; multi-fidelity ... could then choose
fidelity").

Task: learn a surrogate of the lamina compliance (lamina.py, CFM open-hole
lamina) over two design parameters, fibre angle theta in [0, 90] deg and hole
radius r in [1.5, 5] mm, at the least compute. Every simulation has a cost (wall
time) and a mesh level m in {1, 2, 4, 8}; the cheaper levels carry a
discretization error.

The link to verification: the verification agent measured that error on the
lamina family (lamina_ladder.json: compliance 2.37% at m = 1, 0.20% at m = 2,
0.012% at m = 4, 0.0008% at m = 8). The verification-aware learner uses these
verified errors as the observation uncertainty of each level; a naive
multi-fidelity learner treats every level as exact.

Strategies compared (same Gaussian process, same budget, 5 seeds):
  random        random designs at m = 8
  al_fine       active learning (integrated-variance reduction) at m = 8 only
  mf_naive      active learning over (design, level), all levels taken as exact
  mf_verified   active learning over (design, level) per unit cost, level noise =
                the verified discretization error
Metric: RMS relative error of the surrogate over the 11 x 11 design grid against
the m = 8 solutions, as a function of cumulative cost.

    python -m omar_pfem.verification_agent.active_learning table     (solves, ~10 min)
    python -m omar_pfem.verification_agent.active_learning run
"""
import json
import os
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "al_lamina_table.json")
OUT = os.path.join(HERE, "al_lamina_result.json")
PREREG = os.path.join(HERE, "al_lamina_preregistration.json")
LEVELS = (1, 2, 4, 8)
THETAS = np.linspace(0.0, 90.0, 11)
RADII = np.linspace(1.5, 5.0, 11)
# verified relative discretization error of the compliance per level (nominal
# design, lamina_ladder.json, against m = 32)
VERIFIED_REL_ERR = {1: 2.3686e-2, 2: 1.9950e-3, 4: 1.2089e-4, 8: 7.803e-6}


def build_table():
    from omar_pfem.verification_agent.lamina import solve_level
    rows = json.load(open(TABLE)) if os.path.exists(TABLE) else []
    done = {(r["theta"], r["rh"], r["m"]) for r in rows}
    solve_level(1)                                   # warm-up (imports, JIT): not timed
    for th in THETAS:
        for rh in RADII:
            for m in LEVELS:
                key = (float(th), float(rh), m)
                if key in done:
                    continue
                r = solve_level(m, theta_deg=float(th), rh=float(rh))
                rows.append({"theta": float(th), "rh": float(rh), "m": m,
                             "compliance": r["compliance"],
                             "hole_gk_gamma12_avg": r["hole_gk_gamma12_avg"],
                             "cost_s": r["wall_clock_s"]})
        json.dump(rows, open(TABLE, "w"), indent=1)
        print(f"theta {th:.0f}: {len(rows)} rows", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Gaussian process (Matern 5/2, ARD), known per-point noise
# ---------------------------------------------------------------------------
def matern52(A, B, ls, var):
    d = np.sqrt((((A[:, None, :] - B[None, :, :]) / ls) ** 2).sum(-1))
    s5 = np.sqrt(5.0) * d
    return var * (1 + s5 + s5 ** 2 / 3.0) * np.exp(-s5)


class GP:
    def __init__(self, X, y, noise_var, hyper=None):
        self.X, self.noise = X, noise_var
        self.mu, self.sd = y.mean(), y.std() if y.std() > 0 else 1.0
        self.y = (y - self.mu) / self.sd
        self.nv = noise_var / self.sd ** 2
        self._fit(hyper)

    def _nll(self, p):
        ls, var = np.exp(p[:2]), np.exp(p[2])
        K = matern52(self.X, self.X, ls, var) + np.diag(self.nv + 1e-8)
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        a = cho_solve(c, self.y)
        return 0.5 * self.y @ a + np.log(np.diag(c[0])).sum()

    def _fit(self, hyper=None):
        if hyper is not None:                      # reuse hyperparameters (refit every k steps)
            self.ls, self.var = hyper
            K = matern52(self.X, self.X, self.ls, self.var) + np.diag(self.nv + 1e-8)
            self.c = cho_factor(K, lower=True)
            self.alpha = cho_solve(self.c, self.y)
            return
        best = None
        for p0 in ([np.log(0.3), np.log(0.3), 0.0], [np.log(1.0), np.log(1.0), 0.0]):
            r = minimize(self._nll, p0, method="L-BFGS-B",
                         bounds=[(np.log(0.03), np.log(5)), (np.log(0.03), np.log(5)),
                                 (np.log(1e-3), np.log(1e3))])
            if best is None or r.fun < best.fun:
                best = r
        self.ls, self.var = np.exp(best.x[:2]), np.exp(best.x[2])
        K = matern52(self.X, self.X, self.ls, self.var) + np.diag(self.nv + 1e-8)
        self.c = cho_factor(K, lower=True)
        self.alpha = cho_solve(self.c, self.y)

    def predict(self, Xs):
        Ks = matern52(Xs, self.X, self.ls, self.var)
        return self.mu + self.sd * (Ks @ self.alpha)

    def post_cov(self, Xs):
        Ks = matern52(Xs, self.X, self.ls, self.var)
        return matern52(Xs, Xs, self.ls, self.var) - Ks @ cho_solve(self.c, Ks.T)


# ---------------------------------------------------------------------------
def load_table():
    rows = json.load(open(TABLE))
    grid = [(float(t), float(r)) for t in THETAS for r in RADII]
    idx = {g: i for i, g in enumerate(grid)}
    val = {m: np.zeros(len(grid)) for m in LEVELS}
    cost = {m: [] for m in LEVELS}
    for r in rows:
        i = idx[(r["theta"], r["rh"])]
        val[r["m"]][i] = r["compliance"]
        cost[r["m"]].append(r["cost_s"])
    X = np.array([[(t - THETAS[0]) / (THETAS[-1] - THETAS[0]),
                   (r - RADII[0]) / (RADII[-1] - RADII[0])] for t, r in grid])
    return X, val, {m: float(np.median(c)) for m, c in cost.items()}


def run_strategy(name, X, val, cost, seed, budget, n_init=4, max_obs=400, refit_every=5):
    rng = np.random.default_rng(seed)
    truth = np.log(val[8])
    n = len(X)
    if name in ("random", "al_fine"):
        levels = (8,)
    else:
        levels = (2, 4, 8) if name == "mf_verified" else (1, 2, 4, 8)
    init_level = 8 if levels == (8,) else 2
    obs = [(int(i), init_level) for i in rng.choice(n, n_init, replace=False)]
    spent = sum(cost[m] for _, m in obs)
    hist = []

    def noise(m):
        if name == "mf_verified":
            return VERIFIED_REL_ERR[m] ** 2            # log-space variance
        return 1e-10                                   # treated as exact

    hyper = None
    while True:
        Xo = X[[i for i, _ in obs]]
        yo = np.array([np.log(val[m][i]) for i, m in obs])
        refit = hyper is None or len(obs) % refit_every == 0
        gp = GP(Xo, yo, np.array([noise(m) for _, m in obs]), None if refit else hyper)
        hyper = (gp.ls, gp.var)
        pred = gp.predict(X)
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        hist.append({"cost_s": spent, "rmse_rel": rmse, "n_obs": len(obs),
                     "levels": {str(m): sum(1 for _, k in obs if k == m) for m in LEVELS}})
        if spent >= budget or len(obs) >= max_obs:
            break
        taken = set(obs)
        if name == "random":
            free = [j for j in range(n) if (j, 8) not in taken]
            i, m = int(rng.choice(free)), 8
        else:
            C = gp.post_cov(X)
            v = np.clip(np.diag(C), 1e-30, None)
            best, choice = -1.0, None
            for m in levels:
                nm = noise(m) / gp.sd ** 2
                gain = (C ** 2).sum(axis=0) / (v + nm)     # integrated variance reduction
                score = gain / cost[m] if name.startswith("mf") else gain
                score = np.where([(jj, m) in taken for jj in range(n)], -np.inf, score)
                j = int(np.argmax(score))
                if score[j] > best:
                    best, choice = score[j], (j, m)
            i, m = choice
        obs.append((i, m))
        spent += cost[m]
    return hist


def cost_to_reach(hist, target):
    for h in hist:
        if h["rmse_rel"] <= target:
            return h["cost_s"]
    return None


def run(seeds=5, budget=40.0):
    X, val, cost = load_table()
    truth = val[8]
    # how well the nominal verified errors describe the whole design space
    act = {m: np.abs(val[m] / truth - 1) for m in (1, 2, 4)}
    spread = {m: {"median": float(np.median(a)), "max": float(a.max()),
                  "verified_nominal": VERIFIED_REL_ERR[m]} for m, a in act.items()}
    res = {"cost_per_level_s": cost, "error_vs_m8_over_designs": spread, "strategies": {}}
    for name in ("random", "al_fine", "mf_naive", "mf_verified"):
        runs = [run_strategy(name, X, val, cost, s, budget) for s in range(seeds)]
        final = [r[-1]["rmse_rel"] for r in runs]
        reach = {str(t): [cost_to_reach(r, t) for r in runs] for t in (1e-2, 1e-3)}
        res["strategies"][name] = {"final_rmse_rel": final, "cost_to_reach": reach,
                                   "history": runs}
        print(f"{name:<12} final RMS rel error {np.median(final):.2e} "
              f"[{min(final):.1e}, {max(final):.1e}]; cost to 1%: {reach['0.01']}; "
              f"to 0.1%: {reach['0.001']}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("level errors over the design grid (median / max vs nominal):",
          {m: (f"{v['median']:.2e}", f"{v['max']:.2e}", f"{v['verified_nominal']:.2e}")
           for m, v in spread.items()})
    return res


if __name__ == "__main__":
    if sys.argv[1:] == ["table"]:
        build_table()
    elif sys.argv[1:] == ["run"]:
        run()
