# PFEM / Transolver — Hyperelasticity Neural Operator

A physics-informed neural operator (Transolver) trained to predict the
displacement field of 2D hyperelastic solids under randomly sampled
material and loading fields, without labeled data — trained by directly
minimizing the discrete total potential energy of the body. Six
benchmark cases: two geometries (B1, B2) × three material models
(Neo-Hookean, Mooney-Rivlin, Arruda-Boyce).

Author: Omar Amro. Advisor: Prof. Timon Rabczuk.

**Start with `PROJECT_STATUS.md`** at the repo root — it is the living,
authoritative record of what is done, what is in progress, and what is
still pending. This README only orients you to the repo layout.

## Repository layout

- `Practical_Examples/omar_pfem/` — the actual project: data generation,
  training, the GPU-native FEM solver, and every measurement behind the
  report (`point{2,5,6,7a,7b,8,9}_results/`).
- `Practical_Examples/report_builders/` — the scripts that build and
  version the report and summary `.docx` files from that data
  (`make_vN.py`, `make_summary_vN.py`), plus `check_report_tables.py`.
- `Practical_Examples/zeroshot_notebooks/` — the Colab notebooks used to
  run training/evaluation and to print results for cross-checking.
- `advisor_feedback/` — the advisor's feedback emails, stored verbatim.
- `Comparative_Examples/`, `Integration/`, and the top-level `.py` files
  and `utils/` under `Practical_Examples/` — third-party code, see below.

## Third-party code: VINO

This repository's root `README.md` used to be VINO's own, unedited,
because an early exploration of this project (commit `f3d78f0`,
2026-07-03) vendored the code from
[`eshaghi-ms/VINO`](https://github.com/eshaghi-ms/VINO) as-is — the
files now under `Comparative_Examples/`, `Integration/`, and
`Practical_Examples/utils/` plus the top-level `Practical_Examples/*.py`
scripts. A few days later (commit `fff41c6`, 2026-07-06) that prototype
briefly used VINO's own closed-form energy-integration method, in a
now-abandoned directory (`Practical_Examples/omar/`).

**None of that vendored code or method is part of the active project.**
The actual pipeline behind every result in the report
(`Practical_Examples/omar_pfem/`) is a separate implementation, started
independently on 2026-07-09 ("isolated from omar/"), using ordinary
Gauss-quadrature energy assembly. The one real connection that remains:
`materials_torch.py`'s Mooney-Rivlin and Arruda-Boyce strain-energy
densities were cross-checked against VINO's own implementation for
correctness (documented in the report, Section 2.4) — not adopted as
the training method.

VINO is cited in the report as reference [4]:

> Eshaghi, M.S., Anitescu, C., Thombre, M., Wang, Y., Zhuang, X., and
> Rabczuk, T. "Variational Physics-informed Neural Operator (VINO) for
> Learning Partial Differential Equations." Computer Methods in Applied
> Mechanics and Engineering, 437, 117785, 2025.
> [doi:10.1016/j.cma.2025.117785](https://doi.org/10.1016/j.cma.2025.117785)

The original VINO README (its own description, abstract, dataset links,
and author contacts) is preserved unmodified at
[`Comparative_Examples/VINO_README.md`](Comparative_Examples/VINO_README.md).
