# PFEM / Transolver — Hyperelasticity Neural Operator

A physics-informed neural operator (Transolver) trained to predict the
displacement field of 2D hyperelastic solids under randomly sampled
material and loading fields, without labeled data — trained by directly
minimizing the discrete total potential energy of the body. Six
benchmark cases: two geometries (B1, B2) × three material models
(Neo-Hookean, Mooney-Rivlin, Arruda-Boyce).

Author: Omar Amro. Advisor: Prof. Timon Rabczuk.

**Start with `PROJECT_STATUS.md`** at the repo root for the current
status of the project.

## Repository layout

- `Practical_Examples/omar_pfem/` — the project: data generation,
  training, the GPU-native FEM solver, and every measurement behind the
  report.
- `Practical_Examples/report_builders/` — scripts that build the report
  and summary `.docx` files.
- `Practical_Examples/zeroshot_notebooks/` — Colab notebooks for
  training, evaluation, and result verification.
- `advisor_feedback/` — the advisor's feedback emails, stored verbatim.
- `Comparative_Examples/`, `Integration/`, and the remaining top-level
  files under `Practical_Examples/` — third-party code from
  [VINO](https://github.com/eshaghi-ms/VINO) (Eshaghi et al., 2025),
  used for reference during development. See
  `Comparative_Examples/VINO_README.md` for its own documentation and
  citation.
