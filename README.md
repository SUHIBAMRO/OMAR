# PFEM / Transolver — Hyperelasticity Neural Operator

A physics-informed neural operator (Transolver) trained to predict the
displacement field of 2D hyperelastic solids under randomly sampled
material and loading fields, without any labeled displacement data. The
network is trained by directly minimizing the discrete total potential
energy of the body (the Deep Energy Method), evaluated by standard
Gauss-quadrature finite-element assembly. Six benchmark cases: two
geometries (B1, B2) × three material models (Neo-Hookean, Mooney-Rivlin,
Arruda-Boyce).

**Author:** Omar Amro
**Advisor:** Prof. Dr.-Ing. Timon Rabczuk

## Highlights

- **Physics-informed neural operator** (Transolver) trained without any
  labeled displacement data, by directly minimizing the discrete total
  potential energy (Deep Energy Method) via standard finite-element
  Gauss-quadrature assembly.
- **Custom GPU-native finite-element solver**, built from scratch in
  PyTorch: a matrix-free Newton–Krylov solve with a geometric multigrid
  preconditioner, plus a from-scratch sparse-assembly and direct-solve
  path, benchmarked against established finite-element libraries
  (torch-fem, TensorMesh) on an NVIDIA A100.
- **Six benchmark cases** spanning two geometries and three nonlinear
  hyperelastic material models (Neo-Hookean, Mooney-Rivlin, Arruda-Boyce).
- **Verification discipline throughout**: convergence studies against a
  ~10-million-DOF numerical reference, manufactured-solution checks, and
  cross-validation against independent finite-element implementations
  before any timing or accuracy claim is trusted.

See `PROJECT_STATUS.md` for the current, detailed state of the project.

## Status

`PROJECT_STATUS.md` at the repo root is the authoritative, living record
of what is done, in progress, or pending on this project — read it
before assuming anything about current state.

## Requirements

- Python 3.12
- PyTorch 2.11 (CUDA build; developed and benchmarked on an NVIDIA
  A100-SXM4-80GB)
- NumPy, SciPy
- `einops`, `timm`, `h5py`, `jax`, `tqdm`

Training and evaluation were run on Google Colab; there is no separate
`requirements.txt` because the notebooks in `zeroshot_notebooks/`
install the extra packages inline (`pip install -q einops timm h5py jax
tqdm`) on top of Colab's preinstalled PyTorch/NumPy/SciPy.

## Repository structure

```
Practical_Examples/
├── omar_pfem/            the project: data generation, training,
│                         the GPU-native FEM solver, and every
│                         measurement behind the report
│                         (point{2,5,6,7a,7b,8,9}_results/)
├── report_builders/      scripts that build/version the report and
│                         summary .docx files, and check_report_tables.py
├── zeroshot_notebooks/   Colab notebooks for training, evaluation,
│                         and result verification
├── Comparative_Examples/ third-party code (VINO, see below)
└── Integration/          third-party code (VINO, see below)
advisor_feedback/         advisor feedback emails, stored verbatim
PROJECT_STATUS.md         current project status (read this first)
```

## Usage

Each benchmark case is trained with `omar_pfem/train_B1.py` or
`train_B2.py`; see each script's `--help` for the full set of options.
Every result in the report was produced by running the corresponding
cell in `Practical_Examples/zeroshot_notebooks/` on Colab — start there
to reproduce a specific table rather than invoking the scripts directly.

## Results

See the report and summary `.docx` files (built from
`report_builders/`) and `PROJECT_STATUS.md` for what has been measured
and where.

## Third-party code

`Comparative_Examples/`, `Integration/`, and the remaining top-level
files under `Practical_Examples/` are vendored, unmodified code from
[VINO](https://github.com/eshaghi-ms/VINO) (Eshaghi et al., 2025), kept
for reference during development. See
[`Comparative_Examples/VINO_README.md`](Comparative_Examples/VINO_README.md)
for its own documentation and citation. It is cited in the report as
reference [4]; two material energy-density formulas in this project
were cross-checked against its implementation for correctness (see the
report, Section 2.4).

**Unresolved licensing concern, distinct from the Transolver case
below**: checked the upstream VINO repository directly -- it carries
**no LICENSE file of its own** (confirmed: its root listing has no
`LICENSE`/`LICENSE.md`, and its own `README.md`, copied verbatim here as
`VINO_README.md`, has no license section either). Unlike Transolver
(MIT, fixed by adding the required notice), there is no license here to
attribute -- by default, copyright law reserves all rights to the VINO
authors, so vendoring this code is not self-evidently permitted for
redistribution regardless of attribution. **This needs explicit
permission from the VINO authors (Eshaghi et al.) before
`Comparative_Examples/`/`Integration/` are included in any public
release**, not just a notice; until that permission is obtained (or
this code is dropped from what gets published), treat this as a real
blocker for public release, separate from -- and more serious than --
the dependency-licensing items below.

`Practical_Examples/omar_pfem/model/Transolver_Irregular_Mesh.py`,
`Physics_Attention.py`, and `Embedding.py` are adapted from
[Transolver](https://github.com/thuml/Transolver) (Wu et al., ICML
2024, THUML @ Tsinghua University), MIT License, copyright and license
text retained at the top of each file. Modified for this project's
energy-based (Deep Energy Method) training objective; cited in the
report/paper as the architecture used throughout.

## Dependency licensing

Checked before any public release, per the advisor's request (see
`PROJECT_STATUS.md`): every third-party package this project calls is
used only as an installed dependency (via `pip`), never vendored or
redistributed by this repository, so each package's own license governs
a user's installation of it, not this repository's own license.

- **torch-fem** (reference finite-element solver) — MIT License.
- **torch-sla** and **nvmath-python** (sparse linear algebra /
  cuDSS Python bindings used by the assembled+direct GPU solver) —
  Apache License 2.0.
- **NVIDIA cuDSS** (`nvidia-cudss-cu12`, the actual GPU sparse-solver
  binary, pulled in by `nvmath-python[cu12]`) — NVIDIA's own
  proprietary SDK license (not OSI-approved), accepted separately by
  whoever installs it; this repository does not bundle or redistribute
  the cuDSS binary itself.
- **TensorMesh** (`tensormesh-fem`, comparison-only baseline) — Apache
  License 2.0. The assembled+direct solver code actually released
  (`assembled_direct_solver.py`'s `build_sparse_jac_fn`) has no runtime
  dependency on it at all -- confirmed by code audit, every
  `tensormesh` import in the codebase is deferred inside the
  comparison-only functions that are never called by the released
  solver path.

## License

Not yet decided — pending the advisor's input on whether a public
release needs institutional sign-off and which license to use. See
`PROJECT_STATUS.md` for the current status of that question. Whatever
license is chosen for this project's own original code, it must remain
compatible with retaining the Transolver files' own MIT notice (see
above) -- any permissive license (MIT, Apache-2.0, BSD) satisfies this.

## Contact

Omar Amro, advised by Prof. Dr.-Ing. Timon Rabczuk (Institute of
Structural Mechanics, Bauhaus-Universität Weimar).
