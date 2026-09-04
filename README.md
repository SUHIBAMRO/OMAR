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

## License

Not yet decided — pending the advisor's input on whether a public
release needs institutional sign-off and which license to use. See
`PROJECT_STATUS.md` for the current status of that question.

## Contact

Omar Amro, advised by Prof. Dr.-Ing. Timon Rabczuk (Institute of
Structural Mechanics, Bauhaus-Universität Weimar).
