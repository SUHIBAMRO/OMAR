"""Wraps the two round-6 cells into real .ipynb notebooks.

The .py files are meant to be pasted into a Colab cell. Handing them to
Colab as files makes it try to parse them as notebook JSON, which fails with
`Unexpected token '#'`. These are actual notebooks, so a Colab link opens
them directly.

Every source line keeps its trailing newline -- nbformat stores a cell's
source as a list of lines and Jupyter concatenates it verbatim, so a missing
"\\n" glues lines together into a SyntaxError. That bug shipped once already;
check_notebooks.py exists to catch it and is run at the end of this script.
"""
import os
import json
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

NOTEBOOKS = {
    "Round6_Check_Results.ipynb": (
        "cell_check_results.py",
        ["# What is actually on Drive?\n",
         "\n",
         "Read-only. Shows which results exist and how far each got, so a run\n",
         "that printed nothing can be told apart from one that did nothing.\n",
         "Safe to run at any time, including while other notebooks are working.\n"]),
    "Round6_OOD_Progressive.ipynb": (
        "cell_ood_progressive.py",
        ["# Progressive OOD shift — Timon round 6, point 1\n",
         "\n",
         "Separates **material stiffness** from **loading** and sweeps the shift\n",
         "from 0 to 3σ, instead of the single 2–2.5σ pair Table 11 measured.\n",
         "\n",
         "* **CPU is fine** — no GPU needed.\n",
         "* **~1.5–4 h** for the full 19-cell grid at 10 samples each.\n",
         "* **Resumable**: every cell is written to Drive as it finishes and\n",
         "  skipped on a re-run. After a disconnect, just run the cell again.\n",
         "\n",
         "Edit `GEOMETRY`, `MATERIAL` and `CKPT` in the cell to pick the case.\n"]),
    "Round6_Data_Driven.ipynb": (
        "cell_data_driven.py",
        ["# Data-driven operator — Timon round 5, point 7b\n",
         "\n",
         "The same Transolver, same mesh, same 800/200 split, same optimizer and\n",
         "the **same optimizer-step budget** — trained on FEM solutions instead of\n",
         "the energy functional. Only the loss differs, so the comparison isolates\n",
         "the training principle rather than the architecture.\n",
         "\n",
         "* **No new FEM solves** — the labels are already in the dataset.\n",
         "* **GPU strongly preferred**; CPU works but will be slow.\n",
         "* Expected to finish inside the physics-informed run's 48 minutes,\n",
         "  since a data loss skips the per-step energy assembly.\n",
         "\n",
         "The script also reports the **label-generation cost** the data-driven\n",
         "model implicitly requires and the physics-informed one does not — about\n",
         "5.7 h of CPU for 800 B1 × Neo-Hookean solves at Table 4a's measured rate.\n"]),
    "Round6_MMS.ipynb": (
        "cell_mms.py",
        ["# Method of manufactured solutions — round 5, point 9 (FEM half)\n",
         "\n",
         "Timon: *\"compare Q4, Q9 and the physics-informed Transolver against\n",
         "exactly the same analytical solution in L2, H1 and energy norms and\n",
         "also examine stress errors\"*, and *\"this is the last thing to do\"*.\n",
         "\n",
         "**This notebook is the FEM half: Q4 and Q9.** The operator half needs\n",
         "a body-force term in the energy functional and a body-force input\n",
         "channel — a separate piece of work, described in `mms_study.py`.\n",
         "\n",
         "**The fork Timon left open is resolved here as body force**, because a\n",
         "body-force-free exact solution on this geometry is a homogeneous\n",
         "deformation, which Q4 reproduces to machine precision — the study\n",
         "would measure round-off and distinguish nothing.\n",
         "\n",
         "* **GPU optional.** FP64, so a T4/L4 buys little; `--cpu` is fine.\n",
         "* **Minutes** for N up to 33.\n",
         "* **Resumable**: each (order, N) row is appended as it finishes.\n",
         "\n",
         "The study validates itself: a wrong body force collapses the observed\n",
         "convergence rates. The cell prints them against theory and a verdict.\n"]),
    "Round6_PI_OneCycle.ipynb": (
        "cell_pi_onecycle.py",
        ["# The missing cell of the point-7b 2×2 — round 5, point 7b\n",
         "\n",
         "Three of the four boxes are measured:\n",
         "\n",
         "| | Adam 2e-3 | AdamW+OneCycle |\n",
         "|---|---|---|\n",
         "| physics-informed | **0.0959** | ← **this run** |\n",
         "| data-driven | 0.1307 | 0.0826 |\n",
         "\n",
         "Read the first column and the physics-informed loss wins by 36%.\n",
         "Read the diagonal and the data-driven one wins by 16%. They\n",
         "disagree because the diagonal changes the **optimizer and the loss\n",
         "at the same time** — the confound this project keeps hitting. Until\n",
         "the empty box is filled the two training *principles* cannot be\n",
         "ranked at all.\n",
         "\n",
         "* **NEEDS A GPU.** ~48 min, the same budget as the other three.\n",
         "* The recipe is copied exactly from the data-driven run: AdamW\n",
         "  lr=1e-3, wd=1e-5, OneCycleLR, 75,000 steps. Early stopping is off\n",
         "  on purpose — OneCycleLR converges in its final phase, so stopping\n",
         "  early would measure a truncated recipe.\n",
         "* The cell prints the finished 2×2 and says which reading holds.\n"]),
    "Round6_MMS_Operator.ipynb": (
        "cell_mms_operator.py",
        ["# MMS, the operator third — completes point 9's three-way\n",
         "\n",
         "Run **Round6_MMS.ipynb** first (Q4 and Q9). This trains the\n",
         "physics-informed operator on the same manufactured family and scores\n",
         "it with the **same error routine**, so all three are comparable.\n",
         "\n",
         "**Read before interpreting**: the operator minimizes the *same*\n",
         "discrete functional over the *same* Q4 space as the Q4 solver, and\n",
         "the minimizer of that functional **is** the Q4 solution. So the\n",
         "operator cannot beat Q4 here — that is arithmetic, not a finding.\n",
         "The number that means something is **operator / Q4**: 1.0 would mean\n",
         "the network has fully solved the variational problem. Below 1.0 means\n",
         "something is broken, and the cell says so.\n",
         "\n",
         "* **GPU preferred**; FP32, so T4/L4 are fine.\n",
         "* **~20–40 min** at N=17.\n",
         "* The cell **verifies the energy functional first** — the Q4 solution\n",
         "  must be the exact minimizer of the Π the network will minimize.\n",
         "* Labels are free (u\\* is analytic) but are **not used in training**;\n",
         "  the loss is the energy. They are only the scoring truth.\n"]),
    "Round6_OOD_Mitigation.ipynb": (
        "cell_ood_mitigation.py",
        ["# Does normalization fix the OOD degradation? — round 6, point 1b\n",
         "\n",
         "Timon: *\"If this diagnosis suggests a relatively straightforward\n",
         "mitigation, such as changing the training range or normalization, it\n",
         "would be useful to test it.\"* Section 8.6 named exactly that. This\n",
         "**tests** it instead of proposing it.\n",
         "\n",
         "* **GPU** for stage 1 (retraining). Any GPU — this is FP32 training,\n",
         "  not the FP64 solver sweep, so T4/L4 are fine.\n",
         "* **Stage 1 ~45–60 min**: retrain B1 × Neo-Hookean under the identical\n",
         "  protocol with `--normalize_inputs 1` as the only change.\n",
         "* **Stage 2 ~20–40 min**: re-run both OOD sweeps and compare. Fast\n",
         "  because the FEM references are **cached** — they do not depend on the\n",
         "  checkpoint, so the ~190 solves are reused, not repeated.\n",
         "* **Resumable** at both stages.\n",
         "\n",
         "A null result is a real result here. Standardizing is an affine\n",
         "rescaling, so a shifted E is still outside the trained range — if the\n",
         "curves barely move, that is the mechanism in §8.6 confirmed, and it is\n",
         "what gets reported.\n"]),
    "Round6_GPU_FEM_Sweep.ipynb": (
        "cell_gpu_fem_sweep.py",
        ["# GPU-native FEM scaling sweep + cost breakdown\n",
         "\n",
         "Timon round 5 point 8, extended by round 6: smaller intermediate sizes\n",
         "and an assembly-versus-solver cost breakdown.\n",
         "\n",
         "* **NEEDS A GPU** — Runtime → Change runtime type → T4/A100.\n",
         "  The cell stops with a clear message if there is none.\n",
         "* 0.02M → 3.93M DOF over eight resolutions. The four small ones take\n",
         "  minutes between them; the last two are hours each.\n",
         "* **Resumable**: each resolution is appended to the JSON as it\n",
         "  finishes, and each solve checkpoints internally.\n"]),
}


def build(nb_name, py_name, md_lines):
    with open(os.path.join(HERE, py_name)) as f:
        code = f.read()
    # splitlines(keepends=True) preserves every "\n"; the last line may lack
    # one, which is the only place nbformat allows it.
    src = code.splitlines(keepends=True)
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": md_lines},
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": src},
        ],
        "metadata": {
            "accelerator": "GPU" if "GPU_FEM" in nb_name else "None",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }
    path = os.path.join(HERE, nb_name)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {nb_name}  ({len(src)} source lines)")
    return path


if __name__ == "__main__":
    for nb_name, (py_name, md) in NOTEBOOKS.items():
        build(nb_name, py_name, md)
    print()
    subprocess.run([sys.executable, os.path.join(HERE, "check_notebooks.py")],
                   check=True)
