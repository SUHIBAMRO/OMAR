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
