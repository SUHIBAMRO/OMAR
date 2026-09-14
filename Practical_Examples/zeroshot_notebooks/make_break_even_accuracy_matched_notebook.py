"""Builds ONE notebook: the accuracy-matched break-even calc (Timon round-10
point 5), using the already-verified NO@N=1401 timing (point 3) and the
new multi-res checkpoint's own coarsest-suitable-FEM crossover (N=11 at
N=1401, from no_peak_stress_fixed_location_multires.json) plus its real
confirmed training wall-clock (41881.28s). See
cell_break_even_accuracy_matched.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_break_even_accuracy_matched.py"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines]}


def build():
    with open(CELL_FILE) as f:
        cell_src = f.read()
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "name": "Round6_BreakEven_AccuracyMatched.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# Accuracy-matched break-even (Timon round-10, point 5)",
                "",
                "Measures the ONE missing number (GPU-FEM per-sample cost at N=11, "
                "the multi-res checkpoint's own coarsest-suitable FEM at N=1401) and "
                "combines it with already-verified numbers (NO@N=1401 timing from "
                "point 3, real training wall-clock 41,881.28s) to answer: does the "
                "NO's one-off training cost ever get repaid against a FEM baseline "
                "that only needs to be as accurate as the NO actually is? Cheap, "
                "single-N run -- should take well under a minute of GPU time.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/Round6_BreakEven_AccuracyMatched.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
