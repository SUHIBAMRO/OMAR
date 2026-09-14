"""Builds ONE notebook: the fixed-location peak-stress check + multi-QoI
crossover for the NEW multi-resolution-retrained checkpoint (N=21,33,101,201),
mirroring Round6_NO_Peak_Stress_Fixed_Location.ipynb but pointed at the new
checkpoint instead of the old one. See cell_no_peak_stress_fixed_location_multires.py
for why this exists and what it computes.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_no_peak_stress_fixed_location_multires.py"


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
            "colab": {"provenance": [], "name": "Round6_NO_Peak_Stress_Fixed_Location_MultiRes.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# NO (multi-res checkpoint) fixed-location peak-stress + corrected crossover",
                "",
                "Same check as `Round6_NO_Peak_Stress_Fixed_Location.ipynb`, but for the "
                "NEW checkpoint retrained on N=21,33,101,201 instead of the old N=21,33-only "
                "one. Fills in the one QoI (peak PK1 stress at a fixed physical location) not "
                "already covered by the multi-res retrain notebook's own accuracy comparison, "
                "so the full multi-QoI 'coarsest suitable FEM' crossover can be redone correctly "
                "for the new checkpoint instead of reusing stale numbers from the old one.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/Round6_NO_Peak_Stress_Fixed_Location_MultiRes.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
