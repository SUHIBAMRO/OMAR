"""Builds ONE notebook: a standalone CPU-AMG diagnostic for B8-final's
real 105,456-element failure (ConvergenceError on GPU/Jacobi). See
cell_b8_amg_diagnostic.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_b8_amg_diagnostic.py"


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
            "colab": {"provenance": [], "name": "B8_AMG_CPU_Diagnostic.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# B8-final: فحص AMG على CPU لنفس الحالة يلي فشلت (تشخيصي)",
                "",
                "B8-final فشل بخطأ حقيقي عند 105,456 عنصر (ConvergenceError: "
                "CG ما وصل لـ1e-08، Newton-Raphson ما تقارب بالخطوة 4 بعد "
                "10 محاولات). هاي الخلية بتجرب **نفس الحالة بالضبط** "
                "(نفس الهندسة، المواد، منطقة القياس، الدقة، مسار التحميل، "
                "وحدود التقارب) بس بمحلّل AMG بدل Jacobi -- على CPU لأنه "
                "AMG على GPU محتاج مكتبة AmgX (شوف "
                "`AmgX_Build_And_Test.ipynb` لو بدك تجرب هاديك بدل هاي).",
                "",
                "**مهم**: هاي بتشتغل على CPU مش GPU، فمتوقع ياخد وقت "
                "طويل (ممكن ساعة أو أكتر). شغلها وسيبها بالخلفية.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B8_AMG_CPU_Diagnostic.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
