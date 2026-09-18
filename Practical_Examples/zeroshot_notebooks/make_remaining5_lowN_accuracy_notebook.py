"""Builds ONE notebook: closes the last real gap in round-12 point 1
(Omar's own catch, 2026-09-18) -- the operator's classical QoIs at
N=3,4,5,6,9,11 for the five cases beyond B1xNeo-Hookean. See
cell_no_accuracy_degradation_sweep_remaining5_lowN.py for the full
rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_no_accuracy_degradation_sweep_remaining5_lowN.py"


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
            "colab": {"provenance": [], "name": "Remaining5_LowN_Accuracy.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# سد آخر فجوة حقيقية بـround-12 نقطة 1: دقة الـoperator عند N منخفض لخمس حالات",
                "",
                "عمر لاحظ (بحق) إنو النسخة السابقة من الجدول ما كانت \"for each "
                "resolution\" فعليًا لكل الحالات: حالة B1×Neo-Hookean عبينا "
                "فجوتها بنوتة سابقة، بس باقي الخمس حالات (B1×Mooney-Rivlin، "
                "B1×Arruda-Boyce، B2×Neo-Hookean، B2×Mooney-Rivlin، "
                "B2×Arruda-Boyce) لسا خانات الـoperator (L2/H1/energy/reaction) "
                "عندها \"n/a\" عند N=3,4,5,6,9,11.",
                "",
                "**هاي مو تدريب جديد ولا شغل GPU ثقيل** -- بس evaluation إضافي "
                "بنفس الكود المُختبر أصلاً (`run_accuracy_degradation_sweep`)، "
                "وبيكمّل كل ملف موجود مسبقًا (N=13-49 موجودة، رح يحسب بس الست "
                "قيم الناقصة الأرخص أصلاً بكل المجموعة).",
                "",
                "**الكلفة المتوقعة**: رخيصة جدًا، بنفس رتبة نوتة B1×Neo-Hookean "
                "السابقة (46 ثانية حساب فعلي، دقيقتين وربع إجمالي) -- بس مضروبة "
                "بخمس حالات، فمتوقع **10-15 دقيقة إجمالي**.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/Remaining5_LowN_Accuracy.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
