"""Builds ONE notebook: the properly-fixed round-12 point 1 computation
(2026-09-18) -- FEM and the operator scored on the SAME ParametricField
realization, both against ONE real fine reference, for all six cases. See
cell_round12_consistent_field_all_cases.py for the full rationale (Omar
caught two separate real bugs in the original table: same-N-not-fine-
reference, then a deeper FEM-vs-operator field-realization mismatch).
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_round12_consistent_field_all_cases.py"


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
            "colab": {"provenance": [], "name": "Round12_ConsistentField_QoI_AllCases.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# Round-12 نقطة 1: إصلاح حقيقي للحساب (مش بس نص)",
                "",
                "عمر لقى مشكلتين حقيقيتين بالجدول الأصلي:",
                "",
                "1. عمود الـoperator (L2/H1/Energy/Reaction) كان يقارن مع حل FEM "
                "**عند نفس N الواطي**، مش مع مرجع دقيق -- خلافًا لجدول الـCauchy "
                "يلي كان صحيح من الأساس.",
                "2. حتى لو صلحنا (1)، عمود FEM (من `run_qoi_study`، بيستخدم "
                "`AnalyticField`) وعمود الـoperator (بيستخدم `ParametricField`, "
                "seed=0) كانوا عم يحلّوا **مسألتين فيزيائيتين مختلفتين تمامًا** "
                "بنفس الصف من نفس الجدول -- مش بس دقة شبكة مختلفة.",
                "",
                "لما اقترحت نضيف تصريح بالنص بدل ما نعيد الحساب، عمر رفض "
                "(\"بدنا نصلح الداله والكود ونعمل التحليل عشان ما نصرح بالنص\") "
                "-- وهاد صح، لأنو التصريح بيغطي المشكلة مش يحلها.",
                "",
                "**الحل الحقيقي**: دالتين جديدتين "
                "(`run_qoi_study_consistent_field_b1/b2`) بتحلّوا FEM "
                "والـoperator على **نفس الحقل الفيزيائي بالضبط** "
                "(`ParametricField(seed)` -- نفس الحقل يلي تدرب/اتفحص عليه "
                "الـoperator أصلاً)، وكلاهم بيتقارنوا مع **مرجع دقيق حقيقي "
                "واحد** (N=201) -- لكل من L2, H1, Energy, Reaction (B1 بس)، "
                "وإجهاد Cauchy بالمنطقة الثابتة.",
                "",
                "**تم التحقق على CPU قبل أي وقت GPU** (identity check: لما "
                "coarse==fine، كل الأخطاء طلعت صفر تمامًا؛ وزوج حقيقي "
                "coarse-vs-fine محلول بـCPU طلع أرقام منطقية وغير متدهورة) -- "
                "راجع محضر المحادثة لهاد التحقق.",
                "",
                "**الكلفة المتوقعة**: 15-30 دقيقة تقريبًا للحالات الستة كلهم "
                "(أكتر من Remaining5 لأنو هلق بنحل FEM من جديد بكل N كمان، مش "
                "بس نعيد استخدام كاش قديم).",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/Round12_ConsistentField_QoI_AllCases.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
