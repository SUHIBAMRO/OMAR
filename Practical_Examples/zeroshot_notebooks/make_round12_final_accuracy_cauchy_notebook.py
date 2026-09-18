"""Builds ONE notebook: Timon round-12 point 1's final accuracy-vs-
resolution table, all six cases, FEM vs. NO, using each case's own
FINAL (retrained where applicable) checkpoint, with the new fixed-
region Cauchy-stress QoI replacing a bare pointwise stress maximum.
See cell_round12_final_accuracy_cauchy_all_cases.py for the full
rationale and explicit scope note (LOW_N range only, fine_N=201 -- NOT
extended to N=1401 here, which would need a much more expensive fresh
fine_N=2236 reference for five cases that have never had one).
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_round12_final_accuracy_cauchy_all_cases.py"


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
            "colab": {"provenance": [], "name": "Round12_FinalAccuracy_Cauchy_AllCases.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# الجدول النهائي: دقة FEM مقابل NO، بإجهاد Cauchy بدل PK1 -- round-12 نقطة 1",
                "",
                "**طلب تيمون**: جدول نهائي واحد، بس بالشيكبوينتات النهائية (بعد الـretrain)، "
                "يقارن FEM وNO بنفس المرجع الدقيق، بـL2، H1/الطاقة، قوة رد الفعل، "
                "**وإجهاد Cauchy (مش PK1) كمقياس هندسي أساسي** -- وطلب صراحة نتجنب "
                "أقصى إجهاد نقطي (ممكن يكون singular عند الزوايا حتى لـFEM)، وبدالها "
                "نحدد **منطقة فيزيائية ثابتة** حول تركز الإجهاد ونحسب فيها متوسط موزون "
                "بالمساحة، أو نسبة مئوية 95/99، أو متوسط أعلى 1% -- والقيمة القصوى "
                "الحقيقية تنذكر لحالها كرقم إضافي بس.",
                "",
                "**كود جديد بالكامل، اتفحص على CPU قبل هالنوتبوك**: `_cauchy_stress_at` "
                "(push-forward من PK1 المتحقق منه أصلاً)، `select_fixed_region` "
                "(نقاط Gauss من المرجع الدقيق فقط، ثابتة عبر كل الدقات)، "
                "`compute_region_cauchy_stress_error` (خطأ الحقل + متوسط موزون + نسبة "
                "مئوية + أعلى 1% + القيمة القصوى). خمس فحوصات CPU نجحت كلها: تقارب "
                "Cauchy مع PK1 عند انفعال صغير (نسبة ~2e-6)، تناظر Cauchy تام "
                "(3.5e-15) رغم إنه PK1 نفسه غير متناظر حقيقةً، فحص هوية كامل عند "
                "N=11 حقيقي (كل الأخطاء = 0 بالضبط لما coarse=fine).",
                "",
                "**نطاق هاي النوتة، محدد بوضوح مش مفترض بصمت**: الدقات هي نفس مدى "
                "\"LOW_N\" المُتحقق من كلفته سابقاً هالأسبوع (3 لـ49)، بمرجع دقيق "
                "fine_N=201 (رخيص وآمن حسب قاعدة الـ4x الخاصة بالمشروع). **ما "
                "بتمتد لـN=1401 هون** -- هاد كان بده مرجع دقيق أغلى بكثير "
                "(fine_N=2236) لخمس حالات ما عندهاش هيك مرجع أصلاً، بالظبط نفس "
                "غلطة الـ\"150-240 ساعة\" يلي انصلحت هالأسبوع بنوتة تانية -- لو "
                "بدنا نمدها لـN=1401، هاي طلبية منفصلة لازم تتقيّم كلفتها لحالها "
                "الأول.",
                "",
                "**الوقت المتوقع**: رخيص -- كل مرجع دقيق (fine_N=201) محفوظ مسبقاً "
                "من نوتة round-11 نقطة 2 (اليوم قبل هلق)، وكل نقطة torch-fem بمدى "
                "LOW_N رخيصة. متوقع أقل من 20-30 دقيقة إجمالاً للحالات الست.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/Round12_FinalAccuracy_Cauchy_AllCases.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
