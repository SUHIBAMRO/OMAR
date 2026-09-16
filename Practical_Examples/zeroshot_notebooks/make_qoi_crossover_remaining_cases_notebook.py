"""Builds ONE notebook: the multi-QoI crossover analysis (Timon
round-11 point 2) for the five (geometry, material) cases beyond
B1xNeo-Hookean, which already has this analysis published in the
Report (Table 18-R10e: coarsest suitable FEM bound by the tangent-
energy norm at N=11). See cell_qoi_crossover_remaining_cases.py for the
full rationale, what is reused unchanged (each case's own already-
computed NO accuracy sweep), and the real run_qoi_study bug found and
fixed (2026-09-16) to make this safe for B2 cases.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_qoi_crossover_remaining_cases.py"


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
            "colab": {"provenance": [], "name": "Round11_QoI_Crossover_RemainingCases.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# شو المقياس (QoI) يلي بيحدد دقة FEM المطابقة؟ -- الحالات الخمس المتبقية",
                "",
                "**طلب تيمون (round-11, نقطة 2)**: عنده تحليل B1×Neo-Hookean بس "
                "(موجود بالتقرير، Table 18-R10e -- المقياس الحاسم هناك هو الـtangent "
                "energy norm عند N=11). طلب صراحة نتأكد هل نفس المقياس بيحدد الدقة "
                "المطابقة بكل الحالات الست، ولا مختلف من حالة لحالة -- **توقّع بنفسه "
                "إنه مش رح يكون نفس المقياس بكل مرة**.",
                "",
                "**شو هاد النوتبوك بيعيد استخدامه بدون أي تغيير**: حساب دقة الـNO "
                "الخاص بكل حالة عند N=1401 (بكل الـQoIs: L2، H1، الطاقة، الإجهاد "
                "الأقصى، وردة الفعل لـB1) -- **موجود أصلاً من task #22**، محفوظ "
                "على Drive، ما بنعيد حسابه.",
                "",
                "**الشي الوحيد الجديد هون**: حل torch-fem عند أحجام صغيرة (N=3 لـ49، "
                "نفس المدى يلي استُخدم لتحليل B1×Neo-Hookean الأصلي) لكل حالة من "
                "الخمس، وبعدها مقارنة كل QoI لحاله لنلاقي أصغر N من torch-fem يطابق "
                "أو يتفوق على دقة الـNO عند N=1401 -- والمقياس \"الحاسم\" هو أكبر N "
                "بين كل المقاييس (لازم يحقق الكل مع بعض، مش مقياس واحد بس).",
                "",
                "**bug حقيقي لقيناه وصلحناه قبل هاد النوتبوك (2026-09-16)**: دالة "
                "`run_qoi_study` عندها معامل `geometry`، بس **كل استخدام سابق لها "
                "بالمشروع كان بس لـB1** (القيمة الافتراضية) -- يعني bug حقيقي ضل "
                "مخفي: كانت بتستخدم `AnalyticFieldB1` دايماً بغض النظر عن `geometry`، "
                "يلي كان رح يعطي أرقام غلط بصمت لأي حالة B2 (بيقرأ حقل المادة الغلط "
                "بنقاط استعلام مقصودة لحلقة B2). صلحناها تختار `AnalyticFieldB1` أو "
                "`AnalyticFieldB2` حسب `geometry`، وتخطي مقياس ردة الفعل لـB2 (مافي "
                "قاعدة معروفة له هناك، نفس القرار المتبع بباقي المشروع). **اتفحصت "
                "فعلياً على CPU حقيقي عند N=7** لـB1 وB2 معاً قبل الثقة فيها: أرقام "
                "B1 ما تغيرت (فحص رجوع)، وأرقام B2 صارت منطقية وما في كراش.",
                "",
                "**الوقت المتوقع**: رخيص -- 16 حجم صغير × 5 حالات، كل حل أصغر بكثير "
                "من حل N=1401 الوحد يلي أخد ~134-208 ثانية بالخلية السابقة. متوقع "
                "أقل من 15-20 دقيقة إجمالاً على GPU حقيقي.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/Round11_QoI_Crossover_RemainingCases.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
