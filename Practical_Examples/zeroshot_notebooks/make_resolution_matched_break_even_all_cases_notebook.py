"""Builds ONE notebook: the resolution-matched break-even (NO vs.
torch-fem, both at N=1401) for ALL 6 cases -- completes Timon round-11
points 1 (keep both break-even comparisons) and 2 (extend to all 6
cases) together. See cell_resolution_matched_break_even_all_cases.py for
the full rationale and the three real bugs found/fixed in
torchfem_comparison.py to make this possible (material-arity
generalization, a NaN-Hessian det() sharp edge, and B2's per-Gauss-point
material sampling needing a documented per-element-averaging
approximation for torch-fem's own API).
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_resolution_matched_break_even_all_cases.py"


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
            "colab": {"provenance": [], "name": "Round6_ResolutionMatchedBreakEven_AllCases.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# مقارنة resolution-matched (NO مقابل torch-fem، الاثنين على N=1401) لكل الحالات الست",
                "",
                "**طلب تيمون (round-11)**: نقطة 1 طلبت نخلي مقارنتين break-even (مش وحدة "
                "بديلة عن التانية)، ونقطة 2 طلبت نوسّع كل شي لكل الحالات الست. هاي الخلية "
                "بتكمّل الاثنين مع بعض.",
                "",
                "**شغل تحضيري حقيقي انعمل قبل هاي الخلية** (تفاصيل كاملة بتعليقات "
                "الكود نفسه):",
                "1. مكتبة torch-fem كانت مبرمجة بالمشروع بس لـB1×Neo-Hookean -- عممتها "
                "لكل المواد والهندستين.",
                "2. لقيت bug رياضي حقيقي أثناء التحقق: `det()` بمكتبة PyTorch نفسها "
                "عندها مشكلة بالمشتقة الثانية عند F=I (نفس عائلة bug موثقة سابقاً بالمشروع "
                "لـ`log(det())` تحديداً، بس هون طلعت أوسع من هيك) -- تصلح باستخدام "
                "`slogdet` بدل `det` المباشرة.",
                "3. لقيت قيد حقيقي بمكتبة torch-fem: بتقبل قيمة مادة وحدة **لكل عنصر**، "
                "بينما B2 (باتفاقية المشروع) بتاخد قيمة مختلفة **لكل نقطة Gauss** داخل "
                "العنصر الوحد. الحل العملي: متوسط القيم داخل كل عنصر لـtorch-fem تحديداً "
                "(تقريب صغير موثق، مش ادعاء دقة جديد).",
                "",
                "**تحقق شامل قبل الاستخدام**: قارنت الحل الجديد ضد الحلول المرجعية "
                "البطيئة الموثوقة أصلاً لكل الحالات الست عند N صغيرة -- B1 طابقت بدقة "
                "الآلة (~1e-11)، وB2 طابقت بفارق صغير متوقع (~8e-4، بسبب تقريب المتوسط).",
                "",
                "**الوقت المتوقع**: رخيص جداً -- حل torch-fem الواحد عند N=1401 "
                "معروف من قبل إنو ياخد ~134 ثانية بس (مو ساعات). ست حالات + قياس سرعة "
                "NO لكل وحدة -- المتوقع أقل من 15 دقيقة إجمالاً على GPU حقيقي.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/Round6_ResolutionMatchedBreakEven_AllCases.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
