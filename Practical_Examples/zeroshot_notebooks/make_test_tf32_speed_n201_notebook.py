"""Builds the 3rd TF32-training diagnostic notebook: does TF32 give a
real per-step speedup at N=201, the LARGEST resolution in the multi-res
set (where matmuls are big enough to plausibly be tensor-core bound), or
does the "no speedup" finding from N=21 hold everywhere? See
cell_test_tf32_speed_n201.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_test_tf32_speed_n201.py"


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
            "colab": {"provenance": [], "name": "Test_TF32_Speed_N201.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# متابعة ثانية: هل TF32 بيسرّع فعلياً عند N=201 (أكبر حجم بمجموعة "
                "التدريب متعدد الأحجام)؟",
                "",
                "**وين وصلنا**: `Test_TF32_Training_Convergence.ipynb` (N=21، 3000 "
                "خطوة) أكد إنه TF32 **ما بيضر الدقة** (نفس الـseed، fp32 مقابل "
                "TF32، ضلوا قريبين ومستقرين). **بس السرعة طلعت 1.00x -- ولا فرق "
                "تقريباً**، عكس أول فحص (1.12x) يلي كان على الأغلب مجرد ضجيج "
                "warm-up مش تأثير حقيقي.",
                "",
                "**ليش هاد الفحص**: N=21 أصغر حجم بالمجموعة (21,33,101,201) -- "
                "ممكن يكون العمليات (matmul) صغيرة كتير عشان TF32 يفرق فيها أصلاً. "
                "N=201 هو **أكبر حجم**، وهو يلي فعلاً بيسيطر على وقت أي epoch "
                "(matmuls أكبر بكتير). هاد الفحص بيقيس وقت الخطوة الحقيقي (بعد "
                "warm-up) عند N=201 بس -- مش دقة كاملة من جديد (الدقة اتفحصت صح "
                "عند N=21 والآلية العددية نفسها، ما بتعتمد عالحجم)، بس فحص استقرار "
                "بسيط (لا NaN، نفس رتبة الـloss تقريباً).",
                "",
                "**لو ما في تسريع هون كمان**: معناها TF32 ما بيفيد بأي حجم بهاد "
                "المدى، ونقفل الموضوع نهائياً. **لو في تسريع حقيقي هون**: هاد وين "
                "ممكن نلاقي الساعة يلي عمر بيدور عليها، لأنه N=201 هو الأغلى "
                "بالتدريب.",
                "",
                "**آمن تماماً**: نفس مصدر البيانات (مهمة خلصت، read-only)، فحص "
                "قصير (110 خطوة لكل دقة، بعد استبعاد warm-up) -- المفروض يخلص "
                "بدقايق."
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    path = f"{HERE}/Test_TF32_Speed_N201.ipynb"
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)
