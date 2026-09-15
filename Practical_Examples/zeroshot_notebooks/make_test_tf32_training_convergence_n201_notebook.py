"""Builds the 4th TF32-training diagnostic notebook: does the real 2.10x
per-step speedup measured at N=201 (Test_TF32_Speed_N201.ipynb) hold up
once training runs long enough to see whether the bigger short-term loss
gap seen there closes, the way N=21's own gap did over more steps? See
cell_test_tf32_training_convergence_n201.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_test_tf32_training_convergence_n201.py"


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
            "colab": {"provenance": [], "name": "Test_TF32_Training_Convergence_N201.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# متابعة ثالثة: هل التسريع الحقيقي (2.10x) عند N=201 بيضل صحيح "
                "لما ندرب أطول؟",
                "",
                "**وين وصلنا**: `Test_TF32_Speed_N201.ipynb` (100 خطوة بس) طلع "
                "منها **تسريع حقيقي 2.10x** (547ms مقابل 260ms للخطوة) -- وهاد "
                "مهم لأنه N=201 هو أكبر حجم بمجموعة التدريب وهو يلي فعلاً بيسيطر "
                "على وقت أي epoch. **بس** الفرق بالـloss بعد 100 خطوة كان كبير "
                "(TF32=0.386 مقابل fp32=0.070، ~5.5 أضعاف) -- أكبر من الفرق يلي "
                "شفناه عند N=21 بنفس عدد الخطوات تقريباً، رغم إنه فرق N=21 "
                "بالنهاية سكر لما درّبنا لـ3000 خطوة.",
                "",
                "**هاد النوتبوك بيتأكد**: هل نفس الشي بيصير هون -- الفجوة تسكر مع "
                "الوقت -- ولا هاي مشكلة حقيقية دايمة عند N=201؟ نفس تصميم الفحص "
                "الثلاثي (A: fp32 seed=1234، B: fp32 seed=5678 لقياس التذبذب "
                "الطبيعي، C: TF32 seed=1234) بس عند N=201، لعدد خطوات أقل "
                "(1200 بدل 3000، لأنه كل خطوة هون أغلى بـ15-20 ضعف تقريباً).",
                "",
                "**ملاحظة مهمة من فحص N=21**: تدريبة B (fp32 بseed مختلف) هناك "
                "طلعت فيها قفزة حقيقية غير طبيعية (عدم استقرار، مش TF32 السبب) "
                "خلت مقياس 'التذبذب الطبيعي' غير موثوق تماماً. هاد النوتبوك بيفحص "
                "نفس الشي تلقائياً هون (بيطبع تحذير إذا صار)، وبيعتمد أساساً على "
                "المقارنة المباشرة A مقابل C (نفس الـseed) مش بس على النسبة.",
                "",
                "**الوقت المتوقع**: ~25-30 دقيقة (تدريبتين fp32 على A100 لكل "
                "وحدة ~11 دقيقة تقريباً، وتدريبة TF32 وحدة ~5 دقايق، زائد فحص "
                "دقة دوري). آمن تماماً -- نفس مصدر بيانات المهمة الخالصة، "
                "read-only، بدون ما يلمس أي شي شغال."
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    path = f"{HERE}/Test_TF32_Training_Convergence_N201.ipynb"
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)
