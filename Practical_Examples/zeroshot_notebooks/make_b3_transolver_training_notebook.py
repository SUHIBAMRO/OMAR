"""Builds ONE notebook: the fourth real GPU run of train_B3.py's Deep
Energy Method training loop for the 3D Transolver, this time with
--normalize_inputs 1. See cell_b3_transolver_training.py for the full
rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_b3_transolver_training.py"


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
            "colab": {"provenance": [], "name": "B3_Transolver_Training.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# B3: تدريب الترانسولفر -- التشغيلة الرابعة (مع input normalization)",
                "",
                "**ملخص سريع للتاريخ**: تشغيلة 1 انفجرت (bug حقيقي، "
                "انصلح بـ`OUTPUT_SCALE=0.02`). تشغيلة 2 (2000 تكرار) "
                "كانت مستقرة رقميًا بس دقتها ضعيفة (35.7% خطأ مجتمع، "
                "uy=99.95%). تشغيلة 3 (50,000 تكرار) **ما حسّنت** -- "
                "وأسوأ من هيك: `B3_Checkpoint_Sweep.ipynb` (بيانات GPU "
                "حقيقية، held-out set نظيف) بيّن إنه uy بتتذبذب بين 79% "
                "و111% عبر الـ10 checkpoints المحفوظة، وإنه آخر "
                "checkpoint (50,000) **أسوأ فعليًا** من وحدة أبكر "
                "(35,000) -- توقيع واضح لعدم استقرار بالتدريب.",
                "",
                "**الإصلاح الحقيقي، 2026-09-26/27**: E~1000، nu~0.45، "
                "phi~0.05 كانوا يدخلوا الشبكة **خام بلا أي تطبيع** -- "
                "فرق 3 مراتب بالحجم. اختبار A/B مضبوط (نفس الـ8 عينات "
                "الثابتة، نفس الـseed، 4000 تكرار لكل حالة):",
                "",
                "| | خطأ مجتمع | uy | الخسارة |",
                "|---|---|---|---|",
                "| بلا تطبيع (الوضع الحالي) | 10.7% | 54.1% | متذبذبة |",
                "| **مع تطبيع** | **1.1%** | **4.0%** | ناعمة، مستقرة |",
                "",
                "**تحسّن 10 أضعاف تقريبًا، والتذبذب اختفى بالكامل.** "
                "أوضح وأقوى نتيجة بكل تحقيق B3.",
                "",
                "**تنبيه مهم**: هاد الاختبار كان \"حفظ\" (memorization) "
                "على 8 عينات ثابتة، مش الوضع الحقيقي (عينات عشوائية "
                "جديدة كل تكرار). **هاي التشغيلة هي الاختبار الحقيقي** "
                "-- 50,000 تكرار، نفس إعدادات تشغيلة 3، بس مع "
                "`--normalize_inputs 1`.",
                "",
                "**بتحفظ بمجلد منفصل** (`b3_training_normalized`) عشان "
                "ملف التطبيع الجديد ما يتطبّق غلط على checkpoints "
                "تشغيلة 3 غير المُطبَّعة.",
                "",
                "**الوقت المتوقع**: نفس تشغيلة 3 تقريبًا (~2 ساعة).",
                "",
                "**بعد ما تخلص**: شغّل `B3_Checkpoint_Sweep.ipynb` من "
                "جديد (موجّه أوتوماتيكيًا على المجلد الجديد) لنشوف "
                "الدقة الحقيقية على الـheld-out set النظيف.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B3_Transolver_Training.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
