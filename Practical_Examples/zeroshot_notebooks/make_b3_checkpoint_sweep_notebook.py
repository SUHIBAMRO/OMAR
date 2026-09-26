"""Builds ONE notebook: evaluates every saved B3 checkpoint (5k-50k) from
the third training run against the clean held-out set, to find the real
best checkpoint. See cell_b3_checkpoint_sweep.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_b3_checkpoint_sweep.py"


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
            "colab": {"provenance": [], "name": "B3_Checkpoint_Sweep.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# B3: تقييم كل الـcheckpoints المحفوظة (5k إلى 50k) لإيجاد الأفضل الحقيقي",
                "",
                "**السياق**: `train_B3.py` **ما فيه أي متابعة دقة أو "
                "اختيار \"أفضل checkpoint\" أثناء التدريب إطلاقًا** -- "
                "بعكس B1/B2 يلي عندهم آلية `model_best.pt` جاهزة. لهيك "
                "افتراض إنه آخر checkpoint (50,000) هو الأفضل **مش "
                "مضمون** -- ممكن يكون checkpoint أبكر (مثلاً 20k أو "
                "30k) أحسن فعليًا.",
                "",
                "**سابقة حقيقية من مشروعنا نفسه بتدعم هالشك**: بحالة B2 "
                "قديمة، النتيجة قعدت عالقة عند ~1.0 خطأ لأسابيع من "
                "التشخيصات المعقّدة، وطلع السبب checkpoint غلط "
                "اختير بسبب bug بمعيار الاختيار -- مش قيد بنيوي. هاد "
                "الدفتر بيتجنّب نفس الغلطة بتقييم **كل** الـcheckpoints "
                "المحفوظة (كل 5000 تكرار) بدل الاكتفاء بالأخير.",
                "",
                "**بيستخدم الداتاسيت النظيف** (`b3_dataset_clean_holdout/`, "
                "seed=99999) **مش القديم** -- القديم فيه تداخل seeds حقيقي "
                "مع بيانات التدريب (92 من 100 عينة). **شغّل "
                "`B3_Dataset_Clean_Holdout.ipynb` أولاً لو ما شغّلته "
                "بعد.**",
                "",
                "**اتفحص محليًا على CPU قبل هيك**: تشغيلة كاملة (داتاسيت "
                "FEM حقيقي صغير + 3 checkpoints toy) اشتغلت من غير أخطاء "
                "ورجّعت جدول نتائج منطقي.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B3_Checkpoint_Sweep.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
