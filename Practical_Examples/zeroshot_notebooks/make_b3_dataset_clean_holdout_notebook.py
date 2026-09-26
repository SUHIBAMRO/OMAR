"""Builds ONE notebook: regenerates B3's held-out validation set with a
genuinely disjoint seed, fixing the seed-overlap bug found 2026-09-26.
See cell_b3_dataset_clean_holdout.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_b3_dataset_clean_holdout.py"


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
            "colab": {"provenance": [], "name": "B3_Dataset_Clean_Holdout.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# B3: توليد held-out set نظيف (بلا تداخل seeds مع التدريب)",
                "",
                "**المشكلة الحقيقية يلي انلقت، 2026-09-26** (بمراجعة عمر "
                "المنهجية): داتاسيت التقييم الأصلي (`b3_dataset/dataset.h5`) "
                "استخدم `seed=0`، يعني seeds 0-99. بس `train_B3.py` "
                "بيستخدم `seed = it*batch_size + b` بالتدريب، يلي عند "
                "`it=1..12` بيغطي seeds 8-103. يعني **92 من الـ100 عينة "
                "\"held-out\" فعليًا استخدمت seeds متطابقة كـinputs "
                "حقيقية بالتدريب** خلال أول 12 تكرار من أي تشغيلة (تشغيلة "
                "2 وتشغيلة 3 الاثنين) -- تأكدت بحساب مباشر، مش افتراض.",
                "",
                "**هاد الدفتر بيولّد بديل نظيف**: نفس الـ100 عينة، نفس "
                "الدقة، بس بـ`seed=99999` (نطاق sample_seed من 999990 "
                "لـ1000089) -- بعيد كليًا عن أي seed ممكن يوصله أي تدريب "
                "حقيقي (حتى تشغيلة 50,000 تكرار بس توصل لـseed ~400,007).",
                "",
                "**اتفحص محليًا على CPU قبل هيك**: نفس الـseed بالضبط "
                "اتولّد 100/100 بنجاح محليًا (بدون GPU) قبل هاي التشغيلة "
                "-- هاي التشغيلة بس بتعيد نفس الشي أسرع بكتير على GPU.",
                "",
                "**استخدم هاد الملف الجديد (`b3_dataset_clean_holdout/`) "
                "من هلق وطالع لأي مقارنة دقة -- مش الملف القديم.**",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B3_Dataset_Clean_Holdout.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
