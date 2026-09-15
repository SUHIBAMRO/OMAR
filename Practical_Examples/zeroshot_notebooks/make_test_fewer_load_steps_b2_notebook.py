"""Builds ONE small diagnostic notebook: tests whether B2's multi-res
data generation (N=21,33,101,201, both materials) can use FEWER
incremental load steps than the current default (nsteps=10) -- the
same question already verified YES for B1 x Neo-Hookean at N=1401
(Test_FewerLoadSteps_N1401.ipynb, 2026-09-14), asked again here since
that verification was specific to N=1401 and says nothing about these
much smaller resolutions. See cell_test_fewer_load_steps_b2_multires.py
for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_test_fewer_load_steps_b2_multires.py"


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
            "colab": {"provenance": [], "name": "Test_FewerLoadSteps_B2_MultiRes.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# فحص سريع: هل نقدر نقلل خطوات التحميل لتوليد بيانات B2؟",
                "",
                "**ليش هاد النوتبوك**: تأكدنا سابقاً (`Test_FewerLoadSteps_N1401.ipynb`) "
                "إنه `nsteps=3` آمن وأسرع 1.64x من `nsteps=10` -- **بس هاد التحقق كان "
                "عند N=1401 بس**. B2 (إعادة التدريب على أحجام أوسع) بتولّد بيانات "
                "عند أحجام أصغر بكتير (21, 33, 101, 201)، وما فحصناش هل نفس التسريع "
                "آمن عندهم -- محتمل يكون آمن (مسائل أصغر عادة أسهل تتقارب)، بس مش "
                "مؤكد بدون فحص حقيقي.",
                "",
                "**توقعات واقعية**: توليد بيانات B2 أصلاً بياخد ~3 ساعات و10 دقايق "
                "(رقم حقيقي من B1×Mooney-Rivlin بنفس عدد العينات والأحجام) -- أرخص "
                "بكتير من N=1401، فالتوفير المحتمل هون أصغر (ساعة أو ساعتين، مش أكتر).",
                "",
                "**هاد النوتبوك آمن تماماً**: بيجرب 3 عينات لكل توليفة (مادة × حجم × "
                "nsteps) = 72 حل إجمالاً، بدون ما يكتب أي شي عالمجلدات الحقيقية -- "
                "تشخيص بحت، بتاب منفصل، ما بأثر على أي نوتبوك تاني شغال.",
                "",
                "**قاعدة القرار**: قيمة `nsteps` تُعتبر آمنة **بس** إذا تقاربت كل "
                "التوليفات (كل مادة، كل حجم، كل seed) بدونها -- إذا فشلت ولو توليفة "
                "وحدة، ما نستخدمها، حتى لو نجحت الباقي.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    path = f"{HERE}/Test_FewerLoadSteps_B2_MultiRes.ipynb"
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)
