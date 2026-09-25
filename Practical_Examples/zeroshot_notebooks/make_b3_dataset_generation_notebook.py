"""Builds ONE notebook: the first real GPU production run of B3's
Transolver training-dataset generator (data_generate_B3_dataset.py).
See cell_b3_dataset_generation.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_b3_dataset_generation.py"


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
            "colab": {"provenance": [], "name": "B3_Dataset_Generation.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# B3 (الأخدود الأحدّ): توليد داتاسيت تدريب الترانسولفر",
                "",
                "**السياق**: بعد تأكيد تيمون (2026-09-25) إنه \"VINO\" يقصد "
                "فيها الترانسولفر تبعنا بالذات (مش معماريّة تانية)، ما "
                "في داعي أي تغيير بالمعماريّة أو طريقة التدريب -- "
                "بنكمل بنفس منهجية B1/B2: الهندسة ثابتة، حقول المادة "
                "E وnu بتتغيّر مكانيًا لكل عينة (حقل عشوائي غاوسي "
                "ثلاثي الأبعاد)، وزاوية الميلان φ بتتغيّر كرقم واحد "
                "لكل عينة.",
                "",
                "**الكود اتفحص محليًا على CPU قبل هيك** (مش افتراض): عدة "
                "seeds مختلفة اتحلت بنظافة، زاوية خارج النطاق الآمن "
                "فشلت بخطأ حقيقي (يعني معالجة الفشل شغالة)، ولقينا "
                "وصلحنا باگ حقيقي بمنطق الاستئناف (resume) كان رح "
                "يطيح أي محاولة ثانية.",
                "",
                "**الدقة المستخدمة (21,20,19)=6,840 عنصر**: مش رقم جديد "
                "مجرّب لأول مرة -- هاي بالضبط نفس النقطة يلي جربناها "
                "فعلاً بدراسة تقارب B3 (~4.86 ثانية للحالة على A100 "
                "حقيقي)، واختيرت لأنها موازية لإعداد B2 الافتراضي "
                "(Ntheta=Nr=21) لتوليد الداتاسيت.",
                "",
                "**هاي أول تشغيلة GPU حقيقية لهاد الكود بالذات** -- راقب "
                "أول كم صف بعناية قبل ما تفترض إنه الباقي رح يتصرف "
                "نفس الشي. عدد العينات المستهدف هلق: 100 (نفس الإعداد "
                "الافتراضي المستخدم بمولّد داتاسيت B2).",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B3_Dataset_Generation.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
