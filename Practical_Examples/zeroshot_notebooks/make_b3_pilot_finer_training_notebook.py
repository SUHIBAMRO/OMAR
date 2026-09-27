"""Builds ONE notebook: a clean, minimal pilot training run at a finer
mesh resolution (43,400 elements, 36 region Gauss points), to test
whether resolution is the real lever behind B3's region-stress gap
before committing to a long/expensive full retrain. See
cell_b3_pilot_finer_training.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_b3_pilot_finer_training.py"


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
            "colab": {"provenance": [], "name": "B3_Pilot_Finer_Training.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# B3: تجربة تشخيصية نظيفة -- هل الدقة الأعلى بتحسّن دقة "
                "إجهاد المنطقة؟",
                "",
                "**السياق**: التقييم الأدق (43,400 عنصر) طلع القيمة "
                "\"الصحيحة\" لإجهاد المنطقة نفسها بتتغير كتير (2x لمعظم "
                "العينات، 20-79x وحتى بتنقلب إشارتها للعينات القريبة من "
                "الصفر) بين دقة التدريب (6,840) والدقة الأنعم. بس تنبؤ "
                "الشبكة بالكاد تحرّك بين الدقتين -- يعني لسا مش واضح هل "
                "**الدقة** هي المشكلة، أو **وزن المنطقة الصغيرة داخل "
                "الطاقة الكلية** (هدف التدريب)، أو الاثنين معًا.",
                "",
                "**تصحيح دقيق (بناء على ملاحظة عمر)**: ما بنقدر نقول "
                "\"الشبكة تعلّمت تتجاهل المنطقة\" كحقيقة مؤكدة -- الأدق: "
                "شبكة 6,840 عنصر + هدف الطاقة الكلية ما أعطوا دقة كافية "
                "لتدرّجات الإجهاد المحلي بمنطقة الأخدود. السبب ممكن يكون "
                "دقة الشبكة، أو صغر وزن المنطقة بالطاقة الكلية، أو "
                "الاثنين -- وهاي التجربة مصممة تفرّق بينهم.",
                "",
                "**تجربة نظيفة، مش تشغيلة إنتاج**: نفس الـarchitecture، "
                "نفس normalization (نفس ملف `input_norm.json` من "
                "التشغيلة 4، منسوخ بلا إعادة حساب)، نفس توزيعات المادة/"
                "الحمل، نفس كل الإعدادات الأخرى -- **الشيء الوحيد يلي "
                "بيتغيّر هو دقة الشبكة (من 6,840 لـ43,400 عنصر) وعدد "
                "التكرارات (5,000 تجريبي، مش 50,000 الكاملة)**. "
                "التقييم على مجموعة FEM مستقلة **محلولة من جديد فعليًا** "
                "عند نفس دقة 43,400 (مش نفس مجموعة الـ6,840 المعاد "
                "استكمالها) -- seed منفصل تمامًا عن أي seed التدريب "
                "بيلمسه.",
                "",
                "**معيار الحكم على التجربة (مو خسارة التدريب)**: هل "
                "`region_cauchy_field_rel` بينزل بوضوح من ~72% (دقة "
                "الإنتاج) نحو 20-30%، مع بقاء الإزاحة/الطاقة/رد الفعل "
                "منيحين؟ **لو نعم** -- الدقة عامل حقيقي، يستاهل استثمار "
                "تشغيلة أطول أو multi-resolution. **لو ظل 60-80% رغم "
                "هيك** -- المشكلة الأرجح بهدف التدريب نفسه (متل حد صريح "
                "لخسارة الإجهاد المحلي)، مش بحجم الشبكة، ومنعرف ما "
                "نصرف ساعات عالفكرة نفسها.",
                "",
                "**تنبيه**: 43,400 عنصر هون بس تجربة تشخيصية، **مش رقم "
                "نهائي للتقرير** -- درسة تقارب الشبكة (FEM بس) يلي "
                "عملناها سابقًا بيّنت إنه هدف الدقة الحقيقي (5-10%) "
                "محتاج قرابة 480 ألف عنصر (مقابل مرجع 950 ألف).",
                "",
                "**كل قطعة بالكود معاد استخدامها بلا تعديل** من كود "
                "مُتحقّق منه أصلاً: توليد البيانات "
                "(`data_generate_B3_dataset.generate_dataset`)، التدريب "
                "(`train_B3.train`/`get_args`، بارامتر `resolution`)، "
                "والتقييم (`evaluate_B3_qois.main` -- أصلاً مرن بالكامل "
                "بالنسبة للدقة عبر `--resolution`/`--dataset`) -- ما في "
                "فيزياء أو مقاييس جديدة، بس تنظيم/orchestration جديد، "
                "اتفحص محليًا بحجم toy قبل هاي التشغيلة الحقيقية.",
                "",
                "**تحذير التكلفة**: كل عنصر بالمعادلة عم يكبر (43,400 "
                "بدل 6,840 = 6.3x)، فكل تكرار تدريب رح ياخد وقت أطول -- "
                "مش معروف كم بالضبط لسا. راقب معدل الثانية/تكرار "
                "بالمخرجات الأولى، ولو طالع وقت غير معقول، وقف "
                "(Runtime > Interrupt execution) واحكيلي المعدل يلي "
                "شفته حتى نعدّل عدد التكرارات.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B3_Pilot_Finer_Training.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
