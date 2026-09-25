"""Builds ONE notebook: the first real GPU production run of train_B3.py's
Deep Energy Method training loop for the 3D Transolver. See
cell_b3_transolver_training.py for the full rationale.
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
                "# B3 (الأخدود الأحدّ): تدريب الترانسولفر (Deep Energy Method)",
                "",
                "**السياق**: بعد ما ولّدنا داتاسيت FEM حقيقي (100 عينة، بس "
                "للتحقق لاحقًا مش للتدريب)، هلق بنبلش التدريب الفعلي. "
                "التدريب هون **ما بيحتاج أي بيانات FEM محلولة** -- "
                "الشبكة العصبية بتتعلم مباشرة من تصغير طاقة الشكل "
                "المرن (physics-informed / Deep Energy Method)، نفس "
                "منهجية B1/B2 بالضبط، بس بفرق حقيقي واحد: حمل B3 كله "
                "إزاحة مفروضة (دوران الـcore)، مش قوة مفروضة -- فشروط "
                "الحدود التلاتة (الداخل، الخارج، التماثل) متحققة "
                "**بشكل دقيق ومضمون رياضيًا** جوا بنية الشبكة نفسها، "
                "مش بعقوبة تقريبية.",
                "",
                "**اتفحص محليًا على CPU قبل هيك** (مش افتراض): تشغيلة "
                "كاملة صغيرة اشتغلت من غير أخطاء، واختبار أدق (نفس "
                "الـbatch ثابت لـ200 تكرار) بيّن إنه الطاقة بتنزل من "
                "130.7 لحد ما تستقر عند ~2.25 -- دليل حقيقي إنه "
                "الگراديان عم يمشي صح من الطاقة لحد أوزان الشبكة.",
                "",
                "**هاي أول تشغيلة GPU حقيقية للتدريب الفعلي بالذات** "
                "(مش batch ثابت -- كل تكرار بياخد عينة عشوائية جديدة). "
                "قسنا الوقت على CPU: ~20-22 ثانية/تكرار بحجم batch=8 "
                "-- الـGPU المفروض يكون أسرع بكتير، بس ما جربناها قبل "
                "هلق، فراقب أول كم صف بعناية.",
                "",
                "**ملاحظة مهمة**: بما إنه كل تكرار بياخد عينة عشوائية "
                "جديدة (مش نفس البيانات)، الخسارة المطبوعة **مش "
                "المفروض تنزل بشكل رتيب من تكرار لتكرار** -- كل batch "
                "إله مقياس طاقة مختلف. راقب الاتجاه العام عبر كتير "
                "صفوف، مش التغيّر بين خطوة وخطوة.",
                "",
                "**إعدادات هاي التشغيلة**: 2000 تكرار، batch_size=8، "
                "عند نفس دقة الداتاسيت (21,20,19)=6,840 عنصر. "
                "الـcheckpoints بتنحفظ كل 200 تكرار على درايفك.",
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
