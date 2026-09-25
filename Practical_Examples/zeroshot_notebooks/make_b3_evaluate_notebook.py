"""Builds ONE notebook: real accuracy/latency evaluation of the trained B3
Transolver checkpoint against the 100-sample FEM validation dataset. See
cell_b3_evaluate.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_b3_evaluate.py"


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
            "colab": {"provenance": [], "name": "B3_Evaluate.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# B3: تقييم دقة الترانسولفر المدرَّب مقابل FEM الحقيقي",
                "",
                "**السياق**: التدريب (`B3_Transolver_Training.ipynb`) بيصغّر "
                "طاقة الشكل المرن بس (Deep Energy Method) -- **ما بيشوف ولا "
                "إزاحة FEM حقيقية أبدًا**. يعني إنه الخسارة ضلت مستقرة طول "
                "2000 تكرار (النتيجة الناجحة يلي وصلتنا) بيثبت بس إنه "
                "التحسين (optimization) ما انفجر -- **مش** إنه الحقل يلي "
                "اتعلمته الشبكة قريب من الحل الحقيقي. هاد الدفتر هو "
                "**التحقق الفعلي** يلي كان ناقص.",
                "",
                "**شو بيعمل بالضبط**:",
                "1. بيحمّل الـcheckpoint المدرَّب (`checkpoint_2000.pt`، "
                "التشغيلة الثانية الناجحة بعد إصلاح `OUTPUT_SCALE`).",
                "2. بيحمّل الـ100 عينة FEM الحقيقية (`dataset.h5`) -- هاي "
                "العينات **ما استخدمت أبدًا بالتدريب**، كل تكرار تدريب كان "
                "عم يولّد عينة عشوائية جديدة بنفس التوزيع، فهاي مقارنة "
                "حقيقية بيانات لم تُرَ من قبل (held-out).",
                "3. بيحسب **الخطأ النسبي L2** (relative L2 error) لكل مركبة "
                "إزاحة (ux, uy, uz) ومجتمعة، بنفس منهجية B1/B2 بالضبط "
                "(`evaluate_dataset_hyperelastic_Q4`).",
                "4. بيقيس **زمن الاستدلال** (inference latency) للشبكة "
                "المدرَّبة مقابل زمن حل FEM الحقيقي المسجّل بنفس الداتاسيت، "
                "لمقارنة سرعة مباشرة.",
                "",
                "**تحذير مهم**: لسا ما في أي رقم دقة حقيقي لهاد الـcheckpoint "
                "-- منحن ندخل هالخلية وإحنا **ما منعرف** إذا الدقة كويسة "
                "ولا لأ. إذا الخطأ النسبي طلع كبير، هاد مش فشل بالكود، هاد "
                "معناه لازم نعيد ضبط (hyperparameters) أو نزود عدد "
                "التكرارات ونعيد التدريب -- قرار حقيقي بناءً على أرقام "
                "حقيقية، مش افتراض.",
                "",
                "**اتفحص محليًا على CPU قبل هيك**: تشغيلة كاملة صغيرة "
                "(داتاسيت FEM حقيقي بـ4 عينات + موديل toy مدرَّب 20 تكرار) "
                "اشتغلت من غير أخطاء ورجّعت أرقام منطقية (خطأ نسبي مختلف "
                "باختلاف المركبة، زمن استدلال بالميلي ثانية) -- دليل إنه "
                "الأسلاك (wiring) صحيحة قبل ما نجربها على الـcheckpoint "
                "الحقيقي.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B3_Evaluate.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
