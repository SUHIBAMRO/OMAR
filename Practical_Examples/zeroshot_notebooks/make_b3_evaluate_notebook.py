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
                "**السياق**: التدريب بيصغّر طاقة الشكل المرن بس (Deep "
                "Energy Method) -- **ما بيشوف ولا إزاحة FEM حقيقية "
                "أبدًا**. يعني استقرار الخسارة أثناء التدريب بيثبت بس "
                "إنه التحسين (optimization) ما انفجر -- **مش** إنه "
                "الحقل يلي اتعلمته الشبكة قريب من الحل الحقيقي.",
                "",
                "**نتيجة حقيقية سابقة، 2026-09-26**: قيّمنا هالدفتر "
                "checkpoint_2000.pt (التشغيلة التانية) وطلع الخطأ "
                "النسبي L2 عالي كتير: 32.0% (ux)، **99.95% (uy)**، "
                "36.8% (uz)، 35.7% (مجتمعة). فحصنا فرضية إنه "
                "`OUTPUT_SCALE=0.02` الثابت هو السبب (قيمة أكبر، وقيمة "
                "قابلة للتعلّم) بتجربة تشخيصية حقيقية -- **انثبت غلطها**، "
                "ما تحسّنت uy بأي متغير. السبب الأرجح: عدد خطوات التدريب "
                "قليل جدًا (2000 خطوة بس مقابل ~350,000 بتدريب B2 "
                "الناجح). لهيك التشغيلة الثالثة رفعت `n_iters` لـ20,000، "
                "وهالدفتر هلق موجّه على `checkpoint_20000.pt` (نتيجة "
                "هاي التشغيلة الأطول).",
                "",
                "**شو بيعمل بالضبط**:",
                "1. بيحمّل الـcheckpoint المدرَّب.",
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
                "**تحذير مهم**: لسا ما منعرف إذا التدريب الأطول حسّن "
                "الدقة فعليًا. إذا ضلت uy عالقة قريب من 100% حتى بعد "
                "20,000 خطوة، هيك بنعرف إنه المشكلة أعمق من مجرد قلة "
                "التدريب ولازم نراجع تصميم شرط الحدود لـuy تحديدًا.",
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
