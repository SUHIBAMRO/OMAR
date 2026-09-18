"""Builds ONE notebook: fills the real gap found while writing round-12
point 1 into the Report/Summary -- B1xNeo-Hookean's own operator-side
L2/H1/energy/reaction sweep, LOW_N=[3..49], against its FINAL
(multi-resolution-retrained) checkpoint. See
cell_no_accuracy_degradation_sweep_b1nh_final_lowN.py for the full
rationale, including why the existing no_accuracy_degradation_sweep.json
on Drive was checked and confirmed NOT usable (old, pre-retrain
checkpoint).
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_no_accuracy_degradation_sweep_b1nh_final_lowN.py"


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
            "colab": {"provenance": [], "name": "B1NH_FinalCheckpoint_LowN_Accuracy.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# سد فجوة: دقة B1×Neo-Hookean (الشيكبوينت النهائي) عند دقات منخفضة",
                "",
                "لما كتبنا round-12 نقطة 1 (جدول الـCauchy stress) بالتقرير، طلع إنو "
                "حالة B1×Neo-Hookean (الحالة الرئيسية) **ناقصة كليًا** بيانات دقة "
                "الـoperator (L2, H1, energy, reaction) بمدى الدقات المنخفضة "
                "(N=3 لـ49) -- الملف المتوقع (`no_accuracy_degradation_sweep_"
                "B1_neo_hookean.json`) مش موجود على الـDrive.",
                "",
                "**فيه ملف قديم بنفس المعنى تقريبًا** (`no_accuracy_degradation_"
                "sweep.json`, بدون اسم الحالة) -- بس تأكدنا (يدويًا، عبر رقم الـ"
                "checkpoint\\_fingerprint المخزّن فيه، وعبر خطأ N=1401 فيه "
                "~39-45% وهو رقم يخص **الشيكبوينت القديم قبل إعادة التدريب**، "
                "مش الرقم المعتمد حاليًا 5.85%) إنو هاد الملف **مش صالح** "
                "للاستخدام هون -- استخدامه كان رح يخلط بيانات شيكبوينتين "
                "مختلفين تمامًا بنفس الجدول.",
                "",
                "**هاي النوتة** بتعيد نفس الفحص (نفس الكود، `run_accuracy_"
                "degradation_sweep`، بدون أي تعديل) بس على **الشيكبوينت "
                "النهائي المُعاد تدريبه** (`zeroshot_B1_neo_hookean_multires/"
                "model_best.pt`)، وبنفس مدى الدقات المستخدم بجدول الـCauchy "
                "(N=3 لـ49، مش لحد N=1401 -- ما في داعي للجزء الغالي).",
                "",
                "**الكلفة المتوقعة**: رخيصة جدًا -- كل نقطة هون N<=49، وأرخص "
                "بكثير من الـN=1401 الي كان يستهلك معظم وقت هاد الفحص قبل. "
                "بناءً على وقت نفس نوع العمل بنفس المدى (نوتة الـCauchy اليوم)، "
                "المتوقع **دقايق معدودة** للحساب الفعلي، وحوالي 10-15 دقيقة "
                "إجمالي بالنوتة (تحميل + تثبيت مكتبات + حساب).",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B1NH_FinalCheckpoint_LowN_Accuracy.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
