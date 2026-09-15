"""Builds ONE small diagnostic notebook: does TF32 matmul precision
(torch.set_float32_matmul_precision('high')) help the TRAINING loop, not
just inference (already confirmed there: 4.67x alone, N=1401)? See
cell_test_tf32_training.py for the full rationale and decision rule.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_test_tf32_training.py"


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
            "colab": {"provenance": [], "name": "Test_TF32_Training.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# فحص سريع: هل TF32 بيسرّع التدريب نفسه، مش بس الاستدلال؟",
                "",
                "**ليش هاد النوتبوك**: أكدنا قبل هيك إنه `torch.set_float32_matmul_"
                "precision('high')` بيسرّع **الاستدلال** حقيقي (4.67x لحاله عند "
                "N=1401، متحقق منه بمقارنة المخرجات مع مرجع fp32 دقيق) -- بس ما "
                "جربناه على **حلقة التدريب** نفسها. فرق مهم: خطوة استدلال وحدة "
                "فحص أضعف بكثير من مئات خطوات Adam، ووين ممكن فروق عددية صغيرة "
                "بكل خطوة تتراكم على مسار مختلف بعد وقت.",
                "",
                "**آمن تماماً**: بيقرأ بيانات جاهزة من مهمة **خلصت خلاص** "
                "(`zeroshot_B1_neo_hookean_multires/samples_cache.pt`) بشكل "
                "read-only بس -- ما بكتب ولا بيلمس أي نوتبوك تاني شغال. بيدرب "
                "200 خطوة حقيقية بس عند N=21 (أصغر وأرخص حجم موجود بالكاش) --  "
                "المفروض يخلص بدقايق.",
                "",
                "**كيف المقارنة عادلة**: نفس الـseed (torch/numpy/random) قبل كل "
                "تشغيلة مباشرة -- نفس تهيئة الموديل، نفس ترتيب الدُفعات، نفس "
                "أقنعة dropout -- الفرق الوحيد هو TF32 بس.",
                "",
                "**قاعدة القرار**: TF32 يُعتبر **آمن للتدريب** بس إذا (أ) ما "
                "طلع أي NaN/Inf بأي تشغيلة، و(ب) مسار الخسارة (loss) عبر "
                "الخطوات كله متقارب مع النسخة الأصلية (متوسط الفرق النسبي "
                "بكل خطوة أقل من 5%) -- مش بس أول كم خطوة.",
                "",
                "**إذا طلعت النتيجة 'SAFE'**: نضيف سطر واحد "
                "(`torch.set_float32_matmul_precision('high')`) لبداية التدريب "
                "الحقيقي، نفس الطريقة المستخدمة أصلاً للاستدلال.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    path = f"{HERE}/Test_TF32_Training.ipynb"
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)
