"""Builds ONE notebook: finer-resolution region-stress evaluation for
B3's best checkpoint (run 4, normalized inputs), to separate an
evaluation-resolution artifact from a genuine network accuracy gap.
See cell_b3_qois_finer.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_b3_qois_finer.py"


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
            "colab": {"provenance": [], "name": "B3_QoIs_Finer_Resolution.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# B3: تقييم إجهاد المنطقة على دقة أعلى (بدون إعادة تدريب)",
                "",
                "**السياق**: `region_cauchy_field_rel` (المقياس المحصّن من "
                "artifacts القياس) طلع mean=71.77%/median=68.45% على "
                "checkpoint_50000.pt عند **دقة الإنتاج** (6,840 عنصر، بس "
                "**6 نقاط Gauss** بمنطقة الأخدود). هاد المقياس صحّح كيف "
                "نقارن الحقلين، بس لسا بيقارنهم عند 6 نقاط بس -- ما بيجاوب "
                "سؤال: هل 6 نقاط كافية أصلاً لأخذ عينة من المنطقة؟",
                "",
                "**هاد الدفتر**: ياخد عيّنة فرعية من نفس الـ100 عينة "
                "held-out، **بلا إعادة تدريب**: (1) بيستكمل (interpolate) "
                "حقل E/nu الموجود أصلاً لكل عينة على شبكة أنعم بكتير "
                "(41×36×19 = 43,400 عنصر، **36 نقطة Gauss** بالمنطقة بدل "
                "6) -- نفس آلية الاستكمال (`_field_interpolator`) يلي "
                "المشروع مستخدمها أصلاً لمقارنة الدقات المختلفة (نفس "
                "العينة الفيزيائية، مش عينة عشوائية جديدة)؛ (2) بيحل FEM "
                "حقيقي جديد على الشبكة الأنعم بنفس المادة المستكملة ونفس "
                "phi؛ (3) بيستعلم نفس الشبكة المدرّبة (checkpoint_50000، "
                "بلا أي تغيير) zero-shot عند نقاط الشبكة الأنعم؛ (4) "
                "بيحسب `region_cauchy_field_rel` (نفس الدالة، بلا أي "
                "تعديل) عند منطقة الشبكة الأنعم.",
                "",
                "**لو الرقم الجديد نزل كتير عن 71.77%** -- جزء كبير من "
                "الفجوة كان بسبب قلة نقاط العينة عند دقة الإنتاج. **لو "
                "ظل قريب من 71.77%** -- الفجوة حقيقية ومستقلة عن دقة "
                "التقييم.",
                "",
                "**قيد معروف بالتصميم**: استكمال حقل GRF الخشن على شبكة "
                "أنعم ما بيرجع عينة GRF أدق فعليًا (مولّد GRF الطيفي "
                "بيربط الضجيج مباشرة بحجم الشبكة المستهدفة، فنفس الـseed "
                "بدقة مختلفة بيعطي حقل عشوائي مختلف تمامًا -- تأكدنا من "
                "هاد بقراءة `grf.py` مباشرة قبل ما نفترض إن الاستكمال هو "
                "الحل الصح). استكمال العينة الموجودة أصلاً هو الطريقة "
                "الصحيحة للحفاظ على نفس المسألة الفيزيائية وبس تنعيم "
                "الشبكة/التقييم -- بالضبط السؤال المطروح هون -- بس هاد "
                "يعني إنه هاد الاختبار بيفحص \"هل التقييم الأنعم بيغيّر "
                "الدقة المقاسة لنفس الحقل\"، مش \"هل التدريب على بيانات "
                "فيزيائية أدق فعليًا رح يغيّر تنبؤات الشبكة\".",
                "",
                "**اتفحص محليًا على CPU قبل هيك بثلاث خطوات حقيقية**: (1) "
                "فحص identity -- استكمال حقل على نفس شبكته الخشنة بيرجعه "
                "بالضبط (فرق = 0.0)؛ (2) فحص الحدود -- الاستكمال على "
                "شبكة أنعم بيخلي E/nu بحدودهم الفيزيائية الأصلية؛ (3) "
                "تشغيلة كاملة بحجم toy (بيانات صغيرة -> تدريب قصير -> "
                "إعادة حل على شبكة أنعم -> استعلام zero-shot -> "
                "`region_cauchy_field_rel`) بلا أي أخطاء.",
                "",
                "**التكلفة الحقيقية مش معروفة لسا على GPU حقيقي**: كل "
                "عينة من الـ`--n_samples` تحتاج حل FEM جديد فعليًا عند "
                "43,400 عنصر (بدل 6,840). لو الوقت طويل جدًا، وقف "
                "وشغّل بعدد عينات أقل (حتى 3-5 عينات بتعطي إشارة حقيقية، "
                "بس أضجّ شوي).",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B3_QoIs_Finer_Resolution.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
