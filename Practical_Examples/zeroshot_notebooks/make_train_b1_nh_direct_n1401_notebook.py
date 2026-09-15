"""Builds ONE notebook: trains a NEW B1 x Neo-Hookean checkpoint DIRECTLY
at N=1401 (single resolution), as an ablation against the multi-res
(N=21,33,101,201 -> zero-shot N=1401) checkpoint. Timon round-11 point 4.
See cell_train_b1_nh_direct_n1401.py for the full rationale and the
reduced sample count (100+20, not 400+100) Omar chose given the real
per-sample N=1401 ground-truth cost.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_train_b1_nh_direct_n1401.py"


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
            "colab": {"provenance": [], "name": "B1_NeoHookean_Direct_N1401_Ablation.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# تدريب B1 × Neo-Hookean مباشرة على N=1401 (ablation)",
                "",
                "**طلب تيمون (round-11, نقطة 4)**: درّب موديل جديد مباشرة على N=1401 "
                "(مش zero-shot) كمقارنة (ablation) ضد موديل الـmulti-resolution "
                "(N=21,33,101,201 → zero-shot على N=1401). نتيجة الـzero-shot **بتضل "
                "منفصلة تماماً** -- هاي مش بديل إلها، هاي تجربة مقارنة إضافية.",
                "",
                "**عدد العينات المخفّض (100 تدريب + 20 تحقق، مش 400+100 متل الـmulti-res)** "
                "-- قرار Omar الصريح، بهدف تقليل التكلفة لدراسة مقارنة (مش النتيجة الرئيسية).",
                "",
                "**تصحيح حقيقي، 2026-09-14 (لقيناه بعد تشغيل حقيقي على GPU)**: التكلفة "
                "المقدّرة أصلاً (58.54 ثانية/عينة) كانت لحل دفعة-وحدة (`nsteps=1`)، "
                "يلي **ما بيتقارب** عند N=1401 مع حقول مواد عشوائية (اكتشاف سابق، "
                "2026-09-12). الرقم الحقيقي بـ`nsteps=10`: **587 ثانية/عينة**.",
                "",
                "**تسريع حقيقي لقيناه وطبّقناه، 2026-09-14 "
                "(`Test_FewerLoadSteps_N1401.ipynb`, تشغيل حقيقي على A100)**: "
                "كل خطوة من الـ10 كانت بتتقارب لدقة أعلى بكثير من المطلوب فعلاً -- "
                "جرّبنا `nsteps=5` و`nsteps=3` مباشرة (3 seeds لكل وحدة، فحص تقارب "
                "مستقل، مو تخمين): **الاثنين تقاربو بالكامل**، و`nsteps=3` طلعت "
                "أسرع بـ1.64x (358.1 ثانية/عينة مقابل 588.9 للمرجع) **وبدقة أفضل** "
                "(أسوأ residual: 4.459e-12 مقابل 3.859e-10). هاد النوتبوك هلق "
                "بيستخدم `--nsteps 3`. يعني 120 عينة بتاخد **~11.9 ساعة GPU** "
                "لتوليد البيانات (مش ~19.6 ساعة بـnsteps=10، ومش ~1.9 ساعة متل "
                "التقدير الأصلي الخاطئ). خلط عينات اتولدت بـnsteps=10 مع عينات "
                "جديدة بـnsteps=3 آمن -- بس بيغيّر طريقة الحل، مش الحل النهائي نفسه.",
                "",
                "**الخطوات**: (1) توليد بيانات FEM حقيقية عند N=1401 (الحل السريع)، "
                "(2) التدريب (نفس البروتوكول: 2000 epoch كحد أقصى، early stopping صبر 8، "
                "lr=2e-3)، (3) فحص الدقة الحقيقية عند N=1401 مقابل FEM حقيقي، "
                "(4) قياس سرعة الاستدلال، (5) مقارنة مباشرة مع أرقام الـmulti-res الموثقة "
                "أصلاً.",
                "",
                "**تصحيح حقيقي، 2026-09-15 (OOM حقيقي على GPU)**: `batch_size=8` "
                "(نفس قيمة الـmulti-res، كانت تمام لغاية N=201) فشلت فوراً بأول "
                "forward pass عند N=1401 -- `CUDA out of memory`. السبب: N=1401 "
                "معناه شبكة كاملة 1402×1402 ≈ **1.97 مليون عقدة لكل عينة** (مش "
                "1401 عقدة)، فتنسور واحد بحجم (batch=8, عقد=1.97M, hidden=256) "
                "لحاله محتاج ~15GB -- والتدريب (بعكس الاستدلال) لازم يحتفظ بكل "
                "التفعيلات لل-backward، فالمطلوب أكبر بكثير من الـ80GB المتوفرة.",
                "",
                "**تصحيح ثاني، نفس اليوم**: تصغير `batch_size` لـ1 **ما كفاش "
                "لحاله** -- طلع OOM تاني بس أعمق جوا الموديل (جوا MLP الطبقة "
                "الثانية)، لأنه حتى عينة وحدة لازم تحتفظ بتفعيلات كل الطبقات "
                "الأربع (n_layers=4) سوا لل-backward، وكل تنسور منها لحاله "
                "~4GB عند هاد الحجم. **الحل الحقيقي: gradient checkpointing** "
                "(مضاف لملف الموديل، اختياري تماماً، مفعّل بس هون عبر "
                "`--grad_checkpoint 1`) -- بيعيد حساب كل طبقة وقت الـbackward "
                "بدل ما يحتفظ فيها بالذاكرة، بنفس الـgradients بالضبط (اتفحص "
                "على CPU: فرق صفري تماماً بين تفعيل/تعطيل الخاصية) -- بس أبطأ "
                "شوي (~1.3-2x) بسبب إعادة الحساب.",
                "",
                "**تصحيح ثالث، نفس اليوم**: حتى مع الاثنين (batch_size=1 + "
                "grad_checkpoint)، طلع OOM تالت -- بس هالمرة وصل لآخر الـ"
                "forward pass وبلش فعلياً بـ`loss.backward()` قبل ما يفشل، "
                "طالب 7.49GB وعنده \"7.42GB حرة\" بس فشل -- توقيع كلاسيكي "
                "لتجزئة الذاكرة (fragmentation) مش نفاذ حقيقي. الحل: "
                "`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (نفس "
                "الحل يلي اقترحته رسالة الخطأ نفسها) -- إعداد لطريقة حجز "
                "الذاكرة بس، ما بأثر على أي نتيجة حسابية.",
                "",
                "**كل شي محفوظ على Drive تحت `zeroshot_B1_neo_hookean_direct_n1401/`** -- "
                "منفصل تماماً عن checkpoint الـmulti-res والـzero-shot الأصلي، ما بيلمسهم.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B1_NeoHookean_Direct_N1401_Ablation.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
