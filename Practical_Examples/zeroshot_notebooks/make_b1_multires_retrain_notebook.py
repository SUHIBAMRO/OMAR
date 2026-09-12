"""Builds ONE notebook: retrain B1 x Neo-Hookean on a WIDER set of
resolutions, to address the real (corrected-checkpoint) finding that
accuracy degrades smoothly from ~7% (best, near the original training
resolutions 21/33) to ~34% by N=701 and further beyond.

Uses --fast_solver 1 (resolution_invariance_zeroshot.py, added 2026-09-12):
FEM ground truth for new training samples comes from solve_b1_fast_gpu
instead of the original CPU-only solve_hyperelastic_TL_spatial, which was
measured at 7.3 HOURS to generate N=21's own 500 samples alone -- making
any resolution beyond the original 21/33 impractical without this.
Verified bit-identical to the original solver at N=13 (relative L2 diff
0.0) before being trusted for this.

Writes to a NEW out_dir (zeroshot_B1_neo_hookean_multires), never touching
the existing zeroshot_B1_neo_hookean checkpoint -- that stays as the
baseline to compare against.
"""
import json
import os

BRANCH = "claude/claude-code-question-d307wp"
HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
OUT_NAME = "zeroshot_B1_neo_hookean_multires"

# Widening the original 21,33 with two resolutions inside the region the
# corrected sweep showed real, growing error (15.0% at 101, 22.3% at 201) --
# staged deliberately: prove this helps at a moderate cost before
# considering an even wider range (e.g. adding 401/701) in a follow-up.
TRAIN_RESOLUTIONS = "21,33,101,201"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines]}


def build():
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "name": "B1_NeoHookean_MultiRes_Retrain.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md("# إعادة تدريب B1 × Neo-Hookean على مدى أحجام أوسع",
               "",
               "**ليش هاد النوتبوك**: بعد ما صلحنا bug اختيار الـ checkpoint، طلعت",
               f"دقة الموديل الحقيقية: أفضل دقة (~7%) عند N=33-37 (نفس أحجام",
               "التدريب الأصلية)، وبتزيد بالتدريج كل ما ابتعدنا (15% عند N=101،",
               "22% عند N=201، 34% عند N=701). هاد مش عطل — هاد فجوة تعميم",
               "(generalization gap) طبيعية ومعروفة، وأفضل حل لها هو تدريب",
               "الموديل على مدى أحجام أوسع، مش بس N=21 و33.",
               "",
               "**الموديل الأصلي ما رح يتلمس** — هاد النوتبوك بيكتب لمجلد",
               f"جديد (`{OUT_NAME}`) عالـ Drive، فالـ checkpoint الأصلي يضل",
               "موجود نقارن معه.",
               "",
               "**السرعة**: بيستخدم `--fast_solver 1` (خاصية جديدة اليوم) —",
               "الحل الأصلي كان بياخد **7.3 ساعة** بس لتوليد عينات N=21! الحل",
               "الجديد (نفس دقة الحل تماماً، فرق صفر) بياخد أقل من ثانية للعينة.",
               "",
               "| خلية | شو بتعمل | الوقت المتوقع |",
               "|---|---|---|",
               "| 1 | تجهيز | دقيقة |",
               "| 2 | توليد بيانات FEM (4 أحجام: 21,33,101,201) | من نص ساعة",
               "    لساعتين تقريباً (الخلية نفسها بتطبع تقدير حي) |",
               "| 3 | التدريب | ساعة إلى بضع ساعات |",
               "| 4 | مقارنة مع الموديل الأصلي على نفس الأحجام | دقايق |",
               "",
               "**كل شي محفوظ على Drive أول بأول** — إذا وقف الاتصال، رجّع",
               "شغّل نفس الخلية وبتكمّل من وين وقفت."),

            md("## خلية 1 — التجهيز"),
            code(
                "from google.colab import drive",
                "drive.mount('/content/drive')",
                "",
                "import os, shutil, sys, subprocess",
                "os.chdir('/content')",
                "if os.path.exists('/content/OMAR'):",
                "    shutil.rmtree('/content/OMAR')",
                f"subprocess.run(['git', 'clone', '-q', '-b', '{BRANCH}',",
                "                'https://github.com/SUHIBAMRO/OMAR.git', '/content/OMAR'], check=True)",
                "",
                "WORK = '/content/OMAR/Practical_Examples'",
                "os.chdir(WORK); sys.path.insert(0, WORK)",
                "assert os.path.isdir(os.path.join(WORK, 'omar_pfem')), 'clone فشل'",
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',",
                "                'einops', 'timm', 'h5py', 'jax', 'tqdm'], check=True)",
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-sla'], check=True)",
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',",
                "                'nvmath-python[cu12]==0.9.0'], check=True)",
                "",
                "import torch",
                "print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())",
                "assert torch.cuda.is_available(), 'لازم GPU: Runtime > Change runtime type'",
                "print('GPU:', torch.cuda.get_device_name(0))",
                "",
                "from torch_sla.backends import is_cudss_available",
                "assert is_cudss_available(), 'cuDSS مش متوفر -- الحل السريع محتاجه'",
                "print('cuDSS متوفر -- الحل السريع جاهز')",
                "",
                f"OUT = '/content/drive/MyDrive/pfem_run/{OUT_NAME}'",
                "os.makedirs(OUT, exist_ok=True)",
                "print('OUT =', OUT)",
                "for f in sorted(os.listdir(OUT)):",
                "    print('  موجود:', f)",
            ),

            md("## خلية 2 — توليد بيانات FEM (بالحل السريع)",
               "",
               f"بتولّد 400 عينة تدريب + 100 تحقّق لكل حجم من الأربعة "
               f"({TRAIN_RESOLUTIONS}).",
               "",
               "**بتحفظ على Drive كل 25 عينة** — إذا وقف النوتبوك، رجّع شغّل",
               "نفس الخلية وبتكمّل من وين وقفت (`[resume] found X/400 ...`).",
               "",
               "الخلية بتطبع تقدير وقت حي (`~Y min left`) لكل حجم أثناء",
               "التوليد — راقبه لتعرف الوقت الحقيقي المتبقي."),
            code(
                "!python -m omar_pfem.resolution_invariance_zeroshot train \\",
                "    --geometry B1 --material neo_hookean \\",
                f"    --train_resolutions {TRAIN_RESOLUTIONS} \\",
                "    --n_train_per_res 400 --n_val_per_res 100 \\",
                "    --fast_solver 1 \\",
                "    --gen_chunk 25 --stop_after_generation \\",
                "    --out_dir \"$OUT\"",
            ),

            md("## خلية 3 — التدريب",
               "",
               "نفس بروتوكول التدريب الأصلي (2000 epoch كحد أقصى، "
               "early stopping بصبر 8، batch_size=8، lr=2e-3) — بس على",
               "أربع أحجام بدل حجمين. بيقرا البيانات من خلية 2 (ما بيعيد",
               "توليدها). بيحفظ حالته كل تحقّق، فلو انفصل رجّع شغّل الخلية."),
            code(
                "!python -m omar_pfem.resolution_invariance_zeroshot train \\",
                "    --geometry B1 --material neo_hookean \\",
                f"    --train_resolutions {TRAIN_RESOLUTIONS} \\",
                "    --n_train_per_res 400 --n_val_per_res 100 \\",
                "    --fast_solver 1 \\",
                "    --epochs 2000 --validate_every 25 --batch_size 8 \\",
                "    --early_stop_patience 8 --lr 2e-3 \\",
                "    --out_dir \"$OUT\"",
            ),

            md("## خلية 4 — مقارنة مباشرة: الموديل الجديد مقابل الأصلي",
               "",
               "بتشغّل نفس فحص الدقة الصارم (نفس المرجع الحقيقي المحلول لكل N،",
               "نفس الـ QoIs) على **الموديلين الاثنين** عند نفس الأحجام",
               "(13 لغاية 1401)، وتطبع جدول جنب بعض. هاد الجدول هو الدليل",
               "المباشر إذا كان التوسيع فعلاً حسّن الدقة."),
            code(
                "import json, os, sys",
                "sys.path.insert(0, '/content/OMAR/Practical_Examples')",
                "sys.path.insert(0, '/content/OMAR/Practical_Examples/report_builders')",
                "for m in list(sys.modules):",
                "    if m == 'omar_pfem' or m.startswith('omar_pfem.'):",
                "        del sys.modules[m]",
                "",
                "import torch, argparse",
                "from omar_pfem.measure_inference_latency import build_model",
                "from omar_pfem.no_accuracy_at_n1401 import run_accuracy_degradation_sweep",
                "from omar_pfem.resolve_b1_checkpoint import sha256_of",
                "",
                "device = torch.device('cuda')",
                "args = argparse.Namespace(",
                "    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,",
                "    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,",
                "    use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_out=2.0,",
                ")",
                "",
                "R = '/content/drive/MyDrive/pfem_run'",
                "NEW_CKPT = os.path.join(OUT, 'model_best.pt')",
                "OLD_CKPT = f'{R}/zeroshot_B1_neo_hookean/model_best.pt'",
                "assert os.path.exists(NEW_CKPT), f'new checkpoint not found: {NEW_CKPT}'",
                "assert os.path.exists(OLD_CKPT), f'old checkpoint not found: {OLD_CKPT}'",
                "",
                "RESOLUTIONS = [13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 101, 201, 401, 701, 1001, 1401]",
                "",
                "def run_one(ckpt, out_json):",
                "    model = build_model(args, device).to(torch.float32)",
                "    model.load_state_dict(torch.load(ckpt, map_location=device))",
                "    fp = sha256_of(ckpt)",
                "    return run_accuracy_degradation_sweep(model, args, RESOLUTIONS, out_json, device,",
                "                                           checkpoint_fingerprint=fp)",
                "",
                "print('=== OLD (original 21,33-only) checkpoint ===')",
                "old_rows = run_one(OLD_CKPT, f'{R}/no_accuracy_degradation_sweep.json')",
                "",
                "print('\\n=== NEW (multi-resolution) checkpoint ===')",
                "new_rows = run_one(NEW_CKPT, f'{OUT}/no_accuracy_degradation_sweep_multires.json')",
                "",
                "old_by_n = {r['N']: r for r in old_rows}",
                "new_by_n = {r['N']: r for r in new_rows}",
                "print('\\n' + '=' * 70)",
                "print(f\"{'N':<8}{'OLD disp_rel_L2':<18}{'NEW disp_rel_L2':<18}{'better?':<10}\")",
                "for N in RESOLUTIONS:",
                "    o, n = old_by_n.get(N), new_by_n.get(N)",
                "    if o and n:",
                "        oe, ne = o['fp32']['disp_rel_L2'], n['fp32']['disp_rel_L2']",
                "        print(f\"{N:<8}{oe:<18.4e}{ne:<18.4e}{'YES' if ne < oe else 'no':<10}\")",
            ),
        ],
    }


if __name__ == "__main__":
    nb = build()
    path = os.path.join(HERE, "B1_NeoHookean_MultiRes_Retrain.ipynb")
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)
