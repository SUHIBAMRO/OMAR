"""Builds TWO notebooks: retrain B2 x Neo-Hookean and B2 x Mooney-Rivlin
on the SAME wider set of resolutions (21,33,101,201) that fixed B1's own
N=1401 accuracy for all three materials -- see
make_b1_multires_retrain_notebook.py (Neo-Hookean) and
make_b1_mr_ab_multires_retrain_notebooks.py (Mooney-Rivlin/Arruda-Boyce),
the originals this mirrors.

WHY THIS EXISTS: the resolution-matched break-even sweep for all 6 cases
(2026-09-14, resolution_matched_break_even_all_cases.json) could compute a
real break-even only for B1 x Neo-Hookean -- its own checkpoint is the
ONLY one with a recorded training wall-clock (41881.28s, from the
already-completed multi-res retrain). B2 x Neo-Hookean and B2 x
Mooney-Rivlin are STILL on their ORIGINAL zero-shot checkpoints (trained
at N=21,33 only, same as B1's originals before its own retrain) -- these
predate write_manifest/metrics_history.json being added to the training
script, so their training cost is genuinely lost, not merely unmeasured.
Omar's own explicit choice once this gap was found: retrain both under
the same multi-res protocol rather than leave the break-even permanently
"unknown" for these two cases.

(B2 x Arruda-Boyce is NOT included here -- it currently fails outright at
N=1401 with a real torch-fem OOM inside its own Hessian computation,
being fixed separately via a chunked HyperelasticPlaneStrain variant in
torchfem_comparison.py. Once that fix is verified on real GPU, whether
B2 x Arruda-Boyce ALSO needs this same multi-res retrain is a separate,
later decision -- not assumed here.)

Identical protocol to the B1 notebooks in every respect except
--geometry B2, the old checkpoint's "_fixedsel" suffix (B2's own
established naming convention for its Round-6-corrected checkpoints,
see cell_n1401_b1_other_materials.py), and using the B2-specific
accuracy-sweep function (run_accuracy_degradation_sweep_b2) in the final
comparison cell -- deliberately not a different recipe otherwise, so the
comparison to B1's own multi-res results stays apples-to-apples.
"""
import json
import os

BRANCH = "claude/claude-code-question-d307wp"
HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
TRAIN_RESOLUTIONS = "21,33,101,201"

MATERIAL_INFO = {
    "neo_hookean": {
        "arabic_name": "Neo-Hookean",
        "notebook_name": "B2_NeoHookean_MultiRes_Retrain.ipynb",
    },
    "mooney_rivlin": {
        "arabic_name": "Mooney-Rivlin",
        "notebook_name": "B2_MooneyRivlin_MultiRes_Retrain.ipynb",
    },
}


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines]}


def build(material):
    info = MATERIAL_INFO[material]
    out_name = f"zeroshot_B2_{material}_multires"
    old_ckpt_dir = f"zeroshot_B2_{material}_fixedsel"

    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "name": info["notebook_name"]},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(f"# إعادة تدريب B2 × {info['arabic_name']} على مدى أحجام أوسع",
               "",
               "**ليش هاد النوتبوك**: الـcheckpoint الأصلي لهاي الحالة متدرب "
               f"بس على N=21,33 (نفس أحجام B1 الأصلية قبل إعادة تدريبها)، "
               "ما عنده سجل وقت تدريب (`metrics_history.json`) لأنه أقدم من "
               "إضافة هاد التسجيل للسكربت -- يعني break-even لهاي الحالة "
               "\"غير معروف\" بشكل دائم ما لم نعيد التدريب بنفسنا. نفس الحل "
               f"المستخدم لكل مواد B1 الثلاث (توسيع أحجام التدريب لـ"
               f"{TRAIN_RESOLUTIONS}) -- نفس البروتوكول بالضبط، بدون أي فرق، "
               "عشان تبقى المقارنة مع نتائج B1 عادلة.",
               "",
               "**الموديل الأصلي ما رح يتلمس** — هاد النوتبوك بيكتب لمجلد",
               f"جديد (`{out_name}`) عالـ Drive، فالـ checkpoint الأصلي يضل",
               "موجود نقارن معه.",
               "",
               "**السرعة**: بيستخدم `--fast_solver 1` (نفس الحل السريع المستخدم "
               "لكل الحالات الأخرى، تحقق منه لـB2 بكل المواد الثلاث).",
               "",
               "| خلية | شو بتعمل | الوقت المتوقع |",
               "|---|---|---|",
               "| 1 | تجهيز | دقيقة |",
               f"| 2 | توليد بيانات FEM (4 أحجام: {TRAIN_RESOLUTIONS}) | من نص ساعة",
               "    لساعتين تقريباً (الخلية نفسها بتطبع تقدير حي) |",
               "| 3 | التدريب | ساعة إلى بضع ساعات (نفس B1 تقريباً، ~11.6 ساعة) |",
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
                "# نفس الإصلاح الدفاعي المطبق بكل نوتبوكات هالمشروع (2026-09-14):",
                "# JAX بيحجز ~90% من ذاكرة الـGPU كلها أول ما يلمسها، بشكل غير",
                "# مرئي لإحصائيات torch.cuda الخاصة -- فرضه هون قبل أي استيراد",
                "# آخر (بما فيها خلايا !python -m ... تحت، يلي بترث نفس البيئة).",
                "os.environ['JAX_PLATFORMS'] = 'cpu'",
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
                "import jax",
                "assert all(str(d.platform) == 'cpu' for d in jax.devices()), \\",
                "    f'JAX NOT forced to CPU: {jax.devices()}'",
                "print('JAX devices:', jax.devices())",
                "",
                f"OUT = '/content/drive/MyDrive/pfem_run/{out_name}'",
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
               "نفس الخلية وبتكمّل من وين وقفت."),
            code(
                "!python -m omar_pfem.resolution_invariance_zeroshot train \\",
                f"    --geometry B2 --material {material} \\",
                f"    --train_resolutions {TRAIN_RESOLUTIONS} \\",
                "    --n_train_per_res 400 --n_val_per_res 100 \\",
                "    --fast_solver 1 \\",
                "    --gen_chunk 25 --stop_after_generation \\",
                "    --out_dir \"$OUT\"",
            ),

            md("## خلية 3 — التدريب",
               "",
               "نفس بروتوكول B1 بالضبط (2000 epoch كحد أقصى، "
               "early stopping بصبر 8، batch_size=8، lr=2e-3) — بس على",
               "أربع أحجام بدل حجمين. B2 بتحتاج loss_force_norm تلقائياً",
               "(الكود بيفعّلها لوحده لما geometry=B2، مش شي إضافي هون)."),
            code(
                "!python -m omar_pfem.resolution_invariance_zeroshot train \\",
                f"    --geometry B2 --material {material} \\",
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
               "(13 لغاية 1401)، وتطبع جدول جنب بعض."),
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
                "from omar_pfem.no_accuracy_at_n1401 import run_accuracy_degradation_sweep_b2",
                "from omar_pfem.resolve_b1_checkpoint import sha256_of",
                "",
                "device = torch.device('cuda')",
                "args = argparse.Namespace(",
                "    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,",
                "    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,",
                "    use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_out=2.0,",
                ")",
                f"MATERIAL = '{material}'",
                "",
                "R = '/content/drive/MyDrive/pfem_run'",
                "NEW_CKPT = os.path.join(OUT, 'model_best.pt')",
                f"OLD_CKPT = f'{{R}}/{old_ckpt_dir}/model_best.pt'",
                "assert os.path.exists(NEW_CKPT), f'new checkpoint not found: {NEW_CKPT}'",
                "assert os.path.exists(OLD_CKPT), f'old checkpoint not found: {OLD_CKPT}'",
                "",
                "RESOLUTIONS = [13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 101, 201, 401, 701, 1001, 1401]",
                "",
                "def run_one(ckpt, out_json):",
                "    model = build_model(args, device).to(torch.float32)",
                "    model.load_state_dict(torch.load(ckpt, map_location=device))",
                "    fp = sha256_of(ckpt)",
                "    return run_accuracy_degradation_sweep_b2(model, args, RESOLUTIONS, out_json, device,",
                "                                              material=MATERIAL, checkpoint_fingerprint=fp)",
                "",
                "print('=== OLD (original 21,33-only) checkpoint ===')",
                f"old_rows = run_one(OLD_CKPT, f'{{R}}/no_accuracy_degradation_sweep_b2_{material}.json')",
                "",
                "print('\\n=== NEW (multi-resolution) checkpoint ===')",
                f"new_rows = run_one(NEW_CKPT, f'{{OUT}}/no_accuracy_degradation_sweep_b2_{material}_multires.json')",
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
    for material in MATERIAL_INFO:
        nb = build(material)
        path = os.path.join(HERE, MATERIAL_INFO[material]["notebook_name"])
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        print("wrote", path)
