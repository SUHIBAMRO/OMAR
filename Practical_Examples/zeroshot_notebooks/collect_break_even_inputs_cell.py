# ============================================================
# جمع المدخلات الحقيقية لنقطة 3 (التسريع ونقطة التعادل)
# ============================================================
# شغّله بـRound5 بعد خلية 1.
#
# بيعمل إشيين:
#   1. بيحسب التسريع بأحجام دفعات متطابقة — هاد ما بده زمن تدريب،
#      فبيطلع رقم نهائي دقيق لكل حالة عندها الملفين.
#   2. بيطبع محتوى ملفات train.log عشان نطلع منها زمن التدريب
#      الحقيقي لكل حالة — بدونه ما في نقطة تعادل، وما بدنا نخمّنها.
import os
import glob
import json
import subprocess
import sys

R = '/content/drive/MyDrive/pfem_run'
WORK = '/content/OMAR/Practical_Examples'
BRANCH = 'claude/claude-code-question-d307wp'

# The clone may predate the scripts this cell calls, which fails as a bare
# "No module named ...". Pull first so it cannot.
subprocess.run(['git', '-C', '/content/OMAR', 'fetch', '--quiet', 'origin', BRANCH], check=True)
subprocess.run(['git', '-C', '/content/OMAR', 'reset', '--hard', '--quiet',
                f'origin/{BRANCH}'], check=True)
os.chdir(WORK)
print('code at:', subprocess.run(['git', '-C', '/content/OMAR', 'rev-parse', '--short', 'HEAD'],
                                 capture_output=True, text=True).stdout.strip())

print('=' * 74)
print('1. ملفات توقيت GPU FEM')
print('=' * 74)
fem_files = sorted(glob.glob(f'{R}/**/*timing*.json', recursive=True))
for f in fem_files:
    try:
        d = json.load(open(f))
        rows = d.get('rows', [])
        tag = f"{d.get('geometry')}_{d.get('material')}"
        per = ', '.join(f"bs{r['batch_size']}={r['per_sample_ms']:.1f}" for r in rows)
        print(f"  {tag:22s} N={d.get('N')}  {per}")
        print(f"  {'':22s} {f}")
    except Exception as e:
        print(f"  [تعذّر قراءته] {f}: {e}")
if not fem_files:
    print('  ما في ولا ملف')

print()
print('=' * 74)
print('2. ملفات زمن استدلال Transolver')
print('=' * 74)
nn_files = sorted(glob.glob(f'{R}/inference_latency_by_batch_*.json'))
for f in nn_files:
    d = json.load(open(f))
    per = ', '.join(f"bs{r['batch_size']}={r['per_sample_ms']:.4f}" for r in d['rows'])
    print(f"  {d['geometry']}_{d['material']:15s} {per}")
if not nn_files:
    print('  ما في ولا ملف — شغّل خلية نقطة 4 الأول')

print()
print('=' * 74)
print('3. التسريع بأحجام دفعات متطابقة (رقم نهائي — ما بده زمن تدريب)')
print('=' * 74)
# Pair the two benchmarks by case. Speed-up needs only these two files, so
# it is exact right now; break-even is deliberately left out until the
# real training cost is known rather than assumed.
by_case_nn = {}
for f in nn_files:
    d = json.load(open(f))
    by_case_nn[f"{d['geometry']}_{d['material']}"] = f
by_case_fem = {}
for f in fem_files:
    try:
        d = json.load(open(f))
        c = f"{d.get('geometry')}_{d.get('material')}"
        # Two GPU timing sets coexist on Drive with slightly different
        # numbers. PROJECT_STATUS records that the report's Table 10 uses
        # the per-case gpu_fem_solver/ files, so prefer those and say so,
        # rather than letting glob order decide which one a result uses.
        if c in by_case_fem:
            keep = f if 'gpu_fem_solver' in f else by_case_fem[c]
            drop = by_case_fem[c] if keep == f else f
            print(f"  [مكرر] {c}: باخد {keep}")
            print(f"  {'':9} وبتجاهل {drop}")
            by_case_fem[c] = keep
        else:
            by_case_fem[c] = f
    except Exception:
        pass

paired = sorted(set(by_case_nn) & set(by_case_fem))
print(f"  حالات عندها الملفين: {paired if paired else 'ولا وحدة'}")
for c in paired:
    print(f"\n----- {c} -----")
    r = subprocess.run([sys.executable, '-m', 'omar_pfem.break_even_analysis',
                        '--fem_json', by_case_fem[c], '--nn_json', by_case_nn[c],
                        '--out_json', f'{R}/break_even_{c}.json'],
                       capture_output=True, text=True)
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print(r.stderr[-1500:])

print()
print('=' * 74)
print('4. زمن التدريب — من train.log (هاد الي ناقصنا)')
print('=' * 74)
# B1 has no train.log at the path B2 uses, so search by CONTENT: any text
# file carrying a "Total wall clock" line is a training log whatever it is
# called. Without this the B1 break-even stays unknown.
cands = set(glob.glob(f'{R}/**/train.log', recursive=True))
for pat in ('*.log', '*.txt', '*.out', 'log*', '*history*'):
    cands |= set(glob.glob(f'{R}/**/{pat}', recursive=True))
logs = []
for f in sorted(cands):
    try:
        if os.path.getsize(f) > 200_000_000:
            continue
        if 'Total wall clock' in open(f, errors='replace').read():
            logs.append(f)
    except Exception:
        pass
print(f'{len(logs)} ملف فيه سطر "Total wall clock"\n')
for f in logs:
    print(f'----- {f} -----')
    try:
        lines = [l.rstrip() for l in open(f, errors='replace').read().split('\n') if l.strip()]
        # whatever line carries a total/elapsed figure is what we need; print
        # the candidates plus the tail, since the format is not known yet
        for l in lines:
            if 'Total wall clock' in l:
                print('   >>>', l.strip()[:170])
    except Exception as e:
        print('   تعذّر:', e)
    print()

print('=' * 74)
print('ابعت كل المخرجات لكلود.')
print('=' * 74)
