# =====================================================================
#  BOOTSTRAP CELL — runs a round-6 cell straight from the repo
#
#  Why this exists. The other cells in this directory are pasted INTO a
#  Colab notebook, so the notebook holds its own copy of the code. Their
#  first few lines run `git reset --hard origin/<branch>`, which updates
#  the checkout at /content/OMAR -- but NOT the cell that is running. So
#  after a fix is pushed, re-running the cell re-runs the old code, with
#  an identical traceback, and it looks as though the fix did nothing.
#  That happened once with the dataset path in cell_ood_mitigation.py.
#
#  This cell has no logic of its own to go stale: it updates the repo and
#  then executes whichever cell_*.py you name, as it exists on the branch
#  right now. Paste THIS into Colab once; after that every pushed fix is
#  picked up by simply re-running it.
#
#  To switch tasks, change CELL below. The choices are the cell_*.py files
#  in Practical_Examples/zeroshot_notebooks/; a wrong name prints the whole
#  list rather than a bare assertion.
#
#  THIS IS THE ONLY WAY THESE CELLS SHOULD BE RUN. It has now cost three
#  runs: the Pareto cell printed "ALREADY DONE, will skip" under a commit
#  that says "COMPLETE (9/9)", the single-resolution cell recommended
#  hours of FEM that a later commit had already ruled out, and the B1
#  metric cell printed a verdict that the checked-out commit had
#  explicitly withdrawn -- each time with the NEW commit hash printed
#  directly above the OLD output.
# =====================================================================
CELL = 'cell_b1_metric_recheck.py'

import os
import subprocess
import sys

REPO = '/content/OMAR'
BRANCH = 'claude/claude-code-question-d307wp'


def run(cmd):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


from google.colab import drive
drive.mount('/content/drive')

if not os.path.isdir(REPO):
    run(['git', 'clone', '-b', BRANCH,
         'https://github.com/SUHIBAMRO/OMAR.git', REPO])
else:
    run(['git', '-C', REPO, 'fetch', 'origin', BRANCH])
    run(['git', '-C', REPO, 'checkout', BRANCH])
    run(['git', '-C', REPO, 'reset', '--hard', f'origin/{BRANCH}'])

# Print the commit actually being run. Without this there is no way to tell
# from the output whether a fix is present, which is the whole problem.
run(['git', '-C', REPO, 'log', '--oneline', '-1'])

PATH = f'{REPO}/Practical_Examples/zeroshot_notebooks/{CELL}'
if not os.path.exists(PATH):
    here = f'{REPO}/Practical_Examples/zeroshot_notebooks'
    have = sorted(f for f in os.listdir(here) if f.startswith('cell_'))
    raise SystemExit(f'{CELL} not found at\n  {PATH}\n\navailable:\n  '
                     + '\n  '.join(have))
print(f'\n--- running {CELL} from the commit above ---\n', flush=True)

# exec, not import: these files are top-level scripts with no __main__
# guard, and exec keeps their globals here so anything they define stays
# inspectable in the notebook afterwards.
exec(compile(open(PATH).read(), PATH, 'exec'), globals())
