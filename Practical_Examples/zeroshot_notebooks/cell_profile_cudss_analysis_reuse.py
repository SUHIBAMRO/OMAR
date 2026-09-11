# =====================================================================
#  CELL -- QUICK DIAGNOSTIC (2026-09-11), not a production run: is
#  cuDSS's own ANALYSIS phase (fill-reducing reordering, depends only on
#  the matrix's sparsity PATTERN, not its values) a big enough share of
#  one direct solve's cost to be worth reusing across Newton iterations?
#
#  WHY THIS MATTERS: reading torch_sla's own installed source directly
#  showed that every single Newton iteration of solve_assembled_direct
#  currently pays for a brand-new cuDSS handle + ANALYSIS + FACTORIZATION
#  + SOLVE, even though the sparsity PATTERN (which (row,col) entries are
#  nonzero) never changes within one Newton solve -- only the VALUES do.
#  Confirmed on CPU already (before spending any GPU time): the CSR
#  structure from build_sparse_jac_fn really is byte-identical across
#  different displacement fields (only values differ) -- so the
#  assumption this diagnostic depends on is verified, not just plausible.
#
#  WHAT'S NOT YET KNOWN, and what this cell actually measures:
#  1. How big a fraction of ONE full solve ANALYSIS actually is (could be
#     small, in which case none of this is worth building further).
#  2. Whether cuDSS's own API even supports "reuse ANALYSIS, refactorize
#     a matrix with new values on the SAME handle/descriptor" the way
#     the diagnostic assumes -- checked via a real correctness comparison
#     against the current (always-redo-everything) behavior, not assumed.
#
#  THIS CELL DOES NOT CHANGE solve_assembled_direct OR ANY OTHER SOLVER.
#  It is read-only diagnostics: if the idea doesn't pan out (correctness
#  fails, or ANALYSIS turns out to be a tiny fraction of the cost), no
#  further engineering is warranted and this stays exactly what it is --
#  a quick, cheap check before committing real effort.
# =====================================================================
import os
import subprocess
import sys


def run(cmd):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
    p.wait()
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


REPO = '/content/OMAR'
if not os.path.isdir(REPO):
    run(['git', 'clone', '-b', 'claude/claude-code-question-d307wp',
         'https://github.com/SUHIBAMRO/OMAR.git', REPO])
else:
    run(['git', '-C', REPO, 'fetch', 'origin', 'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'checkout', 'claude/claude-code-question-d307wp'])
    run(['git', '-C', REPO, 'reset', '--hard', 'origin/claude/claude-code-question-d307wp'])

run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-sla'])
run([sys.executable, '-m', 'pip', 'install', '-q', 'nvmath-python[cu12]==0.9.0'])

WORK = f'{REPO}/Practical_Examples'
os.chdir(WORK)
sys.path.insert(0, WORK)

for _mod_name in list(sys.modules):
    if _mod_name == 'omar_pfem' or _mod_name.startswith('omar_pfem.'):
        del sys.modules[_mod_name]

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else 'NONE -- this diagnostic needs CUDA (cuDSS has no CPU path)')
if not torch.cuda.is_available():
    raise RuntimeError('No CUDA device -- cuDSS diagnostics cannot run on CPU.')

from torch_sla.backends import is_cudss_available
if not is_cudss_available():
    raise RuntimeError(
        "cuDSS is NOT available after installing nvmath-python[cu12] -- check "
        "the pip install output above.")
print('cuDSS is available.')

# Small N first -- cheap, fast, and the fraction ANALYSIS takes shouldn't
# depend much on N (it's a share of one solve's own cost either way), so
# there is no reason to spend minutes on a large N just to see this.
run([sys.executable, '-u', '-m', 'omar_pfem.profile_cudss_analysis_reuse', '401'])
