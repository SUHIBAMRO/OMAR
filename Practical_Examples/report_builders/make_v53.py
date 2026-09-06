"""v52 -> v53. Timon's round-7 reply (2026-09-06) gave the actual
reference for the continual-learning paper he'd mentioned in round 6
without a title or arXiv ID: https://arxiv.org/abs/2605.04832. Fetched
the real title/authors/abstract from arXiv directly rather than
guessing -- "Replay-Based Continual Learning for Physics-Informed
Neural Operators" (Wang, Eshaghi, Zhuang, Rabczuk, Liu), which uses the
same Transolver architecture this report does, with a replay/distillation
strategy against catastrophic forgetting when adapting to new data.

Adds one sentence at the end of Section 8.6 (right after the OOD
mitigation diagnosis concludes, where Timon originally suggested it:
"Otherwise, we would leave it for future research. You can actually
mention our paper on continual learning in this context if you wish"),
framed accurately as a candidate for INCREMENTAL adaptation to new
distributions, distinct from this report's own zero-shot (no retraining
at all) approach -- not a claim that it solves the same problem.
Reference [5] added.
"""
from docx import Document
import copy

SRC, DST = 'PFEM_Transolver_Report_v52.docx', 'PFEM_Transolver_Report_v53.docx'

doc = Document(SRC)
ORIGINAL = list(doc.paragraphs)


def find_para(prefix):
    hits = [p for p in ORIGINAL if p.text.strip().startswith(prefix)]
    assert len(hits) == 1, (
        f'{len(hits)} paragraphs start with {prefix!r}; expected exactly 1')
    return hits[0]


def insert_after(prefix, text):
    p = find_para(prefix)
    new_p = copy.deepcopy(p._p)
    p._p.addnext(new_p)
    from docx.text.paragraph import Paragraph
    np = Paragraph(new_p, p._parent)
    for r in list(np.runs):
        r._r.getparent().remove(r._r)
    np.add_run(text)
    return np


insert_after(
    'This diagnosis covers B1 × Neo-Hookean. Whether the same '
    'attribution holds',
    'Since normalization is not a clean fix and the underlying '
    'sensitivity to distribution shift remains, a mitigation that '
    'accepts some retraining rather than insisting on a single fixed '
    'zero-shot model is a candidate direction for future work: '
    'continual learning, which adapts a trained operator to new data '
    'distributions incrementally while limiting forgetting of what it '
    'already learned. Wang et al. [5] apply exactly this — a '
    'replay-and-distillation strategy — to physics-informed operators '
    'built on the same Transolver architecture used throughout this '
    'report, without labeled data. This is a different approach from '
    'the zero-shot generalization studied above (no retraining at all) '
    'and would only apply if some retraining budget were acceptable.'
)

insert_after(
    '[4] Eshaghi, M.S., Anitescu, C., Thombre, M., Wang, Y., Zhuang, X., '
    'and Rabczuk, T. “Variational Physics-informed',
    '[5] Wang, Y., Eshaghi, M.S., Zhuang, X., Rabczuk, T., and Liu, Y. '
    '“Replay-Based Continual Learning for Physics-Informed Neural '
    'Operators.” arXiv:2605.04832, 2026.'
)

doc.save(DST)
print('wrote', DST)
