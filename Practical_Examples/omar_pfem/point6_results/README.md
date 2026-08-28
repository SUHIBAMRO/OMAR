# Point 6 results — where the OOD degradation actually comes from

B1 × Neo-Hookean, 19/19 cells, 10 samples each, on the study's own N=21 mesh,
scored with Tables 5/11's per-component relative L2 so the degradation column
reads directly against Table 11's factors.

In-distribution baseline: **0.0867**.

| k (σ) | loading | material | both | | loading | material | both |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.0869 | 0.1409 | 0.1427 | | 1.00× | 1.63× | 1.65× |
| 1.0 | 0.0888 | 0.2073 | 0.2066 | | 1.02× | 2.39× | 2.38× |
| 1.5 | 0.0897 | 0.2763 | 0.2682 | | 1.03× | 3.19× | 3.09× |
| 2.0 | 0.0871 | 0.3486 | 0.3294 | | 1.00× | 4.02× | 3.80× |
| 2.5 | 0.0862 | 0.4265 | 0.3957 | | 0.99× | 4.92× | 4.56× |
| 3.0 | 0.0927 | 0.5112 | 0.4601 | | 1.07× | 5.90× | 5.31× |

## The answer, and it is unambiguous

**The loading shift causes no degradation at all.** Across the entire sweep,
from 0 to 3σ, the error moves between 0.99× and 1.07× of the
in-distribution value — that is noise on ten samples, not a trend.

**The material shift causes all of it**, from 1.63× at half a sigma to 5.90×
at three. Table 11's single 4.11× figure for this case, measured with both
factors shifted at roughly 2–2.5σ, sits exactly inside this curve (3.80× at
k=2.0, 4.56× at k=2.5). The two studies agree; this one says which factor
was responsible.

**The damage accumulates smoothly.** Degradation increments per half sigma are
0.77, 0.80, 0.83, 0.90, 0.98 — slightly accelerating, with no threshold and no
cliff. There is no safe shift below which the operator is unaffected; there is
only a slope.

**The two shifts do not compound — they partially cancel.** "Both" is up to
10% *less* degraded than material alone at k=3.0. A plausible mechanism, not
tested here: raising E stiffens the body and shrinks the displacement, while
raising the load magnitude grows it, so the two move the solution's scale in
opposite directions. Stated as a hypothesis because nothing in this run
isolates it.

## Why loading extrapolates and stiffness does not

Worth saying, because it points straight at a mitigation. Displacement
depends on load magnitude close to *linearly* — double the traction and, in
the small-strain limit, the field roughly doubles. A network that has learned
that proportionality extrapolates it without difficulty, which is exactly what
the flat loading row shows.

Stiffness enters the other way: displacement scales roughly as 1/E. An inverse
relationship learned over a narrow band of E does not extrapolate — the
network has never seen the part of the curve it is being asked to evaluate.
That asymmetry, linear in load and inverse in stiffness, is the simplest
explanation consistent with these numbers.

## The mitigation this diagnosis suggests

Timon's round-6 wording was: *"If this diagnosis suggests a relatively
straightforward mitigation, such as changing the training range or
normalization, it would be useful to test it."*

It does, and it is the second of the two he named. Since the damage comes
entirely through E, and E is one of the network's four input channels, the
candidates in order of cost are:

1. **Normalize the E channel** by the training distribution's own mean and
   standard deviation, so the network sees a standardized quantity rather than
   a raw stiffness. Cheapest to test — one training run, no new FEM data.
2. **Train over a wider E range.** Requires regenerating the dataset, so it
   costs the full 5.7 h of FEM per case, and it does not fix extrapolation, it
   only moves the boundary.
3. **Predict a compliance-like scaled output** (e.g. u·E) so the inverse
   dependence is removed from what the network has to learn. A design change,
   not a knob.

Option 1 is the one worth testing, and this diagnosis is what justifies
spending a run on it rather than guessing.

## Scope

One case, B1 × Neo-Hookean. Poisson's ratio is deliberately not swept — the
field clips it to (0.2, 0.4), so a shifted mean saturates against the clip and
any curve drawn through it would be measuring the clip rather than the physics.
