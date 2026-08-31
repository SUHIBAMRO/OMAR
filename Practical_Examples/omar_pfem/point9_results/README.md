# Point 9 — method of manufactured solutions (neo_hookean)

Timon's point 9, and his "this is the last thing to do". His design: *"compare Q4, Q9 and the physics-informed Transolver against exactly the same analytical solution in L2, H1 and energy norms and also examine stress errors"*.

**All three legs are measured.** Q4 and Q9 are below; the physics-informed operator is at the bottom of this file. The operator needed a body-force term in the energy functional and a body-force input channel, neither of which the report's trained checkpoints have, so it is a separately trained model (`omar_pfem/mms_operator.py`) — not a Table 5 checkpoint.

## The manufactured solution

```
u* = 0.05*(sin(pi x)sin(pi y), 0.7*sin(pi x)sin(pi y)); vanishes on the boundary
```

* **Body force**: b = -Div P(F*), by nested autodiff, checked against a central finite difference.

* **Boundary conditions**: homogeneous Dirichlet on all four edges, exact for this u*.

* **Material**: uniform E=1000, nu=0.3, plane strain.

* **Precision**: float64 on cpu.


### Why a body force, and not a body-force-free exact solution

Timon left this fork open and it had to be settled to write any code. A body-force-free exact solution of finite-strain elasticity on this domain is, in practice, a homogeneous deformation — a constant deformation gradient — which a bilinear Q4 element reproduces to machine precision. The study would measure round-off, both orders would "converge" instantly, and it would distinguish nothing. Manufacturing the solution keeps the geometry, material and discretization exactly as they are everywhere else in the report. **This is a decision made here, not one Timon confirmed**, and it is the first thing to raise if he wants the study shaped differently.


## Results

| order | N | DOF | L2 | H1 semi | stress | energy |
|---|---|---|---|---|---|---|
| Q4 | 5 | 50 | 5.266e-02 | 2.252e-01 | 2.270e-01 | 4.959e-02 |
| Q4 | 9 | 162 | 1.351e-02 | 1.132e-01 | 1.143e-01 | 1.268e-02 |
| Q4 | 17 | 578 | 3.403e-03 | 5.666e-02 | 5.724e-02 | 3.191e-03 |
| Q9 | 5 | 162 | 3.373e-03 | 2.314e-02 | 2.355e-02 | 5.246e-04 |
| Q9 | 9 | 578 | 4.155e-04 | 5.763e-03 | 5.957e-03 | 3.327e-05 |
| Q9 | 17 | 2,178 | 5.163e-05 | 1.438e-03 | 1.495e-03 | 2.088e-06 |

## Convergence rates — this is what validates the study

An MMS study is self-validating: if the body force were wrong by a sign or a factor, the discrete solution would converge to the wrong function and these rates would collapse. They do not.


**Q4**

| norm | observed | theory | pairwise | |
|---|---|---|---|---|
| L2 | 1.98 | 2 | 1.96, 1.99 | OK |
| H1_semi | 1.00 | 1 | 0.99, 1.00 | OK |
| stress | 0.99 | 1 | 0.99, 1.00 | OK |
| energy | 1.98 | 2 | 1.97, 1.99 | OK |

**Q9**

| norm | observed | theory | pairwise | |
|---|---|---|---|---|
| L2 | 3.01 | 3 | 3.02, 3.01 | OK |
| H1_semi | 2.00 | 2 | 2.01, 2.00 | OK |
| stress | 1.99 | 2 | 1.98, 1.99 | OK |
| energy | 3.99 | 4 | 3.98, 3.99 | OK |

* **Q4: as expected**
* **Q9: as expected**

## Q4 vs Q9 at equal cost

Meshes where both orders have the same number of degrees of freedom, which is the only fair way to compare them:

| DOF | Q4 L2 | Q9 L2 | Q9 advantage | Q4 H1 | Q9 H1 | Q9 advantage |
|---|---|---|---|---|---|---|
| 162 | 1.351e-02 | 3.373e-03 | **4.0×** | 1.132e-01 | 2.314e-02 | **4.9×** |
| 578 | 3.403e-03 | 4.155e-04 | **8.2×** | 5.666e-02 | 5.763e-03 | **9.8×** |

## What the error columns mean


## Solver settings, and why they do not affect the numbers

Newton tol None, CG tol None, None load steps.



A caveat about the shared solver, recorded in PROJECT_STATUS: its CG stops on `||r||/||b|| < cg_tol`, and on the last Newton iteration of each load step `b` is the already-converged residual, so the target becomes unreachable and CG runs to its iteration cap. Harmless here — Newton has already converged and the cap is set proportional to the problem size — but it inflates the wall-clock column.


---


## The operator third — the three-way, complete (N=17)

Before any of this was trained, `omar_pfem/test_mms_operator.py` established that the network is minimizing the same thing the FEM solver solves: Π(u_FEM) = -7.999050 is a true minimum of the operator's functional — the interpolated u\* does not beat it, all 36 admissible perturbations raise Π, and the excess grows quadratically (ratio 4.000, 4.0 expected). The meta-check: a W wrongly divided by 8 moves the minimum to scale 0.125, so those checks can fail.

| method | L2 | H1 semi | stress | energy |
|---|---|---|---|---|
| Q4 (same mesh) | 3.403e-03 | 5.666e-02 | 5.724e-02 | 3.191e-03 |
| Q9 (same N) | 5.163e-05 | 1.438e-03 | 1.495e-03 | 2.088e-06 |
| operator | 8.238e-03 | 5.831e-02 | 5.886e-02 | 9.914e-03 |

**operator / Q4 in L2 = 2.42×.** Above 1.0. The ceiling constrains Π, not L2: the Q4 solution minimizes Π over this space, so nothing in it reaches a lower Π, but L2 error against u\* is a different functional. A field that does not minimize Π can sit closer to u\* in L2 by partially cancelling Q4's own discretization bias, and the three-mesh sweep saw exactly that at N=9 (0.37×). The norms that stayed above 1.0 at every mesh are H1 semi and stress.

The four ratios are **not** the same number: L2 2.42×, H1 semi 1.03×, stress 1.03×, energy 3.11×. 
The gradient-based norms are the ones the operator has essentially closed — 1.03× in H1 and 1.03× in stress means it recovers the strain and stress fields about as well as the Q4 optimum it is chasing — while it is 2.42× behind in L2 and 3.11× in energy. That is the **opposite of the usual ordering**, where L2 is the forgiving norm and the derivative norms are the strict ones. The energy functional is what is being minimized, and it is built from the deformation gradient, so the quantities it sees directly are the ones that come out closest; the displacement itself is only pinned down through them, up to what the boundary mask fixes.


**Is this the budget or the method?** Best test L2 was 1.429e-02 at the halfway point and 8.826e-03 at the end, an improvement of 38% over the second half of training — still falling, but slowly. The last twenty validations span 8.826e-03 to 1.055e-01, a factor of 12, so single-epoch scores are noisy and the reported number is the best checkpoint (epoch 1900), not the last one. A longer run would close some of the remaining L2 gap; nothing here shows how much.

Training cost: 16,000 optimizer steps, 8.2 min on an NVIDIA A100-SXM4-80GB, and **no labels** — u\* is analytic but is never used in training, only in scoring.


---


## The operator third — a DEMONSTRATION run only, not a result

`operator_demo_N9_undertrained.json`. **Do not quote these numbers in the report.** It is a 1,600-optimizer-step CPU run at N=9, kept only because it proves the pipeline end to end and because its ceiling check passed. For scale, the report's own physics-informed models were trained for 75,000 steps.

| method | L2 | H1 semi | stress | energy |
|---|---|---|---|---|
| Q4 (same mesh) | 1.351e-02 | 1.132e-01 | 1.143e-01 | 1.268e-02 |
| Q9 (same N) | 4.155e-04 | 5.763e-03 | 5.957e-03 | 3.327e-05 |
| operator (undertrained) | 6.366e-02 | 1.525e-01 | 1.569e-01 | 8.877e-02 |

**operator / Q4 in L2 = 4.71×.** Above 1.0. The ceiling constrains Π, not L2: the Q4 solution minimizes Π over this space, so nothing in it reaches a lower Π, but L2 error against u\* is a different functional. A field that does not minimize Π can sit closer to u\* in L2 by partially cancelling Q4's own discretization bias, and the three-mesh sweep saw exactly that at N=9 (0.37×). The norms that stayed above 1.0 at every mesh are H1 semi and stress.

The four ratios are **not** the same number: L2 4.71×, H1 semi 1.35×, stress 1.37×, energy 7.00×. The same inversion the production run shows, on a different mesh and a different device — which is the reason it is read there as a property of the training principle rather than an artefact of one run.

The error was **still falling at the last epoch** (see `history` in the JSON), so this ratio reflects the step budget, not the method. The reportable number needs `Round6_MMS_Operator.ipynb`: N=17, 2000 epochs, 20–40 min on any GPU.


---


## Does the operator have a convergence rate of its own?

Section 8.11 used to say it could not be asked, because the operator had been trained at one mesh. `mms_operator_rate_B1_neo_hookean.json` trains it at three under the identical protocol and asks.

| N | DOF | operator L2 | Q4 L2 | op/Q4 | operator H1 | Q4 H1 | op/Q4 |
|---|---|---|---|---|---|---|---|
| 9 | 162 | 5.035e-03 | 1.351e-02 | **0.37×** | 1.141e-01 | 1.132e-01 | 1.01× |
| 17 | 578 | 8.238e-03 | 3.403e-03 | **2.42×** | 5.831e-02 | 5.666e-02 | 1.03× |
| 33 | 2,178 | 1.136e-02 | 8.525e-04 | **13.33×** | 3.850e-02 | 2.834e-02 | 1.36× |

**Fitted rates in h**: operator L2 **-0.59**, Q4 L2 1.99; operator H1 0.78, Q4 H1 1.00. The Q4 figures are the control and they land on Table 23's measured 1.98 and 1.00, so the operator's can be quoted beside them.

**the operator does not converge. Its L2 error gets WORSE with refinement -- 5.035e-03 at N=9, 8.238e-03 at N=17, 1.136e-02 at N=33 -- for a fitted rate of -0.59, while Q4 falls at 1.99. Its H1 error does improve, but at 0.78 against Q4's 1.00.**

the operator's error is dominated by OPTIMIZATION error, not discretization error. Refining the mesh reduces the discretization error the Q4 solver is limited by, and leaves the network's own optimization error roughly where it was -- while making the problem it has to optimize larger. So the gap widens: 0.37x, 2.42x, 13.33x.

at N=9 the discretization is coarse enough that Q4's own error (1.35e-02) exceeds the operator's optimization error, and the operator is actually AHEAD in L2. By N=33 Q4 is 13x better. Somewhere between the two the discretization stops being the limiting factor and the network becomes it.


### The ceiling was stated too strongly, and this run showed it

the ceiling is a statement about Pi. Q4 minimizes Pi over the Q4 space, so no field in that space -- the operator's included -- can achieve a lower Pi. But Pi is NOT any of the four error metrics reported. L2 error against u* is a different functional entirely, and nothing forbids a non-minimizer of Pi from sitting closer to u* in L2 than the minimizer does: Q4's discretization error is a systematic bias, and the operator's optimization error can partially cancel it.

at N=9 the operator's L2 is 0.37x Q4's. The runner flagged it as 'should be impossible' and told the reader to check the mask, the quadrature and the energy term. It is not impossible, and there is nothing to check.

the derivative-based norms. operator/Q4 in the H1 semi-norm is 1.01, 1.03, 1.36 -- above one at every mesh -- and in stress 1.01, 1.03, 1.33, likewise. For a LINEAR problem Galerkin optimality would guarantee this, since the Galerkin solution minimizes the energy norm of the error and H1 semi is equivalent to it. This problem is nonlinear so that guarantee does not formally transfer, but it held at all three meshes here.

*'energy' in these tables is the relative error in the scalar internal strain energy, not the energy NORM of the error. It dips to 0.89 at N=9, which is a cancellation in a scalar and carries none of the protection the H1 column does.*


## What is NOT here

* **The operator at more than one mesh.** It is trained and scored at N=17 only, so it has no convergence rate of its own — the two rate tables above are FEM only. Retraining at each N is the missing work, and it is a training run per mesh, not a solve.

* **The operator's cost, on comparable terms.** Its 8.2 min of GPU training is not commensurable with a CPU FP64 Newton solve, and no attempt is made here to force them onto one axis.

* **One material and one geometry**, and a single manufactured solution rather than the parametrised family Timon called the ideal. The family is parametrised in the code by (alpha, beta); only one member is run.

