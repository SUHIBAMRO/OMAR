# Point 9 — method of manufactured solutions (neo_hookean)

Timon's point 9, and his "this is the last thing to do". His design: *"compare Q4, Q9 and the physics-informed Transolver against exactly the same analytical solution in L2, H1 and energy norms and also examine stress errors"*.

**This is the FEM half — Q4 and Q9. The operator half is not done**; it needs a body-force term in the energy functional and a body-force input channel, neither of which the trained checkpoints have. See `omar_pfem/mms_study.py` and PROJECT_STATUS.md.

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


## The operator third — a DEMONSTRATION run only, not a result

`operator_demo_N9_undertrained.json`. **Do not quote these numbers in the report.** It is a 1,600-optimizer-step CPU run at N=9, kept only because it proves the pipeline end to end and because its ceiling check passed. For scale, the report's own physics-informed models were trained for 75,000 steps.

| method | L2 | H1 semi | stress | energy |
|---|---|---|---|---|
| Q4 (same mesh) | 1.351e-02 | 1.132e-01 | 1.143e-01 | 1.268e-02 |
| Q9 (same N) | 4.155e-04 | 5.763e-03 | 5.957e-03 | 3.327e-05 |
| operator (undertrained) | 6.366e-02 | 1.525e-01 | 1.569e-01 | 8.877e-02 |

**operator / Q4 in L2 = 4.71×.** Above 1.0, which is the required outcome: the operator minimizes the same functional over the same Q4 space, so the Q4 solution is the minimizer and a ratio below 1.0 would mean a bug, not a win.

In the H1 semi-norm the ratio is 1.35×. That it is so much smaller than the L2 ratio is worth re-checking on a longer run — it is the opposite of the usual ordering, where L2 is the more forgiving norm.

The error was **still falling at the last epoch** (see `history` in the JSON), so this ratio reflects the step budget, not the method. The reportable number needs `Round6_MMS_Operator.ipynb`: N=17, 2000 epochs, 20–40 min on any GPU.


## What is NOT here

* **The Transolver.** The comparison Timon asked for is three-way; this is two-way. The operator cannot be run on this problem as things stand: its energy functional has no body-force term and its inputs have no body-force channel.

* **One material and one geometry**, and a single manufactured solution rather than the parametrised family Timon called the ideal. The family is parametrised in the code by (alpha, beta); only one member is run.

