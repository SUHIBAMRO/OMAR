"""Is the B3 radial-grading inversion a load-step effect, like the cylinder's? (round K)

On the 3D cylinder (Section 31.1) a graded mesh gave an inverted Gauss point because
the Newton solve converged to an inverted state with too few load increments. B3 (the
bushing groove) also reported inverted elements for radial grading r_grading >= 1.75.
This script checks the UNDEFORMED mesh: det J of the Q20 map at the 27 points of a
3x3x3 rule, level (21, 18, 31) = 10,200 elements, for several gradings. A negative det J
before any load means the inversion is geometric (the Q20 midside nodes snapped to the
curved groove surface bulge through the collapsed first radial layer), and more load
increments cannot remove it.

    python -m omar_pfem.verification_agent.b3_mesh_jacobian
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "b3_mesh_jacobian.json")
GRADINGS = (1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0)


def main():
    import numpy as np
    import torch
    torch.set_default_dtype(torch.float64)
    from torchfem.elements import Hexa2
    from omar_pfem.verification_agent import b3_global as B
    Nt, Nr, Nz = B._family_resolutions()[2]
    g = [-1.0, 0.0, 1.0]
    pts = torch.tensor([[a, b, c] for a in g for b in g for c in g])
    dN = Hexa2().B(pts)
    rows = []
    for rg in GRADINGS:
        nodes, el, *_ = B.build_quadratic_mesh_with_curved_surfaces(
            Nt, Nr, Nz, r_grading=rg, groove_depth=B.GROOVE_DEPTH)
        X = torch.as_tensor(np.asarray(nodes))
        E = torch.as_tensor(np.asarray(el), dtype=torch.long)
        dJ = torch.linalg.det(torch.einsum("pdn,enk->epdk", dN, X[E]))
        rows.append({"r_grading": rg, "n_elem": int(E.shape[0]),
                     "min_detJ_undeformed": float(dJ.min()),
                     "negative_points": int((dJ <= 0).sum()),
                     "elements_with_negative_points": int((dJ <= 0).any(dim=1).sum())})
        print(rows[-1], flush=True)
    json.dump({"level": [Nt, Nr, Nz], "rows": rows}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
