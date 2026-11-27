"""What the pipeline can and cannot do, and the capability assessment of a problem (27 Nov,
Professor Rabczuk: "the agent should determine whether it can handle the problem and, if not,
identify the missing capabilities rather than automatically implementing them").

REGISTRY is a PREDEFINED table written from the code (each entry names the module that provides
it). requirements() is an AUTONOMOUS DECISION by the LLM (it reads the paper and states what the
problem needs); assess() is a PREDEFINED PROCEDURE (code compares the two and estimates the size).
Nothing here implements a missing capability.
"""
import math

# value -> (supported?, provided by / what would be needed)
REGISTRY = {
    "dimension": {
        "2D": (True, "geometry2d.py, pipeline2d.py"),
        "3D extruded section": (True, "geometry3d.py, geometry3d_nh.py (section x thickness, faces with "
                                      "their own conditions)"),
        "3D general": (False, "a 3D geometry kernel and tetrahedral meshing of arbitrary solids (gmsh OCC "
                              "import, revolved or CAD bodies), 3D corner/edge/vertex singularity analysis"),
    },
    "geometry": {
        "straight edges": (True, "geometry2d.py"),
        "inclined edges": (True, "geometry2d.py"),
        "circular holes": (True, "geometry2d.py (curved edges on circles)"),
        "polygonal holes": (True, "geometry2d.py"),
        "curved outer boundary": (False, "arcs/splines on the outer boundary in the description, the mesher "
                                         "and the corner analysis"),
        "elliptical or spline holes": (False, "ellipse/spline curves in geometry2d"),
        "cylindrical hole through the thickness": (True, "geometry3d.py (extruded circular hole)"),
        "spherical or revolved surfaces": (False, "general 3D geometry (see dimension)"),
        "notch with a small root radius": (True, "geometry2d.py if the notch is made of straight edges and "
                                                 "circular arcs as holes; a curved outer notch is not"),
    },
    "material": {
        "linear elastic, isotropic": (True, "pipeline2d.py, geometry2d.py, geometry3d.py"),
        "compressible neo-Hookean": (True, "pipeline2d_nh.py, geometry3d_nh.py"),
        "nearly incompressible (nu > 0.49)": (False, "a mixed u-p (or F-bar / enhanced-strain) formulation "
                                                     "with its own code verification; displacement elements lock"),
        "Mooney-Rivlin": (False, "the strain-energy function, its stress and tangent, MMS body force"),
        "Ogden": (False, "principal-stretch energy, stress, tangent, MMS"),
        "anisotropic / fibre-reinforced": (False, "anisotropic invariants, fibre fields"),
        "plasticity or damage": (False, "history variables, return mapping, incremental verification"),
        "orthotropic linear elastic": (False, "anisotropic elasticity tensor"),
    },
    "loading": {
        "constant traction (dead)": (True, "all solvers"),
        "pressure (linear)": (True, "pipeline2d.py, geometry2d.py"),
        "follower pressure (finite strain)": (False, "configuration-dependent load and its tangent"),
        "prescribed displacement": (True, "linear solvers; neo-Hookean: not yet"),
        "point load": (False, "singular by nature; a regularised load or a quantity away from it"),
        "non-constant distributed load": (False, "load functions in the description"),
        "body force (gravity)": (False, "body loads in the description (MMS body forces exist internally)"),
    },
    "boundary": {
        "clamped": (True, "all"), "symmetry / roller": (True, "all"),
        "contact": (False, "contact formulation and its verification"),
        "periodic": (False, "periodic constraints"),
    },
    "analysis": {
        "static": (True, "all"),
        "load stepping (finite strain)": (True, "pipeline2d_nh.py, geometry3d_nh.py"),
        "dynamics": (False, "time integration and its verification"),
        "buckling / limit points": (False, "arc-length continuation, stability analysis"),
    },
}
LOCAL_DOF = 450_000          # experiment.VERIF_MAX_DOF default
LOCAL_GB = 13.0
CLUSTER_GB = 120.0


def requirements_schema():
    cats = {k: sorted(v) for k, v in REGISTRY.items()}
    return {
        "type": "object", "additionalProperties": False,
        "required": ["example", "needs", "unlisted_needs", "size", "reference_values", "quotes"],
        "properties": {
            "example": {"type": "string"},
            "needs": {"type": "object", "additionalProperties": False,
                      "required": list(cats),
                      "properties": {k: {"type": "array", "items": {"type": "string", "enum": v}}
                                     for k, v in cats.items()}},
            "unlisted_needs": {"type": "array", "items": {"type": "string"}},
            "size": {"type": "object", "additionalProperties": False,
                     "required": ["dimension", "dofs_in_paper", "elements_in_paper"],
                     "properties": {"dimension": {"type": "integer"},
                                    "dofs_in_paper": {"type": "number"},
                                    "elements_in_paper": {"type": "number"}}},
            "reference_values": {"type": "array", "items": {"type": "string"}},
            "quotes": {"type": "array", "items": {"type": "string"}}}}


REQ_PROMPT = (
    "You assess a published benchmark for a verification agent. Read the passage and state what "
    "the example REQUIRES, choosing from the listed values only (several per category when "
    "needed). Anything the example needs that is not in a list goes to unlisted_needs, in a few "
    "words. Give the dofs or elements of the paper's finest mesh if stated (0 if not), the "
    "reference values the paper reports (with their meaning), and short verbatim quotes that "
    "support the material, geometry and loading. Do not judge whether the agent can do it.\n\n")


def requirements(backend, passage, hint):
    return backend.ask("requirements", REQ_PROMPT + f"EXAMPLE: {hint}\n\nPASSAGE:\n{passage}",
                       requirements_schema())


def assess(req):
    """Code: supported? the missing capabilities with what they need; a size estimate."""
    missing = []
    for cat, vals in req["needs"].items():
        for v in vals:
            ok, what = REGISTRY[cat][v]
            if not ok:
                missing.append({"category": cat, "need": v, "would_require": what})
    for u in req.get("unlisted_needs", []):
        missing.append({"category": "unlisted", "need": u,
                        "would_require": "not in the registry: a person must judge"})
    dim = req["size"].get("dimension") or 2
    if dim == 3:
        # the 3D solvers exist, but the modeler that turns a paper into a description is 2D only
        missing.append({"category": "automation", "need": "3D problem definition from a paper",
                        "would_require": "a 3D description schema for the modeler (extrusion "
                                         "thickness, face conditions) with its code checks"})
    dofs = req["size"].get("dofs_in_paper") or 0
    # quadratic elements of this pipeline need roughly the paper's dofs for comparable accuracy;
    # the convergence ladder needs about 1 + 1/2^dim + ... = 2^dim / (2^dim - 1) of the finest
    lad = dofs * (2 ** dim) / (2 ** dim - 1) if dofs else 0
    where = ("unknown (no size in the paper)" if not dofs else
             "local machine" if dofs <= LOCAL_DOF else
             "cluster (manual submission)" if dofs <= 40 * LOCAL_DOF else "beyond the cluster budget")
    return {"feasible": not missing, "missing": missing,
            "size": {"finest_dofs_in_paper": dofs, "ladder_dofs_estimate": lad, "where": where},
            "decision": ("can be carried through" if not missing else
                         f"cannot be carried through: {len(missing)} missing capabilit"
                         f"{'y' if len(missing) == 1 else 'ies'} (listed, not implemented)")}
