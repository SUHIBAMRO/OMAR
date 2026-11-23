"""torch-fem Solid with the FULL 3x3x3 Gauss rule for 20-node hexahedra (2026-11-23).

torch-fem's Hexa2 uses the reduced 2x2x2 rule, which admits spurious (hourglass-type)
deformation on strongly graded meshes: the B3 local-refinement study found
sign-alternating Gauss-point stresses, and the 3D benchmark's graded strategy
(g = 2) an inverted Gauss point. SolidFull swaps in the 27-point rule and nothing
else (same shape functions, material, solver).
"""
from math import sqrt

import torch
from torchfem import Solid
from torchfem.elements import Hexa2

_G = [-sqrt(3.0 / 5.0), 0.0, sqrt(3.0 / 5.0)]
_W = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]


class _ClassProperty:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, cls):
        return self.f(cls)


class Hexa2Full(Hexa2):
    """Hexa2 with the 3x3x3 Gauss rule."""

    ipoints = _ClassProperty(lambda cls: torch.tensor(
        [[a, b, c] for c in _G for b in _G for a in _G]))
    iweights = _ClassProperty(lambda cls: torch.tensor(
        [wa * wb * wc for wc in _W for wb in _W for wa in _W]))


class SolidFull(Solid):
    element_types = tuple(Hexa2Full if e is Hexa2 else e for e in Solid.element_types)
