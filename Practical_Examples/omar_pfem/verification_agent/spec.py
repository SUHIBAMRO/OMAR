"""Problem specifications and GRAFT-style semantic fingerprints.

A fingerprint is the set of descriptor tags a problem instantiates; similarity
is the Jaccard index of two fingerprints (1 - Jaccard is a metric on non-empty
sets, Toscano et al., GRAFT, Prop. S3).
"""
from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    geometry: str
    dim: int
    material: str
    loading: str
    regularity: str
    element: str
    reference: str
    qois: tuple = field(default_factory=tuple)

    def fingerprint(self):
        tags = {
            f"geometry:{self.geometry}",
            f"dim:{self.dim}",
            f"material:{self.material}",
            f"loading:{self.loading}",
            f"regularity:{self.regularity}",
            f"element:{self.element}",
            f"reference:{self.reference}",
        }
        tags |= {f"qoi:{q}" for q in self.qois}
        return frozenset(tags)

    def to_dict(self):
        d = asdict(self)
        d["qois"] = list(self.qois)
        return d


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
