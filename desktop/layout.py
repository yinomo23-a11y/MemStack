"""Lay memories out on the native console's three-dimensional pool shells."""

import math
import operator
import random

DIM = 1024
SWEEPS = 16
SEED = 20261122
POOL_SLOT = {"active": 0, "shortterm": 1, "intimate": 2, "cold": 3, "working": 4}
mul = operator.mul


def usable(embedding):
    return isinstance(embedding, list) and len(embedding) == DIM


def _shell(unit, slot, strength, age_days, rad_override=None):
    """Per-pool radius rule with 1.58 outer ring cap so stars stay inside the chassis."""
    s = max(0.0, min(100.0, float(strength or 0.0))) / 100.0
    if rad_override is not None:
        rad = float(rad_override)
    elif slot == 4:
        rad = 0.14 + (1.0 - s) * 0.20
    elif slot == 0:
        rad = 0.24 + (1.0 - s) * 0.74 + (0.08 if age_days > 7 else 0.0)
    elif slot == 1:
        rad = 0.42 + (1.0 - s) * 0.74
    elif slot == 2:
        rad = 0.42 + (1.0 - s) * 0.56
    else:
        rad = min(1.58, 1.10 + (1.0 - s * 0.5) * 0.56)
    rad = max(0.12, min(1.58, rad))
    ux, uy, uz = unit
    if slot == 2:  # the protected belt rides above the disc
        uy = abs(uy) * 0.72 + 0.22
    return [round(ux * rad, 3), round(uy * rad * 0.65, 3), round(uz * rad, 3)]


def _svd3_numpy(matrix):
    import numpy as np

    m = np.asarray(matrix, dtype=np.float64)
    m = m - m.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(m, full_matrices=False)
    return [[float(c[0]), float(c[1]), float(c[2])] for c in (m @ vt[:3].T)]


def _svd3_pure(rows, sweeps=SWEEPS, seed=SEED):
    n, d = len(rows), len(rows[0])
    mean = [0.0] * d
    for row in rows:
        mean = [a + b for a, b in zip(mean, row)]
    mean = [a / n for a in mean]
    centred = [[a - b for a, b in zip(row, mean)] for row in rows]

    rng = random.Random(seed)
    basis = []
    for _ in range(3):
        v = [rng.random() - 0.5 for _ in range(d)]
        for _ in range(sweeps):
            acc = [0.0] * d
            for row in centred:
                s = sum(map(mul, row, v))
                acc = [a + s * b for a, b in zip(acc, row)]
            for b in basis:
                p = sum(map(mul, acc, b))
                acc = [a - p * c for a, c in zip(acc, b)]
            norm = math.sqrt(sum(map(mul, acc, acc)))
            if norm < 1e-12:
                break
            v = [a / norm for a in acc]
        basis.append(v)
    return [[sum(map(mul, row, b)) for b in basis] for row in centred]


def positions(units):
    """units: list of dicts with embedding / slot / strength / age_days.

    Returns one [x, y, z] per unit, in the same order. Entries without a usable
    embedding get snapshot.py's deterministic fallback so nothing lands on 0,0,0.
    """
    out = []
    indexed = [(i, u) for i, u in enumerate(units) if usable(u.get("embedding"))]
    coords = {}
    if indexed:
        matrix = [u["embedding"] for _, u in indexed]
        try:
            projected = _svd3_numpy(matrix)
        except ImportError:
            projected = _svd3_pure(matrix)
        for k, (i, _) in enumerate(indexed):
            coords[i] = projected[k]

    for i, u in enumerate(units):
        c = coords.get(i)
        if c is None:
            ang = (i * 2.39996) % (2 * math.pi)
            unit = (math.cos(ang), 0.15, math.sin(ang))
        else:
            norm = math.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2])
            if norm <= 1e-6:
                ang = (i * 2.39996) % (2 * math.pi)
                unit = (math.cos(ang), 0.15, math.sin(ang))
            else:
                unit = (c[0] / norm, c[1] / norm, c[2] / norm)
        out.append(_shell(unit, u.get("slot", 0), u.get("strength"), u.get("age_days", 0.0)))
    return out
