"""Build the native star map using the console's three kinds of edges."""

from collections import Counter, defaultdict
import math

import numpy as np

import layout

# Golden angle used by the fallback bearing in star_positions().
GOLDEN_ANGLE = 2.39996
TAU = 2 * math.pi

# Above this size the map uses the adapter's lightweight label links.
MAX_SEMANTIC_NODES = 4096
SECONDS_PER_DAY = 86400
SIMILARITY_BATCH = 128


def rule_words(value):
    if not isinstance(value, (list, tuple)):
        return ()
    return [word for word in value[:512] if isinstance(word, str) and 0 < len(word) <= 100]


def entity_rules(settings):
    hints = settings.get("ENTITY_HINTS")
    if not isinstance(hints, dict):
        return None
    aliases = settings.get("ALIASES")
    blocks = settings.get("SINGLE_CHAR_BLOCK")
    aliases = aliases if isinstance(aliases, dict) else {}
    blocks = blocks if isinstance(blocks, dict) else {}
    variants = {}
    for words in list(hints.values())[:32]:
        for name in rule_words(words):
            variants.setdefault(name, name)
            for alias in rule_words(aliases.get(name)):
                variants.setdefault(alias, name)
    blocked = {
        word: [(compound, compound.index(word)) for compound in rule_words(blocks.get(word)) if word in compound]
        for word in variants if word in blocks
    }
    return variants, blocked


def extract_entity_names(content, rules):
    variants, blocked = rules
    first = {}
    for variant, canonical in variants.items():
        position = content.find(variant)
        while position != -1:
            if not any(
                position >= offset and content.startswith(compound, position - offset)
                for compound, offset in blocked.get(variant, ())
            ):
                first[canonical] = min(position, first.get(canonical, position))
                break
            position = content.find(variant, position + 1)
    return sorted(first, key=lambda name: (first[name], name))


def vectors_for(units):
    matrix = np.zeros((len(units), layout.DIM), dtype=np.float32)
    valid = []
    for index, unit in enumerate(units):
        embedding = unit.get("embedding")
        if not layout.usable(embedding):
            continue
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                vector = np.asarray(embedding, dtype=np.float32)
                norm = np.linalg.norm(vector)
        except (TypeError, ValueError, OverflowError):
            continue
        if vector.shape == (layout.DIM,) and np.isfinite(norm) and norm > 1e-8:
            matrix[index] = vector / norm
            valid.append(index)
    return matrix, valid


def _canonical_axes(matrix):
    """SVD whose output does not depend on the order of the rows.

    A right singular vector is only defined up to sign, and LAPACK picks the
    sign from the input layout. Feeding the same records in a different order
    therefore flips the axes — measured at exactly 180 degrees for cold's 719
    vectors, which mirrors the whole ring on screen. Pinning each axis so its
    largest component is positive, and ordering axes by singular value, removes
    that freedom while leaving every point's position relative to the others
    unchanged.
    """
    _, singular, axes = np.linalg.svd(matrix, full_matrices=False)
    fixed = axes[:3].copy()
    for row in fixed:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1
    order = sorted(range(len(fixed)), key=lambda k: (-round(float(singular[k]), 9), k))
    return fixed[order]


def star_positions(units, matrix, valid):
    """Place every star from its own pool's density-balanced semantic axes.

    Three properties this keeps:

    * Per pool. One SVD over every pool made cold's ring a side effect of how
      many shortterm records existed.
    * Order independent. See _canonical_axes.
    * Density balanced. Near-duplicate bursts (cos >= 0.82, e.g. repetitive tool
      logs) are deweighted in SVD and fanned out smoothly across an airy radial
      and angular nebula band instead of collapsing into one overexposed blob.
    """
    n_units = len(units)
    directions = np.zeros((n_units, 3), dtype=np.float64)
    per_slot = {}
    for index in valid:
        per_slot.setdefault(units[index]["slot"], []).append(index)

    for indices in per_slot.values():
        if len(indices) < 4:
            continue
        sub = matrix[indices].astype(np.float64)
        count = len(indices)
        dup_deg = np.ones(count, dtype=np.float64)
        for start in range(0, count, SIMILARITY_BATCH):
            batch = sub[start:start + SIMILARITY_BATCH] @ sub.T
            dup_deg[start:start + SIMILARITY_BATCH] = np.maximum(1.0, np.sum(batch >= 0.82, axis=1))
        w = 1.0 / np.power(dup_deg, 0.92)
        w = w / np.sum(w)
        mu = np.sum(sub * w[:, None], axis=0, keepdims=True)
        centered = sub - mu
        axes = _canonical_axes(centered * np.sqrt(w)[:, None])
        raw_c = centered @ axes.T
        scales = np.maximum(1e-4, np.percentile(np.abs(raw_c), 88, axis=0))
        raw_c = raw_c / scales[None, :]
        norms = np.linalg.norm(raw_c, axis=1, keepdims=True)
        safe = norms[:, 0] > 1e-6
        if np.any(safe):
            raw_c[safe] /= norms[safe]
            for local_i, global_i in enumerate(indices):
                if safe[local_i]:
                    directions[global_i] = raw_c[local_i]

    # Quantize unit vectors into sector patches per pool to disperse dense clumps cleanly
    cell_totals = Counter()
    cell_keys = []
    for index, unit in enumerate(units):
        ux, uy, uz = map(float, directions[index])
        if ux == 0.0 and uy == 0.0 and uz == 0.0:
            a = index * GOLDEN_ANGLE
            ux, uy, uz = math.cos(a), 0.14 * math.sin(index * 0.7), math.sin(a)
            nrm = math.hypot(ux, uy, uz) or 1.0
            ux, uy, uz = ux / nrm, uy / nrm, uz / nrm
            directions[index] = (ux, uy, uz)
        ck = (unit["slot"], round(ux * 2.4), round(uy * 2.4), round(uz * 2.4))
        cell_keys.append(ck)
        cell_totals[ck] += 1

    cell_rank = Counter()
    golden = 2.399963229728653
    for index, unit in enumerate(units):
        ux, uy, uz = map(float, directions[index])
        ck = cell_keys[index]
        tot = cell_totals[ck]
        rank = cell_rank[ck]
        cell_rank[ck] += 1

        ref = (0.0, 1.0, 0.0) if abs(uy) < 0.85 else (1.0, 0.0, 0.0)
        t1x = uy * ref[2] - uz * ref[1]
        t1y = uz * ref[0] - ux * ref[2]
        t1z = ux * ref[1] - uy * ref[0]
        t1n = math.hypot(t1x, t1y, t1z) or 1.0
        t1x, t1y, t1z = t1x / t1n, t1y / t1n, t1z / t1n
        t2x = uy * t1z - uz * t1y
        t2y = uz * t1x - ux * t1z
        t2z = ux * t1y - uy * t1x

        ang = rank * golden + (index % 13) * 0.19
        spread = 0.045
        clump_mix = 0.0
        if tot > 6:
            clump_mix = min(1.0, (tot - 6) / 65.0)
            spread = min(0.82, 0.085 + 0.62 * math.sqrt(rank / max(1, tot - 1)) * min(1.0, math.sqrt(tot / 42.0)))
        ux += (t1x * math.cos(ang) + t2x * math.sin(ang)) * spread
        uy += (t1y * math.cos(ang) + t2y * math.sin(ang)) * spread
        uz += (t1z * math.cos(ang) + t2z * math.sin(ang)) * spread
        norm = math.hypot(ux, uy, uz) or 1.0
        direction = (ux / norm, uy / norm, uz / norm)

        slot = unit["slot"]
        s_norm = max(0.0, min(100.0, float(unit["strength"] or 0.0))) / 100.0
        rad_override = None
        if clump_mix > 0.0 and slot in (0, 1):
            base_r = (0.24 + (1.0 - s_norm) * 0.74 + (0.08 if unit["age_days"] > 7 else 0.0)) if slot == 0 else (0.42 + (1.0 - s_norm) * 0.74)
            strat_r = 0.30 + 0.90 * ((rank * 0.61803398875) % 1.0)
            rad_override = base_r * (1.0 - 0.78 * clump_mix) + strat_r * (0.78 * clump_mix)
        yield layout._shell(direction, slot, unit["strength"], unit["age_days"], rad_override=rad_override)


def links_for(index, scores, tags, peers, times):
    scores[index] = -1
    size = len(scores)
    count = min(6, size)
    candidates = np.argpartition(scores, size - count)[-count:]
    links = [
        [int(peer), 0, round(float(scores[peer]), 2)]
        for peer in sorted(candidates, key=lambda peer: float(scores[peer]), reverse=True)[:2]
        if scores[peer] >= 0.50
    ]
    seen = {peer for peer, _, _ in links} | {index}

    for tag in tags:
        best, best_score = None, -1.0
        for peer in peers[tag]:
            if peer not in seen and scores[peer] > best_score:
                best, best_score = peer, float(scores[peer])
                if best_score >= 0.68:
                    break
        if best is not None:
            links.append([best, 2, round(max(0.55, 0.50 + max(0, best_score) * 0.45), 2)])
            seen.add(best)
            break

    if times[index] is not None:
        nearby = (
            peer for peer in range(max(0, index - 60), min(size, index + 60))
            if peer not in seen and times[peer] is not None
            and abs(times[index] - times[peer]) <= 3 * SECONDS_PER_DAY and scores[peer] >= 0.42
        )
        peer = max(nearby, key=lambda peer: float(scores[peer]), default=None)
        if peer is not None:
            links.append([peer, 1, round(min(0.95, max(0.45, float(scores[peer])) + 0.12), 2)])

    if not links and size > 1:
        peer = int(np.argmax(scores))
        if scores[peer] > 0:
            links.append([peer, 0, round(float(scores[peer]), 2)])
    return links


def project(units, records):
    if not units:
        return
    if len(units) > MAX_SEMANTIC_NODES:
        return

    matrix, valid = vectors_for(units)
    try:
        positions = list(star_positions(units, matrix, valid))
    except np.linalg.LinAlgError:
        positions = list(star_positions(units, matrix, []))
    peers = defaultdict(list)
    for index, row in enumerate(records):
        for label in row["cosmos_entities"]:
            peers[label].append(index)
    times = [unit["created"] for unit in units]

    for start in range(0, len(units), SIMILARITY_BATCH):
        scores_batch = matrix[start:start + SIMILARITY_BATCH] @ matrix.T
        for index, scores in enumerate(scores_batch, start):
            row = records[index]
            row["pos"] = positions[index]
            row["cosmos_links"] = links_for(index, scores, row["cosmos_entities"], peers, times)

