"""Reproducible public performance & recall-quality benchmark for MemStack.

Zero personal data and zero external network/Ollama dependency: uses
deterministic unit-normalized vectors so any developer can reproduce exact
recall-quality metrics and measure latency on their own hardware.

Usage:
    python tests/benchmark_public.py          # full benchmark
    python tests/benchmark_public.py --quick  # fast CI smoke run
    python tests/benchmark_public.py --json   # JSON output
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import statistics
import sys
import time

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import recall_engine as re_mod  # noqa: E402
from episode import build_episodes  # noqa: E402
from schema import MemoryItem  # noqa: E402


def _unit(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else [0.0] * len(vec)


def _seeded_vec(rng: random.Random, dim: int):
    return _unit([rng.gauss(0.0, 1.0) for _ in range(dim)])


def _mix_vec(base, noise, target_sim: float):
    """Construct a unit vector whose cosine similarity with `base` is ~target_sim."""
    # Remove projection of noise onto base so base and ortho are perpendicular.
    dot = sum(a * b for a, b in zip(base, noise))
    ortho = _unit([n - dot * b for b, n in zip(base, noise)])
    s = max(-0.99, min(0.99, target_sim))
    c = math.sqrt(max(0.0, 1.0 - s * s))
    return _unit([s * b + c * o for b, o in zip(base, ortho)])


def _stats(samples_ms):
    ordered = sorted(samples_ms)
    p95_idx = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)
    return {
        "runs": len(ordered),
        "min_ms": round(ordered[0], 2),
        "median_ms": round(statistics.median(ordered), 2),
        "p95_ms": round(ordered[p95_idx], 2),
        "max_ms": round(ordered[-1], 2),
    }


def bench_episode_aggregate(n_items: int, runs: int = 5):
    rng = random.Random(42 + n_items)
    base_ts = 1760000000.0
    topics = ["project", "travel", "dining", "study", "schedule"]
    mems = []
    ts = base_ts
    for i in range(n_items):
        # Insert >600s gap every ~4 turns to exercise episode boundary cuts.
        ts += 720.0 if (i > 0 and i % 4 == 0) else 45.0
        topic = topics[(i // 4) % len(topics)]
        m = MemoryItem(
            content=f"Turn {i}: discussing {topic} milestones and next action item",
            mem_type="short_term",
            tags=[topic],
            create_time=ts,
            last_mentioned_at=ts,
            strength=80.0,
            session_id="bench-session",
        )
        m.embedding = _seeded_vec(rng, 128)
        mems.append(m)

    samples = []
    episodes_out = 0
    import contextlib
    for _ in range(runs):
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            eps = build_episodes("bench-session", mems)
        samples.append((time.perf_counter() - t0) * 1000.0)
        episodes_out = len(eps)
    return {"items": n_items, "episodes": episodes_out, **_stats(samples)}


def bench_recall_latency(n_mems: int, dim: int = 1024, runs: int = 5):
    rng = random.Random(100 + n_mems)
    now = time.time()
    q_vec = _seeded_vec(rng, dim)
    mems = []
    for i in range(n_mems):
        m = MemoryItem(
            content=f"Synthetic memory item #{i} about release checklist and config",
            tags=["bench", f"topic_{i % 10}"],
            strength=30.0 + (i % 70),
            create_time=now - 3600.0,
            last_recalled_at=now - 3600.0,
        )
        m.embedding = _mix_vec(q_vec, _seeded_vec(rng, dim), 0.30 + 0.50 * (i / max(1, n_mems)))
        mems.append(m)

    hyb_samples, rrf_samples = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        re_mod.hybrid_rerank(
            q_vec, ["checklist"], {"valence": "neutral", "intensity": 0},
            mems, top_k=5, update_strength=False, query_text="release checklist",
            min_similarity=0.45,
        )
        hyb_samples.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        re_mod.rrf_recall(
            q_vec, ["checklist"], {"valence": "neutral", "intensity": 0},
            mems, top_k=5, query_text="release checklist", min_similarity=0.45,
        )
        rrf_samples.append((time.perf_counter() - t0) * 1000.0)

    return {
        "memories": n_mems,
        "dim": dim,
        "hybrid_rerank": _stats(hyb_samples),
        "rrf_recall": _stats(rrf_samples),
    }


def bench_recall_quality(n_cases: int = 40, dim: int = 128):
    """Controlled quality suite checking strength-invariance, threshold & entity bonus."""
    rng = random.Random(2026)
    now = time.time()

    hyb_hits = 0
    rrf_hits = 0
    entity_hits = 0
    neg_rejected = 0

    for i in range(n_cases):
        q_vec = _seeded_vec(rng, dim)
        # Target: higher relevance (sim=0.76), low strength (35)
        target = MemoryItem(
            content=f"Case {i}: weekly lab schedule updated for module alpha",
            tags=["schedule"],
            strength=35.0,
            create_time=now,
            last_recalled_at=now,
        )
        target.embedding = _mix_vec(q_vec, _seeded_vec(rng, dim), 0.76)

        # Distractor: slightly lower relevance (sim=0.70), very high strength (98)
        distractor = MemoryItem(
            content=f"Case {i}: repeated old discussion about general lab notes",
            tags=["notes"],
            strength=98.0,
            create_time=now,
            last_recalled_at=now,
        )
        distractor.embedding = _mix_vec(q_vec, _seeded_vec(rng, dim), 0.70)

        # Low-similarity noise (sim=0.25)
        noise = MemoryItem(
            content=f"Case {i}: unrelated weather remark",
            tags=["misc"],
            strength=99.0,
            create_time=now,
            last_recalled_at=now,
        )
        noise.embedding = _mix_vec(q_vec, _seeded_vec(rng, dim), 0.25)

        pool = [distractor, noise, target]
        h_res = re_mod.hybrid_rerank(
            q_vec, [], {"valence": "neutral", "intensity": 0},
            pool, top_k=3, update_strength=False, min_similarity=0.45,
        )
        if h_res and h_res[0][0].mem_id == target.mem_id:
            hyb_hits += 1

        r_res = re_mod.rrf_recall(
            q_vec, [], {"valence": "neutral", "intensity": 0},
            pool, top_k=3, min_similarity=0.45,
        )
        if r_res and r_res[0][0].mem_id == target.mem_id:
            rrf_hits += 1

        # Negative query test: all candidates < 0.45 must be filtered out.
        neg_pool = [noise]
        neg_res = re_mod.hybrid_rerank(
            q_vec, [], {"valence": "neutral", "intensity": 0},
            neg_pool, top_k=3, update_strength=False, min_similarity=0.45,
        )
        if not neg_res:
            neg_rejected += 1

        # Entity bonus tie-breaker: equal sim (0.68), entity match wins.
        e_target = MemoryItem(
            content=f"Case {i}: Atlas deployment checklist approved",
            tags=["Atlas"],
            strength=40.0,
            create_time=now,
            last_recalled_at=now,
        )
        e_other = MemoryItem(
            content=f"Case {i}: generic deployment checklist approved",
            tags=["ops"],
            strength=95.0,
            create_time=now,
            last_recalled_at=now,
        )
        shared_vec = _mix_vec(q_vec, _seeded_vec(rng, dim), 0.68)
        e_target.embedding = shared_vec
        e_other.embedding = list(shared_vec)
        e_res = re_mod.hybrid_rerank(
            q_vec, ["Atlas"], {"valence": "neutral", "intensity": 0},
            [e_other, e_target], top_k=2, update_strength=False, min_similarity=0.45,
        )
        if e_res and e_res[0][0].mem_id == e_target.mem_id:
            entity_hits += 1

    return {
        "cases": n_cases,
        "hybrid_strength_adversarial_r1": round(hyb_hits / n_cases, 4),
        "rrf_strength_adversarial_r1": round(rrf_hits / n_cases, 4),
        "entity_tiebreak_r1": round(entity_hits / n_cases, 4),
        "min_sim_045_neg_rejection": round(neg_rejected / n_cases, 4),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Fast subset for CI smoke checks")
    ap.add_argument("--json", action="store_true", help="Emit JSON only")
    args = ap.parse_args(argv)

    runs = 3 if args.quick else 7
    agg_sizes = [45, 150] if args.quick else [45, 150, 450]
    recall_sizes = [250, 1000] if args.quick else [250, 1000, 5000, 10000]
    q_cases = 20 if args.quick else 40

    report = {
        "python": sys.version.split()[0],
        "vector_backend": "numpy" if re_mod._np is not None else "python",
        "weights": {
            "WEIGHT_SIM": re_mod.WEIGHT_SIM,
            "WEIGHT_STRENGTH": re_mod.WEIGHT_STRENGTH,
            "WEIGHT_EMOTION": re_mod.WEIGHT_EMOTION,
            "LAYER_DECAY_IN_RANKING": re_mod.LAYER_DECAY_IN_RANKING,
            "ENTITY_BONUS_MAX": re_mod.ENTITY_BONUS_MAX,
        },
        "episode_aggregate": [bench_episode_aggregate(n, runs=runs) for n in agg_sizes],
        "recall_latency": [bench_recall_latency(n, dim=1024, runs=runs) for n in recall_sizes],
        "recall_quality": bench_recall_quality(n_cases=q_cases),
    }

    # Hard quality assertions so CI catches regressions.
    rq = report["recall_quality"]
    assert rq["hybrid_strength_adversarial_r1"] == 1.0, rq
    assert rq["rrf_strength_adversarial_r1"] == 1.0, rq
    assert rq["entity_tiebreak_r1"] == 1.0, rq
    assert rq["min_sim_045_neg_rejection"] == 1.0, rq

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print("=== MemStack Reproducible Public Benchmark ===")
    print(f"Python {report['python']} | backend={report['vector_backend']} | weights={report['weights']}")
    print("\n[Episode Aggregation]")
    for row in report["episode_aggregate"]:
        print(f"  n={row['items']:4d} -> {row['episodes']:3d} eps | median={row['median_ms']:6.2f} ms | p95={row['p95_ms']:6.2f} ms")
    print("\n[Recall Latency (1024-dim vectors)]")
    for row in report["recall_latency"]:
        h, r = row["hybrid_rerank"], row["rrf_recall"]
        print(f"  n={row['memories']:4d} | hybrid median={h['median_ms']:6.2f} ms (p95={h['p95_ms']:6.2f}) | rrf median={r['median_ms']:6.2f} ms (p95={r['p95_ms']:6.2f})")
    print("\n[Controlled Recall Quality]")
    for k, v in rq.items():
        if k != "cases":
            print(f"  {k:34s}: {v:.1%} ({rq['cases']} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
