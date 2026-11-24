# Benchmarks & Recall Evaluation

This document records reproducible latency benchmarks and recall-ranking ablation
results so external developers can inspect and re-verify MemStack's ranking
choices and performance.

## Reproducing locally

```bash
python tests/benchmark_public.py          # full synthetic latency + quality suite
python tests/benchmark_public.py --quick  # fast CI smoke run
python tests/benchmark_public.py --json   # machine-readable JSON report
```

`tests/benchmark_public.py` requires **no external database, no network access,
and no personal data**. It uses deterministic unit-normalized vectors so quality
metrics are identical on any machine and latency can be measured directly on
your own hardware.

---

## 1. Synthetic latency & controlled quality (`tests/benchmark_public.py`)

Measured on Windows / Python 3.12.1 (7 runs per workload, 1024-dim vectors for
recall, default weights `WEIGHT_SIM=0.95`, `WEIGHT_STRENGTH=0.0`,
`WEIGHT_EMOTION=0.05`, `ENTITY_BONUS_MAX=0.10`, `RECALL_SEMANTIC_MIN_SIM=0.45`):

### Latency

| Workload | Size | Output / Dim | Median (ms) | P95 (ms) |
|---|---:|---|---:|---:|
| Episode aggregation (`build_episodes`) | 45 turns | 12 episodes | 6.14 | 7.63 |
| Episode aggregation (`build_episodes`) | 150 turns | 38 episodes | 25.17 | 31.31 |
| Episode aggregation (`build_episodes`) | 450 turns | 113 episodes | 81.18 | 86.64 |
| `hybrid_rerank` | 250 items | 1024-dim | 70.81 | 75.87 |
| `rrf_recall` (2-channel) | 250 items | 1024-dim | 71.91 | 75.68 |
| `hybrid_rerank` | 1,000 items | 1024-dim | 288.06 | 350.99 |
| `rrf_recall` (2-channel) | 1,000 items | 1024-dim | 297.60 | 359.72 |

### Controlled ranking invariants (40 deterministic cases)

| Check | Result |
|---|---:|
| `hybrid_rerank` strength-adversarial R@1 (target `sim=0.76, strength=35` vs distractor `sim=0.70, strength=98`) | 100.0% |
| `rrf_recall` strength-adversarial R@1 (same adversarial pair, 2-channel semantic + lexical) | 100.0% |
| Entity bonus tie-break R@1 (equal `sim=0.68`, entity-matched item vs higher-strength distractor) | 100.0% |
| `min_similarity=0.45` low-similarity noise rejection (`sim=0.25`) | 100.0% |

---

## 2. Offline ablation on a real conversational store (aggregate summary)

To evaluate the ranking choices on real embeddings (`bge-m3`, 1024-dim), we ran
a read-only ablation on a 730-memory active store with 40 labeled queries (29
positive queries with ground-truth IDs + 11 negative queries; store text is
private and omitted here):

| Configuration | R@1 | R@3 | R@5 | MRR |
|---|---:|---:|---:|---:|
| Old baseline (`sim=0.80, str=0.15, emo=0.05`, `min_sim=None`) | 13.8% (4/29) | 27.6% (8/29) | 31.0% (9/29) | 0.1983 |
| Old baseline (`sim=0.80, str=0.15, emo=0.05`, `min_sim=0.45`) | 17.2% (5/29) | 34.5% (10/29) | 37.9% (11/29) | 0.2500 |
| **New default (`sim=0.95, str=0.00, emo=0.05`, `min_sim=0.45`)** | **24.1% (7/29)** | **34.5% (10/29)** | **37.9% (11/29)** | **0.3017** |
| No strength, `min_sim=0.40` or `0.35` | 17.2% (5/29) | 31.0% (9/29) | 34.5% (10/29) | 0.2500 |
| No strength, `ENTITY_BONUS_MAX=0.05` (`min_sim=0.45`) | 24.1% (7/29) | 34.5% (10/29) | 37.9% (11/29) | 0.3017 |
| No strength, `ENTITY_BONUS_MAX=0.00` (`min_sim=0.45`) | 20.7% (6/29) | 31.0% (9/29) | 31.0% (9/29) | 0.2586 |
| Old RRF 3-channel (`sem + lex + strength`, `min_sim=0.45`) | 37.9% (11/29) | 62.1% (18/29) | 79.3% (23/29) | 0.5299 |
| **New RRF 2-channel (`sem + lex`, no strength, `min_sim=0.45`)** | **41.4% (12/29)** | **75.9% (22/29)** | **79.3% (23/29)** | **0.5661** |

Key takeaways from the ablation:

1. **Decoupling `strength` from ranking** improved both `hybrid_rerank`
   (`MRR 0.2500 → 0.3017`, `R@1 +6.9pp`) and `rrf_recall`
   (`R@3 62.1% → 75.9%`, `MRR 0.5299 → 0.5661`), while aligning runtime behavior
   with the design principle that strength governs lifecycle only.
2. **Entity bonus (`ENTITY_BONUS=0.05`, `ENTITY_BONUS_MAX=0.10`)** materially
   helped compared to disabling it (`ENTITY_BONUS_MAX=0.00` dropped `R@5` from
   `37.9%` to `31.0%` and `MRR` from `0.3017` to `0.2586`).
3. **Semantic similarity cutoff (`RECALL_SEMANTIC_MIN_SIM=0.45`)** produced
   **0/29 false drops** on positive queries (ground-truth cosine similarity
   range: `min=0.4959`, `median=0.6290`, `max=0.7681`) while improving `MRR`
   over lower thresholds (`0.40` / `0.35` / `None`).
