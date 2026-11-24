# Benchmarks & Recall Evaluation

This document records reproducible latency benchmarks, real-store ablation
studies, and public **LongMemEval-S** (ICLR 2025) retrieval evaluation results so
external developers can independently verify MemStack's ranking quality and
runtime performance.

## Reproducing locally

```bash
# 1. Zero-dependency synthetic latency (up to 10k items) & invariant checks
python tests/benchmark_public.py          # full suite (250 .. 10,000 items)
python tests/benchmark_public.py --quick  # fast CI smoke run
python tests/benchmark_public.py --json   # machine-readable JSON

# 2. Public LongMemEval-S retrieval benchmark (requires local Ollama bge-m3)
python tests/eval_longmemeval.py --per-type 20   # 120 stratified questions (6 categories)
python tests/eval_longmemeval.py --per-type 0    # full 500 questions
```

---

## 1. Public Benchmark: LongMemEval-S (ICLR 2025)

Evaluated via [tests/eval_longmemeval.py](../tests/eval_longmemeval.py) on 120
deterministically stratified questions (`--per-type 20 --seed 42`, 20 per
official category: 115 answerable / 195 ground-truth evidence sessions + 5
abstention questions; ~48 haystack sessions per question) using local Ollama
`bge-m3` (1024-dim) embeddings.

### Overall retrieval comparison

| System | R@1 | R@3 | R@5 | All-Evidence@5 | MRR@5 | Median Latency |
|---|---:|---:|---:|---:|---:|---:|
| `pure_cosine` (vector-only baseline, no cutoff) | 88.7% | 93.9% | 95.7% | 87.8% | 0.9097 | 1.17 ms |
| `memstack_legacy_v300` (`str=0.15` + `_layer_weight`) | 79.1% | 89.6% | 91.3% | 75.6% | 0.8391 | 3.16 ms |
| **`memstack_hybrid_v310` (default, `min_sim=0.45`)** | **87.8%** | **92.2%** | **93.9%** | **78.3%** | **0.8981** | **2.35 ms** |
| **`memstack_hybrid_v310` (`min_sim=0.35`)** | **87.8%** | **93.9%** | **95.7%** | **87.0%** | **0.9054** | **2.35 ms** |
| `memstack_rrf_v310` (2-channel RRF, `min_sim=0.45`) | 72.2% | 87.8% | 90.4% | 74.8% | 0.7964 | 2.45 ms |

### Per-category breakdown (`memstack_hybrid_v310` at `min_sim=0.45` vs `pure_cosine`)

| Official LongMemEval Category | n | MemStack R@1 | MemStack R@5 | MemStack MRR | Cosine R@1 | Cosine R@5 | Cosine MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `knowledge-update` | 19 | 94.7% | 100.0% | 0.9649 | 94.7% | 100.0% | 0.9649 |
| `multi-session` | 17 | 100.0% | 100.0% | 1.0000 | 100.0% | 100.0% | 1.0000 |
| `single-session-assistant` | 20 | 100.0% | 100.0% | 1.0000 | 100.0% | 100.0% | 1.0000 |
| `single-session-preference` | 20 | 85.0% | 95.0% | 0.8767 | 85.0% | 95.0% | 0.8767 |
| `single-session-user` | 19 | 73.7% | 84.2% | 0.7719 | 73.7% | 84.2% | 0.7719 |
| `temporal-reasoning` | 20 | 75.0% | 85.0% | 0.7875 | 80.0% | 95.0% | 0.8542 |

### Key findings on LongMemEval-S

1. **Decoupling strength and idle-age penalties (`v3.0.0 → v3.1.0`)** recovered
   **+8.7pp R@1** (`79.1% → 87.8%`) and **+0.0590 MRR** (`0.8391 → 0.8981`) on a
   100% external public dataset.
2. **Threshold trade-off (`RECALL_SEMANTIC_MIN_SIM`)**: `0.45` filtered out
   31/195 secondary evidence sessions in multi-session/temporal English queries,
   reducing `All@5` from `87.0%` (`min_sim=0.35`) to `78.3%` (`min_sim=0.45`).
   Meanwhile, the 5 abstention (`_abs`) queries still had top distractor cosines
   in `0.452–0.574`, confirming that abstention cannot rely on a static cosine
   threshold alone and requires downstream evidence/LLM verification.

---

## 2. Offline Ablation on a Real Conversational Store (730 memories)

Read-only ablation on a 730-memory personal active store with 40 labeled queries
(29 positive queries with ground-truth IDs + 11 negative queries, `bge-m3`
1024-dim; private store text omitted):

| Configuration | R@1 | R@3 | R@5 | MRR |
|---|---:|---:|---:|---:|
| `v3.0.0` legacy (`sim=0.80, str=0.15, layer=True`, `min_sim=None`) | 13.8% (4/29) | 27.6% (8/29) | 31.0% (9/29) | 0.1983 |
| `v3.0.0` legacy (`sim=0.80, str=0.15, layer=True`, `min_sim=0.45`) | 17.2% (5/29) | 34.5% (10/29) | 37.9% (11/29) | 0.2500 |
| `v3.0.1` (`sim=0.95, str=0.00, layer=True`, `min_sim=0.45`) | 24.1% (7/29) | 34.5% (10/29) | 37.9% (11/29) | 0.3017 |
| `pure_cosine` baseline (`sim >= 0.45`) | 55.2% (16/29) | 72.4% (21/29) | 75.9% (22/29) | 0.6408 |
| `rrf_recall` 2-channel (`sem + lex`, `min_sim=0.45`) | 41.4% (12/29) | 75.9% (22/29) | 79.3% (23/29) | 0.5661 |
| **`v3.1.0` default (`sim=0.95, str=0.00, layer=False`, `min_sim=0.45`)** | **55.2% (16/29)** | **79.3% (23/29)** | **79.3% (23/29)** | **0.6552** |

Takeaways:
- Removing `_layer_weight` from ranking (`LAYER_DECAY_IN_RANKING = False`) more
  than doubled `hybrid_rerank` accuracy (`R@1 24.1% → 55.2%`,
  `R@3 34.5% → 79.3%`, `MRR 0.3017 → 0.6552`), outperforming both pure cosine
  (`R@3 +6.9pp`, `MRR +0.0144`) and 2-channel RRF (`R@1 +13.8pp`, `MRR +0.0891`).
- Entity bonus (`ENTITY_BONUS=0.05`, `ENTITY_BONUS_MAX=0.10`) accounts for the
  `+6.9pp` `R@3` edge over pure cosine on this store.

---

## 3. Synthetic Latency Scaling & Controlled Invariants (`tests/benchmark_public.py`)

Measured on Windows / Python 3.12.1 (7 runs per workload, 1024-dim vectors,
optional `numpy` batch matrix backend with cache):

### Latency scaling (`250` to `10,000` memories)

| Workload | Size | Pure Python (v3.0.1) | Optional NumPy (v3.1.0) | P95 (v3.1.0) |
|---|---:|---:|---:|---:|
| Episode aggregation (`build_episodes`) | 45 turns | 6.14 ms | 5.95 ms | 6.74 ms |
| Episode aggregation (`build_episodes`) | 150 turns | 25.17 ms | 26.79 ms | 29.66 ms |
| Episode aggregation (`build_episodes`) | 450 turns | 81.18 ms | 70.49 ms | 72.09 ms |
| `hybrid_rerank` (1024-dim) | 250 items | 70.81 ms | **11.44 ms** (6.2×) | 19.89 ms |
| `hybrid_rerank` (1024-dim) | 1,000 items | 288.06 ms | **53.88 ms** (5.3×) | 81.14 ms |
| `hybrid_rerank` (1024-dim) | 5,000 items | — | **240.20 ms** | 307.27 ms |
| `hybrid_rerank` (1024-dim) | 10,000 items | — | **478.87 ms** | 683.40 ms |
| `rrf_recall` 2-channel (1024-dim) | 1,000 items | 297.60 ms | **50.92 ms** (5.8×) | 53.60 ms |
| `rrf_recall` 2-channel (1024-dim) | 10,000 items | — | **504.57 ms** | 537.78 ms |

*Note: In the synthetic latency stress test, 70% of generated items exceed
`min_similarity=0.45`, exercising downstream regex/metadata gates on ~7,000 of
10,000 candidates. On sparser real queries (where ~5–10% pass `0.45`), recall
latency is substantially lower (e.g., `2.35 ms` median on LongMemEval-S).*

### Controlled ranking invariants (40 deterministic cases)

| Check | Result |
|---|---:|
| `hybrid_rerank` strength-adversarial R@1 (`sim=0.76, strength=35` vs `sim=0.70, strength=98`) | 100.0% |
| `rrf_recall` strength-adversarial R@1 (2-channel semantic + lexical) | 100.0% |
| Age-invariance check (`60d old, sim=0.95` vs `1d old, sim=0.85`, in `recall_engine` selftest) | PASS |
| Entity bonus tie-break R@1 (equal `sim=0.68`, entity match vs higher-strength distractor) | 100.0% |
| `min_similarity=0.45` low-similarity noise rejection (`sim=0.25`) | 100.0% |

---

## 4. Scope & Current Limitations

- **Retrieval-layer vs end-to-end QA**: `eval_longmemeval.py` evaluates session
  retrieval quality (`R@K`, `All@5`, `MRR`) without an LLM judge so any
  developer can reproduce it locally at zero API cost. End-to-end LLM answer
  accuracy on LongMemEval / LoCoMo is planned as an optional next layer.
- **Scale boundary**: MemStack targets local, single-agent active working sets
  (hundreds to ~10,000 items, kept bounded by decay and cold archiving). Exact
  matrix scoring scales linearly; 100k+ multi-tenant workloads should pair with
  an approximate nearest-neighbor (ANN) index.
