# Benchmarks & Recall Evaluation

This document records reproducible evaluations across four dimensions so
external developers can independently verify both **static retrieval quality**
and **long-horizon lifecycle management value**:

1. **210-Day Managed Lifecycle vs Unmanaged Append-Only RAG** (`tests/eval_lifecycle_value.py`)
2. **Full 500-Question Official LongMemEval-S (ICLR 2025)** (`tests/eval_longmemeval.py`)
3. **Real Conversational Store Ablation (730 memories)**
4. **Synthetic Latency Scaling up to 10,000 items** (`tests/benchmark_public.py`)

## Reproducing locally

```bash
# 1. Zero-dependency 210-day managed lifecycle vs unmanaged RAG (runs in CI)
python tests/eval_lifecycle_value.py --embed offline
python tests/eval_lifecycle_value.py --embed ollama   # real Ollama bge-m3 embeddings

# 2. Full 500-question official LongMemEval-S retrieval benchmark (Ollama bge-m3)
python tests/eval_longmemeval.py --per-type 0         # all 500 official questions
python tests/eval_longmemeval.py --per-type 20        # fast 120-question stratified subset

# 3. Zero-dependency synthetic latency (250 .. 10,000 items) & invariant checks
python tests/benchmark_public.py
python tests/benchmark_public.py --quick
```

---

## 1. 210-Day Managed Lifecycle vs Unmanaged Append-Only RAG

Evaluated via [tests/eval_lifecycle_value.py](../tests/eval_lifecycle_value.py)
using real production modules (`MemoryManager`, `services/correction_service`,
`services/recall_policy`, `decay_logic.should_archive`, and `recall_engine`)
against an unmanaged append-only cosine vector store holding the exact same
messages and `bge-m3` (1024-dim) embeddings over a simulated 210-day timeline
(332 total written records: 15 multi-revision fact update chains, 12 recurring
time-bounded schedules with expired past weeks, 10 high-importance core facts
aged 205 days, and 240 low-importance casual trivia turns aged 5–215 days).

### Real `bge-m3` (1024-dim) results

| Scenario | Queries | MemStack Current-Truth R@1 | Unmanaged RAG R@1 | MemStack Stale/Expired@5 | Unmanaged Stale/Expired@5 |
|---|---:|---:|---:|---:|---:|
| Knowledge updates & corrections (v1 → v2 → v3) | 15 | **100.0%** | 20.0% | **0.0%** | 38.7% |
| Expired time-bounded schedules (past vs current week) | 12 | **91.7%** | 66.7% | **0.0%** | 36.7% |
| Core facts after 210-day decay (`importance=0.90`) | 10 | **80.0%** | 70.0% | **0.0%** | 0.0% |
| **Overall Current-Truth R@1** | **37** | **91.9%** | **48.6%** (**+43.2pp**) | **0.0%** | **30.3%** |

- **Active search space after 210 days**: MemStack keeps **211 eligible active items** (automatically archiving **66** idle low-importance trivia items to `cold.json` while keeping **10/10** 205-day-old high-importance core facts protected in `active.json`) vs **332** items in the unmanaged append-only store (**-36.4% active search space**).
- **CI offline mode** (`--embed offline`, zero Ollama dependency) verifies the same structural invariants on every commit: `0.0%` stale/expired leakage and `+43.2pp` current-truth R@1 gain (`59.5%` vs `16.2%`).

---

## 2. Full 500-Question Official Benchmark: LongMemEval-S (ICLR 2025)

Evaluated via [tests/eval_longmemeval.py](../tests/eval_longmemeval.py) on **all
500 official LongMemEval-S questions** (`--per-type 0`: 470 answerable questions
across all 6 official categories + 30 abstention `_abs` questions; 23,867 total
haystack session instances / 18,939 unique embedded texts) using local Ollama
`bge-m3` (1024-dim).

### Overall 500-question retrieval comparison

| System | R@1 | R@3 | R@5 | All-Evidence@5 | MRR@5 | Median Latency |
|---|---:|---:|---:|---:|---:|---:|
| `pure_cosine` (vector-only baseline, no cutoff) | 86.4% | 95.1% | 96.8% | 87.9% | 0.9076 | 1.41 ms |
| `memstack_legacy_v300` (`str=0.15`, `layer=True`, fixed `0.45`) | 78.9% | 89.1% | 90.2% | 71.1% | 0.8380 | 3.23 ms |
| `memstack_fixed045_v310` (`str=0.0`, `layer=False`, fixed `0.45`) | 84.0% | 90.6% | 91.7% | 74.0% | 0.8738 | 2.51 ms |
| **`memstack_hybrid_v320` (default, dynamic cutoff `0.45→0.35`)** | **84.0%** | **91.7%** | **92.8%** | **84.3%** | **0.8781** | **9.50 ms** |
| `memstack_rrf_v320` (2-channel RRF, dynamic cutoff) | 55.1% | 74.3% | 83.2% | 57.7% | 0.6580 | 10.12 ms |

### Per-category breakdown across all 470 answerable questions (`v3.2.0` vs `pure_cosine`)

| Official LongMemEval Category | n | MemStack R@1 | MemStack R@5 | MemStack MRR | Cosine R@1 | Cosine R@5 | Cosine MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `knowledge-update` | 72 | 88.9% | 98.6% | 0.9352 | 90.3% | 100.0% | 0.9491 |
| `multi-session` | 121 | 90.9% | 95.0% | 0.9277 | 93.4% | 98.4% | 0.9545 |
| `single-session-assistant` | 56 | 98.2% | 98.2% | 0.9821 | 100.0% | 100.0% | 1.0000 |
| `single-session-preference` | 30 | 76.7% | 93.3% | 0.8289 | 76.7% | 93.3% | 0.8289 |
| `single-session-user` | 64 | 71.9% | 82.8% | 0.7682 | 75.0% | 90.6% | 0.8190 |
| `temporal-reasoning` | 127 | 76.4% | 89.8% | 0.8196 | 79.5% | 96.1% | 0.8618 |

### Key takeaways from the full 500-question run

1. **Dynamic similarity cutoff (`v3.1.0 → v3.2.0`)** lifted full-evidence
   multi-session retrieval **`All@5` by +10.3pp** (`74.0% → 84.3%`, and
   `+13.2pp` vs `v3.0.0`'s `71.1%`), closing most of the gap to uncapped pure
   cosine (`87.9%`) while keeping the `0.45` entry gate for low-similarity
   queries.
2. **Static retrieval vs lifecycle management**: On raw unmanaged LongMemEval-S
   haystacks (where no explicit correction/window APIs are invoked), MemStack's
   `0.45` entry gate still filters ~4% of low-cosine positive queries compared
   to uncapped `pure_cosine` (`92.8%` vs `96.8%` R@5). MemStack's decisive
   advantage appears when its lifecycle management runs over time (Section 1:
   **+43.2pp** current-truth R@1 and **0% vs 30.3%** stale/expired pollution).

---

## 3. Offline Ablation on a Real Conversational Store (730 memories)

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
| **`v3.2.0` default (`str=0.0, layer=False`, dynamic `0.45→0.35`)** | **55.2% (16/29)** | **79.3% (23/29)** | **79.3% (23/29)** | **0.6552** |

Dynamic cutoff (`v3.2.0`) preserves 100% of the `v3.1.0` gain on the real
conversational store (`R@3 +6.9pp`, `MRR +0.0144` over pure cosine) with zero
regression.

---

## 4. Synthetic Latency Scaling & Controlled Invariants (`tests/benchmark_public.py`)

Measured on Windows / Python 3.12.1 (7 runs per workload, 1024-dim vectors,
optional `numpy` batch matrix backend with cache):

| Workload | Size | Pure Python (v3.0.1) | Optional NumPy (v3.1+) | P95 |
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

Controlled ranking invariants (`recall_engine.py` + `tests/benchmark_public.py`):
- Strength-adversarial R@1 (`hybrid_rerank` & `rrf_recall`): **100.0%**
- Age-invariance (`60d old` vs `1d old`): **PASS**
- Dynamic cutoff (rejects `max_sim < 0.45`, preserves secondary clue `0.40` when primary `0.62` clears `0.45`): **PASS**
- Entity bonus tie-break R@1 & low-similarity (`0.25`) rejection: **100.0%**

---

## 5. Scope & Current Limitations

- **Retrieval-layer vs end-to-end LLM QA**: `eval_longmemeval.py` and
  `eval_lifecycle_value.py` evaluate retrieval and lifecycle correctness without
  an external paid LLM judge so anyone can reproduce them locally at zero API
  cost. End-to-end LLM answer grading on LongMemEval / LoCoMo remains optional
  future work.
- **Scale boundary**: MemStack targets local, single-agent active working sets
  (hundreds to ~10,000 items, kept bounded by decay and cold archiving). Exact
  matrix scoring scales linearly; 100k+ multi-tenant workloads should pair with
  an approximate nearest-neighbor (ANN) index.
