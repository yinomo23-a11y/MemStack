"""Reproducible public LongMemEval-S retrieval benchmark for MemStack.

Evaluates MemStack against a pure cosine-similarity baseline and legacy v3.0.0
weights on the official LongMemEval-S dataset (ICLR 2025):
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

Requires a local Ollama embedding endpoint (default: http://127.0.0.1:11434,
model: bge-m3). Zero paid LLM calls required for retrieval evaluation.

Usage:
    python tests/eval_longmemeval.py --per-type 20
    python tests/eval_longmemeval.py --per-type 0          # full 500 questions
    python tests/eval_longmemeval.py --dataset /path/to/longmemeval_s_cleaned.json --json
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import io
import json
import math
import os
import random
import sqlite3
import statistics
import sys
import tempfile
import time
import urllib.request

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import recall_engine as re_mod  # noqa: E402
from entity_graph import extract_entity_names  # noqa: E402
from schema import MemoryItem  # noqa: E402

DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    "resolve/main/longmemeval_s_cleaned.json"
)
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")


def _default_cache_dir():
    return os.path.join(tempfile.gettempdir(), "memstack_longmemeval_cache")


def ensure_dataset(path: str | None) -> str:
    if path and os.path.isfile(path):
        return path
    target = path or os.path.join(_default_cache_dir(), "longmemeval_s_cleaned.json")
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    if not os.path.isfile(target):
        print(f"Downloading LongMemEval-S to {target} ...", file=sys.stderr)
        req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "MemStack-Eval"})
        with urllib.request.urlopen(req, timeout=180) as resp, open(target, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    return target


def _parse_lme_date(date_str: str, fallback_ts: float) -> float:
    # Format: "2023/05/20 (Sat) 02:21"
    try:
        head = str(date_str or "").strip()
        parts = head.split()
        if len(parts) >= 3:
            dt = datetime.datetime.strptime(f"{parts[0]} {parts[2]}", "%Y/%m/%d %H:%M")
            return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    except Exception:
        pass
    return fallback_ts


def _format_session(turns: list[dict], max_chars: int = 2400) -> str:
    # Prioritize user turns + concise assistant context within embedding budget.
    lines = []
    total = 0
    for turn in turns:
        role = str(turn.get("role") or "user")
        content = " ".join(str(turn.get("content") or "").split())
        if not content:
            continue
        piece = f"{role}: {content[:500]}"
        if total + len(piece) > max_chars and lines:
            break
        lines.append(piece)
        total += len(piece) + 1
    return "\n".join(lines)[:max_chars]


class EmbedCache:
    """SQLite-backed embedding cache keyed by (model, sha1(text))."""

    def __init__(self, db_path: str, model: str = EMBED_MODEL):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.model = model
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embeds ("
            "model TEXT NOT NULL, key TEXT NOT NULL, vec BLOB NOT NULL, "
            "PRIMARY KEY (model, key))"
        )
        self.conn.commit()
        self.mem: dict[str, list[float]] = {}

    def _key(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

    def get_many(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        import array

        keys = [self._key(t) for t in texts]
        missing_idx = []
        missing_keys = {}

        for i, k in enumerate(keys):
            if k in self.mem:
                continue
            missing_idx.append(i)

        if missing_idx:
            uniq_lookup = list({keys[i] for i in missing_idx})
            for off in range(0, len(uniq_lookup), 400):
                chunk = uniq_lookup[off : off + 400]
                qmarks = ",".join("?" * len(chunk))
                rows = self.conn.execute(
                    f"SELECT key, vec FROM embeds WHERE model=? AND key IN ({qmarks})",
                    [self.model, *chunk],
                ).fetchall()
                for k, blob in rows:
                    arr = array.array("f")
                    arr.frombytes(blob)
                    self.mem[k] = list(arr)

        for i in missing_idx:
            k = keys[i]
            if k not in self.mem and k not in missing_keys:
                missing_keys[k] = texts[i]

        if missing_keys:
            items = list(missing_keys.items())
            for off in range(0, len(items), batch_size):
                batch = items[off : off + batch_size]
                vecs = self._ollama_batch([t for _, t in batch])
                rows_to_put = []
                for (k, _), v in zip(batch, vecs):
                    self.mem[k] = v
                    arr = array.array("f", v)
                    rows_to_put.append((self.model, k, arr.tobytes()))
                self.conn.executemany(
                    "INSERT OR REPLACE INTO embeds (model, key, vec) VALUES (?, ?, ?)",
                    rows_to_put,
                )
                self.conn.commit()

        return [self.mem[k] for k in keys]

    def _ollama_batch(self, texts: list[str]) -> list[list[float]]:
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/embed",
            data=json.dumps({"model": self.model, "input": texts}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        embs = payload.get("embeddings")
        if not isinstance(embs, list) or len(embs) != len(texts):
            raise RuntimeError(f"Ollama /api/embed returned {len(embs or [])} vectors for {len(texts)} inputs")
        return embs

    def close(self):
        self.conn.close()


def select_questions(data: list[dict], per_type: int, seed: int = 42) -> list[dict]:
    if per_type <= 0:
        return list(data)
    rng = random.Random(seed)
    by_type = collections.defaultdict(list)
    for row in data:
        by_type[row.get("question_type", "unknown")].append(row)
    picked = []
    for qtype in sorted(by_type):
        group = list(by_type[qtype])
        rng.shuffle(group)
        picked.extend(group[:per_type])
    return picked


def _build_candidates(row: dict, sess_texts: list[str], sess_vecs: list[list[float]]) -> list[MemoryItem]:
    q_ts = _parse_lme_date(row.get("question_date", ""), time.time())
    now = time.time()
    cands = []
    for idx, (sid, sdate, text, vec) in enumerate(
        zip(row["haystack_session_ids"], row["haystack_dates"], sess_texts, sess_vecs)
    ):
        orig_ts = _parse_lme_date(sdate, q_ts - (len(sess_texts) - idx) * 86400.0)
        age_sec = max(0.0, q_ts - orig_ts)
        # Map dataset session age onto current wall clock so age-dependent logic
        # (like legacy _layer_weight) reflects the true conversation timeline.
        mapped_ts = now - age_sec
        age_days = age_sec / 86400.0
        # Simulate realistic decayed strength: older sessions decay lower, recent stay high.
        sim_strength = max(25.0, min(98.0, 98.0 - age_days * 1.6))
        m = MemoryItem(
            content=text,
            mem_type="long_term",
            tags=[],
            strength=sim_strength,
            create_time=mapped_ts,
            last_recalled_at=mapped_ts,
            session_id=str(sid),
        )
        m.mem_id = str(sid)
        m.embedding = vec
        cands.append(m)
    return cands


def _run_system(system: str, q_text: str, q_vec: list[float], q_ents: list[str], cands: list[MemoryItem], top_k: int = 5):
    t0 = time.perf_counter()
    if system == "pure_cosine":
        sims = re_mod._batch_cosine_similarities(q_vec, cands)
        ranked = sorted(zip(cands, sims), key=lambda x: x[1], reverse=True)[:top_k]
        out = [m.mem_id for m, _ in ranked]
    elif system == "memstack_legacy_v300":
        orig = (
            re_mod.WEIGHT_SIM, re_mod.WEIGHT_STRENGTH, re_mod.WEIGHT_EMOTION,
            re_mod.LAYER_DECAY_IN_RANKING, re_mod.LAYER_L1_WEIGHT, re_mod.LAYER_L2_WEIGHT,
        )
        re_mod.WEIGHT_SIM, re_mod.WEIGHT_STRENGTH, re_mod.WEIGHT_EMOTION = 0.80, 0.15, 0.05
        re_mod.LAYER_DECAY_IN_RANKING, re_mod.LAYER_L1_WEIGHT, re_mod.LAYER_L2_WEIGHT = True, 0.8, 0.5
        try:
            res = re_mod.hybrid_rerank(
                q_vec, q_ents, {"valence": "neutral", "intensity": 0},
                cands, top_k=top_k, update_strength=False, query_text=q_text, min_similarity=0.45,
            )
            out = [m.mem_id for m, _ in res]
        finally:
            (
                re_mod.WEIGHT_SIM, re_mod.WEIGHT_STRENGTH, re_mod.WEIGHT_EMOTION,
                re_mod.LAYER_DECAY_IN_RANKING, re_mod.LAYER_L1_WEIGHT, re_mod.LAYER_L2_WEIGHT,
            ) = orig
    elif system == "memstack_hybrid_v310":
        res = re_mod.hybrid_rerank(
            q_vec, q_ents, {"valence": "neutral", "intensity": 0},
            cands, top_k=top_k, update_strength=False, query_text=q_text, min_similarity=0.45,
        )
        out = [m.mem_id for m, _ in res]
    elif system == "memstack_rrf_v310":
        res = re_mod.rrf_recall(
            q_vec, q_ents, {"valence": "neutral", "intensity": 0},
            cands, top_k=top_k, query_text=q_text, min_similarity=0.45,
        )
        out = [m.mem_id for m, _ in res]
    else:
        raise ValueError(system)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return out, elapsed_ms


def evaluate(questions: list[dict], cache: EmbedCache) -> dict:
    # Pre-embed all unique session texts and questions in batches.
    t_embed0 = time.perf_counter()
    all_texts = []
    q_texts = []
    per_q_texts = []
    for row in questions:
        q_texts.append(str(row["question"]))
        st = [_format_session(s) for s in row["haystack_sessions"]]
        per_q_texts.append(st)
        all_texts.extend(st)
    cache.get_many(all_texts + q_texts, batch_size=32)
    embed_sec = round(time.perf_counter() - t_embed0, 2)

    systems = ["pure_cosine", "memstack_legacy_v300", "memstack_hybrid_v310", "memstack_rrf_v310"]
    stats = {
        s: {
            "n_answerable": 0, "r1": 0, "r3": 0, "r5": 0, "all5": 0, "mrr": 0.0,
            "n_abs": 0, "abs_empty": 0,
            "lat_ms": [],
            "by_type": collections.defaultdict(lambda: {"n": 0, "r1": 0, "r3": 0, "r5": 0, "all5": 0, "mrr": 0.0}),
        }
        for s in systems
    }

    for idx, row in enumerate(questions):
        q_text = q_texts[idx]
        q_vec = cache.get_many([q_text])[0]
        s_vecs = cache.get_many(per_q_texts[idx])
        cands = _build_candidates(row, per_q_texts[idx], s_vecs)
        q_ents = extract_entity_names(q_text) or []
        is_abs = str(row.get("question_id", "")).endswith("_abs")
        gold = set(str(x) for x in (row.get("answer_session_ids") or []))
        qtype = str(row.get("question_type", "unknown"))

        for sys_name in systems:
            top_ids, lat_ms = _run_system(sys_name, q_text, q_vec, q_ents, cands, top_k=5)
            st = stats[sys_name]
            st["lat_ms"].append(lat_ms)
            if is_abs or not gold:
                st["n_abs"] += 1
                if not top_ids:
                    st["abs_empty"] += 1
                continue
            st["n_answerable"] += 1
            bt = st["by_type"][qtype]
            bt["n"] += 1
            rank = next((r for r, sid in enumerate(top_ids, 1) if sid in gold), None)
            if rank is not None:
                if rank <= 1:
                    st["r1"] += 1
                    bt["r1"] += 1
                if rank <= 3:
                    st["r3"] += 1
                    bt["r3"] += 1
                if rank <= 5:
                    st["r5"] += 1
                    bt["r5"] += 1
                st["mrr"] += 1.0 / rank
                bt["mrr"] += 1.0 / rank
            if gold.issubset(set(top_ids)):
                st["all5"] += 1
                bt["all5"] += 1

    summary = {
        "dataset": "LongMemEval-S (xiaowu0162/longmemeval-cleaned)",
        "embed_model": EMBED_MODEL,
        "questions_evaluated": len(questions),
        "embed_seconds": embed_sec,
        "systems": {},
    }
    for sys_name in systems:
        st = stats[sys_name]
        n = max(1, st["n_answerable"])
        lats = sorted(st["lat_ms"])
        p95 = lats[min(len(lats) - 1, int(math.ceil(0.95 * len(lats))) - 1)] if lats else 0.0
        by_type_out = {}
        for qtype, bt in sorted(st["by_type"].items()):
            bn = max(1, bt["n"])
            by_type_out[qtype] = {
                "n": bt["n"],
                "r1": round(bt["r1"] / bn, 4),
                "r3": round(bt["r3"] / bn, 4),
                "r5": round(bt["r5"] / bn, 4),
                "all5": round(bt["all5"] / bn, 4),
                "mrr": round(bt["mrr"] / bn, 4),
            }
        summary["systems"][sys_name] = {
            "answerable": st["n_answerable"],
            "r1": round(st["r1"] / n, 4),
            "r3": round(st["r3"] / n, 4),
            "r5": round(st["r5"] / n, 4),
            "all5": round(st["all5"] / n, 4),
            "mrr": round(st["mrr"] / n, 4),
            "abstention_cases": st["n_abs"],
            "abstention_rejected_at_045": st["abs_empty"],
            "median_ms": round(statistics.median(lats), 2) if lats else 0.0,
            "p95_ms": round(p95, 2),
            "by_type": by_type_out,
        }
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="Path to longmemeval_s_cleaned.json")
    ap.add_argument("--cache-db", default=None, help="SQLite embedding cache path")
    ap.add_argument("--per-type", type=int, default=20, help="Questions per category (0 = all 500)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    ds_path = ensure_dataset(args.dataset)
    with open(ds_path, encoding="utf-8") as fh:
        data = json.load(fh)
    questions = select_questions(data, args.per_type, seed=args.seed)
    cache_db = args.cache_db or os.path.join(_default_cache_dir(), f"embeds_{EMBED_MODEL}.sqlite")
    cache = EmbedCache(cache_db, model=EMBED_MODEL)
    try:
        report = evaluate(questions, cache)
    finally:
        cache.close()

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(f"=== {report['dataset']} | model={report['embed_model']} | n={report['questions_evaluated']} ===")
    print(f"Embedding time (cached+new): {report['embed_seconds']}s\n")
    print(f"{'System':24s} | {'R@1':7s} | {'R@3':7s} | {'R@5':7s} | {'All@5':7s} | {'MRR@5':7s} | {'Med ms':6s}")
    print("-" * 82)
    for name, s in report["systems"].items():
        print(
            f"{name:24s} | {s['r1']:6.1%} | {s['r3']:6.1%} | {s['r5']:6.1%} | "
            f"{s['all5']:6.1%} | {s['mrr']:7.4f} | {s['median_ms']:6.2f}"
        )
    print("\n[Per-Category Breakdown: memstack_hybrid_v310 vs pure_cosine]")
    hyb = report["systems"]["memstack_hybrid_v310"]["by_type"]
    cos = report["systems"]["pure_cosine"]["by_type"]
    for qtype in sorted(hyb):
        h, c = hyb[qtype], cos[qtype]
        print(
            f"  {qtype:26s} (n={h['n']:2d}) | "
            f"MemStack R@1={h['r1']:5.1%} R@5={h['r5']:5.1%} MRR={h['mrr']:.4f} | "
            f"Cosine R@1={c['r1']:5.1%} R@5={c['r5']:5.1%} MRR={c['mrr']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
