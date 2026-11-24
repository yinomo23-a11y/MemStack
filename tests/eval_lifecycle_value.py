"""Reproducible 210-day Managed Lifecycle vs Unmanaged Append-Only RAG experiment.

Directly tests whether MemStack's lifecycle management (explicit correction
supersede chains, validity time windows, importance-protected retention, and
180-day cold archiving) outperforms an unmanaged append-only vector store over
a 210-day conversational timeline.

Supports both:
    --embed offline   Deterministic local char/token n-gram vectors (zero deps, CI-safe)
    --embed ollama    Real local Ollama bge-m3 1024-dim embeddings

Usage:
    python tests/eval_lifecycle_value.py --embed offline
    python tests/eval_lifecycle_value.py --embed ollama
    python tests/eval_lifecycle_value.py --embed ollama --json
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
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

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")


# ── Deterministic offline embedder (char + word n-gram hashing, 256-dim) ─────

def _offline_embed(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    norm = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", norm)
    feats = list(tokens)
    for i in range(len(tokens) - 1):
        feats.append(tokens[i] + "_" + tokens[i + 1])
    for tok in feats:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) else -1.0
        vec[idx] += sign
    n = math.sqrt(sum(v * v for v in vec))
    return [v / n for v in vec] if n > 0 else vec


def _make_embedder(mode: str):
    cache: dict[str, list[float]] = {}
    if mode == "offline":
        def embed_batch(texts: list[str]) -> list[list[float]]:
            out = []
            for t in texts:
                if t not in cache:
                    cache[t] = _offline_embed(t)
                out.append(cache[t])
            return out
        return embed_batch

    def embed_batch_ollama(texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        for off in range(0, len(missing), 32):
            batch = missing[off : off + 32]
            req = urllib.request.Request(
                f"{OLLAMA_BASE}/api/embed",
                data=json.dumps({"model": EMBED_MODEL, "input": batch}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                embs = json.loads(resp.read().decode("utf-8")).get("embeddings") or []
            if len(embs) != len(batch):
                raise RuntimeError("Ollama /api/embed count mismatch")
            for t, v in zip(batch, embs):
                cache[t] = v
        return [cache[t] for t in texts]

    return embed_batch_ollama


# ── Scenario definitions (100% synthetic, zero personal data) ────────────────

UPDATE_CASES = [
    ("city", "Where do I live now?",
     "I currently live in Shanghai near Jing'an district.",
     "Update: I moved from Shanghai to Hangzhou Binjiang.",
     "Latest residence: I relocated to Shenzhen Nanshan this quarter."),
    ("role", "What is my current job role?",
     "My current job role is frontend engineer on the web team.",
     "Job update: I switched to backend infrastructure engineer.",
     "Current role: staff reliability engineer leading storage systems."),
    ("db", "Which database does our production service use?",
     "Our production service uses MySQL 8.0 for primary storage.",
     "Architecture change: migrated primary storage from MySQL to PostgreSQL 16.",
     "Final production database choice is SQLite WAL for local-first nodes."),
    ("port", "What port does the staging gateway listen on?",
     "The staging gateway listens on port 8080.",
     "Staging gateway port moved to 9090 after firewall rules changed.",
     "Staging gateway now runs on port 7443 with TLS enabled."),
    ("standup", "When is our weekly team standup?",
     "Our weekly team standup is every Monday at 9:30 AM.",
     "Moved weekly team standup to Tuesday at 10:00 AM.",
     "Weekly team standup is now fixed at Wednesday 11:00 AM."),
    ("editor", "What code editor do I prefer?",
     "My preferred code editor is Sublime Text for daily coding.",
     "Switched my preferred code editor to VS Code.",
     "Now I prefer Neovim as my primary daily code editor."),
    ("allergy", "What food allergy do I need to avoid?",
     "Food allergy note: I avoid peanuts in all meals.",
     "Medical update: peanut test was negative, but shellfish triggers hives.",
     "Confirmed allergy record: strictly avoid sesame oil and shellfish."),
    ("branch", "What is the default release git branch?",
     "Our default release git branch is named master.",
     "Renamed the default release git branch to release-prod.",
     "Default release git branch is standardized to main."),
    ("coffee", "How do I usually take my coffee?",
     "I usually order an iced caramel latte with whole milk.",
     "Switched my daily coffee order to hot oat flat white.",
     "Current coffee preference is black pour-over with no sugar."),
    ("backup", "How often do automated backups run?",
     "Automated backups run once a week on Sunday night.",
     "Automated backups changed to twice a week on Wednesday and Sunday.",
     "Automated backups now run every night at 02:00 UTC."),
    ("lead", "Who is the current mobile project lead?",
     "Current mobile project lead is Alice Chen.",
     "Mobile project lead handed over to David Park.",
     "Current mobile project lead is Elena Rivera."),
    ("budget", "What is our monthly cloud budget cap?",
     "Our monthly cloud budget cap is set to 500 USD.",
     "Raised the monthly cloud budget cap to 1200 USD for launch.",
     "Trimmed the monthly cloud budget cap down to 800 USD."),
    ("gym", "Which days do I go to the gym?",
     "I go to the gym on Monday and Thursday evenings.",
     "Adjusted gym schedule to Tuesday and Friday mornings.",
     "Current gym routine is Wednesday, Saturday, and Sunday mornings."),
    ("lang", "What primary programming language are we using for the CLI?",
     "The CLI tool is written in Node.js TypeScript.",
     "Rewrote the CLI prototype in Go 1.22.",
     "Final CLI implementation is maintained in Python 3.12."),
    ("office", "Which building is my desk located in?",
     "My desk is located in Building A on the 3rd floor.",
     "Moved desks to Building B on the 5th floor.",
     "Current desk location is Building C room 204."),
]

SCHEDULE_CASES = [
    ("sched_mon", "What is my Monday class or meeting schedule this week?",
     ["Monday schedule (expired week): 09:00 Linear Algebra, 14:00 Lab Prep.",
      "Monday schedule (expired week): 10:00 Compilers, 15:00 Design Review.",
      "Monday schedule (expired week): 08:30 Physics Lab, 13:30 Reading Group."],
     "Monday schedule (current active week): 10:00 Distributed Systems, 16:00 Sprint Demo."),
    ("sched_tue", "What is scheduled for Tuesday morning this week?",
     ["Tuesday morning (expired week): 09:00 dentist checkup downtown.",
      "Tuesday morning (expired week): 09:30 vendor onboarding call."],
     "Tuesday morning (current active week): 09:00 architecture review with platform team."),
    ("sched_wed", "What is my Wednesday afternoon plan this week?",
     ["Wednesday afternoon (expired week): 14:00 database migration rehearsal.",
      "Wednesday afternoon (expired week): 15:00 quarterly tax filing."],
     "Wednesday afternoon (current active week): 14:30 benchmark review and release prep."),
    ("sched_thu", "What deadline falls on Thursday this week?",
     ["Thursday deadline (expired week): submit Q1 expense receipts.",
      "Thursday deadline (expired week): publish v2.0 migration checklist."],
     "Thursday deadline (current active week): freeze v3.2 release notes by 18:00."),
    ("sched_fri", "What is planned for Friday evening this week?",
     ["Friday evening (expired week): badminton session at West Court.",
      "Friday evening (expired week): airport pickup for visiting colleague."],
     "Friday evening (current active week): team dinner at Riverside Bistro at 19:00."),
    ("oncall", "Who is on-call for production alerts this week?",
     ["Production on-call rotation (expired week): Marcus is primary on-call.",
      "Production on-call rotation (expired week): Priya is primary on-call."],
     "Production on-call rotation (current active week): Hannah is primary on-call."),
    ("promo", "What active discount campaign is running this week?",
     ["Discount campaign (expired week): Spring 15% off coupon SPRING15.",
      "Discount campaign (expired week): Flash 20% off coupon FLASH20."],
     "Discount campaign (current active week): Autumn bundle 10% off coupon AUTUMN10."),
    ("sprint", "What is the current sprint goal this week?",
     ["Sprint goal (expired sprint): refactor JSON file locking retries.",
      "Sprint goal (expired sprint): add SQLite vector cache invalidation."],
     "Sprint goal (current active sprint): ship dynamic recall cutoff and LongMemEval suite."),
    ("parking", "What is the temporary visitor parking code this week?",
     ["Visitor parking code (expired week): gate PIN 4821.",
      "Visitor parking code (expired week): gate PIN 7390."],
     "Visitor parking code (current active week): gate PIN 6154."),
    ("wifi", "What is the guest Wi-Fi password for the conference room this week?",
     ["Guest Wi-Fi password (expired week): MapleTree-402.",
      "Guest Wi-Fi password (expired week): SilverLake-819."],
     "Guest Wi-Fi password (current active week): CedarPeak-930."),
    ("duty", "Who leads the code review rotation this week?",
     ["Code review rotation (expired week): Kevin leads PR triage.",
      "Code review rotation (expired week): Sofia leads PR triage."],
     "Code review rotation (current active week): Noah leads PR triage."),
    ("maint", "When is the scheduled maintenance window this week?",
     ["Maintenance window (expired week): Saturday 01:00-02:00 UTC.",
      "Maintenance window (expired week): Sunday 03:00-04:00 UTC."],
     "Maintenance window (current active week): Sunday 04:00-05:00 UTC."),
]

CORE_FACTS = [
    ("core_1", "What is my emergency contact phone policy?",
     "Core rule: always notify my sister Maya first in any medical emergency."),
    ("core_2", "What is my non-negotiable privacy policy for local memory?",
     "Core privacy rule: personal memory files must stay local-only and never sync to third-party clouds."),
    ("core_3", "What operating system constraint applies to my deployment?",
     "Core system fact: production workstation runs Windows 11 with Python 3.12 and local Ollama."),
    ("core_4", "What is my long-term goal for the memory engine?",
     "Core design goal: keep active memory lean through decay and cold archiving without losing key identity facts."),
    ("core_5", "What backup rule must run before schema migrations?",
     "Core operational rule: always snapshot active.json and cold.json before running any schema migration."),
    ("core_6", "What naming convention do we use for release tags?",
     "Core release policy: tag every public milestone with semantic version vX.Y.Z after strict privacy scan passes."),
    ("core_7", "What is my favorite annual holiday destination?",
     "Long-term personal fact: my favorite annual winter getaway is hot springs in Hakone."),
    ("core_8", "What musical instrument have I played since childhood?",
     "Long-term personal fact: I have played classical acoustic guitar for over twelve years."),
    ("core_9", "What pet do I have at home?",
     "Long-term personal fact: I have an orange tabby cat named Mochi who loves cardboard boxes."),
    ("core_10", "What marathon milestone did I complete?",
     "Long-term personal milestone: I finished my first half-marathon in 1 hour 52 minutes."),
]

TRIVIA_TEMPLATES = [
    "Casual chat day {d}: weather felt slightly humid outside while walking to lunch.",
    "Casual chat day {d}: traffic near the main intersection was slow for ten minutes.",
    "Casual chat day {d}: had a quick noodle bowl and scrolled news during break.",
    "Casual chat day {d}: tried a new sparkling water flavor from the corner store.",
    "Casual chat day {d}: random chatter about weekend movie trailers and popcorn.",
    "Casual chat day {d}: minor reminder to stretch shoulders after staring at monitor.",
    "Casual chat day {d}: passing thought about desk lamp brightness and keyboard click.",
    "Casual chat day {d}: overheard someone talking about coffee prices and schedules.",
    "Casual chat day {d}: briefly checked transit app for bus arrival estimate.",
    "Casual chat day {d}: idle remark about clouds clearing up before sunset.",
]


def run_experiment(embed_mode: str = "offline", n_trivia: int = 240) -> dict:
    tmp_dir = tempfile.mkdtemp(prefix="mm_lifecycle_eval_")
    old_env = os.environ.get("MEMORY_DATA_DIR")
    os.environ["MEMORY_DATA_DIR"] = tmp_dir
    try:
        # Import with MEMORY_DATA_DIR pointed at our isolated temp store.
        import importlib
        import config
        import memory_manager
        import recall_engine
        from decay_logic import calculate_current_strength, should_archive
        from schema import MemoryItem
        from services.correction_service import correct
        from services.memory_tools import mcp_add
        from services.recall_policy import RecallContext, select as filter_records

        importlib.reload(config)
        importlib.reload(memory_manager)

        embed_batch = _make_embedder(embed_mode)

        # Pre-compute all texts in batches so Ollama mode is fast.
        all_texts = []
        for _, q, v1, v2, v3 in UPDATE_CASES:
            all_texts.extend([q, v1, v2, v3])
        for _, q, expired_list, curr in SCHEDULE_CASES:
            all_texts.extend([q, *expired_list, curr])
        for _, q, fact in CORE_FACTS:
            all_texts.extend([q, fact])
        trivia_texts = [
            TRIVIA_TEMPLATES[i % len(TRIVIA_TEMPLATES)].format(d=(i % 85) + 1)
            for i in range(n_trivia)
        ]
        all_texts.extend(trivia_texts)
        embed_batch(all_texts)

        def single_embed(txt: str) -> list[float]:
            return embed_batch([txt])[0]

        mgr = memory_manager.MemoryManager(embedding_func=single_embed)
        now = time.time()
        day = 86400.0

        # Also maintain the Unmanaged Append-Only Vector Store baseline with identical texts/vectors.
        unmanaged_pool: list[MemoryItem] = []

        def _append_unmanaged(mem_id: str, text: str, create_ts: float, status_tag: str):
            m = MemoryItem(content=text, mem_type="long_term", create_time=create_ts, last_recalled_at=create_ts)
            m.mem_id = mem_id
            m.embedding = single_embed(text)
            m.context = {"eval_status": status_tag}
            unmanaged_pool.append(m)

        # ── 1. Ingest Knowledge-Update / Correction chains (Days 1 -> 30 -> 60) ──
        update_truth = {}
        for key, q, v1, v2, v3 in UPDATE_CASES:
            r1 = mcp_add(mgr, {"session_id": "life", "content": v1, "is_manual": True, "operation_id": f"{key}_1"})
            r2 = correct(mgr, r1["mem_id"], v2, operation_id=f"{key}_2", reason="updated fact", session_id="life")
            r3 = correct(mgr, r2["mem_id"], v3, operation_id=f"{key}_3", reason="latest fact", session_id="life")
            update_truth[key] = {
                "query": q,
                "current_id": r3["mem_id"],
                "stale_ids": {r1["mem_id"], r2["mem_id"]},
            }
            _append_unmanaged(r1["mem_id"], v1, now - 89 * day, "stale")
            _append_unmanaged(r2["mem_id"], v2, now - 60 * day, "stale")
            _append_unmanaged(r3["mem_id"], v3, now - 30 * day, "current")

        # ── 2. Ingest Time-Bounded Weekly Schedules (expired past weeks + current active week) ──
        schedule_truth = {}
        for key, q, expired_list, curr in SCHEDULE_CASES:
            stale_ids = set()
            for idx, exp_txt in enumerate(expired_list):
                weeks_ago = len(expired_list) - idx
                ef = now - (weeks_ago * 7 + 3) * day
                et = now - (weeks_ago * 7 - 3) * day
                res = mcp_add(mgr, {
                    "session_id": "life", "content": exp_txt,
                    "effective_from": ef, "effective_to": et,
                    "operation_id": f"{key}_exp_{idx}",
                })
                stale_ids.add(res["mem_id"])
                _append_unmanaged(res["mem_id"], exp_txt, ef, "expired")
            res_cur = mcp_add(mgr, {
                "session_id": "life", "content": curr,
                "effective_from": now - 2 * day, "effective_to": now + 5 * day,
                "operation_id": f"{key}_cur",
            })
            schedule_truth[key] = {
                "query": q,
                "current_id": res_cur["mem_id"],
                "stale_ids": stale_ids,
            }
            _append_unmanaged(res_cur["mem_id"], curr, now - 2 * day, "current")

        # ── 3. Ingest High-Importance Core Facts (Day 1) + 90-Day Low-Importance Trivia ──
        core_truth = {}
        for key, q, fact in CORE_FACTS:
            res = mcp_add(mgr, {"session_id": "life", "content": fact, "is_manual": True, "operation_id": key})
            core_truth[key] = {"query": q, "current_id": res["mem_id"], "stale_ids": set()}
            _append_unmanaged(res["mem_id"], fact, now - 88 * day, "current")

        trivia_ids = set()
        for i, txt in enumerate(trivia_texts):
            # Span 5..215 days old so idle trivia >= 180d triggers real cold archiving.
            age_days = 215.0 - (i % 210)
            ts = now - age_days * day
            m = MemoryItem(
                content=txt, mem_type="long_term", source="auto",
                consolidation_level=1, strength=60.0, importance=0.15,
                create_time=ts, last_recalled_at=ts, session_id="life",
            )
            m.last_active = ts
            m.strength_at = ts
            m.embedding = single_embed(txt)
            mgr.active_mems.append(m)
            trivia_ids.add(m.mem_id)
            _append_unmanaged(m.mem_id, txt, ts, "trivia")

        # Simulate 210-day lifecycle decay & cold-archiving pass on MemStack active_mems.
        # Backdate core facts by 205 days (>180d) with importance=0.90 to verify protection.
        core_ids = {v["current_id"] for v in core_truth.values()}
        kept_active = []
        archived_cold = []
        for m in mgr.active_mems:
            if m.mem_id in core_ids:
                m.importance = 0.90
                m.consolidation_level = 3
                m.create_time = now - 205 * day
                m.last_active = now - 205 * day
                m.last_recalled_at = now - 205 * day
                m.strength_at = now - 205 * day
            if m.mem_id in trivia_ids:
                updated, _ = calculate_current_strength(m, current_timestamp=now)
                if should_archive(updated, current_timestamp=now):
                    archived_cold.append(updated)
                    continue
                kept_active.append(updated)
            elif m.mem_id in core_ids:
                updated, _ = calculate_current_strength(m, current_timestamp=now)
                assert not should_archive(updated, current_timestamp=now), "high-importance core fact archived!"
                kept_active.append(updated)
            else:
                kept_active.append(m)
        mgr.active_mems = kept_active
        mgr.cold_mems.extend(archived_cold)

        # Query at current wall-clock time AFTER all writes & corrections settled.
        eval_now = time.time() + 1.0
        min_sim = 0.45 if embed_mode == "ollama" else 0.15

        # ── Evaluate MemStack (managed) vs Unmanaged Append-Only Vector Store ──
        def eval_bucket(cases_dict: dict, label: str):
            ms_r1 = ms_r5 = ms_stale_slots = ms_total_slots = 0
            un_r1 = un_r5 = un_stale_slots = un_total_slots = 0
            ms_lats, un_lats = [], []
            n = len(cases_dict)

            for item in cases_dict.values():
                q = item["query"]
                qv = single_embed(q)
                curr_id = item["current_id"]
                stale_ids = item["stale_ids"]

                # MemStack managed recall: policy eligibility filter -> v3.2 hybrid_rerank
                t0 = time.perf_counter()
                ctx = RecallContext(query=q, session_id="life", as_of=eval_now)
                eligible_cands = filter_records(mgr.active_mems, ctx, pool="active")
                ms_res = recall_engine.hybrid_rerank(
                    qv, [], {"valence": "neutral", "intensity": 0},
                    eligible_cands, top_k=5, update_strength=False,
                    query_text=q, min_similarity=min_sim,
                )
                ms_lats.append((time.perf_counter() - t0) * 1000.0)
                ms_ids = [m.mem_id for m, _ in ms_res]
                if ms_ids and ms_ids[0] == curr_id:
                    ms_r1 += 1
                if curr_id in ms_ids:
                    ms_r5 += 1
                ms_total_slots += len(ms_ids)
                ms_stale_slots += sum(1 for mid in ms_ids if mid in stale_ids)

                # Unmanaged append-only vector store: pure cosine over everything ever written
                t0 = time.perf_counter()
                sims = recall_engine._batch_cosine_similarities(qv, unmanaged_pool)
                ranked = sorted(zip(unmanaged_pool, sims), key=lambda x: x[1], reverse=True)[:5]
                un_lats.append((time.perf_counter() - t0) * 1000.0)
                un_ids = [m.mem_id for m, _ in ranked]
                if un_ids and un_ids[0] == curr_id:
                    un_r1 += 1
                if curr_id in un_ids:
                    un_r5 += 1
                un_total_slots += len(un_ids)
                un_stale_slots += sum(1 for mid in un_ids if mid in stale_ids)

            return {
                "scenario": label,
                "queries": n,
                "memstack_r1": round(ms_r1 / n, 4),
                "memstack_r5": round(ms_r5 / n, 4),
                "memstack_stale_rate_at5": round(ms_stale_slots / max(1, ms_total_slots), 4),
                "memstack_median_ms": round(statistics.median(ms_lats), 2),
                "unmanaged_r1": round(un_r1 / n, 4),
                "unmanaged_r5": round(un_r5 / n, 4),
                "unmanaged_stale_rate_at5": round(un_stale_slots / max(1, un_total_slots), 4),
                "unmanaged_median_ms": round(statistics.median(un_lats), 2),
            }

        b_upd = eval_bucket(update_truth, "knowledge_updates_and_corrections")
        b_sch = eval_bucket(schedule_truth, "expired_time_bounded_schedules")
        b_cor = eval_bucket(core_truth, "core_facts_after_210d_decay")
        buckets = [b_upd, b_sch, b_cor]

        total_q = sum(b["queries"] for b in buckets)
        ms_overall_r1 = sum(b["memstack_r1"] * b["queries"] for b in buckets) / total_q
        un_overall_r1 = sum(b["unmanaged_r1"] * b["queries"] for b in buckets) / total_q

        # Count eligible active items (excluding superseded/expired) vs unmanaged total
        eligible_now = len(filter_records(mgr.active_mems, RecallContext(query="", session_id="life", as_of=eval_now), pool="active"))
        return {
            "embed_mode": embed_mode,
            "embed_model": EMBED_MODEL if embed_mode == "ollama" else "offline-ngram-256d",
            "simulated_days": 210,
            "pool_stats": {
                "unmanaged_total_items": len(unmanaged_pool),
                "memstack_active_eligible_items": eligible_now,
                "memstack_cold_archived_items": len(archived_cold),
                "active_search_space_reduction": round(1.0 - eligible_now / max(1, len(unmanaged_pool)), 4),
            },
            "overall": {
                "queries": total_q,
                "memstack_current_truth_r1": round(ms_overall_r1, 4),
                "unmanaged_current_truth_r1": round(un_overall_r1, 4),
                "gain_r1": round(ms_overall_r1 - un_overall_r1, 4),
            },
            "scenarios": buckets,
        }
    finally:
        if old_env is None:
            os.environ.pop("MEMORY_DATA_DIR", None)
        else:
            os.environ["MEMORY_DATA_DIR"] = old_env
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", choices=("offline", "ollama"), default="offline")
    ap.add_argument("--trivia", type=int, default=240)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    report = run_experiment(embed_mode=args.embed, n_trivia=args.trivia)

    # Hard lifecycle assertions so CI catches regressions.
    by_name = {s["scenario"]: s for s in report["scenarios"]}
    assert by_name["knowledge_updates_and_corrections"]["memstack_stale_rate_at5"] == 0.0, by_name
    assert by_name["expired_time_bounded_schedules"]["memstack_stale_rate_at5"] == 0.0, by_name
    assert report["overall"]["memstack_current_truth_r1"] > report["overall"]["unmanaged_current_truth_r1"], report["overall"]
    assert report["pool_stats"]["memstack_cold_archived_items"] > 0, report["pool_stats"]

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    ps = report["pool_stats"]
    ov = report["overall"]
    print(f"=== MemStack 210-Day Managed Lifecycle vs Unmanaged RAG ({report['embed_model']}) ===")
    print(
        f"Active search space: MemStack={ps['memstack_active_eligible_items']} eligible items "
        f"({ps['memstack_cold_archived_items']} trivia archived to cold) vs "
        f"Unmanaged={ps['unmanaged_total_items']} items "
        f"(-{ps['active_search_space_reduction']:.1%} search space)\n"
    )
    print(f"{'Scenario':34s} | {'MS R@1':7s} | {'Unm R@1':7s} | {'MS Stale@5':10s} | {'Unm Stale@5':11s}")
    print("-" * 82)
    for s in report["scenarios"]:
        print(
            f"{s['scenario']:34s} | {s['memstack_r1']:6.1%} | {s['unmanaged_r1']:7.1%} | "
            f"{s['memstack_stale_rate_at5']:9.1%} | {s['unmanaged_stale_rate_at5']:10.1%}"
        )
    print("-" * 82)
    print(
        f"{'OVERALL CURRENT-TRUTH R@1':34s} | {ov['memstack_current_truth_r1']:6.1%} | "
        f"{ov['unmanaged_current_truth_r1']:7.1%} | gain = +{ov['gain_r1']:.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
