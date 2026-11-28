# -*- coding: utf-8 -*-
"""Targeted offline regressions for the 4 memory-model audit findings.

Uses only temporary data directories, synthetic data, and mock embeddings.
Never touches production memory directories or online services.
"""
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

MOCK_VEC = [1.0, 0.2, 0.0, 0.0]


def _mock_embed(_text):
    return list(MOCK_VEC)


class AuditRepairBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="audit-repair-4-")
        self.root = Path(self.tmp)
        (self.root / "state").mkdir(exist_ok=True)
        for name in ("active.json", "cold.json", "shortterm.json"):
            (self.root / name).write_text(json.dumps({"memories": []}), encoding="utf-8")
        (self.root / "working_mem.json").write_text(json.dumps({"sessions": {}}), encoding="utf-8")

        self._env = patch.dict(os.environ, {
            "MEMORY_DATA_DIR": self.tmp,
            "PYTHONUTF8": "1",
        })
        self._env.start()

        import config
        import pending_longterm
        import candidate_pool
        import memory_manager
        import daily_consolidation
        import distillation
        from services import deletion_service, vector_cache

        for mod in (config, pending_longterm, candidate_pool, memory_manager,
                    daily_consolidation, distillation):
            importlib.reload(mod)
        deletion_service._cache = None
        if hasattr(vector_cache, "_CACHE"):
            vector_cache._CACHE.clear()

        self.config = config
        self.pl = pending_longterm
        self.mm = memory_manager
        self.dc = daily_consolidation
        self.distill = distillation

    def tearDown(self):
        try:
            self.pl.release_pending(self.pl.read_pending())
        except Exception:
            pass
        self._env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def read_json(self, rel):
        return json.loads((self.root / rel).read_text(encoding="utf-8"))

    def write_json(self, rel, obj):
        (self.root / rel).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    def run_cli(self, action, payload):
        env = dict(os.environ, MEMORY_DATA_DIR=self.tmp, PYTHONUTF8="1")
        cli_path = str(Path(__file__).resolve().with_name("memory_cli.py"))
        proc = subprocess.run(
            [sys.executable, "-B", cli_path, action],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True, text=True, encoding="utf-8", env=env, timeout=20,
        )
        return proc.returncode, json.loads(proc.stdout)


class TestFinding1ConsolidationCrashConsistency(AuditRepairBase):
    """P1: consolidation crash consistency across bidirectional active/cold, working, shortterm, pending."""

    def _seed_full_scenario(self):
        from schema import MemoryItem
        now = time.time()

        # 1. Active record slated for merge reinforcement from working memory
        exist_active = MemoryItem(
            "合成测试：周三下午三点整理周报模板",
            mem_type="long_term", session_id="private:owner",
            strength=50.0, create_time=now - 3600, importance=0.8,
        )
        exist_active.last_active = now
        exist_active.strength_at = now
        exist_active.embedding = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

        # 2. Old active record slated for archiving to cold (> 180d inactive, low importance)
        old_active = MemoryItem(
            "合成测试：半年前随口提过的旧纸箱位置",
            mem_type="long_term", session_id="private:owner",
            strength=15.0, create_time=now - 220 * 86400, importance=0.4,
        )
        old_active.last_active = now - 220 * 86400
        old_active.strength_at = now - 220 * 86400
        old_active.last_recalled_at = now - 220 * 86400
        old_active.embedding = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        # 3. Cold record slated for auto-awakening (important + recently active, no protected archive_reason)
        cold_awaken = MemoryItem(
            "合成测试：重要纪念日备忘记录需要唤醒",
            mem_type="long_term", session_id="private:owner",
            strength=40.0, create_time=now - 10 * 86400, importance=0.92,
        )
        cold_awaken.last_active = now - 3600
        cold_awaken.strength_at = now - 3600
        cold_awaken.last_recalled_at = now - 3600
        cold_awaken.embedding = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

        # 4. Working memories: one new promotion, one duplicate merge-reinforcing exist_active
        w_new = MemoryItem(
            "合成测试：新项目部署检查清单已定稿",
            mem_type="working", session_id="private:owner",
            source="bridge_auto", chunk_priority=5, create_time=now - 60,
        )
        w_new.embedding = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        w_merge = MemoryItem(
            exist_active.content,
            mem_type="working", session_id="private:owner",
            source="bridge_auto", chunk_priority=5, create_time=now - 30,
        )
        w_merge.embedding = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

        # 5. Expired important short-term memory slated for promotion to active
        st_expired = MemoryItem(
            "合成测试：他不喜欢吃香菜和芹菜",
            mem_type="short_term", session_id="private:owner",
            source="bridge_auto", create_time=now - 10 * 86400,
        )
        st_expired.last_mentioned_at = now - 10 * 86400
        st_expired.embedding = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        # 6. Pending commitment record
        self.assertTrue(self.pl.append_pending(
            "我答应下周五整理完归档清单",
            session_id="private:owner", category_at_write="commitment",
            source="auto", create_time=now - 120,
        ))

        self.write_json("active.json", {"memories": [exist_active.to_dict(), old_active.to_dict()]})
        self.write_json("cold.json", {"memories": [cold_awaken.to_dict()]})
        self.write_json("shortterm.json", {"memories": [st_expired.to_dict()]})
        self.write_json("working_mem.json", {
            "sessions": {
                "private:owner": {
                    "working_memory": [w_new.to_dict(), w_merge.to_dict()],
                    "last_active": now,
                }
            }
        })
        return {
            "exist_active": exist_active.mem_id,
            "old_active": old_active.mem_id,
            "cold_awaken": cold_awaken.mem_id,
            "w_new": w_new.mem_id,
            "w_merge": w_merge.mem_id,
            "st_expired": st_expired.mem_id,
        }

    def _assert_converged(self, ids):
        active = {m["mem_id"]: m for m in self.read_json("active.json")["memories"]}
        cold = {m["mem_id"]: m for m in self.read_json("cold.json")["memories"]}
        short = {m["mem_id"]: m for m in self.read_json("shortterm.json")["memories"]}
        working = {
            m["mem_id"]: m
            for m in self.read_json("working_mem.json")["sessions"]["private:owner"]["working_memory"]
        }

        # Cold-to-active awakening succeeded and removed from cold
        self.assertIn(ids["cold_awaken"], active)
        self.assertNotIn(ids["cold_awaken"], cold)
        # Active-to-cold archive succeeded and removed from active
        self.assertIn(ids["old_active"], cold)
        self.assertNotIn(ids["old_active"], active)
        # Working new promoted once (no self-merge +15 inflation!)
        self.assertIn(ids["w_new"], active)
        self.assertAlmostEqual(active[ids["w_new"]]["strength"], 55.0, delta=0.5)
        # Working merge reinforcement applied EXACTLY ONCE (50 + 15 = 65, never 80)
        self.assertIn(ids["exist_active"], active)
        self.assertAlmostEqual(active[ids["exist_active"]]["strength"], 65.0, delta=0.5)
        # Working items marked _已巩固
        self.assertIn("_已巩固", working[ids["w_new"]]["tags"])
        self.assertIn("_已巩固", working[ids["w_merge"]]["tags"])
        # Expired important short-term promoted once & removed from shortterm
        self.assertIn(ids["st_expired"], active)
        self.assertNotIn(ids["st_expired"], short)
        # Pending committed once & queue empty
        pending_matches = [m for m in active.values() if "下周五整理完归档清单" in m.get("content", "")]
        self.assertEqual(len(pending_matches), 1)
        self.assertEqual(self.pl.read_pending(), [])

    def _run_consolidate(self):
        with patch("ollama_embedding.get_embedding", side_effect=lambda _: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), \
             patch.object(self.dc, "run_episode_aggregation", return_value=[]):
            return self.dc.process_working_memory_to_long_term()

    def test_active_write_failure_preserves_cold_awaken_working_shortterm_pending(self):
        ids = self._seed_full_scenario()
        orig_save = self.dc._save_json

        def fail_active(path, data):
            if os.path.abspath(path) == os.path.abspath(self.dc.LONG_TERM_ACTIVE_PATH):
                raise OSError("injected active failure")
            return orig_save(path, data)

        with patch.object(self.dc, "_save_json", side_effect=fail_active):
            with self.assertRaises(OSError):
                self._run_consolidate()

        # After failure: cold_awaken MUST still exist on disk; working MUST NOT be marked _已巩固;
        # shortterm expired record & pending record MUST still be retryable.
        active = {m["mem_id"]: m for m in self.read_json("active.json")["memories"]}
        cold = {m["mem_id"]: m for m in self.read_json("cold.json")["memories"]}
        short = {m["mem_id"]: m for m in self.read_json("shortterm.json")["memories"]}
        working = {
            m["mem_id"]: m
            for m in self.read_json("working_mem.json")["sessions"]["private:owner"]["working_memory"]
        }
        self.assertTrue(ids["cold_awaken"] in cold or ids["cold_awaken"] in active)
        self.assertTrue(ids["old_active"] in active or ids["old_active"] in cold)
        self.assertNotIn("_已巩固", working[ids["w_new"]].get("tags", []))
        self.assertNotIn("_已巩固", working[ids["w_merge"]].get("tags", []))
        self.assertIn(ids["st_expired"], short)
        self.assertEqual(len(self.pl.read_pending()), 1)
        self.assertAlmostEqual(active[ids["exist_active"]]["strength"], 50.0, delta=0.1)

        # Rerun converges cleanly and commits each transition exactly once
        self._run_consolidate()
        self._assert_converged(ids)
        # Second rerun is a clean no-op
        self._run_consolidate()
        self._assert_converged(ids)

    def test_fault_injection_each_post_stage_converges_once(self):
        stages = ("cold_staged", "cold_final", "shortterm", "working", "pending_commit")
        for stage in stages:
            with self.subTest(stage=stage):
                self.tearDown()
                self.setUp()
                ids = self._seed_full_scenario()
                orig_save = self.dc._save_json
                cold_calls = [0]

                def fail_at(path, data, _stage=stage, _cold_calls=cold_calls, _orig=orig_save):
                    ap = os.path.abspath(path)
                    if ap == os.path.abspath(self.dc.LONG_TERM_COLD_PATH):
                        _cold_calls[0] += 1
                        if _stage == "cold_staged" and _cold_calls[0] == 1:
                            raise OSError("injected staged cold failure")
                        if _stage == "cold_final" and _cold_calls[0] == 2:
                            raise OSError("injected final cold failure")
                    elif _stage == "shortterm" and ap == os.path.abspath(self.dc.SHORT_TERM_PATH):
                        raise OSError("injected shortterm failure")
                    elif _stage == "working" and ap == os.path.abspath(self.dc.WORKING_MEM_PATH):
                        raise OSError("injected working failure")
                    return _orig(path, data)

                if stage == "pending_commit":
                    with patch.object(self.pl, "commit_pending", side_effect=OSError("injected commit failure")):
                        with self.assertRaises(OSError):
                            self._run_consolidate()
                else:
                    with patch.object(self.dc, "_save_json", side_effect=fail_at):
                        with self.assertRaises(OSError):
                            self._run_consolidate()

                # At every failure point, every migrated record has at least one durable copy or pending job
                active = {m["mem_id"]: m for m in self.read_json("active.json")["memories"]}
                cold = {m["mem_id"]: m for m in self.read_json("cold.json")["memories"]}
                short = {m["mem_id"]: m for m in self.read_json("shortterm.json")["memories"]}
                self.assertTrue(ids["cold_awaken"] in active or ids["cold_awaken"] in cold)
                self.assertTrue(ids["old_active"] in active or ids["old_active"] in cold)
                self.assertTrue(ids["st_expired"] in active or ids["st_expired"] in short)
                self.assertEqual(len(self.pl.read_pending()), 1)

                # Recovery rerun converges cleanly with single reinforcement
                self._run_consolidate()
                self._assert_converged(ids)


class TestFinding2ShorttermSessionIsolation(AuditRepairBase):
    """P1: add_shortterm_ex session boundary, legacy compatibility, and receipt scope."""

    def test_two_groups_isolated_mentions_recall_and_retire(self):
        from services.deletion_service import retire
        from services.recall_service import recall_core

        mgr = self.mm.MemoryManager(embedding_func=_mock_embed)
        text = "合成消息：今晚八点群里一起联机打副本"
        r1 = mgr.add_shortterm_ex(text, session_id="group:1001", source="bridge_auto")
        r2 = mgr.add_shortterm_ex(text, session_id="group:2002", source="bridge_auto")

        self.assertEqual(r1["reason"], "created")
        self.assertEqual(r2["reason"], "created")
        self.assertNotEqual(r1["mem_id"], r2["mem_id"])

        # Re-mentioning in group:1001 increments ONLY group:1001
        r1_dup = mgr.add_shortterm_ex(text, session_id="group:1001", source="bridge_auto")
        self.assertEqual(r1_dup["reason"], "dedup_updated")
        self.assertEqual(r1_dup["mem_id"], r1["mem_id"])

        by_id = {m.mem_id: m for m in mgr._load_shortterm_mems()}
        self.assertEqual(by_id[r1["mem_id"]].mention_count, 1)
        self.assertEqual(by_id[r2["mem_id"]].mention_count, 0)

        # Recall isolation per group
        rec1 = recall_core(mgr, "今晚八点联机", session_id="group:1001", embedding_func=_mock_embed)
        rec2 = recall_core(mgr, "今晚八点联机", session_id="group:2002", embedding_func=_mock_embed)
        ids1 = {m.mem_id for m, _ in rec1.get("_shortterm_results", [])}
        ids2 = {m.mem_id for m, _ in rec2.get("_shortterm_results", [])}
        self.assertIn(r1["mem_id"], ids1)
        self.assertNotIn(r2["mem_id"], ids1)
        self.assertIn(r2["mem_id"], ids2)
        self.assertNotIn(r1["mem_id"], ids2)

        # Retiring group:1001 record must NOT affect group:2002
        retire(mgr, [r1["mem_id"]], operation_id="retire-g1")
        rec1_after = recall_core(mgr, "今晚八点联机", session_id="group:1001", embedding_func=_mock_embed)
        rec2_after = recall_core(mgr, "今晚八点联机", session_id="group:2002", embedding_func=_mock_embed)
        self.assertEqual({m.mem_id for m, _ in rec1_after.get("_shortterm_results", [])}, set())
        self.assertIn(r2["mem_id"], {m.mem_id for m, _ in rec2_after.get("_shortterm_results", [])})

    def test_receipts_and_dedup_use_the_same_session_identity_as_recall(self):
        manager = self.mm.MemoryManager(embedding_func=_mock_embed)
        message = "Regression fixture: organize the spare-parts shelf"
        first = manager.add_shortterm_ex(message, session_id="group:one", operation_id="op-1")
        second = manager.add_shortterm_ex(message, session_id="group:one ", operation_id="op-1")
        self.assertNotEqual(first["mem_id"], second["mem_id"])
        for session, expected in (("group:one", first), ("group:one ", second)):
            retry = manager.add_shortterm_ex(message, session_id=session, operation_id="op-1")
            self.assertEqual(retry["mem_id"], expected["mem_id"])
            self.assertTrue(retry["duplicate"])
            mention = manager.add_shortterm_ex(message, session_id=session)
            self.assertEqual(mention["mem_id"], expected["mem_id"])

    def test_readding_corrected_text_uses_a_fresh_id(self):
        """Check what happens if same text is re-added when stale shortterm copy is superseded in active."""
        from services.correction_service import correct
        now = time.time()
        text = "合成消息：下午去公园散步"
        from schema import MemoryItem
        m = MemoryItem(text, mem_type="short_term", session_id="private:owner", create_time=now - 60)
        m.embedding = list(MOCK_VEC)
        self.write_json("shortterm.json", {"memories": [m.to_dict()]})
        self.write_json("active.json", {"memories": [dict(m.to_dict(), mem_type="long_term")]})

        mgr = self.mm.MemoryManager(embedding_func=_mock_embed)
        correct(mgr, m.mem_id, "合成消息：下午去河边散步", operation_id="c-dup-1", reason="修正地点")

        # Now user writes the same text again into shortterm:
        r_new = mgr.add_shortterm_ex(text, session_id="private:owner")
        # If it dedup_updated onto superseded m.mem_id, CLI detail hides it!
        _, detail = self.run_cli("shortterm", {"sub": "detail", "mem_id": r_new["mem_id"]})
        self.assertNotEqual(r_new["mem_id"], m.mem_id,
                            f"add_shortterm_ex deduped onto mem_id superseded in active: {r_new}, cli_detail={detail}")
        self.assertTrue(detail["ok"], f"newly written shortterm item hidden by CLI: {detail}")
        retry = mgr.add_shortterm_ex(text, session_id="private:owner")
        self.assertEqual(retry["mem_id"], r_new["mem_id"])

    def test_private_vs_group_and_legacy_isolation(self):
        mgr = self.mm.MemoryManager(embedding_func=_mock_embed)
        text = "合成消息：午饭吃番茄炒蛋配米饭"
        r_priv = mgr.add_shortterm_ex(text, session_id="private:owner", source="bridge_auto")
        r_grp = mgr.add_shortterm_ex(text, session_id="group:3003", source="bridge_auto")
        r_leg1 = mgr.add_shortterm_ex(text, source="bridge_auto")
        r_leg2 = mgr.add_shortterm_ex(text, source="bridge_auto")

        self.assertEqual(len({r_priv["mem_id"], r_grp["mem_id"], r_leg1["mem_id"]}), 3)
        self.assertEqual(r_leg2["reason"], "dedup_updated")
        self.assertEqual(r_leg2["mem_id"], r_leg1["mem_id"])

    def test_operation_receipt_same_session_retry_and_cross_session_isolation(self):
        mgr = self.mm.MemoryManager(embedding_func=_mock_embed)
        text_a = "合成消息：下午去图书馆借书"
        text_b = "合成消息：晚上去操场慢跑三公里"

        # Repro 1: two different group sessions share the same external operation_id
        # with DIFFERENT content (and with SAME content) -> fully isolated writes & replays.
        g1 = mgr.add_shortterm_ex(text_a, session_id="group:1", operation_id="op-shared-1")
        g2 = mgr.add_shortterm_ex(text_b, session_id="group:2", operation_id="op-shared-1")
        self.assertEqual(g1["reason"], "created")
        self.assertEqual(g2["reason"], "created")
        self.assertNotEqual(g1["mem_id"], g2["mem_id"])

        g1_retry = mgr.add_shortterm_ex(text_a, session_id="group:1", operation_id="op-shared-1")
        g2_retry = mgr.add_shortterm_ex(text_b, session_id="group:2", operation_id="op-shared-1")
        self.assertTrue(g1_retry.get("duplicate"))
        self.assertTrue(g2_retry.get("duplicate"))
        self.assertEqual(g1_retry["mem_id"], g1["mem_id"])
        self.assertEqual(g2_retry["mem_id"], g2["mem_id"])
        by_id = {m.mem_id: m for m in mgr._load_shortterm_mems()}
        self.assertEqual(by_id[g1["mem_id"]].mention_count, 0)
        self.assertEqual(by_id[g2["mem_id"]].mention_count, 0)

        # Same operation_id + SAME text across two groups -> also isolated distinct IDs
        text_c = "合成消息：周末一起去植物园拍照"
        g1_same = mgr.add_shortterm_ex(text_c, session_id="group:1", operation_id="op-same-text")
        g2_same = mgr.add_shortterm_ex(text_c, session_id="group:2", operation_id="op-same-text")
        self.assertNotEqual(g1_same["mem_id"], g2_same["mem_id"])
        g2_same_retry = mgr.add_shortterm_ex(text_c, session_id="group:2", operation_id="op-same-text")
        self.assertTrue(g2_same_retry.get("duplicate"))
        self.assertEqual(g2_same_retry["mem_id"], g2_same["mem_id"])

        # Same session reusing its own operation_id with different content MUST still fail
        with self.assertRaises(ValueError):
            mgr.add_shortterm_ex("合成消息：同一会话篡改内容", session_id="group:1", operation_id="op-shared-1")

    def test_legacy_operation_id_cannot_alias_a_scoped_receipt(self):
        from schema import MemoryItem
        from services.write_receipts import record

        message = "Regression fixture: organize the spare-parts shelf"
        legacy = MemoryItem(message, mem_type="short_term", session_id="group:legacy")
        legacy_key = json.dumps(["session", "group:other", "op-1"])
        path = Path(self.config.SHORT_TERM_PATH)
        doc = {"memories": [legacy.to_dict()]}
        record(doc, legacy_key, message, {"saved": True, "mem_id": legacy.mem_id})
        path.write_text(json.dumps(doc), encoding="utf-8")

        manager = self.mm.MemoryManager(embedding_func=_mock_embed)
        created = manager.add_shortterm_ex(message, session_id="group:other", operation_id="op-1")
        self.assertNotEqual(created["mem_id"], legacy.mem_id)
        legacy_retry = manager.add_shortterm_ex(message, session_id="group:legacy", operation_id=legacy_key)
        self.assertEqual(legacy_retry["mem_id"], legacy.mem_id)
        self.assertTrue(legacy_retry["duplicate"])

    def test_legacy_bare_receipt_fallback_by_trusted_owner_and_no_cross_session_leak(self):
        from schema import MemoryItem
        from services.write_receipts import record as legacy_record

        mgr = self.mm.MemoryManager(embedding_func=_mock_embed)
        text = "合成消息：旧版回执兼容性回归条目"

        # Repro 2a: private:a record in shortterm with pre-upgrade bare receipt record(doc, op, content, result)
        row_a = MemoryItem(text, mem_type="short_term", session_id="private:a", source="bridge_auto")
        doc = {"memories": [row_a.to_dict()]}
        legacy_record(doc, "legacy-op-a", text, {"saved": True, "reason": "created", "mem_id": row_a.mem_id})
        self.write_json("shortterm.json", doc)

        # Same session replay hits legacy receipt -> duplicate=True and same mem_id
        replay_a = mgr.add_shortterm_ex(text, session_id="private:a", operation_id="legacy-op-a")
        self.assertTrue(replay_a.get("duplicate"))
        self.assertEqual(replay_a["mem_id"], row_a.mem_id)
        # Same session with different content still raises ValueError
        with self.assertRaises(ValueError):
            mgr.add_shortterm_ex("合成消息：同会话改写内容", session_id="private:a", operation_id="legacy-op-a")
        # Other scoped session and legacy None MUST NOT inherit private:a's mem_id or error out
        other_b = mgr.add_shortterm_ex(text, session_id="private:b", operation_id="legacy-op-a")
        self.assertNotEqual(other_b["mem_id"], row_a.mem_id)
        self.assertFalse(other_b.get("duplicate", False))
        other_legacy = mgr.add_shortterm_ex(text, operation_id="legacy-op-a")
        self.assertNotEqual(other_legacy["mem_id"], row_a.mem_id)
        self.assertFalse(other_legacy.get("duplicate", False))

        # Repro 2b: promoted shortterm record now lives in active.json (not in shortterm)
        row_act = MemoryItem("合成消息：已提升至活跃库的旧回执条目", mem_type="long_term",
                             session_id="private:a", source="tool_result")
        self.write_json("active.json", {"memories": [row_act.to_dict()]})
        st_doc = self.read_json("shortterm.json")
        legacy_record(st_doc, "legacy-op-active", row_act.content,
                      {"saved": True, "reason": "created", "mem_id": row_act.mem_id})
        self.write_json("shortterm.json", st_doc)
        replay_act = mgr.add_shortterm_ex(row_act.content, session_id="private:a", operation_id="legacy-op-active")
        self.assertTrue(replay_act.get("duplicate"))
        self.assertEqual(replay_act["mem_id"], row_act.mem_id)
        cross_act = mgr.add_shortterm_ex(row_act.content, session_id="group:9", operation_id="legacy-op-active")
        self.assertNotEqual(cross_act["mem_id"], row_act.mem_id)

        # Repro 2c: archived record now lives in cold.json
        row_cold = MemoryItem("合成消息：已归档冷存的旧回执条目", mem_type="long_term",
                              session_id="private:a", source="tool_result",
                              archive_reason="tool_card_superseded")
        self.write_json("cold.json", {"memories": [row_cold.to_dict()]})
        st_doc = self.read_json("shortterm.json")
        legacy_record(st_doc, "legacy-op-cold", row_cold.content,
                      {"saved": True, "reason": "created", "mem_id": row_cold.mem_id})
        self.write_json("shortterm.json", st_doc)
        replay_cold = mgr.add_shortterm_ex(row_cold.content, session_id="private:a", operation_id="legacy-op-cold")
        self.assertTrue(replay_cold.get("duplicate"))
        self.assertEqual(replay_cold["mem_id"], row_cold.mem_id)
        cross_cold = mgr.add_shortterm_ex(row_cold.content, operation_id="legacy-op-cold")
        self.assertNotEqual(cross_cold["mem_id"], row_cold.mem_id)

        # Repro 2d: unattributed legacy receipt (mem_id missing from all pools) MUST NOT leak to any caller
        st_doc = self.read_json("shortterm.json")
        legacy_record(st_doc, "legacy-op-orphan", "合成消息：无归属旧回执",
                      {"saved": True, "reason": "created", "mem_id": "mem_missing_untrusted"})
        self.write_json("shortterm.json", st_doc)
        r_orphan_scoped = mgr.add_shortterm_ex("合成消息：无归属旧回执", session_id="private:a", operation_id="legacy-op-orphan")
        self.assertNotEqual(r_orphan_scoped["mem_id"], "mem_missing_untrusted")
        r_orphan_legacy = mgr.add_shortterm_ex("合成消息：无归属旧回执", operation_id="legacy-op-orphan")
        self.assertNotEqual(r_orphan_legacy["mem_id"], "mem_missing_untrusted")

        # Repro 2e: genuine legacy record (session_id=None) matches legacy caller only, not scoped callers
        row_leg = MemoryItem("合成消息：真正无会话的旧记录", mem_type="short_term", session_id=None, source="bridge_auto")
        st_doc = self.read_json("shortterm.json")
        st_doc["memories"].append(row_leg.to_dict())
        legacy_record(st_doc, "legacy-op-nosess", row_leg.content,
                      {"saved": True, "reason": "created", "mem_id": row_leg.mem_id})
        self.write_json("shortterm.json", st_doc)
        replay_leg = mgr.add_shortterm_ex(row_leg.content, operation_id="legacy-op-nosess")
        self.assertTrue(replay_leg.get("duplicate"))
        self.assertEqual(replay_leg["mem_id"], row_leg.mem_id)
        scoped_vs_leg = mgr.add_shortterm_ex(row_leg.content, session_id="group:7", operation_id="legacy-op-nosess")
        self.assertNotEqual(scoped_vs_leg["mem_id"], row_leg.mem_id)


class TestFinding3ShorttermMetadataPreservation(AuditRepairBase):
    """P2: consolidation preserves _operation_receipts & custom metadata across all shortterm write paths."""

    def test_all_consolidation_shortterm_writes_preserve_receipts_and_replay_idempotency(self):
        from schema import MemoryItem

        mgr = self.mm.MemoryManager(embedding_func=_mock_embed)
        text = "合成消息：今天在阳台浇完多肉植物"
        r1 = mgr.add_shortterm_ex(
            text, session_id="private:owner", source="bridge_auto", operation_id="st-op-keep-1",
        )
        self.assertEqual(r1["reason"], "created")

        # Inject:
        # 1) custom top-level metadata alongside _operation_receipts
        # 2) an orphan episode_id on a second fragment (triggers step -1 pre-heal write!)
        # 3) extra fragments in same session (triggers step 0 episode aggregation write!)
        now = time.time()
        st_doc = self.read_json("shortterm.json")
        st_doc["_custom_meta"] = {"schema": "keep-me", "tag": 42}
        orphan = MemoryItem(
            "他说：今天天气真舒服；我回：嗯，适合出去散步",
            mem_type="short_term", session_id="private:owner", source="bridge_auto",
            create_time=now - 600, episode_id="ep_nonexistent_orphan",
        )
        frag2 = MemoryItem(
            "他说：顺路买杯柠檬茶吧；我回：好，少冰三分糖",
            mem_type="short_term", session_id="private:owner", source="bridge_auto",
            create_time=now - 300,
        )
        st_doc["memories"].extend([orphan.to_dict(), frag2.to_dict()])
        self.write_json("shortterm.json", st_doc)

        with patch("ollama_embedding.get_embedding", side_effect=_mock_embed):
            stats = self.dc.process_working_memory_to_long_term()

        # Verify step -1 pre-heal ran and step 0 episode aggregation ran
        self.assertGreaterEqual(stats.get("episodes_pre_healed", 0), 1)

        after_doc = self.read_json("shortterm.json")
        self.assertEqual(after_doc.get("_custom_meta"), {"schema": "keep-me", "tag": 42})
        receipts = after_doc.get("_operation_receipts", {})
        self.assertTrue(any("st-op-keep-1" in k for k in receipts), receipts)

        # Replaying operation_id must return duplicate=True and NOT increment mention_count
        replay = mgr.add_shortterm_ex(
            text, session_id="private:owner", source="bridge_auto", operation_id="st-op-keep-1",
        )
        self.assertTrue(replay.get("duplicate"))
        self.assertEqual(replay["mem_id"], r1["mem_id"])
        row = next(m for m in self.read_json("shortterm.json")["memories"] if m["mem_id"] == r1["mem_id"])
        self.assertEqual(row.get("mention_count", 0), 0)


class TestFinding4ToolCardPromotionAndDuplicateRead(AuditRepairBase):
    """P2: tool card promotion newest-N per (session, tool, action), provenance, correction/retire, dup reads."""

    def _make_tool_card(self, tool, action, artifact, *, session_id="private:owner",
                        create_time=None, status="succeeded"):
        from schema import MemoryItem
        ct = time.time() if create_time is None else create_time
        content = (
            f"[工具] {tool} [操作] {action} [状态] {status}\n"
            f"[产物] 输出完成：{artifact}"
        )
        m = MemoryItem(
            content, mem_type="short_term", session_id=session_id,
            source="tool_result", create_time=ct,
            tool_result={"tool": tool, "action": action, "status": status, "artifact": artifact},
        )
        m.embedding = list(MOCK_VEC)
        return m

    def _consolidate(self):
        with patch("ollama_embedding.get_embedding", side_effect=_mock_embed):
            return self.dc.process_working_memory_to_long_term()

    def test_newer_artifact_replaces_old_respects_keep_n_and_provenance(self):
        now = time.time()
        t1 = self._make_tool_card("fixture_renderer", "远程出图", "render_v1.png", create_time=now - 300)
        self.write_json("shortterm.json", {"memories": [t1.to_dict()]})

        # Run 1: promotes t1, removes t1 from shortterm (no duplicate ID across pools)
        self._consolidate()
        active_ids = [m["mem_id"] for m in self.read_json("active.json")["memories"]]
        short_ids = [m["mem_id"] for m in self.read_json("shortterm.json")["memories"]]
        self.assertIn(t1.mem_id, active_ids)
        self.assertNotIn(t1.mem_id, short_ids)

        # Create an approved distillation skill citing t1 to check provenance preservation
        def provider(_prompt):
            return [{
                "content": "使用 fixture_renderer 远程出图",
                "out_kind": "skill",
                "trigger": "收到 .png 参考图",
                "trigger_kind": "file",
                "action": "调用远程出图工具",
                "steps": ["检查参数", "执行出图"],
                "required_tools": [],
                "out_turns": [t1.mem_id],
                "quotes": {t1.mem_id: "render_v1.png"},
            }]

        with patch("ollama_embedding.get_embedding", side_effect=_mock_embed):
            gen = self.distill.generate("main", llm=provider, model="mock")
            card = gen["cards"][0]
            self.assertEqual(self.distill.review("main", card["id"], reviewer="owner", approved=True), "active")

        # Idempotent rerun
        self._consolidate()
        self.assertEqual(
            [m["mem_id"] for m in self.read_json("active.json")["memories"]].count(t1.mem_id), 1
        )

        # Run 2: newer artifact t2 arrives for same (session, tool, action) with N=1
        t2 = self._make_tool_card("fixture_renderer", "远程出图", "render_v2.png", create_time=now - 60)
        st = self.read_json("shortterm.json")
        st["memories"].append(t2.to_dict())
        self.write_json("shortterm.json", st)

        self._consolidate()
        active_rows = self.read_json("active.json")["memories"]
        cold_rows = self.read_json("cold.json")["memories"]
        short_rows = self.read_json("shortterm.json")["memories"]

        proc_active = [m for m in active_rows if m.get("category") == "procedure"]
        self.assertEqual([m["mem_id"] for m in proc_active], [t2.mem_id])
        self.assertIn("render_v2.png", proc_active[0]["content"])
        self.assertNotIn(t2.mem_id, [m["mem_id"] for m in short_rows])
        # Old t1 archived to cold with provenance intact (and not resurrected on rerun)
        self.assertTrue(any(m["mem_id"] == t1.mem_id and m.get("archive_reason") == "tool_card_superseded"
                            for m in cold_rows))
        self._consolidate()
        self.assertEqual(
            [m["mem_id"] for m in self.read_json("active.json")["memories"] if m.get("category") == "procedure"],
            [t2.mem_id],
        )
        # Distillation gather sees latest active t2; existing skill citing archived t1 keeps valid provenance
        gathered = self.distill.gather("main")
        self.assertIn(t2.mem_id, gathered)
        self.assertNotIn(t1.mem_id, gathered)
        self.assertEqual([c["id"] for c in self.distill.active_cards("main", include_skills=True)], [card["id"]])
        self.assertTrue(self.distill.match_skills(
            "main", session_id="private:owner", has_file=True, file_names=["ref.png"],
        ))

        # Verify N=2 keeps newest 2 artifacts
        t3 = self._make_tool_card("fixture_renderer", "远程出图", "render_v3.png", create_time=now - 10)
        t4 = self._make_tool_card("fixture_renderer", "远程出图", "render_v4.png", create_time=now)
        self.write_json("shortterm.json", {"memories": [t3.to_dict(), t4.to_dict()]})
        with patch.object(self.config, "TOOL_CARD_PROMOTE_KEEP_PER_ACTION", 2):
            self._consolidate()
        proc_ids = {
            m["mem_id"] for m in self.read_json("active.json")["memories"]
            if m.get("category") == "procedure"
        }
        self.assertEqual(proc_ids, {t3.mem_id, t4.mem_id})

    def test_tool_card_cross_session_isolation(self):
        now = time.time()
        c_priv = self._make_tool_card("fixture_renderer", "远程出图", "priv.png",
                                      session_id="private:owner", create_time=now - 30)
        c_g1 = self._make_tool_card("fixture_renderer", "远程出图", "g1.png",
                                    session_id="group:1", create_time=now - 20)
        c_g2 = self._make_tool_card("fixture_renderer", "远程出图", "g2.png",
                                    session_id="group:2", create_time=now - 10)
        self.write_json("shortterm.json", {"memories": [c_priv.to_dict(), c_g1.to_dict(), c_g2.to_dict()]})

        self._consolidate()
        proc_rows = [m for m in self.read_json("active.json")["memories"] if m.get("category") == "procedure"]
        self.assertEqual({m["mem_id"] for m in proc_rows}, {c_priv.mem_id, c_g1.mem_id, c_g2.mem_id})
        self.assertEqual(self.read_json("shortterm.json")["memories"], [])

    def test_tool_card_correction_and_retirement_consistency(self):
        from services.correction_service import correct
        from services.deletion_service import retire
        from services.recall_service import recall_core

        now = time.time()
        t_corr = self._make_tool_card("fixture_renderer", "远程出图", "wrong_cfg.png", create_time=now - 100)
        t_ret_pre = self._make_tool_card("sing_cover", "翻唱", "secret.wav", create_time=now - 90)
        self.write_json("shortterm.json", {"memories": [t_corr.to_dict(), t_ret_pre.to_dict()]})

        mgr = self.mm.MemoryManager(embedding_func=_mock_embed)
        # Retire t_ret_pre before consolidation -> must NEVER be promoted
        retire(mgr, [t_ret_pre.mem_id], operation_id="ret-pre-1")
        self._consolidate()

        active_ids = {m["mem_id"] for m in self.read_json("active.json")["memories"]}
        self.assertIn(t_corr.mem_id, active_ids)
        self.assertNotIn(t_ret_pre.mem_id, active_ids)

        # Correct promoted tool card
        mgr = self.mm.MemoryManager(embedding_func=_mock_embed)
        corr = correct(
            mgr, t_corr.mem_id,
            "[工具] fixture_renderer [操作] 远程出图 [产物] 正确参数 fixed_cfg.png",
            operation_id="corr-tool-1", reason="修正错误产物",
        )
        # Shortterm CLI detail cannot read the original text
        _, detail = self.run_cli("shortterm", {"sub": "detail", "mem_id": t_corr.mem_id})
        self.assertFalse(detail["ok"])
        # Distillation gather excludes original and includes correction
        gathered = self.distill.gather("main")
        self.assertNotIn(t_corr.mem_id, gathered)
        self.assertIn(corr["mem_id"], gathered)
        # Recall returns correction, never original
        recalled = recall_core(mgr, "远程出图", session_id="private:owner", embedding_func=_mock_embed)
        recalled_ids = {m.mem_id for m, _ in recalled.get("results", [])}
        self.assertIn(corr["mem_id"], recalled_ids)
        self.assertNotIn(t_corr.mem_id, recalled_ids)

        # Subsequent newer tool run promotes cleanly without displacing the explicit correction
        t_next = self._make_tool_card("fixture_renderer", "远程出图", "newer_ok.png", create_time=now)
        st = self.read_json("shortterm.json")
        st["memories"].append(t_next.to_dict())
        self.write_json("shortterm.json", st)
        self._consolidate()
        active_after = {m["mem_id"] for m in self.read_json("active.json")["memories"]}
        self.assertIn(corr["mem_id"], active_after)
        self.assertIn(t_next.mem_id, active_after)

        # Retiring promoted card hides it everywhere
        retire(self.mm.MemoryManager(embedding_func=_mock_embed), [t_next.mem_id], operation_id="ret-post-1")
        self.assertNotIn(t_next.mem_id, self.distill.gather("main"))
        _, detail2 = self.run_cli("shortterm", {"sub": "detail", "mem_id": t_next.mem_id})
        self.assertFalse(detail2["ok"])

    def test_retired_cold_correction_stays_out_of_active(self):
        from services.correction_service import correct
        from services.deletion_service import retire
        from services.recall_service import recall_core

        now = time.time()
        c0 = self._make_tool_card("fixture_renderer", "远程出图", "v0.png", create_time=now - 500)
        self.write_json("cold.json", {"memories": [
            dict(c0.to_dict(), mem_type="long_term", category="procedure", archive_reason="tool_card_superseded")
        ]})
        mgr = self.mm.MemoryManager(embedding_func=_mock_embed)
        corr = correct(mgr, c0.mem_id, "[工具] fixture_renderer [操作] 远程出图 [产物] wrong_fix.png",
                       operation_id="corr-ret-1", reason="误更正")
        retire(mgr, [corr["mem_id"]], operation_id="ret-corr-1")

        # Consolidation runs
        with patch("ollama_embedding.get_embedding", side_effect=_mock_embed):
            self.dc.process_working_memory_to_long_term()

        # Check: retired replacement should not be recalled (and check if it moved to active!)
        mgr2 = self.mm.MemoryManager(embedding_func=_mock_embed)
        rec = recall_core(mgr2, "远程出图", session_id="private:owner", embedding_func=_mock_embed)
        self.assertEqual(rec.get("results", []), [])
        active_ids = [m["mem_id"] for m in self.read_json("active.json")["memories"]]
        self.assertNotIn(corr["mem_id"], active_ids,
                         f"Retired cold explicit_correction was rescued into active.json! active_ids={active_ids}")

    def test_archived_tool_card_stays_cold_despite_shortterm_copy(self):
        """Legacy tool card in BOTH active (>180d -> cold) and shortterm (unexpired)."""
        now = time.time()
        old_card = self._make_tool_card("fixture_renderer", "远程出图", "ancient_v1.png",
                                        session_id="private:owner", create_time=now - 220 * 86400)
        active_copy = dict(old_card.to_dict(), mem_type="long_term", category="procedure",
                           tags=["工具卡提升"], strength=15.0, importance=0.4,
                           last_active=now - 220 * 86400, strength_at=now - 220 * 86400,
                           last_recalled_at=now - 220 * 86400)
        # Stale shortterm duplicate still sitting in shortterm.json
        short_copy = dict(old_card.to_dict(), mem_type="short_term", last_mentioned_at=now - 100)
        self.write_json("active.json", {"memories": [active_copy]})
        self.write_json("shortterm.json", {"memories": [short_copy]})

        with patch("ollama_embedding.get_embedding", side_effect=_mock_embed), \
             patch.object(self.dc, "run_episode_aggregation", return_value=[]):
            self.dc.process_working_memory_to_long_term()

        active_ids = [m["mem_id"] for m in self.read_json("active.json")["memories"]]
        cold_ids = [m["mem_id"] for m in self.read_json("cold.json")["memories"]]
        short_ids = [m["mem_id"] for m in self.read_json("shortterm.json")["memories"]]
        self.assertNotIn(old_card.mem_id, active_ids)
        self.assertIn(old_card.mem_id, cold_ids)
        self.assertNotIn(old_card.mem_id, short_ids)

    def test_cold_correction_overrides_stale_shortterm_copy(self):
        """What if a duplicate exists in shortterm (active) AND cold (superseded)?"""
        from services.recall_service import recall_core
        now = time.time()
        card = self._make_tool_card("fixture_renderer", "远程出图", "cold_sup_old.png", create_time=now - 100)
        cold_copy = dict(card.to_dict(), mem_type="long_term", category="procedure",
                         archive_reason="tool_card_superseded", status="superseded",
                         corrected_by="rep_123", effective_to=now - 30)
        short_copy = dict(card.to_dict(), mem_type="short_term", status="active")
        self.write_json("cold.json", {"memories": [cold_copy]})
        self.write_json("shortterm.json", {"memories": [short_copy]})

        # CLI hides it
        _, detail = self.run_cli("shortterm", {"sub": "detail", "mem_id": card.mem_id})
        self.assertFalse(detail["ok"])

        # Distillation gather
        gathered = self.distill.gather("main")
        self.assertNotIn(card.mem_id, gathered,
                         f"distillation.gather leaked shortterm copy superseded in cold! gathered={list(gathered.keys())}")

        # Recall
        mgr = self.mm.MemoryManager(embedding_func=_mock_embed)
        rec = recall_core(mgr, "远程出图", session_id="private:owner", embedding_func=_mock_embed)
        all_ids = {m.mem_id for m, _ in rec.get("results", [])} | {m.mem_id for m, _ in rec.get("_shortterm_results", [])}
        self.assertNotIn(card.mem_id, all_ids,
                         f"recall_core leaked shortterm copy superseded in cold! all_ids={all_ids}")

    def test_preexisting_duplicate_id_reads_honor_correction_without_cleanup(self):
        from services.correction_service import correct
        from services.recall_service import recall_core

        now = time.time()
        # Simulate pre-existing duplicate ID in BOTH active and shortterm (no cleanup script run)
        dup_card = self._make_tool_card("fixture_renderer", "远程出图", "legacy_dup_old.png", create_time=now - 50)
        active_copy = dict(dup_card.to_dict(), mem_type="long_term", category="procedure", tags=["工具卡提升"])
        short_copy = dict(dup_card.to_dict(), mem_type="short_term")
        self.write_json("active.json", {"memories": [active_copy]})
        self.write_json("shortterm.json", {"memories": [short_copy]})

        mgr = self.mm.MemoryManager(embedding_func=_mock_embed)
        corr = correct(
            mgr, dup_card.mem_id,
            "[工具] fixture_renderer [操作] 远程出图 [产物] legacy_dup_fixed.png",
            operation_id="corr-legacy-dup-1", reason="更正双池重复旧卡",
        )

        # Stale shortterm file still contains the raw row (proving no cleanup was needed)
        raw_short_ids = [m["mem_id"] for m in self.read_json("shortterm.json")["memories"]]
        self.assertIn(dup_card.mem_id, raw_short_ids)

        # 1) CLI shortterm detail MUST NOT return uncorrected original
        _, detail = self.run_cli("shortterm", {"sub": "detail", "mem_id": dup_card.mem_id})
        self.assertFalse(detail["ok"])
        _, listing = self.run_cli("shortterm", {"sub": "list"})
        self.assertNotIn(dup_card.mem_id, {item["mem_id"] for item in listing.get("items", [])})

        # 2) Distillation gather MUST NOT leak the stale shortterm duplicate
        gathered = self.distill.gather("main")
        self.assertNotIn(dup_card.mem_id, gathered)
        self.assertIn(corr["mem_id"], gathered)

        # 3) Recall MUST return only the correction, never the stale original
        rec = recall_core(mgr, "远程出图", session_id="private:owner", embedding_func=_mock_embed)
        all_recalled_ids = (
            {m.mem_id for m, _ in rec.get("results", [])}
            | {m.mem_id for m, _ in rec.get("_shortterm_results", [])}
        )
        self.assertIn(corr["mem_id"], all_recalled_ids)
        self.assertNotIn(dup_card.mem_id, all_recalled_ids)

    def test_legacy_active_shortterm_dup_displaced_to_cold_no_fork_and_cold_correct(self):
        from services.correction_service import correct
        from services.memory_tools import cold_search
        from services.recall_service import recall_core

        now = time.time()
        # 1) Legacy shape: old tool card exists in BOTH active and shortterm with same mem_id
        old_card = self._make_tool_card(
            "fixture_renderer", "远程出图", "legacy_v1_artifact.png",
            session_id="private:owner", create_time=now - 200,
        )
        new_card = self._make_tool_card(
            "fixture_renderer", "远程出图", "fresh_v2_artifact.png",
            session_id="private:owner", create_time=now - 10,
        )
        active_old = dict(old_card.to_dict(), mem_type="long_term", category="procedure", tags=["工具卡提升"])
        short_old = dict(old_card.to_dict(), mem_type="short_term")
        self.write_json("active.json", {"memories": [active_old]})
        self.write_json("shortterm.json", {"memories": [short_old, new_card.to_dict()]})

        # Consolidate: new_card enters active, old_card moves to cold, NO stale shortterm duplicate remains
        self._consolidate()
        active_ids = [m["mem_id"] for m in self.read_json("active.json")["memories"]]
        cold_rows = self.read_json("cold.json")["memories"]
        cold_ids = [m["mem_id"] for m in cold_rows]
        short_ids = [m["mem_id"] for m in self.read_json("shortterm.json")["memories"]]

        self.assertIn(new_card.mem_id, active_ids)
        self.assertNotIn(old_card.mem_id, active_ids)
        self.assertIn(old_card.mem_id, cold_ids)
        self.assertNotIn(old_card.mem_id, short_ids)
        self.assertNotIn(new_card.mem_id, short_ids)

        mgr = self.mm.MemoryManager(embedding_func=_mock_embed)
        # Before correction, old_card is still discoverable via cold_search
        before_cold = cold_search(mgr, {"query": "远程出图", "session_id": "private:owner"}, _mock_embed)
        self.assertIn(old_card.mem_id, {item["mem_id"] for item in before_cold["items"]})

        # User corrects old_card directly by original ID (and retry is idempotent)
        corr = correct(
            mgr, old_card.mem_id,
            "[工具] fixture_renderer [操作] 远程出图 [产物] corrected_v1_artifact.png",
            operation_id="corr-displaced-old-1", reason="修正冷存旧产物",
        )
        self.assertEqual(corr["pool"], "cold")
        self.assertTrue(corr["written"])
        corr_retry = correct(
            mgr, old_card.mem_id,
            "[工具] fixture_renderer [操作] 远程出图 [产物] corrected_v1_artifact.png",
            operation_id="corr-displaced-old-1", reason="修正冷存旧产物",
        )
        self.assertTrue(corr_retry.get("duplicate"))
        self.assertEqual(corr_retry["mem_id"], corr["mem_id"])

        # After correction: cold_search, shortterm detail, recall, distillation all follow unified authority
        after_cold = cold_search(mgr, {"query": "远程出图", "session_id": "private:owner"}, _mock_embed)
        self.assertNotIn(old_card.mem_id, {item["mem_id"] for item in after_cold["items"]})
        _, detail = self.run_cli("shortterm", {"sub": "detail", "mem_id": old_card.mem_id})
        self.assertFalse(detail["ok"])
        gathered = self.distill.gather("main")
        self.assertNotIn(old_card.mem_id, gathered)
        self.assertIn(corr["mem_id"], gathered)
        self.assertIn(new_card.mem_id, gathered)
        rec = recall_core(mgr, "远程出图", session_id="private:owner", embedding_func=_mock_embed)
        recalled_ids = {m.mem_id for m, _ in rec.get("results", [])}
        self.assertIn(corr["mem_id"], recalled_ids)
        self.assertNotIn(old_card.mem_id, recalled_ids)

        # 2) Pre-consolidation uncleaned shortterm+cold duplicate: correcting shortterm hides cold copy too
        dup_sc = self._make_tool_card(
            "sing_cover", "翻唱", "dup_sc_old.wav",
            session_id="private:owner", create_time=now - 300,
        )
        st_doc = self.read_json("shortterm.json")
        st_doc["memories"].append(dict(dup_sc.to_dict(), mem_type="short_term"))
        self.write_json("shortterm.json", st_doc)
        cold_doc = self.read_json("cold.json")
        cold_doc["memories"].append(dict(dup_sc.to_dict(), mem_type="long_term", category="procedure",
                                         archive_reason="tool_card_superseded"))
        self.write_json("cold.json", cold_doc)

        mgr2 = self.mm.MemoryManager(embedding_func=_mock_embed)
        corr_sc = correct(
            mgr2, dup_sc.mem_id,
            "[工具] sing_cover [操作] 翻唱 [产物] dup_sc_fixed.wav",
            operation_id="corr-dup-sc-1", reason="修正双池重复记录",
        )
        self.assertEqual(corr_sc["pool"], "shortterm")
        cold_after_sc = cold_search(mgr2, {"query": "翻唱", "session_id": "private:owner"}, _mock_embed)
        self.assertNotIn(dup_sc.mem_id, {item["mem_id"] for item in cold_after_sc["items"]})

    def test_cold_search_preserves_as_of_history_and_cold_correct_is_single_file_atomic(self):
        from atomic_io import atomic_write_json as real_atomic_write
        from services.correction_service import correct
        from services.memory_tools import cold_search
        from services.recall_service import recall_core

        now = time.time()
        # 1) Historical as_of on a superseded cold record MUST still see the original,
        #    while current query excludes it.
        hist_cold = self._make_tool_card(
            "fixture_renderer", "远程出图", "historical_v0.png",
            session_id="private:owner", create_time=now - 500,
        )
        hist_row = dict(
            hist_cold.to_dict(),
            mem_type="long_term", category="procedure",
            archive_reason="tool_card_superseded",
            status="superseded", corrected_by="rep_hist_1",
            effective_to=now - 60,
        )
        # Also test a pre-consolidation shortterm+cold duplicate superseded in shortterm at now-60
        dup_hist = self._make_tool_card(
            "sing_cover", "翻唱", "historical_song.wav",
            session_id="private:owner", create_time=now - 500,
        )
        dup_st_row = dict(
            dup_hist.to_dict(),
            mem_type="short_term", status="superseded",
            corrected_by="rep_song_1", effective_to=now - 60,
        )
        dup_cold_row = dict(
            dup_hist.to_dict(),
            mem_type="long_term", category="procedure",
            archive_reason="tool_card_superseded",
        )
        self.write_json("cold.json", {"memories": [hist_row, dup_cold_row]})
        self.write_json("shortterm.json", {"memories": [dup_st_row]})

        mgr = self.mm.MemoryManager(embedding_func=_mock_embed)
        cur_res = cold_search(mgr, {"query": "远程出图", "session_id": "private:owner"}, _mock_embed)
        self.assertNotIn(hist_cold.mem_id, {i["mem_id"] for i in cur_res["items"]})
        past_res = cold_search(
            mgr, {"query": "远程出图", "session_id": "private:owner", "as_of": now - 120}, _mock_embed,
        )
        self.assertIn(hist_cold.mem_id, {i["mem_id"] for i in past_res["items"]})

        cur_dup = cold_search(mgr, {"query": "翻唱", "session_id": "private:owner"}, _mock_embed)
        self.assertNotIn(dup_hist.mem_id, {i["mem_id"] for i in cur_dup["items"]})
        past_dup = cold_search(
            mgr, {"query": "翻唱", "session_id": "private:owner", "as_of": now - 120}, _mock_embed,
        )
        self.assertIn(dup_hist.mem_id, {i["mem_id"] for i in past_dup["items"]})

        # 2) Single-file atomic commit on correct(cold): injected write failure leaves ZERO split-brain
        target_cold = self._make_tool_card(
            "fixture_renderer", "远程出图", "cold_atomic_old.png",
            session_id="private:owner", create_time=now - 300,
        )
        self.write_json("active.json", {"memories": []})
        self.write_json("shortterm.json", {"memories": []})
        self.write_json("cold.json", {"memories": [dict(
            target_cold.to_dict(), mem_type="long_term", category="procedure",
            archive_reason="tool_card_superseded",
        )]})
        mgr = self.mm.MemoryManager(embedding_func=_mock_embed)

        def fail_cold_write(path, data, **kw):
            if os.path.abspath(path) == os.path.abspath(self.config.LONG_TERM_COLD_PATH):
                raise OSError("injected cold write failure")
            return real_atomic_write(path, data, **kw)

        new_text = "[工具] fixture_renderer [操作] 远程出图 [产物] cold_atomic_fixed.png"
        with patch("atomic_io.atomic_write_json", side_effect=fail_cold_write):
            with self.assertRaises(OSError):
                correct(
                    mgr, target_cold.mem_id, new_text,
                    operation_id="corr-cold-atomic-1", reason="原子冷更正测试",
                    session_id="private:owner",
                )

        # After failure: active still empty, cold still has ONLY active original (no split-brain)
        self.assertEqual(self.read_json("active.json")["memories"], [])
        cold_after_fail = self.read_json("cold.json")["memories"]
        self.assertEqual(len(cold_after_fail), 1)
        self.assertEqual(cold_after_fail[0]["mem_id"], target_cold.mem_id)
        self.assertEqual(cold_after_fail[0]["status"], "active")
        rec_after_fail = recall_core(mgr, "远程出图", session_id="private:owner", embedding_func=_mock_embed)
        self.assertEqual(rec_after_fail.get("results", []), [])
        cs_after_fail = cold_search(mgr, {"query": "远程出图", "session_id": "private:owner"}, _mock_embed)
        self.assertEqual({i["mem_id"] for i in cs_after_fail["items"]}, {target_cold.mem_id})

        # Retry succeeds atomically in cold; second retry returns duplicate=True with same mem_id
        before_corr_ts = time.time() - 10
        corr = correct(
            mgr, target_cold.mem_id, new_text,
            operation_id="corr-cold-atomic-1", reason="原子冷更正测试",
            session_id="private:owner",
        )
        self.assertEqual(corr["pool"], "cold")
        corr_retry = correct(
            mgr, target_cold.mem_id, new_text,
            operation_id="corr-cold-atomic-1", reason="原子冷更正测试",
            session_id="private:owner",
        )
        self.assertTrue(corr_retry.get("duplicate"))
        self.assertEqual(corr_retry["mem_id"], corr["mem_id"])

        # Immediately after atomic cold commit (before consolidation):
        # - normal recall & distillation see the replacement, not the old record
        # - current cold_search excludes old record, historical as_of still sees old record
        rec_now = recall_core(mgr, "远程出图", session_id="private:owner", embedding_func=_mock_embed)
        rec_ids = {m.mem_id for m, _ in rec_now.get("results", [])}
        self.assertIn(corr["mem_id"], rec_ids)
        self.assertNotIn(target_cold.mem_id, rec_ids)
        gathered_now = self.distill.gather("main")
        self.assertIn(corr["mem_id"], gathered_now)
        self.assertNotIn(target_cold.mem_id, gathered_now)
        cs_cur = cold_search(mgr, {"query": "远程出图", "session_id": "private:owner"}, _mock_embed)
        self.assertNotIn(target_cold.mem_id, {i["mem_id"] for i in cs_cur["items"]})
        self.assertIn(corr["mem_id"], {i["mem_id"] for i in cs_cur["items"]})
        cs_hist = cold_search(
            mgr, {"query": "远程出图", "session_id": "private:owner", "as_of": before_corr_ts}, _mock_embed,
        )
        self.assertIn(target_cold.mem_id, {i["mem_id"] for i in cs_hist["items"]})

        # Next consolidation promotes staged replacement from cold -> active,
        # keeps superseded history + receipt in cold, preserves as_of & retry idempotency.
        self._consolidate()
        active_ids_post = {m["mem_id"] for m in self.read_json("active.json")["memories"]}
        cold_rows_post = {m["mem_id"]: m for m in self.read_json("cold.json")["memories"]}
        self.assertIn(corr["mem_id"], active_ids_post)
        self.assertNotIn(target_cold.mem_id, active_ids_post)
        self.assertIn(target_cold.mem_id, cold_rows_post)
        self.assertEqual(cold_rows_post[target_cold.mem_id]["status"], "superseded")
        self.assertNotIn(corr["mem_id"], cold_rows_post)

        mgr_post = self.mm.MemoryManager(embedding_func=_mock_embed)
        cs_hist_post = cold_search(
            mgr_post, {"query": "远程出图", "session_id": "private:owner", "as_of": before_corr_ts}, _mock_embed,
        )
        self.assertIn(target_cold.mem_id, {i["mem_id"] for i in cs_hist_post["items"]})
        cs_cur_post = cold_search(mgr_post, {"query": "远程出图", "session_id": "private:owner"}, _mock_embed)
        self.assertNotIn(target_cold.mem_id, {i["mem_id"] for i in cs_cur_post["items"]})
        corr_post_retry = correct(
            mgr_post, target_cold.mem_id, new_text,
            operation_id="corr-cold-atomic-1", reason="原子冷更正测试",
            session_id="private:owner",
        )
        self.assertTrue(corr_post_retry.get("duplicate"))
        self.assertEqual(corr_post_retry["mem_id"], corr["mem_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
