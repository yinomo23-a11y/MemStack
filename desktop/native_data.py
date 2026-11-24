"""Read only the display fields needed by the six native console pages."""
from collections import Counter, deque
from datetime import datetime
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time

import yaml
import zstandard

from sources import MAX_JSON_BYTES, SourceError, date_string, literal_settings, score, text

SLIDERS = {
    "ep": "EPISODE_DECAY_RATE", "erd": "EMOTION_RESIDUE_DECAY_RATE",
    "we": "WEIGHT_EMOTION", "st": "SHORT_TERM_FORGET_THRESHOLD",
    "cd": "COLD_ARCHIVE_AFTER_DAYS", "wb": "ACTIVE_WINDOW_BONUS",
}
BUDGETS = {"long": "RECALL_MAX_CHARS", "short": "SHORT_TERM_RECALL_MAX_CHARS",
           "companion": "INIT_MAX_TOKEN", "compact": "INIT_MAX_TOKEN_COMPACT"}
PARAMETERS = set(SLIDERS.values()) | set(BUDGETS.values()) | {
    "DECAY_RATE_MAP", "EMOTION_DECAY_PROTECT", "INTIMATE_IGNORES_EMOTION",
    "HANDOVER_TTL_HOURS", "HANDOVER_MAX_CHARS",
}
STATES = ("active", "provisional", "stale", "rejected", "superseded", "archived")
PRIVATE_TEXT = re.compile(
    r"亲密|身体记忆|暧昧|裸|做爱|性交|床上|内衣|吻|"
    r"(?i:\bsk[-_][\w-]{6,}|bearer\s+\S+|(?:api[_ -]?key|access[_ -]?token|password|secret)\s*[:=])"
)


def safe_text(value, limit):
    value = str(value or "")
    return "" if PRIVATE_TEXT.search(value) else text(value, limit)


def object_at(path):
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_JSON_BYTES + 1)
        value = json.loads(data) if len(data) <= MAX_JSON_BYTES else None
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, RecursionError):
        return {}


def stamp(path):
    file = path.stat()
    return file.st_mtime_ns, file.st_size


@lru_cache(maxsize=20)
def fingerprint(path, signature):
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if stamp(path) != signature:
        raise SourceError("记忆池正在写入，下次刷新会重试。")
    return sha


def lines(path):
    try:
        if path.stat().st_size <= MAX_JSON_BYTES:
            with path.open("rb") as stream:
                return sum(1 for row in stream if row.strip())
    except OSError:
        pass
    return None


def settings_values(model, data):
    base = literal_settings(model / "config.py", PARAMETERS)
    overrides = object_at(data / "state" / "runtime_overrides.json")
    return base, {**base, **{k: v for k, v in overrides.items() if k in PARAMETERS}}


def numeric(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def slider_view(values):
    sliders = {key: values.get(name) for key, name in SLIDERS.items()}
    rate = values.get("DECAY_RATE_MAP")
    if isinstance(rate, dict):
        sliders["d2"] = rate.get("2", rate.get(2))
    protect = values.get("EMOTION_DECAY_PROTECT")
    high = protect.get("high") if isinstance(protect, dict) else None
    if isinstance(high, (list, tuple)) and len(high) == 2:
        sliders["epf"] = high[1]
    return {key: value for key, value in sliders.items() if numeric(value) is not None}


def handover_view(state, ttl):
    raw = object_at(state / "handover.json")
    rows = raw.get("sessions")
    rows = rows.values() if isinstance(rows, dict) else [raw]
    visible = []
    for row in rows:
        if not isinstance(row, dict) or row.get("is_intimate") or row.get("private"):
            continue
        age = (time.time() - (numeric(row.get("written_at")) or 0)) / 3600
        if not 0 <= age <= ttl:
            continue
        trivia = row.get("trivia")
        fields = {k: safe_text(row.get(k), 120) for k in ("topic", "emotion", "pending_tail")}
        fields["trivia"] = [safe_text(t, 40) for t in trivia[:3]] if isinstance(trivia, list) else []
        if not any(fields.values()) or any(row.get(k) and not fields[k] for k in ("topic", "emotion", "pending_tail")):
            continue
        visible.append({**fields, "seq": numeric(row.get("seq")), "age_hours": round(age, 1)})
    return min(visible, key=lambda row: row["age_hours"], default=None)


def card_view(state, visible_ids):
    from writer import card_version

    data = object_at(state / "distillation" / "main.json")
    rows = data.get("cards")
    if isinstance(rows, dict):
        rows = list(rows.values())
    if not isinstance(rows, list):
        return None
    batches = data.get("completed_batches")
    batches = [x for x in batches if isinstance(x, str)] if isinstance(batches, list) else []
    by_batch = {key: Counter() for key in batches}
    cards, skills = [], []
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            continue
        status = raw.get("status") or raw.get("distill_status")
        if status not in STATES:
            status = "其他"
        ids = raw.get("out_turns")
        kind = safe_text(raw.get("out_kind"), 30) or "其他"
        body = safe_text(raw.get("content"), 360)
        protected = (not body or raw.get("scope") == "intimate" or raw.get("is_intimate") or raw.get("private")
                or not isinstance(ids, list) or not ids
                or any(not isinstance(i, str) or i not in visible_ids for i in ids))
        if protected:
            body = "受保护卡片"
        raw_id = str(raw.get("id") or index)
        key = hashlib.sha256(raw_id.encode()).hexdigest()
        confidence = raw.get("confidence")
        model = safe_text(raw.get("model"), 80)
        card = {
            "id": "source-" + key, "label": key[:8], "st": status, "kind": kind,
            "txt": body[:80], "full": body, "quote": "", "model": model or "未记录",
            "confLabel": safe_text(confidence, 24) or "未记录", "str": score(raw.get("strength")),
            "sourceRevs": f"{len(ids) if isinstance(ids, list) else 0} 个来源",
            "updatedAt": date_string(raw.get("updated_at") or raw.get("at")) or "未记录",
            "rev": "已审核" if raw.get("review") else "待复核",
            "floorRule": "记忆库实际记录", "note": "来源卡片", "readOnly": True,
            "version": None if protected else card_version(raw),
        }
        cards.append(card)
        batch_key = raw.get("evidence_sha")
        if isinstance(batch_key, str) and batch_key in by_batch:
            by_batch[batch_key].update(("total", status))
        if kind == "skill" and status == "active":
            skill_status = raw.get("skill_status")
            skills.append({"id": card["id"], "name": body[:40],
                           "st": skill_status if skill_status in ("active", "draft", "retired") else "draft",
                           "fails": numeric(raw.get("fail_count")) or 0, "readOnly": True,
                           "version": card["version"]})
    total = Counter()
    history = []
    for batch in by_batch.values():
        total.update(batch)
        history.append([total["total"], total["active"], total["stale"]])
    options = data.get("options")
    model = safe_text(options.get("model"), 80) if isinstance(options, dict) else ""
    return {"cards": cards, "counts": dict(Counter(c["st"] for c in cards)),
            "skills": skills, "batches": len(batches), "history": history,
            "capacity": 300, "model": model or "未记录"}


def yaml_data(path):
    try:
        if path.stat().st_size <= 2 * 1024 * 1024:
            return yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, RecursionError, yaml.YAMLError):
        pass
    return None


def nodes(node):
    # YAML aliases can refer back to the same object.
    stack, seen = [node], set()
    while stack:
        obj = stack.pop()
        if isinstance(obj, (dict, list)) and id(obj) not in seen:
            seen.add(id(obj))
            if isinstance(obj, dict):
                yield obj
                stack.extend(obj.values())
            else:
                stack.extend(obj)


@lru_cache(maxsize=16)
def session_usage(path, signature):
    samples, window = deque(maxlen=24), None
    try:
        with path.open("rb") as stream, zstandard.ZstdDecompressor().stream_reader(stream) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as decoded:
                for line in decoded:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    data = row.get("data") if isinstance(row, dict) else None
                    if not isinstance(data, dict):
                        continue
                    if row.get("type") == "request/context":
                        size = data.get("contextWindow")
                        if type(size) is int and size > 0 and size != window:
                            window = size
                            samples.clear()
                    if row.get("type") == "assistant/message":
                        usage = data.get("usage")
                        value = usage.get("inputTokens") if isinstance(usage, dict) else None
                        if type(value) is int and value >= 0:
                            samples.append(value)
    except (OSError, UnicodeError, zstandard.ZstdError):
        pass
    return list(samples), window


def dsh_view(root):
    groups = {"desktop": ("profiles-desktop--", "default-workspace--"),
              "headless": ("profiles-headless--", "headless--"), "web": ("profiles-web--",)}
    profiles = {}
    for name, suffixes in groups.items():
        paths = []
        for folder in (root / "sessions").glob("*--"):
            if folder.name.endswith(suffixes):
                for path in folder.glob("*/session.v4.jsonl.zstd"):
                    try:
                        paths.append((stamp(path), path))
                    except OSError:
                        continue
        paths.sort(reverse=True)
        samples, window = [], None
        if paths:
            for signature, path in paths[:4]:
                samples, window = session_usage(path, signature)
                if samples:
                    break
        config = yaml_data(root / "profiles" / name / "cordis.patch.yml")
        windows = sorted({n["contextWindow"] for n in nodes(config)
                          if type(n.get("contextWindow")) is int and n["contextWindow"] > 0})
        profiles[name] = {"samples": samples, "window": window, "windows": windows, "files": len(paths),
                          "latest_kb": round(paths[0][0][1] / 1024, 1) if paths else None}
    vacuum = object_at(root / "dsh-session-vacuum" / "config.json")
    slim = next((n.get("config") for n in nodes(yaml_data(root / "cordis.patch.yml"))
                 if n.get("name") == "dsh-prompt-slim"), None)
    return profiles, {"enabled": vacuum.get("enabled"), "dryRun": vacuum.get("dryRun"),
                      "minTools": numeric(slim.get("minTools")) if isinstance(slim, dict) else None,
                      "revisions": {}}


def vault_view(root):
    data = object_at(root / ".distill-last.json")
    skills = []
    for path in (root / "02 经验与方法" / "技能").glob("*.md"):
        try:
            with path.open(encoding="utf-8-sig") as stream:
                header = stream.read(4096).split("---", 2)
            meta = yaml.safe_load(header[1]) if len(header) == 3 and not header[0].strip() else None
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if not isinstance(meta, dict) or meta.get("status") not in ("active", "draft"):
            continue
        skills.append({"name": safe_text(meta.get("name") or path.stem, 60) or "受保护技能",
                       "st": meta["status"]})
    mapping = {"state": "last_state", "mode": "last_mode", "last": "last_distill",
               "run": "last_run", "zone": "tz"}
    output = {key: safe_text(data.get(name), 80) for key, name in mapping.items()}
    for key, name in (("every", "every_days"), ("since", "days_since"), ("calls", "model_calls")):
        output[key] = numeric(data.get(name))
    return {**output, "skills": skills, "notes": sum(1 for _ in root.rglob("*.md")),
            "logs": lines(root / "distill-scheduled.log")}


def load_details(profile, data, files, signatures, records, visible_ids, base, live, flags):
    state = data / "state"
    counts = Counter(row["pool"] for row in records)
    consolidation = object_at(data / "last_consolidation.json")
    decisions = consolidation.get("candidate_decisions")
    decisions = decisions if isinstance(decisions, dict) else {}
    runtime = {key: numeric(consolidation.get(key)) for key in
               ("consolidated", "updated", "episodes_created", "episodes_archived")}
    runtime.update(
        last_consolidate_at=safe_text(consolidation.get("_finished_at_str"), 40),
        consolidate_time=safe_text(consolidation.get("schedule_time"), 30),
        candidate_seen=numeric(decisions.get("seen")), candidate_merged=numeric(decisions.get("merged")),
        candidates_lines=lines(state / "candidates.jsonl"),
        lessons_count=sum(row["kind"] == "lesson" for row in records),
        reflection_count=sum(row["kind"] == "reflection" for row in records),
        oldest_inactive_d=max((row["visual_age"] for row in records if row["pool"] == "active"), default=0),
        flags={key: value for key, value in flags.items() if isinstance(key, str) and type(value) is bool},
    )
    cards = card_view(state, visible_ids)
    budgets = {key: value for key, name in BUDGETS.items() if (value := numeric(live.get(name))) is not None}
    details = {
        "runtime": runtime, "cards": cards, "budgets": budgets,
        "settings": {"values": slider_view(live), "defaults": slider_view(base)},
        "handover": handover_view(state, numeric(live.get("HANDOVER_TTL_HOURS")) or 24),
        "pools": [{"name": path.name, "desc": label, "count": counts[key], "unit": "条",
                   "kb": round(signature[1] / 1024, 1), "sha": fingerprint(path, signature), "status": "只读"}
                  for (key, label, path), signature in zip(files, signatures)],
    }
    if profile.get("dsh_path"):
        dsh = Path(profile["dsh_path"])
        budgets["profiles"], details["dsh"] = dsh_view(dsh)
        budgets["vac_logs"] = lines(dsh / "logs" / "session-vacuum.jsonl")
    if profile.get("vault_path"):
        details["vault"] = vault_view(Path(profile["vault_path"]))
    return details

