import ast
from collections import Counter
from contextlib import closing
from hashlib import sha256
from datetime import datetime, timedelta, timezone
from itertools import count, islice
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import cosmos

PROTOCOL = 1
MAX_RECORDS = 20_000
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
FIELDS = ("content", "strength", "time", "pool", "entities")
RECORD_IDS = ("mem_id", "record_id", "id")
EXTRAS = ("telemetry", "settings", "handover", "cards", "audit")
# Decay floor constants, copied from the memory model this app reads:
# decay_logic.FLOOR_BY_LEVEL / IMPORTANCE_FLOOR_MULT / config.CORE_FLOOR /
# config.EPISODE_FLOOR / the intimate floor / config.CORE_PEOPLE.
# Keep them in step with the model; a mismatch only changes derived card counts.
FLOOR_BY_LEVEL = {3: 35.0, 2: 15.0, 1: 0.0}
IMPORTANCE_FLOOR_MULT = 60.0
CORE_FLOOR = 45.0
EPISODE_FLOOR = 45.0
INTIMATE_FLOOR = 55.0
SHORT_TERM_WINDOW_DAYS = 3.0


def shortterm_strength(row, now):
    """Mirror of decay_logic.calculate_shortterm_strength().

    The shortterm pool stores a flat 60.0 in `strength`; the real score is
    100·(1-(t/3)²) computed at read time from last_mentioned_at. Without this
    the matrix would report every shortterm record as an identical 60.
    """
    last = row.get("last_mentioned_at") or row.get("create_time")
    try:
        last = float(last)
    except (TypeError, ValueError, OverflowError):
        return None
    if last <= 0:
        return None
    days = max(0.0, (now - last) / 86400.0)
    if days >= SHORT_TERM_WINDOW_DAYS:
        return 0.0
    return round(max(0.0, min(100.0, 100.0 * (1.0 - (days / SHORT_TERM_WINDOW_DAYS) ** 2))), 1)
NATIVE_POOLS = (
    ("active", "长期活跃", ("active.json", "owner_active.json")),
    ("shortterm", "短期记忆", ("shortterm.json", "owner_shortterm.json")),
    ("intimate", "私密记忆", ("intimate.json", "owner_intimate.json")),
    ("cold", "冷存记忆", ("cold.json", "owner_cold.json")),
    ("working", "工作记忆", ("working_mem.json",)),
)
SECRET = re.compile(r"\b(?:sk-[\w-]{16,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[\w-]{30,})\b")
MAPPING_HINTS = {
    "content": ("content", "text", "body", "message", "内容"),
    "strength": ("strength", "score", "weight", "强度"),
    "time": ("create_time", "created_at", "timestamp", "time", "date", "时间"),
    "pool": ("pool", "group", "category", "type", "分组"),
    "entities": ("entities", "tags", "labels", "标签"),
}


class SourceError(ValueError):
    """A connection problem that can be corrected in the source dialog."""


def object_list(value, message):
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise SourceError(message)
    return value


def text(value, limit):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        return ""
    return SECRET.sub("[已遮盖]", value.strip())[:limit]


def score(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and abs(number) <= 1e12 else None


def date_string(value):
    try:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            number = None
        if number is not None:
            if abs(number) >= 1e11:
                number /= 1000
            date = datetime.fromtimestamp(number, timezone.utc)
        elif isinstance(value, str):
            date = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        else:
            return None
        return date.astimezone(timezone.utc).isoformat(timespec="seconds")
    except (ValueError, OSError, OverflowError):
        return None


def count_of(value, ceiling=1_000_000):
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if 0 <= number <= ceiling else None


def decay_floor(row, pool, private, content, entities, core_people):
    """Mirror of decay_logic.real_floor() for the API-shaped record.

    Returns None when the source does not carry the inputs, so the matrix can
    show "unknown" rather than invent a floor.
    """
    if private or pool == "intimate" or row.get("category") == "intimate":
        floor = INTIMATE_FLOOR
    else:
        floor = 0.0
    level = count_of(row.get("consolidation_level"), 9)
    importance = score(row.get("importance"))
    if level is None and importance is None and not entities and not content:
        return None
    if level is not None:
        floor = max(floor, FLOOR_BY_LEVEL.get(level, 0.0))
    if importance is not None:
        floor = max(floor, max(0.0, min(1.0, importance)) * IMPORTANCE_FLOOR_MULT)
    if isinstance(content, str) and any(name and name in content for name in core_people):
        floor = max(floor, CORE_FLOOR)
    if row.get("mem_type") == "episode":
        floor = max(floor, EPISODE_FLOOR)
    custom = score(row.get("strength_floor"))
    if custom is not None:
        floor = max(floor, custom)
    return round(min(100.0, floor), 1)


def entity_names(value):
    if isinstance(value, str):
        if value.lstrip().startswith("["):
            try:
                value = json.loads(value)
            except ValueError:
                return []
        else:
            value = re.split(r"[,，;；|]", value)
    if not isinstance(value, list):
        return []
    names = (text(item.get("name") if isinstance(item, dict) else item, 80) for item in value[:32])
    return list(dict.fromkeys(name for name in names if name))[:12]


def record(content, strength=None, time=None, pool=None, entities=None, private=False,
           level=None, floor=None, episode=False, schedule=False, tags=None,
           kind="", emotion=None, mentions=None, recalls=None, pos=None, source_id=None):
    return {
        # Where this memory sits in the cosmos, from layout.py. Absent for sources
        # that carry no embedding; the matrix then falls back to its own layout.
        "pos": pos,
        "source_id": source_id,
        "content": "私密记忆 · 内容已隐藏" if private else text(content, 2400),
        "strength": score(strength),
        "time": date_string(time),
        "pool": text(pool, 100) or "default",
        "entities": [] if private else entity_names(entities),
        "private": private,
        # Structured, content-free fields the memory matrix routes on.
        "level": None if private else level,
        "floor": None if private else floor,
        "episode": False if private else bool(episode),
        "schedule": False if private else bool(schedule),
        "tags": [] if private else [name for name in (text(t, 40) for t in (tags or [])[:8]) if name],
        "kind": "" if private else text(kind, 20),
        "emotion": None if private else emotion,
        "mentions": None if private else mentions,
        "recalls": None if private else recalls,
    }


def snapshot(records=(), pools=(), strength_unit="", features=(), truncated=False):
    rows = list(islice(records, MAX_RECORDS + 1))
    truncated = truncated or len(rows) > MAX_RECORDS
    rows = rows[:MAX_RECORDS]
    labels = dict(pools)
    occurrences = Counter()
    for index, row in enumerate(rows):
        source_id = row.pop("source_id", None)
        identity = [source_id] if type(source_id) in (str, int) and source_id != "" else [
            row["time"], row["content"], row["pool"] if not row["time"] else None,
        ]
        digest = sha256(json.dumps(identity, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
        row["key"] = f"{digest}:{occurrences[digest]}"
        occurrences[digest] += 1
        row["id"] = index
        labels.setdefault(row["pool"], "未分组" if row["pool"] == "default" else row["pool"])
    return {
        "capabilities": {
            "pools": [{"id": key, "label": label} for key, label in labels.items()],
            "strength_unit": strength_unit,
            "features": ["snapshot", *features],
        },
        "records": rows,
        "truncated": truncated,
    }


def read_json(path):
    with path.open("rb") as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES:
        raise SourceError(f"{path.name} 超出 64 MiB 的读取上限。")
    return json.loads(data.decode("utf-8-sig"))


def pool_files(folder):
    for key, label, variants in NATIVE_POOLS:
        for name in variants:
            path = folder / name
            if path.is_file():
                yield key, label, path
                break


def unix_time(value):
    formatted = date_string(value)
    return datetime.fromisoformat(formatted).timestamp() if formatted else None


def live_strength(row, pool, floor, policy, now):
    stored = score(row.get("strength"))
    if pool == "shortterm":
        calculated = shortterm_strength(row, now)
        return stored if calculated is None else calculated
    if pool == "working":
        return stored or 90.0
    if stored is None:
        return None
    anchor = next((timestamp for name in ("strength_at", "last_active", "create_time")
                   if (timestamp := unix_time(row.get(name))) is not None), now)
    intimate = row.get("category") == "intimate" or bool(row.get("is_intimate"))
    level = count_of(row.get("consolidation_level"), 9)
    rates = policy.get("DECAY_RATE_MAP")
    if not isinstance(rates, dict):
        rates = {3: 2.0, 2: 5.0, 1: 8.0}
    custom = score(row.get("decay_rate"))
    default = score(rates.get(level, rates.get(2))) or 5.0
    rate = max(0.0, custom) if custom is not None else 0.3 if intimate else default
    fixed = custom is not None and row.get("decay_rate_mode") == "effective"
    if not fixed:
        if policy.get("half_life_boost", True):
            rate /= max(1, score(row.get("decay_slowdown")) or 1)
        if not (row.get("category") == "intimate" and policy.get("INTIMATE_IGNORES_EMOTION", True)):
            emotion = score(row.get("emotion_intensity")) or 0
            bands = policy.get("EMOTION_DECAY_PROTECT")
            if not isinstance(bands, dict):
                bands = {"extreme": (85, .25), "high": (60, .4), "medium": (40, .7)}
            for name in ("extreme", "high", "medium"):
                item = bands.get(name)
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    threshold, factor = map(score, item)
                    if threshold is not None and factor is not None and emotion >= threshold:
                        rate *= max(0.0, min(1.0, factor))
                        break
    if row.get("mem_type") == "episode":
        rate = score(policy.get("EPISODE_DECAY_RATE")) or 0.8
    strength = stored - max(0, now - anchor) / 86400 * rate
    return round(max(floor or 0.0, min(100.0, strength)), 1)


def native_rows(files, layout_units=None, core_people=(), entity_rules=None, policy=None, visible_ids=None):
    now = time.time()
    policy = policy or {}
    for pool, _, path in files:
        data = read_json(path)
        message = f"{path.name} 的记忆格式无法识别。"
        if pool == "working" and isinstance(data, dict) and "sessions" in data:
            sessions = data["sessions"]
            if not isinstance(sessions, dict) or any(not isinstance(s, dict) for s in sessions.values()):
                raise SourceError(message)
            batches = (s.get("working_memory", []) for s in sessions.values())
        elif isinstance(data, dict) and ("memories" in data or not data):
            batches = [data.get("memories", [])]
        elif isinstance(data, list):
            batches = [data]
        else:
            raise SourceError(message)
        for batch in batches:
            for row in object_list(batch, message):
                if row.get("deleted"):
                    continue
                private = (
                    pool == "intimate" or row.get("category") == "intimate"
                    or any(row.get(key) is True for key in ("private", "intimate", "is_intimate"))
                )
                content = row.get("content")
                if visible_ids is not None and not private and isinstance(row.get("mem_id"), str):
                    visible_ids.add(row["mem_id"])
                entities = row.get("entities") or row.get("tags")
                floor = decay_floor(row, pool, private, content,
                                    entity_names(entities) if not private else [], core_people)
                strength = live_strength(row, pool, floor, policy, now)
                last_active = unix_time(row.get("last_active") or row.get("create_time"))
                age_days = max(0, now - last_active) / 86400 if last_active is not None else 0
                if layout_units is not None:
                    created = unix_time(row.get("create_time") or row.get("last_active"))
                    layout_units.append({
                        "embedding": row.get("embedding"),
                        "slot": {"active": 0, "shortterm": 1, "intimate": 2,
                                 "cold": 3, "working": 4}.get(pool, 0),
                        "strength": score(strength) or 0.0,
                        "age_days": age_days,
                        "created": created,
                    })
                item = record(
                    content, strength,
                    row.get("create_time") or row.get("last_active"), pool,
                    entities, private,
                    level=count_of(row.get("consolidation_level"), 9),
                    floor=floor,
                    episode=bool(row.get("episode_id")),
                    schedule=bool(row.get("effective_from") or row.get("effective_to")),
                    tags=row.get("tags"),
                    kind=row.get("kind") or row.get("mem_type") or "",
                    emotion=score(row.get("emotion_intensity")),
                    mentions=count_of(row.get("mention_count")),
                    recalls=count_of(row.get("recall_count")),
                    source_id=next((row[name] for name in RECORD_IDS if row.get(name) is not None), None),
                )
                item["visual_age"] = round(age_days, 1)
                item["visual_floor"] = floor
                item["cosmos_entities"] = []
                if not private:
                    item["impression"] = score(row.get("impression_strength"))
                    item["event_intensity"] = score(row.get("event_intensity"))
                    names = entity_names(row.get("entities"))
                    if not names and entity_rules is not None:
                        names = entity_names(cosmos.extract_entity_names(text(content, 2400), entity_rules))
                    # The original star map uses the first four recognized entities per record.
                    item["cosmos_entities"] = names[:4]
                yield item


def literal_settings(path, names):
    """Read literal metadata from a selected folder without importing its code."""
    try:
        with path.open("rb") as stream:
            source = stream.read(256 * 1024 + 1)
        if len(source) > 256 * 1024:
            return {}
        tree = ast.parse(source)
    except (OSError, SyntaxError, ValueError, RecursionError):
        return {}
    values = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        keys = [target.id for target in statement.targets if isinstance(target, ast.Name) and target.id in names]
        if keys:
            try:
                value = ast.literal_eval(statement.value)
            except (TypeError, ValueError, RecursionError):
                continue
            values.update(dict.fromkeys(keys, value))
    return values


def configured_core_people(folder):
    value = literal_settings(folder / "config.py", {"CORE_PEOPLE"}).get("CORE_PEOPLE")
    if isinstance(value, (tuple, list)):
        return tuple(name for name in value[:64] if isinstance(name, str) and 0 < len(name) <= 100)
    return ()


def local_recall_days(data_folder):
    path = data_folder / "state" / "recall_audit.jsonl"
    days, level2, level3 = Counter(), Counter(), Counter()
    try:
        with path.open("rb") as stream:
            # Bound auxiliary telemetry independently of the memory pools.
            if path.stat().st_size > MAX_JSON_BYTES:
                return None
            for line in stream:
                try:
                    row = json.loads(line)
                    day = datetime.fromtimestamp(float(row["ts"])).date()
                    levels = {item.get("lvl") for item in row.get("items", ())}
                except (KeyError, TypeError, ValueError, AttributeError, OverflowError, OSError):
                    continue
                days[day] += 1
                level2[day] += 2 in levels
                level3[day] += 3 in levels
    except OSError:
        return None
    if not days:
        return None
    last = max(max(days), datetime.now().date())
    start = max(min(days), last - timedelta(days=89))
    series = []
    longest = running = 0
    while start <= last:
        series.append([start.isoformat(), days[start], level2[start], level3[start]])
        running = running + 1 if days[start] else 0
        longest = max(longest, running)
        start += timedelta(days=1)
    peak = max(days, key=days.get)
    return {"days": series, "total": sum(days.values()), "n": sum(count > 0 for count in days.values()),
            "longest": longest, "peak": days[peak], "peak_day": peak.isoformat()}


def folder_snapshot(profile):
    from native_data import load_details, settings_values, stamp

    path = profile["path"]
    folder = Path(path)
    if not folder.is_dir():
        raise SourceError("找不到选中的数据目录。")
    files = list(pool_files(folder)) or list(pool_files(folder / "data"))
    if not files:
        raise SourceError("此目录没有找到忆栈池文件。请选模型的数据目录。")
    signatures = [stamp(path) for _, _, path in files]
    data = files[0][2].parent
    model = data.parent if data.name == "data" and not (data / "config.py").exists() else folder
    raw_rules = literal_settings(folder / "entity_graph.py", {"ENTITY_HINTS", "ALIASES", "SINGLE_CHAR_BLOCK"})
    entity_rules = cosmos.entity_rules(raw_rules)
    base, live = settings_values(model, data)
    policy = dict(live)
    if isinstance(policy.get("DECAY_RATE_MAP"), dict):
        policy["DECAY_RATE_MAP"] = {int(k): v for k, v in policy["DECAY_RATE_MAP"].items()
                                    if str(k) in ("0", "1", "2", "3")}
    flags = literal_settings(model / "features.py", {"FEATURES"}).get("FEATURES")
    if isinstance(flags, dict):
        policy["half_life_boost"] = flags.get("half_life_boost", True)
    units, visible_ids = [], set()
    records = list(islice(
        native_rows(files, units, configured_core_people(model), entity_rules, policy, visible_ids), MAX_RECORDS + 1
    ))
    try:
        cosmos.project(units, records)
    except (ArithmeticError, MemoryError, ValueError):
        # Keep the generic projection available when stored vectors cannot be decomposed.
        pass
    result = snapshot(records, ((key, label) for key, label, _ in files), "分")
    result["local_recall"] = local_recall_days(data)
    result["details"] = load_details(profile, data, files, signatures, result["records"],
                                     visible_ids, base, live, flags if isinstance(flags, dict) else {})
    if signatures != [stamp(path) for _, _, path in files]:
        raise SourceError("记忆池正在写入，下次刷新会重试。")
    return result


def jsonl_rows(path):
    with Path(path).open("rb") as stream:
        for line_number in count(1):
            line = stream.readline(MAX_LINE_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_LINE_BYTES:
                raise SourceError(f"JSONL 第 {line_number} 行超出 1 MiB。")
            line = line.decode("utf-8-sig" if line_number == 1 else "utf-8").strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise SourceError(f"JSONL 第 {line_number} 行无法解析。") from exc
            if not isinstance(row, dict):
                raise SourceError(f"JSONL 第 {line_number} 行应为 JSON 对象。")
            yield row


def system_table(name):
    return name.casefold().startswith("sqlite_")


def identifier(name):
    return '"' + name.replace('"', '""') + '"'


def open_database(path):
    # The SQLite URI uses escaped path characters and opens the database read-only.
    uri = Path(path).resolve(strict=True).as_uri() + "?mode=ro"
    db = sqlite3.connect(uri, uri=True, timeout=3)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA trusted_schema=OFF")
    return db


def table_names(db):
    return sorted(row["name"] for row in db.execute("PRAGMA main.table_list")
                  if row["type"] == "table" and not system_table(row["name"]))


def columns(db, table):
    if table not in table_names(db):
        raise SourceError("请选择存在的 SQLite 数据表。")
    return [row["name"] for row in db.execute(f"PRAGMA main.table_xinfo({identifier(table)})")
            if row["hidden"] == 0]


def inspect_source(kind, path, table=""):
    if kind == "sqlite":
        with closing(open_database(path)) as db:
            tables = table_names(db)
            if not tables:
                raise SourceError("SQLite 文件没有可读取的数据表。")
            table = table or tables[0]
            keys = columns(db, table)
    elif kind == "jsonl":
        tables = []
        keys = sorted({key for row in islice(jsonl_rows(path), 200) for key in row})
    else:
        raise SourceError("请选择 SQLite 或 JSONL 文件。")
    aliases = {key.casefold(): key for key in keys}
    guesses = {
        field: next((aliases[name] for name in hints if name in aliases), "")
        for field, hints in MAPPING_HINTS.items()
    }
    return {"tables": tables, "table": table, "fields": keys, "guesses": guesses}


def mapped_snapshot(profile):
    mapping = profile.get("mapping")
    if not isinstance(mapping, dict) or not isinstance(mapping.get("content"), str) or not mapping["content"]:
        raise SourceError("请选择内容字段。")
    names = {field: mapping.get(field, "") for field in FIELDS}
    if any(not isinstance(name, str) for name in names.values()):
        raise SourceError("字段映射无效。")

    def adapt(rows, id_column=None):
        for row in rows:
            source_id = (row.get(id_column) if id_column
                         else next((row[name] for name in RECORD_IDS if row.get(name) is not None), None))
            yield record(**{field: row.get(key) for field, key in names.items() if key}, source_id=source_id)

    if profile["kind"] == "jsonl":
        return snapshot(adapt(jsonl_rows(profile["path"])))
    with closing(open_database(profile["path"])) as db:
        table = profile.get("table", "")
        keys = columns(db, table)
        selected = list(dict.fromkeys(key for key in names.values() if key))
        if any(key not in keys for key in selected):
            raise SourceError("映射字段已改变，请重新选择字段。")
        id_column = next((key for name in RECORD_IDS for key in keys if key.casefold() == name), None)
        if id_column and id_column not in selected:
            selected.append(id_column)
        fields = ", ".join(map(identifier, selected))
        rows = db.execute(f"SELECT {fields} FROM main.{identifier(table)} LIMIT ?", (MAX_RECORDS + 1,))
        return snapshot(adapt(map(dict, rows), id_column))


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def remote_url(value):
    try:
        parsed = urlsplit(value.strip())
        valid = bool(parsed.hostname) and parsed.port != 0
    except (AttributeError, ValueError):
        valid = False
    if not valid or parsed.scheme not in ("http", "https") or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise SourceError("请输入 HTTP 或 HTTPS 服务地址。地址中不要放账号、密码或查询参数。")
    return parsed.geturl().rstrip("/")


def remote_json(base, endpoint):
    req = Request(f"{remote_url(base)}/{endpoint}", headers={"Accept": "application/json"})
    try:
        with build_opener(NoRedirect).open(req, timeout=6) as response:
            data = response.read(MAX_JSON_BYTES + 1)
    except HTTPError as exc:
        raise SourceError(f"/{endpoint} 请求失败（HTTP {exc.code}）。") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise SourceError(f"无法访问 /{endpoint}，请检查服务地址和网络。") from exc
    if len(data) > MAX_JSON_BYTES:
        raise SourceError(f"/{endpoint} 超出 64 MiB 的读取上限。")
    try:
        value = json.loads(data)
    except (UnicodeError, ValueError) as exc:
        raise SourceError(f"/{endpoint} 返回的内容不是有效 JSON。") from exc
    if not isinstance(value, dict):
        raise SourceError(f"/{endpoint} 应返回 JSON 对象。")
    return value


def remote_snapshot(base):
    caps = remote_json(base, "capabilities")
    features = caps.get("features", [])
    if type(caps.get("version")) is not int or caps["version"] != PROTOCOL or not isinstance(features, list) or "snapshot" not in features:
        raise SourceError("此接口不支持忆栈协议 v1 的 snapshot。")
    pools = object_list(caps.get("pools", []), "接口的 pools 应为 JSON 对象数组。")
    labels = {
        text(pool.get("id"), 100): text(pool.get("label"), 100) or text(pool.get("id"), 100)
        for pool in pools if text(pool.get("id"), 100)
    }
    payload = remote_json(base, "snapshot")
    rows = object_list(payload.get("records"), "接口的 records 应为 JSON 对象数组。")
    records = (record(*(row.get(key) for key in FIELDS), private=row.get("private") is True,
                      source_id=next((row[name] for name in RECORD_IDS if row.get(name) is not None), None)) for row in rows)
    return snapshot(records, labels.items(), text(caps.get("strength_unit"), 24),
                    [part for part in EXTRAS if part in features], payload.get("truncated") is True)


def remote_module(base, endpoint):
    if endpoint not in EXTRAS:
        raise SourceError("此模块暂不支持。")
    data = remote_json(base, endpoint)
    rows = object_list(data.get("rows"), f"/{endpoint} 应返回 rows 数组。")
    return [
        {"label": text(row.get("label"), 120), "value": text(row.get("value"), 2400),
         "section": text(row.get("section"), 64)}
        for row in rows[:500]
    ]


def demo_snapshot():
    topics = {
        "城市花园": ("排查滴灌管线", "记录薄荷发芽时间", "确认育苗架尺寸", "整理堆肥清单"),
        "潮汐图册": ("绘制港口潮汐曲线", "复核海岸线标注", "校对观测日期", "整理灯塔照片"),
        "轨道书屋": ("对照书架编号", "修订夜间借阅流程", "登记科幻选集", "绘制阅读动线"),
        "云岛电台": ("确认录音设定", "筛选节目片头", "整理采访问题", "测试信号覆盖"),
        "琥珀工坊": ("核对版画样张", "记录木料纹理", "调整作业台灯光", "预约材料试验"),
        "雾岭观景台": ("标记观景路线", "整理指南地图", "复查登顶时间", "记录花期变化"),
    }
    groups = [(key, label) for key, label, _ in NATIVE_POOLS if key != "intimate"]
    now = datetime.now(timezone.utc)
    rows = (
        record(task, 27 + (batch * 17 + topic_index * 11 + task_index * 13) % 72,
               (now - timedelta(days=(batch * 9 + topic_index * 3 + task_index) % 35)).isoformat(),
               groups[(batch + task_index) % len(groups)][0], [topic], source_id=f'{batch}:{topic_index}:{task_index}')
        for batch in range(4)
        for topic_index, (topic, tasks) in enumerate(topics.items())
        for task_index, task in enumerate(tasks)
    )
    return snapshot(rows, groups, "分")


def load_source(profile):
    kind = profile.get("kind")
    if kind == "empty":
        return snapshot()
    if kind == "demo":
        return demo_snapshot()
    if kind == "folder":
        return folder_snapshot(profile)
    if kind in ("sqlite", "jsonl"):
        return mapped_snapshot(profile)
    if kind == "http":
        return remote_snapshot(profile["url"])
    raise SourceError("请选择数据来源。")

