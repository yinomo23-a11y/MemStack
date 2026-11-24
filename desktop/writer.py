"""Reversible writes to a trusted local MemStack model."""
from contextlib import ExitStack, closing, nullcontext, redirect_stdout
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sqlite3
import stat
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
import time

from filelock import FileLock, Timeout

from native_data import PRIVATE_TEXT, SLIDERS, numeric
from sources import MAX_JSON_BYTES, SourceError, pool_files

SETTINGS = "state/runtime_overrides.json"
HANDOVER = "state/handover.json"
CARDS = "state/distillation/main.json"
DSH = ("dsh-session-vacuum/config.json", "cordis.patch.yml")
ACTIONS = {
    "settings": ("记忆参数", (SETTINGS,)), "handover": ("交接单", (HANDOVER,)),
    "dsh": ("DSH 配置", DSH), "generate": ("生成蒸馏候选", (CARDS,)),
    "review": ("审核蒸馏卡片", (CARDS,)), "review_batch": ("批量审核蒸馏卡片", (CARDS,)),
    "maintain": ("校准蒸馏卡片", (CARDS,)), "outcome": ("记录技能结果", (CARDS,)),
}
UNDO_MODULES = {
    "m4": ("settings",),
    "m5": ("handover", "dsh"),
    "m6": ("generate", "review", "review_batch", "maintain", "outcome"),
}
REVIEW_FIELDS = ("id", "status", "content", "source_revisions", "updated_at", "review", "out_kind",
                 "out_turns", "quotes", "trigger", "steps", "skill_status", "evidence_sha")
LIMITS = {"d2": (1, 12), "ep": (.2, 3), "erd": (.1, 1.6), "epf": (.15, 1),
          "we": (0, .30), "st": (10, 60), "cd": (60, 365), "wb": (0, .20)}


def trusted_source(profile):
    if profile.get("kind") != "folder" or profile.get("write_enabled") is not True:
        return None
    model = Path(profile["path"]).resolve()
    if model.name == "data" and not (model / "config.py").is_file():
        model = model.parent
    for name in ("config.py", "runtime_config.py", "distillation.py", "handover.py", "common/locking.py"):
        if not (model / name).is_file():
            return None
    requested = Path(profile["path"])
    files = list(pool_files(requested)) or list(pool_files(requested / "data"))
    if not any(path.name == "owner_active.json" for _, _, path in files):
        return None
    return model, files[0][2].parent.resolve()


def read_bytes(path):
    # Refuse junctions/symlinks that would change the owner of a mapped target.
    if path.resolve() != path.absolute():
        raise SourceError("写入目标经过重定向，无法安全保存。")
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_JSON_BYTES + 1)
    except FileNotFoundError:
        return None
    if len(data) > MAX_JSON_BYTES:
        raise SourceError("写入文件超过 64 MiB，无法创建回滚副本。")
    return data


def digest(value):
    return sha256(value).hexdigest() if value is not None else "missing"


def card_version(card):
    data = {key: card.get(key) for key in REVIEW_FIELDS}
    return sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def target_paths(profile, data, action):
    folder = Path(profile["dsh_path"]).resolve() if action == "dsh" and profile.get("dsh_path") else data
    if action == "dsh" and not profile.get("dsh_path"):
        raise SourceError("先关联 DSH 会话目录。")
    return {name: folder / name for name in ACTIONS[action][1]}


def atomic(path, data):
    if data is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.is_file() else None
    with NamedTemporaryFile(dir=path.parent, prefix="memstack-", delete=False) as stream:
        staged = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if mode is not None:
            staged.chmod(mode)
        for attempt in range(6):
            try:
                staged.replace(path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(.05 * (attempt + 1))
    finally:
        staged.unlink(missing_ok=True)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def json_object(raw):
    value = json.loads(raw) if raw is not None else {}
    if not isinstance(value, dict):
        raise SourceError("无法识别现有文件格式，写入已停止。")
    return value


def handover_bytes(body, before):
    import handover

    names = ("topic", "emotion", "pending_tail", "trivia")
    fields = [body.get(name) for name in names]
    if (not all(isinstance(f, str) and len(f) <= 500 for f in fields[:3])
            or not isinstance(fields[3], list) or len(fields[3]) > 12
            or not all(isinstance(f, str) and len(f) <= 500 for f in fields[3])
            or not handover.should_write(body.get("trigger"))):
        raise SourceError("交接内容或触发条件无效。")
    if digest(before) != body.get("revision"):
        raise SourceError("已有新的交接记录，请刷新后再保存。")
    fitted = handover._fit(*fields)
    if not any(fitted.values()):
        raise SourceError("交接内容为空。")
    protected = body.get("is_intimate") is True or any(PRIVATE_TEXT.search(str(f)) for f in fields)
    with TemporaryDirectory(prefix="memstack-handover-") as folder:
        path = Path(folder) / "handover.json"
        if before is not None:
            json_object(before)
            path.write_bytes(before)
        result = handover.write_handover("memstack-desktop", **fitted, path=str(path), is_intimate=protected)
        if result.get("_verify") != "ok":
            raise SourceError("交接单回读失败。")
        output = path.read_bytes()
        return output, {"seq": result["seq"], "protected": protected, "revision": digest(output)}


def settings_bytes(body, before):
    import config
    import runtime_config
    from native_data import slider_view

    if digest(before) != body.get("revision"):
        raise SourceError("模型参数已更新，请刷新后再保存。")
    values = body.get("values")
    if not isinstance(values, dict) or set(values) - set(LIMITS) or not values:
        raise SourceError("没有有效的可写模型参数。")
    changes = json_object(before)
    current = {name: getattr(config, name) for name in runtime_config.NAMES}
    for key, value in values.items():
        runtime_config.number(value, *LIMITS[key])
        if key == "d2":
            changes["DECAY_RATE_MAP"] = {**current["DECAY_RATE_MAP"], 2: value}
        elif key == "epf":
            protect = current["EMOTION_DECAY_PROTECT"]
            changes["EMOTION_DECAY_PROTECT"] = {**protect, "high": (protect["high"][0], value)}
        else:
            name = SLIDERS[key]
            changes[name] = int(value) if type(current[name]) is int and value == int(value) else value
    validated = runtime_config.validated(changes, current)
    out = json_bytes(validated)
    return out, {"values": slider_view({**current, **validated}), "revision": digest(out)}


def slim_node(text):
    import yaml

    def value(node, key):
        return next((v for k, v in node.value if isinstance(k, yaml.ScalarNode) and k.value == key), None)

    nodes = [yaml.compose(text, Loader=yaml.SafeLoader)]
    seen, found = set(), []
    while nodes:
        node = nodes.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, yaml.MappingNode):
            name = value(node, "name")
            if isinstance(name, yaml.ScalarNode) and name.value == "dsh-prompt-slim":
                cfg = value(node, "config")
                if isinstance(cfg, yaml.MappingNode):
                    found.append(value(cfg, "minTools"))
            else:
                nodes.extend(v for _, v in node.value)
        elif isinstance(node, yaml.SequenceNode):
            nodes.extend(node.value)
    if len(found) == 1 and isinstance(found[0], yaml.ScalarNode) and found[0].tag.endswith(":int"):
        return found[0]
    return None


def dsh_bytes(body, before):
    updates, revisions = body.get("updates"), body.get("revisions")
    if (not isinstance(updates, dict) or not updates or set(updates) - set(DSH)
            or not isinstance(revisions, dict)):
        raise SourceError("没有有效的 DSH 可写字段。")
    output = dict(before)
    result = {}
    for name, fields in updates.items():
        old = before[name]
        if old is None or digest(old) != revisions.get(name):
            raise SourceError("DSH 配置已修改或缺失，请刷新后再保存。")
        if not isinstance(fields, dict) or not fields:
            raise SourceError("DSH 配置字段无效。")
        if name == DSH[0]:
            if set(fields) - {"enabled", "dryRun"} or any(type(value) is not bool for value in fields.values()):
                raise SourceError("DSH 开关参数无效。")
            value = {**json_object(old), **fields}
            output[name] = json_bytes(value)
            result.update(enabled=value.get("enabled"), dryRun=value.get("dryRun"))
        else:
            n = fields.get("minTools")
            if set(fields) != {"minTools"} or type(n) is not int or not 1 <= n <= 500:
                raise SourceError("minTools 必须在 1 到 500 之间。")
            text = old.decode("utf-8-sig")
            node = slim_node(text)
            if node is None:
                raise SourceError("配置没有唯一的 dsh-prompt-slim.minTools 字段。")
            replaced = text[:node.start_mark.index] + str(n) + text[node.end_mark.index:]
            output[name] = replaced.encode("utf-8-sig" if old.startswith(b"\xef\xbb\xbf") else "utf-8")
            result["minTools"] = n
    result["revisions"] = {name: digest(value) for name, value in output.items()}
    return output, result


def generate_cards(model, body, data):
    batches = body.get("batches")
    if type(batches) is not int or batches not in (1, 3, 5):
        raise SourceError("需要选择 1、3 或 5 批。")
    sources = model.gather("main", read_only=True)
    history = data.setdefault("cards", [])
    if not isinstance(history, list):
        raise SourceError("旧版卡片库无法安全更新。")
    current = sum(model.card_view(card)["status"] in ("active", "provisional") for card in history)
    known = {card["id"] for card in history}
    new = []
    now = time.time()
    for mid, row in sorted(sources.items(), key=lambda x: -(numeric(x[1].get("strength")) or 0)):
        raw = str(row.get("content") or "").strip()
        if len(raw) < 12 or PRIVATE_TEXT.search(raw):
            continue
        quote = raw[:180]
        rev = model.revision(row)
        key = sha256(json_bytes(["memstack-local", mid, rev])).hexdigest()
        if key in known:
            continue
        if len(new) >= min(batches * 6, max(0, model.MAX_CURRENT_CARDS - current)):
            break
        new.append({"id": key, "content": quote, "out_kind": "画像", "out_turns": [mid],
                    "quotes": {mid: quote}, "source_revisions": {mid: rev}, "scope": "main",
                    "status": "provisional", "distill_status": "provisional", "mem_type": "distilled",
                    "confidence": "unverified", "model": "memstack-local", "at": now,
                    "strength": 60.0, "strength_at": now, "configuration_sha": data.get("configuration_sha")})
    if body.get("dry_run") is not True:
        history.extend(new)
    return {"sources": len(sources), "pending": len(new), "planned": len(new),
            "state": "dry_run" if body.get("dry_run") is True else "provisional"}


def card_bytes(action, body, before):
    import common.locking
    import distillation as model

    data = json_object(before)
    if action == "generate":
        result = generate_cards(model, body, data)
        return (json_bytes(data) if result["planned"] and body.get("dry_run") is not True else before), result
    raw = None
    batch_targets = []
    if action == "review_batch":
        items = body.get("items")
        if not isinstance(items, list) or not items or len(items) > 300:
            raise SourceError("批量审核项无效（每次最多 300 张）。")
        by_sid = {"source-" + sha256(str(card.get("id")).encode()).hexdigest(): card
                  for card in data.get("cards", []) if isinstance(card, dict)}
        seen = set()
        for entry in items:
            if (not isinstance(entry, dict) or not isinstance(entry.get("id"), str)
                    or type(entry.get("approved")) is not bool or entry["id"] in seen):
                raise SourceError("批量审核参数无效。")
            seen.add(entry["id"])
            card = by_sid.get(entry["id"])
            if card is None or card.get("scope") == "intimate" or card.get("is_intimate"):
                raise SourceError("存在不可写的受保护卡片，请刷新后重试。")
            if card_version(card) != entry.get("version"):
                raise SourceError("部分卡片已变化，请刷新后重试。")
            batch_targets.append((entry["id"], card["id"], entry["approved"]))
    else:
        raw = next((card for card in data.get("cards", [])
                    if "source-" + sha256(str(card.get("id")).encode()).hexdigest() == body.get("id")), None)
        if action in ("review", "outcome"):
            if raw is None or raw.get("scope") == "intimate" or raw.get("is_intimate"):
                raise SourceError("找不到可写的卡片。")
            if card_version(raw) != body.get("version"):
                raise SourceError("卡片已变化，请刷新后重试。")
    if before is None:
        return before, {"current": 0, "skills": 0, "skill_capacity": model.SKILL_MAX, "stale": 0, "archived": 0}
    with TemporaryDirectory(prefix="memstack-cards-") as folder:
        path = Path(folder) / "main.json"
        path.write_bytes(before)
        orig_path = model._path
        graph = model.source_graph
        state_lock = common.locking.state_lock
        model._path = lambda scope: str(path)
        model.source_graph = lambda **kw: graph(read_only=True)
        common.locking.state_lock = lambda name: nullcontext() if name == "distillation-main" else state_lock(name)
        try:
            if action == "maintain":
                result = model.maintain("main")
            elif action == "review_batch":
                import atomic_io

                # Cache source_graph & keep main.json in memory across the batch:
                # avoids N× full-pool reloads and N× multi-MB JSON fsyncs.
                cached_graph = graph(read_only=True)
                mem_data = json.loads(before.decode("utf-8"))
                orig_load = model._load
                orig_write = atomic_io.atomic_write_json
                model.source_graph = lambda **kw: cached_graph
                model._load = lambda _p: mem_data
                atomic_io.atomic_write_json = lambda _p, _d: None
                try:
                    results = []
                    counts = {"active": 0, "rejected": 0, "stale": 0, "blocked": 0}
                    for sid, cid, approved in batch_targets:
                        try:
                            status = model.review("main", cid, reviewer="memstack-desktop", approved=approved)
                        except ValueError as exc:
                            raise SourceError(f"批量审核中止：{exc}") from None
                        results.append({"id": sid, "status": status})
                        if status in counts:
                            counts[status] += 1
                        else:
                            counts["blocked"] += 1
                    result = {"applied": len(results), **counts, "results": results}
                    return json_bytes(mem_data), result
                finally:
                    model._load = orig_load
                    atomic_io.atomic_write_json = orig_write
            elif action == "review":
                if type(body.get("approved")) is not bool:
                    raise SourceError("缺少审核结果。")
                status = model.review("main", raw["id"], reviewer="memstack-desktop", approved=body["approved"])
                result = {"status": status}
            else:
                if type(body.get("ok")) is not bool:
                    raise SourceError("缺少技能执行结果。")
                value = model.record_skill_outcome("main", raw["id"], ok=body["ok"])
                result = {"status": value["skill_status"], "fails": value["fail_count"]}
            return path.read_bytes(), result
        finally:
            model._path = orig_path
            model.source_graph = graph
            common.locking.state_lock = state_lock


def source_key(profile, data):
    return json.dumps([str(data), profile.get("dsh_path")], ensure_ascii=False)


def undo_status(profile, data, app_data):
    path = app_data / "undo.sqlite3"
    if not path.is_file():
        return {}
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        rows = db.execute(
            "SELECT id, action FROM operation WHERE source = ? ORDER BY id DESC", (source_key(profile, data),)
        ).fetchall()
    result = {}
    for module, actions in UNDO_MODULES.items():
        latest = next(((tx, action) for tx, action in rows if action in actions), None)
        if latest is not None:
            tx, action = latest
            result[module] = {"id": tx, "label": ACTIONS[action][0]}
    return result


def revisions(profile, app_data=None):
    info = trusted_source(profile)
    if info is None:
        return {}
    _, data = info
    actions = ["settings", "handover", "generate", "review", "review_batch", "maintain", "outcome"]
    if profile.get("dsh_path") and all((Path(profile["dsh_path"]) / f).is_file() for f in DSH):
        actions.append("dsh")
    names = (SETTINGS, HANDOVER, *(DSH if "dsh" in actions else ()))
    paths = {name: (Path(profile["dsh_path"]) if name in DSH else data) / name for name in names}
    result = {"actions": actions, "revisions": {name: digest(read_bytes(path)) for name, path in paths.items()}}
    if app_data is not None:
        result["undo"] = undo_status(profile, data, app_data)
    return result


def restore(paths, before, after):
    # A synchronized writer may have changed a file since MemStack touched it.
    current = {name: read_bytes(path) for name, path in paths.items()}
    if any(current[name] not in (before[name], after[name]) for name in paths):
        raise SourceError("文件已被其他程序修改。已保留现状及备份。")
    for name, path in paths.items():
        if current[name] == after[name] and after[name] != before[name]:
            if read_bytes(path) != current[name]:
                raise SourceError("文件刚被其他程序修改。已停止回滚。")
            atomic(path, before[name])


def record_paths(db, transaction, profile, data, action):
    paths = target_paths(profile, data, action)
    rows = db.execute("SELECT name, before, after FROM backup WHERE tx = ?", (transaction,)).fetchall()
    before = {name: old for name, old, _ in rows}
    after = {name: new for name, _, new in rows}
    if not before or set(before) - set(paths):
        raise SourceError("回滚记录与当前数据源不匹配。")
    return {k: paths[k] for k in before}, before, after


def lock_for(stack, action, locking):
    if action != "dsh":
        lock = locking.memory_lock(singleton=True)
        if not isinstance(lock, FileLock):
            raise SourceError("记忆模型没有可用的写入锁。")
        stack.enter_context(lock.acquire(timeout=8))
    name = "console-dsh" if action == "dsh" else "distillation-main" if action in ("generate", "review", "review_batch", "maintain", "outcome") else None
    if name:
        lock = locking.state_lock(name)
        if not isinstance(lock, FileLock):
            raise SourceError("记忆模型没有可用的状态锁。")
        stack.enter_context(lock.acquire(timeout=8))


def execute(profile, action, body, app_data):
    source = trusted_source(profile)
    if source is None:
        raise SourceError("当前来源未启用本机写入。")
    model, data = source
    os.environ["MEMORY_DATA_DIR"] = str(data)
    os.environ["DSH_MEMORY_DATA_DIR"] = str(data)  # privacy-scan: allow
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(model))
    import common.locking as locking

    app_data.mkdir(parents=True, exist_ok=True)
    key = source_key(profile, data)
    where, params = "source = ?", (key,)
    if action == "undo" and "module" in body:
        module = body["module"]
        if not isinstance(module, str) or module not in UNDO_MODULES or type(body.get("id")) is not int:
            raise SourceError("回滚参数无效。")
        allowed = UNDO_MODULES[module]
        where += " AND action IN (" + ", ".join("?" for _ in allowed) + ")"
        params += allowed
    with FileLock(str(app_data / "write.lock"), timeout=15), closing(sqlite3.connect(app_data / "undo.sqlite3")) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS operation (
                id INTEGER PRIMARY KEY, source TEXT, action TEXT, status TEXT
            );
            CREATE TABLE IF NOT EXISTS backup (
                tx INTEGER, name TEXT, before BLOB, after BLOB, PRIMARY KEY(tx, name),
                FOREIGN KEY(tx) REFERENCES operation(id) ON DELETE CASCADE
            );
            PRAGMA foreign_keys = ON;
        """)
        query = "SELECT id, action FROM operation WHERE " + where
        if action == "undo":
            latest = db.execute(query + " ORDER BY id DESC LIMIT 1", params).fetchone()
            if latest is None:
                raise SourceError("当前范围没有可回滚的写入。")
            if "module" in body and latest[0] != body["id"]:
                raise SourceError("已有新的写入，请刷新后重新确认回滚。")
        pending = db.execute(query + " AND status != 'done' ORDER BY id", params).fetchall()
        for tx, prior in pending:
            with ExitStack() as stack:
                lock_for(stack, prior, locking)
                paths, before, after = record_paths(db, tx, profile, data, prior)
                restore(paths, before, after)
                with db:
                    db.execute("DELETE FROM operation WHERE id = ?", (tx,))
        if action == "undo":
            if pending:
                prior = pending[-1][1]
                return {"undone": ACTIONS[prior][0], "action": prior}
            tx, prior = latest
            with ExitStack() as stack:
                lock_for(stack, prior, locking)
                paths, before, after = record_paths(db, tx, profile, data, prior)
                if any(read_bytes(p) != after[n] for n, p in paths.items()):
                    raise SourceError("文件已有其他改动，撤销已停止。")
                with db:
                    db.execute("UPDATE operation SET status = 'undoing' WHERE id = ?", (tx,))
                restore(paths, before, after)
                with db:
                    db.execute("DELETE FROM operation WHERE id = ?", (tx,))
            return {"undone": ACTIONS[prior][0], "action": prior}
        if action not in ACTIONS:
            raise SourceError("未知写入操作。")
        paths = target_paths(profile, data, action)
        with ExitStack() as stack:
            lock_for(stack, action, locking)
            before = {name: read_bytes(path) for name, path in paths.items()}
            if action == "dsh":
                after, response = dsh_bytes(body, before)
            else:
                name, old = next(iter(before.items()))
                if action == "settings":
                    content, response = settings_bytes(body, old)
                elif action == "handover":
                    content, response = handover_bytes(body, old)
                else:
                    content, response = card_bytes(action, body, old)
                after = {name: content}
            changed = {name: p for name, p in paths.items() if before[name] != after[name]}
            if not changed:
                return {**response, "written": False}
            with db:
                tx = db.execute("INSERT INTO operation (source, action, status) VALUES (?, ?, 'writing')", (key, action)).lastrowid
                db.executemany("INSERT INTO backup VALUES (?, ?, ?, ?)",
                               [(tx, name, before[name], after[name]) for name in changed])
            try:
                for name, path in changed.items():
                    if read_bytes(path) != before[name]:
                        raise SourceError("文件已被其他程序修改，请刷新后重试。")
                    atomic(path, after[name])
                    if read_bytes(path) != after[name]:
                        raise SourceError("回读不一致。")
            except Exception:
                restore(changed, before, after)
                with db:
                    db.execute("DELETE FROM operation WHERE id = ?", (tx,))
                raise
            with db:
                db.execute("UPDATE operation SET status = 'done' WHERE id = ?", (tx,))
                db.execute("DELETE FROM operation WHERE source = ? AND status = 'done' AND id NOT IN "
                           "(SELECT id FROM operation WHERE source = ? AND status = 'done' ORDER BY id DESC LIMIT 10)", (key, key))
            return {**response, "written": True}


def worker(path):
    request = json.loads(path.read_bytes())
    try:
        with redirect_stdout(io.StringIO()):
            result = execute(request["profile"], request["action"], request["body"], Path(request["app_data"]))
        reply = {"ok": True, "data": result}
    except SourceError as error:
        reply = {"ok": False, "error": str(error)}
    except Timeout:
        reply = {"ok": False, "error": "另一个程序正在写入，请稍后重试。"}
    except Exception as error:
        reply = {"ok": False, "error": "保存操作未完成（" + type(error).__name__ + "）。"}
    atomic(path.with_suffix(".reply"), json_bytes(reply))

