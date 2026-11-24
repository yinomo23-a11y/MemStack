"""MemStack 适配器：把本地 MemStack 记忆池目录暴露为 HTTP v1 接口喂给忆栈 App。

只读。不改任何记忆文件。

用法：
    python memstack_adapter.py                       # 默认读取仓库 data/，监听 127.0.0.1:8787
    python memstack_adapter.py --data-dir ./my-data  # 指定自定义记忆目录
    python memstack_adapter.py --port 9000

然后在 App 里「接入数据 → HTTP」填 http://127.0.0.1:8787
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 池文件（同时兼容公开版与五池命名） → (候选文件名, 池 id, 中文标签)
POOLS = [
    (("active.json", "owner_active.json"), "active", "长期记忆"),
    (("shortterm.json", "owner_shortterm.json"), "shortterm", "短期记忆"),
    (("cold.json", "owner_cold.json"), "cold", "冷存记忆"),
]

_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
DATA_DIR = (
    os.environ.get("MEMORY_DATA_DIR")
    or os.environ.get("MEMSTACK_DATA_DIR")
    or _DEFAULT_DATA_DIR
)

#: 最多返回多少条（App 单快照上限 20000）
MAX_RECORDS = 20000


def _iso(ts) -> str | None:
    """把 Unix 秒/毫秒或 ISO 字符串统一成 ISO 8601。"""
    if ts is None or ts == "":
        return None
    if isinstance(ts, str):
        s = ts.strip()
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return None
    if isinstance(ts, (int, float)):
        v = float(ts)
        if v > 1e12:      # 毫秒
            v /= 1000.0
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()
        except (OSError, ValueError, OverflowError):
            return None
    return None


def _entities(row) -> list[str]:
    out: list[str] = []
    ents = row.get("entities") or row.get("entity_tags") or []
    if isinstance(ents, dict):
        ents = list(ents.keys())
    if isinstance(ents, (list, tuple)):
        for e in ents:
            s = str(e).strip()
            if s and s not in out:
                out.append(s[:40])
    tags = row.get("tags") or []
    if isinstance(tags, list):
        for t in tags:
            s = str(t).strip()
            if s and s not in out:
                out.append(s[:40])
    return out[:12]


def _strength(row) -> float | None:
    for k in ("strength", "current_strength", "impression_strength"):
        v = row.get(k)
        if isinstance(v, (int, float)):
            return round(float(v), 2)
    return None


def _time_of(row) -> str | None:
    for k in ("create_time", "last_active", "said_at", "last_recalled_at", "event_time"):
        v = row.get(k)
        s = _iso(v)
        if s:
            return s
    return None


def _resolve_pool_path(candidates: tuple[str, ...]) -> str | None:
    for fname in candidates:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            return path
    return None


def _load_pool(candidates: tuple[str, ...], pool_id: str) -> list[dict]:
    path = _resolve_pool_path(candidates)
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []

    rows = data.get("memories")
    if not isinstance(rows, list):
        # working_mem 风格：{sessions: {...}} 不在这里处理
        return []

    out: list[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        content = str(r.get("content") or "").strip()
        if not content:
            continue
        rec = {"content": content[:2000], "pool": pool_id}
        s = _strength(r)
        if s is not None:
            rec["strength"] = s
        t = _time_of(r)
        if t:
            rec["time"] = t
        ent = _entities(r)
        if ent:
            rec["entities"] = ent
        if r.get("is_private"):
            rec["private"] = True
        out.append(rec)
    return out


def snapshot() -> tuple[list[dict], bool]:
    """返回 (记录, 是否被截断)。跨池按时间倒序。"""
    all_rows: list[dict] = []
    for candidates, pid, _label in POOLS:
        all_rows.extend(_load_pool(candidates, pid))

    def key(r):
        return r.get("time") or ""

    all_rows.sort(key=key, reverse=True)
    if len(all_rows) > MAX_RECORDS:
        return all_rows[:MAX_RECORDS], True
    return all_rows, False


def capabilities() -> dict:
    present = []
    for candidates, pid, label in POOLS:
        if _resolve_pool_path(candidates):
            present.append({"id": pid, "label": label})
    if not present:
        # 一个池都没有也给出声明，App 才能正常渲染空状态
        present = [{"id": pid, "label": label} for _c, pid, label in POOLS]
    return {
        "version": 1,
        "features": ["snapshot", "audit"],
        "pools": present,
        "strength_unit": "分",
    }


def audit_rows() -> list[dict]:
    rows = [{"label": "数据目录", "value": DATA_DIR, "section": "hashes"}]
    for candidates, _pid, label in POOLS:
        path = _resolve_pool_path(candidates)
        if path:
            st = os.stat(path)
            rows.append({
                "label": label,
                "value": "%d 条 · %.1f MB" % (
                    len(_load_pool(candidates, _pid)), st.st_size / 1024 / 1024),
                "section": "hashes",
            })
        else:
            rows.append({"label": label, "value": "不存在", "section": "hashes"})
    rows.append({"label": "上限", "value": "%d 条/快照" % MAX_RECORDS, "section": "hashes"})
    return rows


class Handler(BaseHTTPRequestHandler):
    server_version = "MemStackAdapter/1.0"

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            if path == "/capabilities":
                self._send(capabilities())
            elif path == "/snapshot":
                recs, trunc = snapshot()
                payload = {"records": recs}
                if trunc:
                    payload["truncated"] = True
                self._send(payload)
            elif path == "/audit":
                self._send({"rows": audit_rows()})
            elif path == "/":
                self._send({"ok": True, "endpoints": ["/capabilities", "/snapshot", "/audit"]})
            else:
                self.send_error(404)
        except Exception as e:      # 任何异常都返回 JSON，别让 App 看到空响应
            self._send({"error": type(e).__name__, "detail": str(e)[:200]})

    def log_message(self, fmt, *args):
        sys.stderr.write("[adapter] " + (fmt % args) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    global DATA_DIR
    if args.data_dir:
        DATA_DIR = args.data_dir

    recs, trunc = snapshot()
    print("数据目录: %s" % DATA_DIR)
    print("快照记录: %d 条%s" % (len(recs), "（已截断）" if trunc else ""))
    print("监听: http://%s:%d" % (args.host, args.port))
    print("在忆栈 App 里：接入数据 → HTTP → 填上面的地址")
    print()

    with ThreadingHTTPServer((args.host, args.port), Handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止")


if __name__ == "__main__":
    main()
