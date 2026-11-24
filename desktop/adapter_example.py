"""Minimal read-only HTTP adapter. Replace records() with a database query."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json


def records():
    return [{
        "content": "为示例项目记录一次设计评审",
        "pool": "project",
        "strength": 72,
        "time": "2026-01-01T12:00:00Z",
        "entities": ["示例项目"],
    }]


class Adapter(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/capabilities":
            payload = {
                "version": 1, "features": ["snapshot", "audit"], "strength_unit": "分",
                "pools": [{"id": "project", "label": "项目记忆"}],
            }
        elif self.path == "/snapshot":
            payload = {"records": records()}
        elif self.path == "/audit":
            payload = {"rows": [{"label": "接口模式", "value": "只读演示"}]}
        else:
            self.send_error(404)
            return
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    with ThreadingHTTPServer(("127.0.0.1", 8787), Adapter) as server:
        server.serve_forever()

