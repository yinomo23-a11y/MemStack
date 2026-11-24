from ctypes import windll
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import NamedTemporaryFile
from threading import Lock
from uuid import uuid4

import webview
from filelock import FileLock, Timeout

from sources import EXTRAS, SourceError, inspect_source, load_source, remote_module, remote_url
from writer import revisions, trusted_source, worker

DATA = Path(os.environ["LOCALAPPDATA"]) / "MemStack"
CONFIG = DATA / "connection.json"
RECONNECT = DATA / "reconnect.signal"
WEB = Path(__file__).resolve().parent / "web"
# 打包(onefile/onedir)和源码运行两种情形的图标位置
ICON = Path(getattr(__import__("sys"), "_MEIPASS", Path(__file__).resolve().parent)) / "packaging" / "icon.ico"
if not ICON.exists():
    ICON = Path(__file__).resolve().parent / "packaging" / "icon.ico"


def read_profile():
    try:
        value = json.loads(CONFIG.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def save_profile(value):
    DATA.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=DATA, mode="w", encoding="utf-8", delete=False) as stream:
        temp = Path(stream.name)
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        temp.replace(CONFIG)
    finally:
        temp.unlink(missing_ok=True)


def guarded(action):
    try:
        return {"ok": True, "data": action()}
    except SourceError as error:
        return {"ok": False, "error": str(error)}
    except FileNotFoundError:
        return {"ok": False, "error": "数据文件已移动或不存在，请重新选择。"}
    except PermissionError:
        return {"ok": False, "error": "没有读取此位置的权限。"}
    except (UnicodeError, json.JSONDecodeError):
        return {"ok": False, "error": "数据无法解析，请检查文件格式和 UTF-8 编码。"}
    except sqlite3.Error:
        return {"ok": False, "error": "无法读取 SQLite 文件，请检查文件格式或重试。"}
    except (OSError, KeyError, TypeError, ValueError):
        return {"ok": False, "error": "无法读取此来源，请检查连接设置并重试。"}


class Desktop:
    def __init__(self):
        self._window = None
        self._lock = Lock()
        self._profile = None
        self._features = set()
        self._connection = ""

    def pick(self, kind):
        if kind == "folder":
            result = self._window.create_file_dialog(webview.FileDialog.FOLDER)
        elif kind == "file":
            result = self._window.create_file_dialog(
                webview.FileDialog.OPEN,
                file_types=("记忆数据 (*.db;*.sqlite;*.sqlite3;*.jsonl;*.ndjson)", "所有文件 (*.*)"),
            )
        else:
            return None
        return result[0] if result else None

    def saved(self):
        return read_profile()

    def inspect(self, kind, path, table=""):
        return guarded(lambda: inspect_source(kind, path, table))

    def connect(self, profile):
        return guarded(lambda: self._connect(profile))

    def _connect(self, profile):
        with self._lock:
            kind = profile.get("kind")
            profile_is_writable = profile.get("write_enabled") is True
            if kind == "http":
                profile = {"kind": kind, "url": remote_url(profile.get("url"))}
            elif kind == "folder":
                profile = {"kind": kind, **{
                    name: str(Path(profile[name]).resolve())
                    for name in ("path", "dsh_path", "vault_path") if profile.get(name)
                }}
                if profile_is_writable:
                    profile["write_enabled"] = True
                    if trusted_source(profile) is None:
                        raise SourceError("此目录缺少可写入的忆栈模型。")
            elif kind in ("sqlite", "jsonl"):
                profile = {key: profile[key] for key in ("kind", "path", "table", "mapping") if key in profile}
            result = load_source(profile)
            if kind != "demo":
                saved = None if kind == "empty" else profile
                if read_profile() != saved:
                    save_profile(saved)
            self._profile = profile
            self._connection = uuid4().hex
            self._features = set(result["capabilities"]["features"])
            return self._decorate(result)

    def _decorate(self, result):
        result["write"] = revisions(self._profile or {}, DATA)
        result["connection"] = self._connection
        return result

    def refresh(self):
        def read():
            with self._lock:
                result = load_source(self._profile or {"kind": "empty"})
                self._features = set(result["capabilities"]["features"])
                return self._decorate(result)
        return guarded(read)

    def mutate(self, connection, action, body):
        def write():
            with self._lock:
                if connection != self._connection or not self._profile:
                    raise SourceError("数据源已切换，请重新操作。")
                if not isinstance(body, dict) or len(json.dumps(body)) > 20000:
                    raise SourceError("写入参数无效。")
                DATA.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(dir=DATA, suffix=".request", delete=False) as stream:
                    path = Path(stream.name)
                    stream.write(json.dumps({"profile": self._profile, "action": action, "body": body,
                                             "app_data": str(DATA)}, ensure_ascii=False).encode())
                reply_path = path.with_suffix(".reply")
                command = [sys.executable]
                if not getattr(sys, "frozen", False):
                    command.append(str(Path(__file__).resolve()))
                try:
                    subprocess.run([*command, "--write-worker", str(path)], timeout=50, check=True,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                    reply = json.loads(reply_path.read_bytes())
                    if not reply["ok"]:
                        raise SourceError(reply["error"])
                    return reply["data"]
                except subprocess.SubprocessError:
                    raise SourceError("写入未完成，请刷新查看状态。") from None
                finally:
                    path.unlink(missing_ok=True)
                    reply_path.unlink(missing_ok=True)
        return guarded(write)

    def reconnect_requested(self):
        if not RECONNECT.exists():
            return False
        RECONNECT.unlink(missing_ok=True)
        return True

    def module(self, endpoint):
        def read():
            with self._lock:
                if endpoint not in EXTRAS or endpoint not in self._features:
                    raise SourceError("此数据来源未接入该模块。")
                return remote_module(self._profile["url"], endpoint)
        return guarded(read)


def main():
    # WebView2's legacy MSHTML fallback cannot render the console's WebGL interface.
    from webview.platforms import winforms
    if not winforms.is_chromium:
        windll.user32.MessageBoxW(
            0, "请安装 Microsoft Edge WebView2 Runtime，然后重新打开忆栈。", "忆栈 MemStack", 0x10,
        )
        return
    api = Desktop()
    api._window = webview.create_window(
        "忆栈 MemStack", str(WEB / "index.html"), js_api=api,
        width=1440, height=920, min_size=(1010, 640),
    )
    # pywebview 的 edgechromium 后端不理会 create_window(icon=...)，
    # 开发态（python.exe）标题栏会是蟒蛇图标。这里在窗口创建后
    # 用 Win32 消息直接换掉，打包后与 exe 图标一致。
    def apply_icon():
        if not ICON.exists():
            return
        try:
            hwnd = windll.user32.FindWindowW(None, "忆栈 MemStack")
            if not hwnd:
                return
            # IMAGE_ICON=1, LR_LOADFROMFILE=0x10, LR_DEFAULTSIZE=0x40
            for size, name in ((16, "small"), (32, "small"), (0, "big")):
                handle = windll.user32.LoadImageW(
                    None, str(ICON), 1, size, size, 0x10 | 0x40
                )
                if handle:
                    windll.user32.SendMessageW(
                        hwnd, 0x0080, 0 if name == "small" else 1, handle
                    )
        except OSError:
            pass

    webview.start(gui="edgechromium", private_mode=True, func=apply_icon)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--write-worker":
        worker(Path(sys.argv[2]))
    else:
        DATA.mkdir(parents=True, exist_ok=True)
        instance = FileLock(str(DATA / "desktop.lock"))
        try:
            with instance.acquire(timeout=0):
                main()
        except Timeout:
            if sys.argv[1:] == ["memstack://reconnect"] or sys.argv[1:] == ["memstack://reconnect/"]:
                RECONNECT.touch()

