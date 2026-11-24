"""打包后把 ICO 原样写回 exe 的图标资源。

为什么需要：PyInstaller 的 --icon 会把 ICO 重新编译成旧的 BMP 图标帧，
丢掉 alpha 通道 —— 48px 四角 alpha 从 0 变成 255，视觉上就是直角。
这个脚本用 Win32 UpdateResource 把 PNG 压缩帧原样塞进去。

用法：
    python patch_exe_icon.py <exe路径> <ico路径>
"""
from __future__ import annotations

import ctypes
import os
import struct
import sys
from ctypes import wintypes

RT_ICON = 3
RT_GROUP_ICON = 14
LANG_NEUTRAL = 0

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

BeginUpdateResourceW = kernel32.BeginUpdateResourceW
BeginUpdateResourceW.restype = wintypes.HANDLE
BeginUpdateResourceW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL]

UpdateResourceW = kernel32.UpdateResourceW
UpdateResourceW.restype = wintypes.BOOL
UpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.LPVOID,
                            wintypes.WORD, wintypes.LPVOID, wintypes.DWORD]

EndUpdateResourceW = kernel32.EndUpdateResourceW
EndUpdateResourceW.restype = wintypes.BOOL
EndUpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.BOOL]


def parse_ico(data: bytes):
    """拆 ICO：返回 [(宽, 高, 色数, 数据), ...]。"""
    reserved, typ, count = struct.unpack_from("<HHH", data, 0)
    if reserved != 0 or typ != 1:
        raise ValueError("不是合法 ICO")
    entries = []
    for i in range(count):
        off = 6 + 16 * i
        w, h, colors, rsv, planes, bpp, size, offset = struct.unpack_from(
            "<BBBBHHII", data, off)
        blob = data[offset:offset + size]
        entries.append((w, h, colors, planes, bpp, blob))
    return entries


def patch(exe_path: str, ico_path: str) -> int:
    if not os.path.exists(exe_path):
        print("找不到 exe:", exe_path)
        return 1
    if not os.path.exists(ico_path):
        print("找不到 ico:", ico_path)
        return 1

    ico = open(ico_path, "rb").read()
    entries = parse_ico(ico)
    print("ICO 帧数:", len(entries))

    handle = BeginUpdateResourceW(exe_path, False)
    if not handle:
        print("BeginUpdateResource 失败:", ctypes.get_last_error())
        return 1

    try:
        # ① 逐个写入 RT_ICON（资源 id 从 1 开始）
        for idx, (w, h, colors, planes, bpp, blob) in enumerate(entries, start=1):
            ok = UpdateResourceW(handle, RT_ICON, ctypes.cast(
                ctypes.c_void_p(idx), wintypes.LPVOID), LANG_NEUTRAL,
                blob, len(blob))
            if not ok:
                print("  写 RT_ICON #%d 失败: %s" % (idx, ctypes.get_last_error()))
                return 1
            print("  RT_ICON #%-3d %dx%d  %d 字节" % (idx, w or 256, h or 256, len(blob)))

        # ② 写 RT_GROUP_ICON：指向刚才那些 id
        group = struct.pack("<HHH", 0, 1, len(entries))
        for idx, (w, h, colors, planes, bpp, blob) in enumerate(entries, start=1):
            group += struct.pack("<BBBBHHIH", w, h, colors, 0, planes, bpp,
                                 len(blob), idx)
        ok = UpdateResourceW(handle, RT_GROUP_ICON, ctypes.cast(
            ctypes.c_void_p(1), wintypes.LPVOID), LANG_NEUTRAL,
            group, len(group))
        if not ok:
            print("  写 RT_GROUP_ICON 失败: %s" % ctypes.get_last_error())
            return 1
        print("  RT_GROUP_ICON  %d 字节" % len(group))
    finally:
        if not EndUpdateResourceW(handle, False):
            print("EndUpdateResource 失败:", ctypes.get_last_error())
            return 1

    print("✅ 已写回:", exe_path)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(patch(sys.argv[1], sys.argv[2]))
