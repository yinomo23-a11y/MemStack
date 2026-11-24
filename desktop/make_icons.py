"""为 ICO 逐尺寸生成圆角遮罩。

问题：PIL 的 save(format='ICO', sizes=[...]) 会用自己内部的重采样把小尺寸
      压出来，四角 alpha 被抹成 255 —— 32/16px 上就变回直角。
      而 Windows 任务栏/桌面恰好主要用 32px。

做法：对每个尺寸**单独**用超采样画圆角遮罩，再合成到缩放后的图上，
      最后手工打包 ICO（每帧一张独立 PNG），不让 PIL 重新采样。
"""
from __future__ import annotations

import os
import struct
import sys

from PIL import Image, ImageDraw

BASE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(BASE, "packaging")
SRC = os.path.join(PKG, "icon-source.png")
WEB = os.path.join(BASE, "web")

SIZES = [16, 24, 32, 48, 64, 128, 256]
RADIUS_RATIO = 0.18
BLACK_SUM = 18


def content_box(img):
    w, h = img.size
    px = img.convert("RGB").load()

    def row_ok(y):
        return any(sum(px[x, y]) > BLACK_SUM for x in range(0, w, 2))

    def col_ok(x):
        return any(sum(px[x, y]) > BLACK_SUM for y in range(0, h, 2))

    top = 0
    while top < h - 1 and not row_ok(top):
        top += 1
    bot = h - 1
    while bot > top and not row_ok(bot):
        bot -= 1
    left = 0
    while left < w - 1 and not col_ok(left):
        left += 1
    right = w - 1
    while right > left and not col_ok(right):
        right -= 1
    return left, top, right + 1, bot + 1


def rounded_mask(size: int, ratio: float) -> Image.Image:
    """按目标尺寸直接画圆角遮罩（8x 超采样），不走缩放。"""
    s = 8
    big = size * s
    m = Image.new("L", (big, big), 0)
    d = ImageDraw.Draw(m)
    # 半径至少 2 像素，保证小尺寸也看得出圆
    r = max(int(round(size * ratio)) * s, 2 * s)
    d.rounded_rectangle((0, 0, big - 1, big - 1), radius=r, fill=255)
    return m.resize((size, size), Image.LANCZOS)


def make_tile(master: Image.Image, size: int, ratio: float) -> Image.Image:
    tile = master.resize((size, size), Image.LANCZOS).convert("RGBA")
    tile.putalpha(rounded_mask(size, ratio))
    return tile


def write_ico(tiles: list[Image.Image], path: str) -> None:
    """手写 ICO：每帧一张独立 PNG（Vista+ 支持）。"""
    frames = []
    for t in tiles:
        buf = os.path.join(os.environ["TEMP"], "_ico_tmp.png")
        t.save(buf, format="PNG")
        frames.append(open(buf, "rb").read())
        os.remove(buf)

    n = len(tiles)
    header = struct.pack("<HHH", 0, 1, n)
    offset = 6 + 16 * n
    entries = b""
    for t, data in zip(tiles, frames):
        w = 0 if t.size[0] >= 256 else t.size[0]
        h = 0 if t.size[1] >= 256 else t.size[1]
        entries += struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32, len(data), offset)
        offset += len(data)

    with open(path, "wb") as f:
        f.write(header)
        f.write(entries)
        for data in frames:
            f.write(data)


def main() -> int:
    src = Image.open(SRC).convert("RGBA")
    box = content_box(src)
    master = src.crop(box)
    w, h = master.size
    side = min(w, h)
    master = master.crop(((w - side) // 2, (h - side) // 2,
                          (w - side) // 2 + side, (h - side) // 2 + side))
    print("母图: %d x %d  圆角 %.0f%%" % (master.size[0], master.size[1], RADIUS_RATIO * 100))

    tiles = [make_tile(master, s, RADIUS_RATIO) for s in SIZES]
    ico_path = os.path.join(PKG, "icon.ico")
    write_ico(tiles, ico_path)
    print("生成 %s (%.1f KB, %d 帧)" % (ico_path, os.path.getsize(ico_path) / 1024, len(tiles)))

    # 逐个校验 alpha
    print("\n逐尺寸校验四角 alpha：")
    ok = True
    for t in tiles:
        s = t.size[0]
        a = [t.getpixel((1, 1))[3], t.getpixel((s - 2, 1))[3],
             t.getpixel((1, s - 2))[3], t.getpixel((s - 2, s - 2))[3]]
        good = max(a) < 128
        if not good:
            ok = False
        print("  %3dpx  四角alpha=%-18s %s" % (s, a, "✅" if good else "❌"))

    # PNG 副本
    for path, size in [
        (os.path.join(PKG, "icon-256.png"), 256),
        (os.path.join(PKG, "icon-1024.png"), 1024),
        (os.path.join(WEB, "icon-192.png"), 192),
        (os.path.join(WEB, "icon-512.png"), 512),
        (os.path.join(WEB, "favicon.png"), 32),
    ]:
        make_tile(master, size, RADIUS_RATIO).save(path, format="PNG", optimize=True)
        print("  生成 %s" % os.path.basename(path))

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
