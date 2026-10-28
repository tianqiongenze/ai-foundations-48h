# -*- coding: utf-8 -*-
"""把真跑捕获的终端文本渲染成「终端窗口」样式的 PNG，供 PPT 证据页插图。

文本逐字来自 capture97.py 捕获的 run.py / pytest / report.txt，不做任何补写；
只负责排版（等宽 ASCII + 中文回退字体、自动换行、窗口标题栏）。
"""
import os
import re
from PIL import Image, ImageDraw, ImageFont

ROOT = "/Users/joelgong/PycharmProjects/ai-course-teaching-resources"
MONO_CAND = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Courier.ttc",
             "/System/Library/Fonts/Monaco.ttf"]
CJK_CAND = [os.path.join(ROOT, "01-AI应用基础/fonts/ZH_Font.ttf"),
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"]
BG = (24, 26, 32)
BAR = (46, 50, 60)
FG = (226, 230, 236)
ACCENT = (120, 200, 140)
WARN = (232, 148, 118)
TITLE = (150, 160, 178)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
CJK = re.compile(r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")


def _font(paths, size):
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def _wrap(line, mono, cjk, max_w):
    """按显示宽度折行，中文/英文各自按字切。"""
    out, cur, cur_w = [], "", 0.0
    for ch in line:
        f = cjk if CJK.search(ch) else mono
        w = f.getlength(ch)
        if cur_w + w > max_w and cur:
            out.append(cur)
            cur, cur_w = ch, w
        else:
            cur += ch
            cur_w += w
    out.append(cur)
    return out or [""]


def render(lines, out_path, title, width=1280, height=720, size=18, fg=FG):
    """lines: 终端文本行（list[str]）；生成 PNG 到 out_path。"""
    mono = _font(MONO_CAND, size)
    cjk = _font(CJK_CAND, size)
    pad, bar_h = 22, 40
    max_w = width - 2 * pad
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, width, bar_h], fill=BAR)
    for i, col in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 88)]):
        d.ellipse([pad + i * 20, 14, pad + i * 20 + 13, 27], fill=col)
    d.text((pad + 80, 11), title, font=cjk, fill=TITLE)
    y = bar_h + 14
    shown = []
    for raw in lines:
        for ln in _wrap(ANSI.sub("", raw), mono, cjk, max_w):
            shown.append(ln)
    # 只保留能放下的行；溢出时首行提示，保尾部（结论在末段）
    lh = int(size * 1.55)
    fit = (height - y - pad) // lh
    if len(shown) > fit:
        shown = ["…（前 %d 行省略，完整输出见 工具/ppt97/evidence/%s）…"
                 % (len(shown) - fit + 1, "运行日志")] + shown[-(fit - 1):]
    for ln in shown:
        x = pad
        for seg, is_cjk in _segments(ln):
            f = cjk if is_cjk else mono
            col = WARN if ("FAIL" in seg or "Error" in seg or "未达标" in seg) else (
                ACCENT if ("PASS" in seg or "passed" in seg or "达标" in seg) else fg)
            d.text((x, y), seg, font=f, fill=col)
            x += f.getlength(seg)
        y += lh
    img.save(out_path, optimize=True)
    return out_path


def _segments(line):
    segs, cur, mode = [], "", None
    for ch in line:
        m = bool(CJK.search(ch))
        if mode is None:
            mode = m
        if m != mode:
            segs.append((cur, mode))
            cur, mode = ch, m
        else:
            cur += ch
    if cur:
        segs.append((cur, mode or False))
    return segs
