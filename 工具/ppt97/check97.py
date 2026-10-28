# -*- coding: utf-8 -*-
"""注入后体检：页码连续性 + 新增页形状是否越界/压到页脚 + 幂等性。

用法: python check97.py
"""
import glob
import os
import sys

from pptx import Presentation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inject97 as I  # noqa: E402

SLIDE_W, SLIDE_H = 13.3333, 7.5
FOOTER_TOP = 7.10          # 页脚分隔线顶边（内容不得压上去）
TOL = 0.02


def inches(v):
    return (v or 0) / 914400.0


def check_shape(sh, page_no, name, bad):
    l, t = inches(sh.left), inches(sh.top)
    w, h = inches(sh.width), inches(sh.height)
    if None in (sh.left, sh.top, sh.width, sh.height):
        return
    r, b = l + w, t + h
    why = None
    if r > SLIDE_W + TOL:
        why = "右边越界 %.2f>%.2f" % (r, SLIDE_W)
    elif b > SLIDE_H + TOL:
        why = "底边越界 %.2f" % b
    elif l < -TOL or t < -TOL:
        why = "负坐标"
    elif 1.5 <= t <= 7.0 and b > FOOTER_TOP + TOL and h < 1.0:
        why = "压住页脚（底边 %.2f）" % b
    if why:
        txt = (sh.text_frame.text[:26] if sh.has_text_frame else sh.shape_type)
        bad.append("%s p%d %s ｜ %s ｜ %s" % (name[:26], page_no, why, txt,
                                            "%.2f,%.2f %.2fx%.2f" % (l, t, w, h)))


def check_page(slide, page_no, name, bad):
    for sh in slide.shapes:
        check_shape(sh, page_no, name, bad)


def main():
    bad, stats = [], {"deck": 0, "page": 0, "card": 0}
    for _k, rel in I.COURSE_DIRS:
        for deck in sorted(glob.glob(os.path.join(I.ROOT, rel, "*.pptx"))):
            prs = Presentation(deck)
            name = os.path.basename(deck)
            stats["deck"] += 1
            hits = [i for i, s in enumerate(prs.slides)
                    if (I.slide_title(s) or "").startswith(
                        (I.E1_KEY, I.E2_KEY, I.E3_KEY))]
            if len(hits) not in (2, 3):
                bad.append("%s 新增页数异常 %s" % (name, hits))
            stats["page"] += len(hits)
            for i in hits:
                check_page(prs.slides[i], i + 1, name, bad)
            for key, head in I.CARD_KEY.items():
                idx = I.find_index(prs, head)
                if idx is None:
                    continue
                for sh in prs.slides[idx].shapes:
                    if sh.has_text_frame and "本次实测" in sh.text_frame.text:
                        stats["card"] += 1
                        check_shape(sh, idx + 1, name + "/卡" + key, bad)
            nums = []
            for s in prs.slides:
                for sh in s.shapes:
                    if sh.has_text_frame:
                        t = sh.text_frame.text.strip()
                        if t.count("/") == 1 and t.replace(" ", "").replace(
                                "/", "").isdigit():
                            nums.append(t.split("/")[1].strip())
            if len(set(nums)) > 1:
                bad.append("%s 页码总数不一致 %s" % (name, sorted(set(nums))))
    print("decks=%(deck)d 新增页=%(page)d 卡片标注=%(card)d" % stats)
    print("越界/异常 %d 处" % len(bad))
    for b in bad[:40]:
        print("  !", b)


if __name__ == "__main__":
    main()
