# -*- coding: utf-8 -*-
"""采集 32 个工程化项目的「实跑证据」：run.py 输出、pytest 输出、报告档位段、产出图。

证据落在 工具/ppt97/evidence/<slug>/，供 inject97.py 生成 PPT 实测页与截图。
截图内容全部来自真跑（不手抄数字）；报告里已有的判定行按原样逐字截取。
"""
import concurrent.futures
import json
import os
import re
import subprocess
import sys

ROOT = "/Users/joelgong/PycharmProjects/ai-course-teaching-resources"
PY = sys.executable or "/opt/anaconda3/bin/python"
EV = os.path.join(ROOT, "工具", "ppt97", "evidence")
COURSES = ["01-AI应用基础", "02-程序设计基础", "05-AI应用基础48课时版"]
TIER_RE = re.compile(r"^\s*(及格\(D\)|良好\(C\)|优秀\(B\)|优\(A\))\s+(.*)$")
CLAIM_RE = re.compile(r"实测达成[^:：]{0,28}[:：]\s*(.*)$")
MANUAL_RE = re.compile(r"人工答辩项\s*(\d+)\s*条")
TIMEOUT = 900


def tail(text, n):
    lines = [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-n:]


def read(path):
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def parse_report(txt):
    rows, claimed, manual_n, manual_items, measured = [], None, 0, [], []
    in_manual = False
    for raw in txt.splitlines():
        ln = raw.rstrip()
        cm = CLAIM_RE.search(ln.strip())
        if cm and claimed is None:
            parts = [p.strip() for p in cm.group(1).split("｜") if p.strip()]
            claimed = parts[0] if parts else None
            measured = parts[1:]
        m = TIER_RE.match(ln)
        if m:
            seg = [s.strip() for s in m.group(2).split("｜") if s.strip()]
            gate = seg[0] if seg else ""
            need = seg[-1] if len(seg) > 1 else ""
            rows.append({"tier": m.group(1),
                         "key": m.group(1)[-3],
                         "gate": gate,
                         "verdict": "未达标" if "未达标" in gate else "达标",
                         "need": need})
        mm = MANUAL_RE.search(ln)
        if mm:
            manual_n = int(mm.group(1))
            in_manual = True
            continue
        if in_manual:
            if ln.startswith("  ") or ln.lstrip().startswith(("D 档", "C 档", "B 档", "A 档", "『")):
                manual_items.append(ln.strip(" ·-—0123456789").strip())
            elif ln.strip().startswith("[") or "绝不自动判达标" in ln:
                continue
            else:
                in_manual = False
    return rows, claimed, manual_n, manual_items[:6], measured


def run(cmd, cwd):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=TIMEOUT, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        return p.returncode, out
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT after %ss" % TIMEOUT
    except Exception as exc:  # noqa: BLE001
        return 125, "ERROR: %s" % exc


def collect(proj, course):
    slug = os.path.basename(proj.rstrip("/"))
    out_dir = os.path.join(EV, slug)
    os.makedirs(out_dir, exist_ok=True)
    rc_run, run_out = run([PY, "run.py"], proj)
    rc_test, test_out = run([PY, "-m", "pytest", "-q", "--no-header",
                             "-p", "no:cacheprovider"], proj)
    report = read(os.path.join(proj, "output", "report.txt"))
    rows, claimed, manual_n, manual_items, measured = parse_report(report)
    imgs = []
    for dirpath, _dirs, files in os.walk(os.path.join(proj, "output")):
        for fn in sorted(files):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                imgs.append(os.path.join(dirpath, fn))
    # 报告的「验收档位判定」整段逐字截取，作为报告实测截图
    seg, cap = [], False
    for ln in report.splitlines():
        if "验收档位判定" in ln:
            cap = True
        if cap:
            seg.append(ln)
        if cap and len(seg) >= 14:
            break
    data = {"slug": slug, "course": course, "run_rc": rc_run, "test_rc": rc_test,
            "claimed": claimed, "rows": rows, "manual_n": manual_n,
            "manual_items": manual_items, "measured": measured[:10],
            "run_tail": tail(run_out, 22), "pytest_tail": tail(test_out, 14),
            "report_tier_seg": seg[:14],
            "images": [os.path.relpath(p, proj) for p in sorted(imgs)[:6]]}
    with open(os.path.join(out_dir, "evidence.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "run.txt"), "w", encoding="utf-8") as fh:
        fh.write(run_out)
    with open(os.path.join(out_dir, "pytest.txt"), "w", encoding="utf-8") as fh:
        fh.write(test_out)
    print("[%s] run=%s test=%s tiers=%s claimed=%s imgs=%d" % (
        slug, rc_run, rc_test, len(rows), claimed, len(imgs)), flush=True)
    return data


def main():
    os.makedirs(EV, exist_ok=True)
    jobs = []
    for c in COURSES:
        base = os.path.join(ROOT, c, "工程化项目")
        for slug in sorted(os.listdir(base)):
            p = os.path.join(base, slug)
            if os.path.isdir(p):
                jobs.append((p, c))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(lambda j: collect(*j), jobs))
    bad = [r["slug"] for r in res if r["run_rc"] != 0 or r["test_rc"] != 0
           or len(r["rows"]) < 4 or not r["claimed"]]
    print("TOTAL=%d  problems=%s" % (len(res), bad or "无"))
    with open(os.path.join(EV, "_index.json"), "w", encoding="utf-8") as fh:
        json.dump([{k: v for k, v in r.items() if k != "images"} for r in res],
                  fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
