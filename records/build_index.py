"""records/ 的索引：每份 run_*.log 对应哪条 B 命令、哪个 C 会话 / 哪个 CSV。

命令行本身没有落盘（板上 .bash_history 是旧的），B 的参数从每趟日志的第 1、3 行
横幅反推 —— 那两行打的就是生效值。C 的会话从 motor_positions 的文件名时间戳
（CSV 名里带启动时刻）和行数（50Hz）还原，再看它和 B 的跑动窗口重叠多少。

用法: py -3 records/build_index.py   → 生成 records/INDEX.md
"""
from __future__ import annotations

import glob
import os
import re
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))


def run_time(path):
    # 早期几趟是 run_HHMM.log（4 位），之后才是 run_HHMMSS.log。
    m = re.search(r"run_(\d{4}|\d{6})\.log$", os.path.basename(path))
    if not m:
        return None
    stamp = m.group(1)
    t = datetime.fromtimestamp(os.path.getmtime(path))
    return t.replace(hour=int(stamp[:2]), minute=int(stamp[2:4]),
                     second=int(stamp[4:6]) if len(stamp) == 6 else 0,
                     microsecond=0)


def csv_window(path):
    m = re.search(r"motor_positions_(\d{8})_(\d{6})_", os.path.basename(path))
    if not m:
        return None
    t0 = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    with open(path, "rb") as handle:
        rows = max(0, sum(1 for _ in handle) - 1)
    return t0, t0 + timedelta(seconds=(rows - 1) / 50.0), rows


def b_params(lines):
    p = {}
    head = next((l for l in lines if l.startswith("Camera 0:")), "")
    for key in ("vx", "max_wz", "yaw_sign", "sl_gain", "center_dead"):
        m = re.search(rf"\b{key}=([\d.+-]+)", head)
        if m:
            p[key] = m.group(1)
    for key in ("red", "line_pitch", "hold-still", "gate", "wz_mode"):
        m = re.search(rf"\b{re.escape(key)}=(\S+?)[;)]", head)
        if m:
            p[key] = m.group(1)
    banner = next((l for l in lines if "离散模式" in l), "")
    m = re.search(r"err ≥ ([\d.]+)cm", banner)
    p["fire"] = m.group(1) if m else "?"
    m = re.search(r"(?:另一侧的|回到) ([\d.]+)cm", banner)
    p["stop"] = m.group(1) if m else "?"
    m = re.search(r"最多 ([\d.]+)s", banner)
    p["turn"] = m.group(1) if m else "?"
    m = re.search(r"强制空 ([\d.]+)s", banner)
    p["gap"] = m.group(1) if m else "?"
    m = re.search(r"\{0, \+([\d.]+)\}", banner)
    p["step"] = m.group(1) if m else "?"
    return p


def main():
    runs = []
    for path in sorted(glob.glob(os.path.join(HERE, "run_*.log"))):
        t0 = run_time(path)
        if t0 is None:
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
        vis = [l for l in lines if l.startswith("[vision]") and "Hz" in l]
        dur = timedelta(seconds=0.5 * max(1, len(vis)))
        runs.append({
            "name": os.path.basename(path),
            "t0": t0,
            "dur": dur,
            "params": b_params(lines),
            "vision": len(vis),
        })
    sessions = []
    for path in sorted(glob.glob(os.path.join(HERE, "motor_positions", "*.csv"))):
        w = csv_window(path)
        if w:
            sessions.append({"name": os.path.basename(path), "t0": w[0],
                             "t1": w[1], "rows": w[2]})
    for r in runs:
        best, best_ov = None, timedelta(0)
        for s in sessions:
            ov = min(r["t0"] + r["dur"], s["t1"]) - max(r["t0"], s["t0"])
            if ov > best_ov:
                best, best_ov = s, ov
        r["session"], r["overlap"] = best, best_ov
    out = ["# records/ 索引（自动生成，py -3 records/build_index.py）", "",
           "⚠️ 命令是从日志横幅**反推**的：命令行本身没落盘。反推不到的只有 `--max-seconds`、",
           "`--camera` 这类不在横幅里的开关；`--no-red-detect` 看 `red=off`。", ""]
    out.append("| run | 起 | 时长 | fire | stop | turn | gap | step | vx | C 会话 | C 晚起 | 重叠 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        p, s = r["params"], r["session"]
        late = "-"
        if s:
            late_s = (s["t0"] - r["t0"]).total_seconds()
            late = f"{late_s:+.0f}s"
        out.append("| {} | {} | {:.0f}s | {} | {} | {} | {} | {} | {} | {} | {} | {:.0f}s |".format(
            r["name"].replace("run_", "").replace(".log", ""),
            r["t0"].strftime("%H:%M:%S"), r["dur"].total_seconds(),
            p.get("fire", "?"), p.get("stop", "?"), p.get("turn", "?"),
            p.get("gap", "?"), p.get("step", "?"), p.get("vx", "?"),
            s["t0"].strftime("%H:%M:%S") if s else "无 C",
            late, r["overlap"].total_seconds()))
    sets = {}
    for r in runs:
        p = r["params"]
        key = (p.get("fire"), p.get("stop"), p.get("turn"), p.get("gap"), p.get("step"))
        sets.setdefault(key, []).append(r["name"].replace("run_", "").replace(".log", ""))
    out += ["", "## 参数组 → 重建的 B 命令", ""]
    for (fire, stop, turn, gap, step), names in sorted(sets.items()):
        out.append(f"### fire={fire} stop={stop} turn={turn} gap={gap} step={step}"
                   f"（{len(names)} 趟：{', '.join(names)}）")
        if "?" in (fire, stop, turn, gap, step):
            out += ["", "（这几趟的横幅里没有上面打 `?` 的项：是加 `--wz-stop-cm` / 改成"
                        "「另一侧」语义之前的代码版本，命令只能从对应日期的代码里查。）", ""]
            continue
        out.append("```bash")
        out.append("python -u new_vision/jetson/run_policy_vision.py --camera 0 --headless \\")
        out.append("    --no-red-detect --bias-straight-cm 0 --card-trigger-dist-cm 30 \\")
        out.append(f"    --wz-fire-cm {fire} --wz-stop-cm {stop} --wz-turn-s {turn} \\")
        out.append(f"    --wz-gap-s {gap} --wz-step {step} \\")
        out.append("    --dump-on-loss records/loss_dump --shape-dump records/shape_dump \\")
        out.append("    --max-seconds 90 2>&1 | tee records/run_$(date +%H%M%S).log")
        out.append("```")
        out.append("")
    out.append("## C 会话（motor_positions CSV ↔ main_*.log ↔ 哪几趟用了它）")
    out.append("")
    for s in sorted(sessions, key=lambda x: x["t0"]):
        users = [r["name"].replace("run_", "").replace(".log", "")
                 for r in runs if r["session"] is s]
        if not users:
            continue
        out.append("- `{}`  {} 起，{:.0f}s / {} 行 → main_logs/main_{}.log；run: {}".format(
            s["name"], s["t0"].strftime("%H:%M:%S"), s["rows"] / 50.0, s["rows"],
            s["t0"].strftime("%H%M"), ", ".join(users)))
    named = [p for p in sorted(glob.glob(os.path.join(HERE, "run_*.log")))
             if run_time(p) is None]
    if named:
        out += ["", "## 有名字的日志（没有时间戳，命令从横幅看）", ""]
        for path in named:
            with open(path, encoding="utf-8", errors="replace") as handle:
                lines = handle.read().splitlines()
            banner = next((l for l in lines if "离散模式" in l), "")
            head = next((l for l in lines if l.startswith("Camera 0:")), "")
            note = banner or head or (lines[0] if lines else "(空)")
            out.append(f"- `{os.path.basename(path)}`（{len(lines)} 行）：{note[:150]}")
    index = os.path.join(HERE, "INDEX.md")
    with open(index, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out) + "\n")
    print(f"wrote {index}: {len(runs)} runs, {len(sessions)} C sessions")


if __name__ == "__main__":
    main()
