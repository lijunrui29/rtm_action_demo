#!/usr/bin/env python3
"""能不能把 Salahzadeh et al. (2014) 的临床 CVA 三档阈值直接套到 45° proxy 上？

=== 这个脚本干什么 ===
拿本项目已录好的 45° 机位三档数据，逐帧取生产 `CvaProxyFeatures` 的角度值，然后：

  [1] 每档的描述统计（帧数 / 中位数 / 均值 / 标准差 / IQR / min / max）
  [2] 把文献阈值当判据套上去，看各档被判成什么、有没有整体塌缩进同一档
  [3] 与**存档的**自标定阈值（68.17 / 61.42 —— 2026-09-29 生产预设已被换成
      文献切点，值存档在 data/archive/measured_basis_2026-09-29.md）做帧级
      一致率 + 混淆矩阵
  [4] 偏移是不是**系统性且跨档恒定**（各档 proxy 中位数 − 文献各档均值，看差值/比值）
  [5] 段级中位数 × 段间极差（段内重复性；顺便看"坐直基线跨次漂移"在这两批里有多大）
  [6] skip-sec 敏感性（切姿势过渡帧丢多少对结论有没有影响）

=== 这个脚本**不**干什么（别指望它给出可用阈值）===
  * 不拟合阈值、不给修正系数、不改任何生产代码（`features.py` / `decision.py` 只读）
  * 不碰 `FhpDecision` 的阈值：生产现值（文献切点）与存档的自标定值 68.17 / 61.42
    只读不改，本脚本的对照一律打**存档基线**那一列
  * 不把"文献阈值能不能用"当成预设结论 —— 数字出来什么样就报什么样，包括
    "全塌缩进同一档"这种难看的结果（用户明确要求：不许发明修正系数把结果调好看）

=== 数据（怎么读出来的 / 缺关键点怎么办）===
  A. `data/molaeifar_frames_45.csv`（2026-09-28，view=45，person=1）
     生产 `CvaProxyFeatures` 在录制时逐帧算过一遍，值在 `cva_proxy_raw`（未平滑）与
     `cva_proxy_raw_ema`（脚本自己 τ=1.0 的 EMA ≈ 生产读数）两列。本脚本**另外**从
     存的四个关键点像素坐标复算一遍（`replay_proxy`）交叉核对——存的是
     像素坐标，先除 W/H 还原成归一化、生产类再乘回同一个 W/H，两处相消，所以复算
     与 W/H 取值无关。缺关键点（置信度 < 生产 `visibility_min`，默认 0.3）的帧复算
     得 None，与录制时同为降级帧；这类帧算不了读数，只计掉帧数，不进统计。
     **复算结果（实测，见数据表）**：1627 帧可比、最大偏差 21.0°、超 1° 的有 13 帧
     —— 这 13 帧全部同时是"存的右肩其实是髋"的帧（`|shR_y − shL_y| > 100px`），
     对应 molaeifar 脚本里 shR 列被髋覆盖那个 2026-09-28 修掉的写入 bug；本批文件里
     还有 25 帧带这个特征、**全部落在未标注帧上**。统计一律读录制列（生产算的），
     所以这些坏关键点不改变任何一个数字——只是想说明"复算不是万能核对"。
  B. `data/f45.csv`（2026-09-23，view=45，person=1）
     就是 68.17 / 61.42 的**样本内**来源（口径：`cva_proxy` = 平滑值 = 生产读数）。
     该文件**没有关键点列**（`cva_proxy_raw` 与 `valid` 整列为空），所以只能读录制值、
     无法复算、也拿不到逐帧置信度。它是"标定用过的同一批"，报它的数字时别当独立验证。

两批数据都是 45° 机位、1 个人、各 1 次录制（Molaeifar 脚本按标签变化切段，每档 3 段）；
`f45` 相对生产阈值是**样本内**，`molaeifar_45` 是**另一次录制**（对生产阈值算样本外，
对文献阈值同样只是同一个人）。

=== 文献阈值的录入口径（重要：两种读法都报，不替用户选）===
Salahzadeh 给的是三档的**区间**（正常 53.2~56.8 / 轻度 46.9~49.1 / 中重度 40.7~43.2），
不是一个切点，所以"≥上界→NORMAL，介于→SLIGHT，<下界→OBVIOUS"这条规则要两个切点，
而区间本身给了两种自洽取法：
    L1 区间边缘：hi=53.2（正常档下缘）、lo=43.2（中重度档上缘）
    L2 均值中点：hi=51.5=(55+48)/2、lo=45.0=(48+42)/2（三档均值 55/48/42 的中点）
两种都算、都放进表里，读数一样就都能看出来；本脚本不声称哪种"才对"。

=== 用法 ===
    python tools/test_salahzadeh_cva_thresholds.py                 # 用默认两批数据
    python tools/test_salahzadeh_cva_thresholds.py --csv a.csv b.csv
    python tools/test_salahzadeh_cva_thresholds.py --skip-sec 0 --csv ...
    python tools/test_salahzadeh_cva_thresholds.py --selftest      # 合成数据自测

输出（默认落 `data/`，写盘前自建父目录）：
    --out      data/salahzadeh_cva_frames.csv   逐帧：两批数据的读数 + 三种判据的档位
    --table-md data/salahzadeh_cva_table.md     上面 [1]~[6] 的全部表格（markdown）
    --report   data/salahzadeh_cva_report.txt   同样的表格 + 结论 + 假设（纯文本）

⚠ 逐帧 CSV 含实拍读数 = 个人身体数据，`.gitignore` 挡着，别提交。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# 本文件在 tools/ 下，上一层才是仓库根（features / decision 在那儿）。
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features import CvaProxyFeatures          # 只读、只调用，不改
from decision import CVA_PROXY_VIEW_PRESETS    # 生产阈值，直接读，绝不重复写死


# ---------- 固定口径（改了要在这里说明原因） ----------

LABEL_ORDER = ("Normal", "Slight_FHP", "Obvious_FHP")

# 关键点置信度下限：与生产 `CvaProxyFeatures` 的 `visibility_min` 默认值同源。
# 直接问类要，别写死 —— 生产改了这里跟着改，不会悄悄错开。
VIS_MIN = CvaProxyFeatures().visibility_min

# 复算用的画面尺寸。**取任意自洽值都行**：CSV 存的是像素坐标，本脚本先除 W/H 还原成
# 归一化，生产类再乘回同一个 W/H，两处相消 —— 角度与 W/H 无关（这就是"像素空间"的定义）。
# 1280x720 只是照录制取景写，换个值不影响任何数字。
REPLAY_W, REPLAY_H = 1280.0, 720.0

# molaeifar 脚本逐帧 CSV 里的四个关键点像素坐标 + 置信度列（COCO 编号 3/4 耳、5/6 肩）
KEYPOINT_COLUMNS = {
    3: ("earL_x", "earL_y", "earL_conf"),
    4: ("earR_x", "earR_y", "earR_conf"),
    5: ("shL_x", "shL_y", "shL_conf"),
    6: ("shR_x", "shR_y", "shR_conf"),
}

# 复算核对用的两条判读线（**只用于诊断输出，不参与任何统计**）：
#   REPLAY_TOL_DEG      复算与录制列算"对得上"的容差
#   HIP_CONTAM_DY_PX    存的"右肩"比左肩低这么多像素 → 那其实是髋
# 后者对应 molaeifar 脚本自己记过的写入 bug（`tools/test_molaeifar_sternum_angle.py`
# 里 shR 列被髋覆盖那条注释，2026-09-28 修）。本批 45° 录制里仍有 25 帧带这个特征，
# **全部落在未标注帧上**，所以四档统计不受影响 —— 但如实报出来，别假装数据是干净的。
REPLAY_TOL_DEG = 1.0
HIP_CONTAM_DY_PX = 100.0

# 文献值：按用户给的区间录入，**没有回原文复核**（写在假设里）。
LIT_SOURCE = "Salahzadeh et al. (2014) 的三档 CVA 区间（数值按需求原文录入，未回原文复核）"
LIT_RANGES = {
    "Normal": (53.2, 56.8),
    "Slight_FHP": (46.9, 49.1),
    "Obvious_FHP": (40.7, 43.2),
}
LIT_MEANS = {"Normal": 55.0, "Slight_FHP": 48.0, "Obvious_FHP": 42.0}

# 文献阈值的两种自洽切法（切点 = 上界 hi / 下界 lo；方向：proxy 越大越直）
LIT_CUT_SETS: Tuple[Tuple[str, float, float], ...] = (
    ("L1_edges_53.2_43.2", 53.2, 43.2),
    ("L2_mean_mid_51.5_45.0", 51.5, 45.0),
)

# 对照基线 = **存档的自标定阈值**（45°），不是生产现值：
# 2026-09-29 生产预设 CVA_PROXY_VIEW_PRESETS["45"] 已被换成文献切点（53.2 / 43.2，
# 见 decision.py 里那个常量的注释），它与本脚本的 "L1" 切法重合 —— 再拿生产现值
# 当对照就没有对照可言了。这两个数是被替换下去的值，原始录制与完整记录见
# data/archive/measured_basis_2026-09-29.md（"实测基线存档"，不入仓库）。
EMP_HI, EMP_LO = 68.17, 61.42         # 存档的自标定值（对照基线）
PROD_HI, PROD_LO = CVA_PROXY_VIEW_PRESETS["45"]   # 生产现值（2026-09-29 起 = 文献切点）
EMP_CUT_LABEL = "存档自标定_45_%.2f_%.2f" % (EMP_HI, EMP_LO)

# 判读措辞用的容差（**只用于本脚本写结论时的措辞，不是标定参数**）：
# 各档偏移的极差 ≤ 此值 → 说"跨档大致恒定"
OFFSET_TOL_DEG = 2.0
RATIO_TOL = 0.05
# 档位是否"分开"：占用档数 ≥ 2 就算没整体塌缩
MIN_OCCUPIED_TIERS = 2

STAT_KEYS = ("n", "median", "mean", "std", "q1", "q3", "iqr", "min", "max")


# ---------- 小工具 ----------

def _f(v) -> Optional[float]:
    """CSV 里的浮点。空串 / None / '-' / 非法 → None。"""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


_LABEL_ALIASES = {
    "1": "Normal", "2": "Slight_FHP", "3": "Obvious_FHP",
    "normal": "Normal", "slight_fhp": "Slight_FHP", "obvious_fhp": "Obvious_FHP",
    "slight": "Slight_FHP", "obvious": "Obvious_FHP",
}


def label_of(raw) -> Optional[str]:
    """标签 → 统一档名；未标注 / 认不出 → None。数字 1/2/3 = Normal/Slight/Obvious。"""
    return _LABEL_ALIASES.get(str(raw or "").strip().lower())


def pct(sorted_vals: Sequence[float], q: float) -> float:
    """线性插值分位数（q ∈ [0,1]）。sorted_vals 必须已升序且非空。"""
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def describe(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    """描述统计。空输入 → 全 None（除以 n-1 的标准差在 n<2 时给 0.0，不报 NaN）。"""
    n = len(vals)
    if n == 0:
        return {k: None for k in STAT_KEYS}
    s = sorted(float(v) for v in vals)
    mean = sum(s) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in s) / (n - 1)   # 样本标准差
        std = math.sqrt(var)
    else:
        std = 0.0
    q1, q3 = pct(s, 0.25), pct(s, 0.75)
    return {
        "n": n, "median": pct(s, 0.5), "mean": mean, "std": std,
        "q1": q1, "q3": q3, "iqr": q3 - q1, "min": s[0], "max": s[-1],
    }


def tier_of(deg: float, hi: float, lo: float) -> int:
    """两切点分三档：deg >= hi → 0(NORMAL)，lo <= deg < hi → 1，deg < lo → 2。

    方向与生产一致：proxy 越大越直（耳在肩正上方 = 90°），越小越前伸。
    边界归更直的一档（与生产 `FhpDecision` 的 ">= 阈值即触发" 写法一致）。
    """
    if deg >= hi:
        return 0
    if deg >= lo:
        return 1
    return 2


def _fmt(v, nd: int = 2) -> str:
    """表格里统一的小数格式。None → 'N/A'。"""
    if v is None:
        return "N/A"
    if isinstance(v, int):
        return str(v)
    return f"{v:.{nd}f}"


# ---------- 数据加载 ----------

class Frame:
    """一帧的读数。两批数据归一化成同一个结构，后面的分析只认它。

    deg      本脚本用来做判定的读数 = 生产口径的**平滑值**（标定阈值就是标在这上面的）
             —— molaeifar 取 `cva_proxy_raw_ema`，f45 取 `cva_proxy`。
    raw      未平滑读数（molaeifar 有；f45 整列为空 → None）。
    replay   从关键点像素坐标复算出的未平滑读数（只有 molaeifar 能算；f45 → None）。
    conf     本帧参与关键点的平均置信度（只有 molaeifar 有）。
    """
    __slots__ = ("src", "seg", "label", "t", "deg", "raw", "replay", "conf")

    def __init__(self, src: str, seg: int, label: str, t: Optional[float],
                 deg: Optional[float], raw: Optional[float],
                 replay: Optional[float], conf: Optional[float]) -> None:
        self.src, self.seg, self.label, self.t = src, seg, label, t
        self.deg, self.raw, self.replay, self.conf = deg, raw, replay, conf


def _read_rows(path: Path) -> List[dict]:
    """读逐帧 CSV（utf-8）。空文件 → 空列表。"""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _rec_id(row: dict) -> Optional[str]:
    """录制段编号；本批数据里恒为 '0'（用户没按 r），所以段的切分实际靠标签变化。"""
    v = row.get("rec_index")
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _pose_from_row(row: dict) -> Optional[dict]:
    """把 CSV 一行还原成 detect_pose() 形状的 pose dict（只填四个关键点）。

    像素 → 归一化用 REPLAY_W/H；置信度直接用录下来的 conf（生产类自己按
    `visibility_min` 过滤，与本脚本的 VIS_MIN 同源）。一个点都没有 → None。
    """
    lm: List[List[float]] = [[0.0, 0.0, 0.0, 0.0] for _ in range(17)]
    got = 0
    for pid, (kx, ky, kc) in KEYPOINT_COLUMNS.items():
        x, y, c = _f(row.get(kx)), _f(row.get(ky)), _f(row.get(kc))
        if x is None or y is None or c is None:
            continue
        lm[pid] = [x / REPLAY_W, y / REPLAY_H, 0.0, c]
        got += 1
    if not got:
        return None
    return {"landmarks": lm, "image_size": (REPLAY_W, REPLAY_H)}


def replay_proxy(frames_rows: List[dict]) -> List[Optional[float]]:
    """逐帧复算生产 `CvaProxyFeatures` 的未平滑读数（交叉核对录下来的 cva_proxy_raw）。

    用**一个**特征实例顺序喂完整个文件 —— 与录制时一样，EMA/预热按 `rec_sec` 推进；
    这里只取 `cva_proxy_raw_deg`（与时间戳无关，所以不受复算时 EMA 起点的影响）。
    """
    feat = CvaProxyFeatures()
    out: List[Optional[float]] = []
    for row in frames_rows:
        pose = _pose_from_row(row)
        res = feat.update(pose, _f(row.get("rec_sec")))
        out.append(res.get("cva_proxy_raw_deg"))
    return out


def load_molaeifar(path: Path) -> Tuple[List[Frame], dict]:
    """读 molaeifar 脚本的逐帧 CSV（带关键点，可复算）。返回 (帧列表, 元信息)。"""
    rows = _read_rows(path)
    replayed = replay_proxy(rows)

    # 交叉核对：录制时算的 raw 与本地复算的 raw 差多少（只统计两边都有值的帧）
    pairs = [(_f(r.get("cva_proxy_raw")), rp, r) for r, rp in zip(rows, replayed)]
    both = [(a, b) for a, b, _r in pairs if a is not None and b is not None]
    diffs = [abs(a - b) for a, b in both]
    max_diff = max(diffs) if diffs else None
    over = sum(1 for a, b in both if abs(a - b) > REPLAY_TOL_DEG)
    # 复算不出来的帧里，有多少录制时其实有值（应当为 0；不为 0 说明口径对不上）
    only_recorded = sum(1 for a, b, _r in pairs if a is not None and b is None)
    # 超差帧里有多少同时是"右肩被髋顶掉"的帧 —— 让复算偏差能被解释，而不是留个哑谜
    over_contam = sum(1 for a, b, r in pairs
                      if a is not None and b is not None
                      and abs(a - b) > REPLAY_TOL_DEG and _hip_contam(r))
    # 存的右肩是不是被髋顶掉了（见 HIP_CONTAM_DY_PX 的注释）
    contaminated = [r for r in rows if _hip_contam(r)]
    contam_labeled = sum(1 for r in contaminated if label_of(r.get("posture_label")))

    frames: List[Frame] = []
    seg_id, cur_key = -1, None
    for row, rp in zip(rows, replayed):
        lab = label_of(row.get("posture_label"))
        if not lab:
            continue
        key = (lab, _rec_id(row))
        if key != cur_key:
            seg_id += 1
            cur_key = key
        conf = _mean_conf(row)
        frames.append(Frame(
            src=path.name, seg=seg_id, label=lab,
            t=_f(row.get("rec_sec")) if _f(row.get("rec_sec")) is not None
            else _f(row.get("label_sec")),
            deg=_f(row.get("cva_proxy_raw_ema")),
            raw=_f(row.get("cva_proxy_raw")),
            replay=rp, conf=conf))

    meta = {
        "file": str(path), "kind": "molaeifar",
        "rows": len(rows), "labeled": len(frames),
        "view": _first(rows, "view"), "person": _first(rows, "person"),
        "session": _first(rows, "session"),
        "has_keypoints": True,
        "replay_n": len(both), "replay_max_diff": max_diff,
        "replay_over": over, "replay_over_contam": over_contam,
        "replay_only_recorded": only_recorded,
        "contam_all": len(contaminated), "contam_labeled": contam_labeled,
        "has_raw": any(r.get("cva_proxy_raw") for r in rows),
        "conf_available": True,
    }
    return frames, meta


def _hip_contam(row: dict) -> bool:
    """存的关键点里"右肩"比左肩低了 HIP_CONTAM_DY_PX 以上 → 那其实是髋。

    对应 molaeifar 脚本里 shR 列被髋覆盖那个已修的写入 bug。**只作诊断**，
    不改任何读数：本脚本的统计一律读录制列（生产算的），复算只用来核对。
    """
    a, b = _f(row.get("shL_y")), _f(row.get("shR_y"))
    return a is not None and b is not None and abs(b - a) > HIP_CONTAM_DY_PX


def load_fhp_csv(path: Path) -> Tuple[List[Frame], dict]:
    """读生产 demo / 标定脚本写的逐帧 CSV（`cva_proxy` = 平滑值 = 生产读数）。"""
    rows = _read_rows(path)

    frames: List[Frame] = []
    seg_id, cur_key = -1, None
    for row in rows:
        lab = label_of(row.get("posture_label"))
        if not lab:
            continue
        key = (lab, _rec_id(row))
        if key != cur_key:
            seg_id += 1
            cur_key = key
        frames.append(Frame(
            src=path.name, seg=seg_id, label=lab,
            t=_f(row.get("timestamp")),
            deg=_f(row.get("cva_proxy")),
            raw=_f(row.get("cva_proxy_raw")),      # 该文件整列为空 → 全 None
            replay=None,                           # 没有关键点列 → 无法复算
            conf=_f(row.get("keypoint_confidence"))))

    meta = {
        "file": str(path), "kind": "fhp_csv",
        "rows": len(rows), "labeled": len(frames),
        "view": _first(rows, "view"), "person": _first(rows, "person"),
        "session": _first(rows, "session"),
        "has_keypoints": False, "replay_n": 0, "replay_max_diff": None,
        "replay_only_recorded": 0,
        "has_raw": any(r.get("cva_proxy_raw") for r in rows),
        "conf_available": any(r.get("keypoint_confidence") for r in rows),
    }
    return frames, meta


def _mean_conf(row: dict) -> Optional[float]:
    """四个关键点置信度的均值（缺的点不计）。都没有 → None。"""
    cs = [_f(row.get(c)) for _, _, c in KEYPOINT_COLUMNS.values()]
    got = [c for c in cs if c is not None]
    return sum(got) / len(got) if got else None


def _first(rows: List[dict], key: str) -> str:
    """取第一行里某列的值（view/person/session 这类整列为常量的标记）。"""
    for r in rows:
        v = r.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return "-"


def load_any(path: Path) -> Tuple[List[Frame], dict]:
    """按文件里有没有关键点列自动选加载器。"""
    rows = _read_rows(path)
    if rows and rows[0].get("earL_x") is not None:
        return load_molaeifar(path)
    return load_fhp_csv(path)


# ---------- 分析 ----------

def drop_segment_heads(frames: List[Frame], skip_sec: float) -> Tuple[List[Frame], int]:
    """丢掉每段开头 skip_sec 秒（切姿势的过渡帧）。t 缺失的帧不丢也不作起点。"""
    if skip_sec <= 0:
        return list(frames), 0
    kept: List[Frame] = []
    dropped = 0
    prev_seg: Optional[int] = None
    run_start: Optional[float] = None
    for fr in frames:
        if fr.seg != prev_seg:
            prev_seg, run_start = fr.seg, fr.t
        if (fr.t is not None and run_start is not None
                and (fr.t - run_start) < skip_sec):
            dropped += 1
            continue
        kept.append(fr)
    return kept, dropped


def segments(frames: List[Frame]) -> List[Tuple[str, int, List[Frame]]]:
    """按 (标签, 段号) 切段，跳过没读数（deg=None）的帧。"""
    order: List[Tuple[str, int]] = []
    buckets: Dict[Tuple[str, int], List[Frame]] = {}
    for fr in frames:
        if fr.deg is None:
            continue
        key = (fr.label, fr.seg)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(fr)
    return [(lab, seg, buckets[(lab, seg)]) for lab, seg in order]


class DatasetResult:
    """一批数据的全部分析结果（表格 + 结论都从这里出）。"""

    def __init__(self, meta: dict, frames: List[Frame], dropped: int,
                 skip_sec: float) -> None:
        self.meta, self.frames, self.dropped, self.skip_sec = meta, frames, dropped, skip_sec
        # frames 已经是「丢掉过渡帧之后」的那批；无读数帧 = 其中 deg 为 None 的
        # （关键点降级 / 没算出来的帧）。它和 dropped 是两回事，别混着报。
        self.n_no_reading = sum(1 for f in frames if f.deg is None)
        self.n_reading = len(frames) - self.n_no_reading
        self.stats: Dict[str, Dict[str, Optional[float]]] = {}
        self.raw_stats: Dict[str, Dict[str, Optional[float]]] = {}
        self.by_tier: Dict[str, List[Frame]] = {}
        self.lit_counts: Dict[str, List[int]] = {}
        self.lit_counts_raw: Dict[str, List[int]] = {}
        self.emp_counts: List[int] = [0, 0, 0]
        self.confusion: Dict[str, List[List[int]]] = {}
        self.agreement: Dict[str, Optional[float]] = {}
        self._analyze()

    def _analyze(self) -> None:
        for lab in LABEL_ORDER:
            rows = [f for f in self.frames if f.label == lab and f.deg is not None]
            self.by_tier[lab] = rows
            self.stats[lab] = describe([f.deg for f in rows])
            self.raw_stats[lab] = describe([f.raw for f in rows if f.raw is not None])

        for name, hi, lo in LIT_CUT_SETS:
            counts = [0, 0, 0]
            raw_counts = [0, 0, 0]
            for lab in LABEL_ORDER:
                for fr in self.by_tier[lab]:
                    counts[tier_of(fr.deg, hi, lo)] += 1
                    if fr.raw is not None:
                        raw_counts[tier_of(fr.raw, hi, lo)] += 1
            self.lit_counts[name] = counts
            self.lit_counts_raw[name] = raw_counts

        for lab in LABEL_ORDER:
            for fr in self.by_tier[lab]:
                self.emp_counts[tier_of(fr.deg, EMP_HI, EMP_LO)] += 1

        # 混淆矩阵：行 = 生产口径判的档，列 = 文献口径判的档（同一批帧）
        for name, hi, lo in LIT_CUT_SETS:
            mat = [[0, 0, 0] for _ in range(3)]
            for lab in LABEL_ORDER:
                for fr in self.by_tier[lab]:
                    mat[tier_of(fr.deg, EMP_HI, EMP_LO)][tier_of(fr.deg, hi, lo)] += 1
            self.confusion[name] = mat
            tot = sum(sum(r) for r in mat)
            self.agreement[name] = (sum(mat[i][i] for i in range(3)) / tot
                                    if tot else None)

    # -- 表格片段 --

    def tier_names_seen(self) -> int:
        """真值档里有几档拿到了读数（样本够不够三档分析）。"""
        return sum(1 for lab in LABEL_ORDER if self.by_tier[lab])

    def has_raw(self) -> bool:
        """有没有未平滑读数（f45 那批整列为空 → False）。"""
        return any(self.raw_stats[lab]["n"] for lab in LABEL_ORDER)


def offsets(ds: DatasetResult) -> List[dict]:
    """各档 proxy 中位数 − 文献各档均值，以及比值（看偏移是不是跨档恒定）。"""
    out = []
    for lab in LABEL_ORDER:
        med = ds.stats[lab]["median"]
        lit = LIT_MEANS[lab]
        out.append({
            "label": lab, "median": med, "lit_mean": lit,
            "diff": (med - lit) if med is not None else None,
            "ratio": (med / lit) if med is not None and lit else None,
        })
    return out


def offset_spread(off: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    """(差值的极差, 比值的极差)。任一档缺读数 → (None, None)。"""
    ds = [o["diff"] for o in off if o["diff"] is not None]
    rs = [o["ratio"] for o in off if o["ratio"] is not None]
    if len(ds) < len(LABEL_ORDER) or len(rs) < len(LABEL_ORDER):
        return None, None
    return max(ds) - min(ds), max(rs) - min(rs)


def verdict_phrase(d: Optional[float], r: Optional[float]) -> str:
    """"差值/比值跨档恒定"的判读措辞。**只用于措辞**，不是标定结论。

    贴线（≥ 判读线的 80%）时额外标"（临界）" —— 免得把 0.049 vs 0.05 这种
    擦边结果说成干脆的"恒定"。
    """
    def near(v: float, tol: float) -> bool:
        return tol * 0.8 <= v <= tol

    if d is None or r is None:
        return "档位不全，算不出恒定与否"
    if d <= OFFSET_TOL_DEG:
        return "差值跨档大致恒定（像纯平移）" + ("（临界）" if near(d, OFFSET_TOL_DEG) else "")
    if r <= RATIO_TOL:
        return "比值跨档大致恒定（像纯比例）" + ("（临界）" if near(r, RATIO_TOL) else "")
    return "差值与比值都不跨档恒定"


def seg_medians(ds: DatasetResult) -> List[dict]:
    """每段的中位数 + 各档的段间极差（段内重复性；也是"基线跨次漂移"的规模）。"""
    out = []
    for lab in LABEL_ORDER:
        meds = [describe([f.deg for f in rows])["median"]
                for l, _s, rows in segments(ds.frames) if l == lab]
        meds = [m for m in meds if m is not None]
        out.append({
            "label": lab, "n_seg": len(meds),
            "meds": meds,
            "spread": (max(meds) - min(meds)) if len(meds) >= 2 else None,
        })
    return out


def skip_sensitivity(frames: List[Frame], skips: Sequence[float]) -> List[dict]:
    """不同 skip-sec 下各档的中位数（看过渡帧丢不丢影不影响结论）。"""
    out = []
    for sk in skips:
        kept, _ = drop_segment_heads(frames, sk)
        row = {"skip": sk}
        for lab in LABEL_ORDER:
            row[lab] = describe([f.deg for f in kept
                                 if f.label == lab and f.deg is not None])["median"]
        out.append(row)
    return out


# ---------- 表格渲染 ----------

class Table:
    """一张表：标题 + 表头 + 行 + 可选脚注。文本和 markdown 两种渲染。"""

    def __init__(self, title: str, header: Sequence[str],
                 rows: Sequence[Sequence[str]], note: str = "") -> None:
        self.title, self.header, self.rows, self.note = title, list(header), list(rows), note

    def text(self) -> str:
        cols = len(self.header)
        widths = [len(self.header[i]) for i in range(cols)]
        for r in self.rows:
            for i in range(cols):
                widths[i] = max(widths[i], len(str(r[i])))
        line = "  ".join("-" * w for w in widths)
        out = [self.title, line,
               "  ".join(self.header[i].ljust(widths[i]) for i in range(cols)), line]
        for r in self.rows:
            out.append("  ".join(str(r[i]).ljust(widths[i]) for i in range(cols)))
        if self.note:
            out.append(self.note)
        return "\n".join(out)

    def markdown(self) -> str:
        out = [f"### {self.title}", "",
               "| " + " | ".join(self.header) + " |",
               "|" + "|".join("---" for _ in self.header) + "|"]
        for r in self.rows:
            out.append("| " + " | ".join(str(c) for c in r) + " |")
        if self.note:
            out += ["", self.note]
        return "\n".join(out)


def _replay_note(meta: dict) -> str:
    """复算核对结论的一句话（由实际数字生成，别写死）。"""
    name = Path(meta["file"]).name
    if not meta["has_keypoints"]:
        return f"{name} 无关键点列、无法复算"
    txt = (f'{name} {meta["replay_n"]} 帧可比、最大偏差 '
           f'{_fmt(meta["replay_max_diff"], 2)}°、超线 {meta["replay_over"]} 帧')
    if meta["replay_over"]:
        txt += (f'（其中 {meta["replay_over_contam"]} 帧同时是“右肩被髋顶掉”的帧，'
                "即 molaeifar 脚本记过的 shR 列串髋 bug）")
    txt += (f'；存量带该特征的帧 {meta["contam_all"]} 个、其中带标签 '
            f'{meta["contam_labeled"]} 个')
    if meta["replay_only_recorded"]:
        txt += f'；另有 {meta["replay_only_recorded"]} 帧录制有值但复算不出（应当为 0）'
    return txt


def build_tables(ds: DatasetResult, skips: Sequence[float]) -> Tuple[List[Table], dict]:
    """某批数据的 [1]~[6] 全部表格 + 结论要用的关键数字。"""
    m = ds.meta
    tags = [t[0] for t in LIT_CUT_SETS]
    tier_cn = {"Normal": "Normal", "Slight_FHP": "Slight", "Obvious_FHP": "Obvious"}
    tables: List[Table] = []

    # [0] 这批数据是什么
    extra_rows = []
    if m["has_keypoints"]:
        extra_rows = [
            ("复算核对（可比帧 / 最大偏差 / 超 %.1f° 的帧）" % REPLAY_TOL_DEG,
             f'{m["replay_n"]} 帧 / '
             + (_fmt(m["replay_max_diff"], 2) if m["replay_max_diff"] is not None else "N/A")
             + f' / {m["replay_over"]} 帧'
             + (f'（另有 {m["replay_only_recorded"]} 帧录制有值但复算不出，应当为 0）'
                if m["replay_only_recorded"] else "")),
            ("其中同时是“右肩被髋顶掉”的帧", str(m["replay_over_contam"])),
            ("存的右肩疑似被髋顶掉（|shR_y−shL_y| > %.0fpx）" % HIP_CONTAM_DY_PX,
             f'全文件 {m["contam_all"]} 帧、其中带标签 {m["contam_labeled"]} 帧'),
        ]
    tables.append(Table(
        "数据", ("项", "值"), [
            ("文件", m["file"]),
            ("机位 / 人 / 次", f'{m["view"]} / {m["person"]} / {m["session"]}'),
            ("总行数", m["rows"]),
            ("带标签帧（丢过渡帧前）", m["labeled"]),
            ("skip-sec 丢掉的过渡帧", ds.dropped),
            ("进统计帧（有读数）", ds.n_reading),
            ("无读数帧（丢掉过渡帧后仍无读数）", ds.n_no_reading),
        ] + extra_rows + [
            ("有未平滑列", "是" if ds.has_raw() else "否（整列为空）"),
            ("有逐帧置信度", "是" if m["conf_available"] else "否"),
        ],
        note="读数 = 平滑值（生产口径，阈值就标在它上面）。"
             "无读数帧（关键点降级）不进任何统计，只计入上面这一行。"
             + ("复算只作核对：全部统计读的是录制列（生产在录制时算的），"
                "所以下面那两行的坏关键点不影响任何数字，别把它们读成“数据不可用”。"
                if m["has_keypoints"] else "")))

    # [1] 每档描述统计
    rows = []
    for lab in LABEL_ORDER:
        s = ds.stats[lab]
        rows.append([tier_cn[lab]] + [_fmt(s[k]) for k in STAT_KEYS])
    tables.append(Table(
        "1 每档 proxy 描述统计（度；读数 = 平滑值）", ("档",) + STAT_KEYS, rows,
        note="文献各档均值依次 55 / 48 / 42（度）；存档自标定 45° 阈值 = "
             f"{_fmt(EMP_HI)} / {_fmt(EMP_LO)}。"))

    # [2] 文献阈值套用后的档位分布
    rows = []
    for name, hi, lo in LIT_CUT_SETS:
        c = ds.lit_counts[name]
        tot = sum(c) or 1
        rows.append([name, f"{_fmt(hi)} / {_fmt(lo)}"] +
                    [f"{c[i]} ({100.0 * c[i] / tot:.1f}%)" for i in range(3)])
    c = ds.emp_counts
    tot = sum(c) or 1
    rows.append([EMP_CUT_LABEL, f"{_fmt(EMP_HI)} / {_fmt(EMP_LO)}"] +
                [f"{c[i]} ({100.0 * c[i] / tot:.1f}%)" for i in range(3)])
    tables.append(Table(
        "2 判据套用后的档位分布（全部带读数帧）",
        ("判据（hi/lo）", "切点", "→ NORMAL", "→ SLIGHT", "→ OBVIOUS"), rows,
        note="文献阈值那两行的只看占比：某一行只有一个非零格 = 整体塌缩进同一档。"
             "存档自标定基线那行是对照，看本批数据在三档上是不是真能分开。"))

    # [3] 混淆矩阵 + 一致率
    for name in tags:
        mat = ds.confusion[name]
        rows = []
        for i, tname in enumerate(("基线: NORMAL", "基线: SLIGHT", "基线: OBVIOUS")):
            rows.append([tname] + [str(mat[i][j]) for j in range(3)] +
                        [str(sum(mat[i]))])
        rows.append(["合计"] + [str(sum(mat[i][j] for i in range(3))) for j in range(3)] +
                    [str(sum(sum(r) for r in mat))])
        agrees = [sum(mat[i][i] for i in range(3)), sum(sum(r) for r in mat)]
        agr_pct = None if ds.agreement[name] is None else 100.0 * ds.agreement[name]
        tables.append(Table(
            f"3 混淆矩阵：存档自标定基线 vs {name}", ("帧数", "文献 NORMAL", "文献 SLIGHT",
                                            "文献 OBVIOUS", "行合计"), rows,
            note=f"对角 = 两套判据同档；一致率 = {agrees[0]}/{agrees[1]} = "
                 f"{_fmt(agr_pct, 1)}%。读法：**任一套判据塌缩成一档时，"
                 "一致率就等于另一套在那个档上的占比** —— 它本身不携带额外信息，"
                 "别把它当“两套判据吻合度”来读（要看得看整张矩阵的形状）。"))

    # [4] 偏移是不是跨档恒定
    off = offsets(ds)
    dspread, rspread = offset_spread(off)
    rows = [[tier_cn[o["label"]], _fmt(o["median"]), _fmt(o["lit_mean"], 1),
             _fmt(o["diff"]), _fmt(o["ratio"], 3)] for o in off]
    rows.append(["跨档极差", "-", "-", _fmt(dspread), _fmt(rspread, 3)])
    tables.append(Table(
        "4 偏移分析：各档中位数 − 文献各档均值", ("档", "proxy 中位数", "文献均值",
                                        "差值", "比值"), rows,
        note=f"差值跨档极差 ≤ {OFFSET_TOL_DEG}° → 说“纯平移”；比值极差 ≤ {RATIO_TOL} "
             "→ 说“纯比例”；两者都超 = 线性标定也解释不了（这两条只是本脚本的措辞判读，"
             "不是标定结论）。"))

    # [5] 段级重复性
    sm = seg_medians(ds)
    rows = []
    for s in sm:
        rows.append([tier_cn[s["label"]], str(s["n_seg"]),
                     " / ".join(_fmt(v) for v in s["meds"]) if s["meds"] else "-",
                     _fmt(s["spread"])])
    tables.append(Table(
        "5 段级中位数与段间极差（段内重复性）",
        ("档", "段数", "各段中位数", "段间极差"), rows,
        note="段间极差大 = 同一档在不同段之间本身就在漂（“坐直”这类档尤其明显）；"
             "这种情况下跨档的比较要先看它，别急着解释阈值。"))

    # [6] skip-sec 敏感性
    rows = []
    for r in skip_sensitivity(ds.frames, skips):
        rows.append([_fmt(r["skip"], 1)] +
                    [_fmt(r[lab]) for lab in LABEL_ORDER])
    tables.append(Table(
        "6 skip-sec 敏感性（各档中位数，度）",
        ("skip-sec",) + tuple(tier_cn[l] for l in LABEL_ORDER), rows,
        note="差别很小 = 结论不是靠切姿势的过渡帧撑起来的。"))

    # [7] 未平滑读数（只在有该列时给）：阈值是标在平滑值上的，这里看结论是否依赖平滑
    if ds.has_raw():
        rows = []
        for lab in LABEL_ORDER:
            s = ds.raw_stats[lab]
            rows.append([tier_cn[lab], _fmt(s["n"]), _fmt(s["median"]),
                         _fmt(s["mean"]), _fmt(s["std"]), _fmt(s["iqr"])])
        occ_s = "/".join(str(sum(1 for v in ds.lit_counts[n] if v > 0))
                         for n, _h, _l in LIT_CUT_SETS)
        occ_r = "/".join(str(sum(1 for v in ds.lit_counts_raw[n] if v > 0))
                         for n, _h, _l in LIT_CUT_SETS)
        tables.append(Table(
            "7 未平滑读数的敏感性（度）",
            ("档", "n", "median", "mean", "std", "iqr"), rows,
            note=f"文献切法的占用档数：平滑口径 {occ_s}，未平滑口径 {occ_r}"
                 "（顺序同表 2 的切法）。平滑只压同一姿势内的抖动、不改档与档之间的"
                 "距离 —— 两者一致说明结论不是平滑撑出来的；不一致就要以平滑口径为准"
                 "（生产判据读的是平滑值）。"))
    else:
        tables.append(Table(
            "7 未平滑读数的敏感性（度）", ("项", "值"),
            [("能不能算", "不能：本批数据的未平滑列整列为空（旧版录制没记这一列）")],
            note="平滑敏感性在本批数据上做不了，结论只能按平滑口径读。"))

    key = {"tags": tags, "off": off, "dspread": dspread, "rspread": rspread,
           "seg_spread": {s["label"]: s["spread"] for s in sm},
           "n_seg": {s["label"]: s["n_seg"] for s in sm}}
    return tables, key


# ---------- 结论 ----------

def build_conclusion(results: List[Tuple[DatasetResult, dict]]) -> List[str]:
    """≤10 行结论。每一句都由前面的数字推出来，不预设“文献阈值不能用”。

    行数上限是硬要求（需求写死的），所以这里的信息密度刻意压高：每批数据的
    差值/比值极差并进同一行，段数并进证据强度那行，不在结论里另开小表。
    """
    lines: List[str] = []
    ok = [(ds, k) for ds, k in results if ds.tier_names_seen() >= 1]
    if not ok:
        return ["没有可用读数：两批数据的带标签帧全部降级（关键点缺失），无法比较。"]

    # 每批数据在两种文献切法下各占用几个档位（1 = 整体塌缩进同一档）+ 非 NORMAL 占比
    collapse = {name: [sum(1 for v in ds.lit_counts[name] if v > 0) for ds, _k in ok]
                for name, _h, _l in LIT_CUT_SETS}
    emp_occ = [sum(1 for v in ds.emp_counts if v > 0) for ds, _k in ok]
    all_lit_one = all(all(v < MIN_OCCUPIED_TIERS for v in collapse[name])
                      for name, _h, _l in LIT_CUT_SETS)
    any_lit_one = any(any(v < MIN_OCCUPIED_TIERS for v in collapse[name])
                      for name, _h, _l in LIT_CUT_SETS)

    names = [Path(ds.meta["file"]).name for ds, _k in ok]
    lines.append(f"[数据] {', '.join(names)}（45° 机位、1 人；逐帧读数 = 生产平滑值）。")

    # [1] 文献切法：占用档数 + 非 NORMAL 占比（占比才是"还能不能分档"的真信号）
    parts = []
    for name, hi, lo in LIT_CUT_SETS:
        shares = []
        for ds, _k in ok:
            c = ds.lit_counts[name]
            tot = sum(c) or 1
            shares.append(f"{100.0 * (c[1] + c[2]) / tot:.1f}%")
        parts.append(f"{name}({_fmt(hi)}/{_fmt(lo)}) 非NORMAL "
                     + "/".join(shares) + f"、占用档数 {collapse[name]}")
    lines.append("[1] 文献切法套用后（占比依次 = " + " / ".join(names) + "）："
                 + "；".join(parts) + "。")

    # [2] 存档自标定基线对照 + 塌缩判定
    emp_shares = []
    for ds, _k in ok:
        tot = sum(ds.emp_counts) or 1
        emp_shares.append("/".join(f"{100.0 * v / tot:.1f}%" for v in ds.emp_counts))
    lines.append(f"[2] 对照：同一批帧用存档自标定基线 {_fmt(EMP_HI)}/{_fmt(EMP_LO)} 时三档占比 "
                 + " 与 ".join(emp_shares) + f"（占用档数 {emp_occ}）。")
    if all_lit_one:
        lines.append("[3] 两套文献切法都把全部帧判进同一档（整体塌缩）——"
                     "**文献阈值不能直接用于本 proxy**；这是数字本身给出的，不是预设结论。")
    elif any_lit_one:
        worst = [n for n, _h, _l in LIT_CUT_SETS if all(v < MIN_OCCUPIED_TIERS
                                                        for v in collapse[n])]
        lines.append(f"[3] 文献切法 {', '.join(worst)} 整体塌缩（非 NORMAL 占比见 [1]）——"
                     "该切法不能直接用；另一套虽然没塌缩，但分出去的比例极小，"
                     "等于把几乎全部姿势都叫“正常”，同样不能当判据。")
    else:
        lines.append("[3] 两套文献切法都没塌缩，但非 NORMAL 占比远低于存档自标定基线那行（见 [1][2]）——"
                     "档位能分开不等于分得对，还要看偏移。")

    agr = [ds.agreement[name] for ds, _k in ok
           for name, _h, _l in LIT_CUT_SETS if ds.agreement[name] is not None]
    if agr:
        lines.append(f"[4] 与存档自标定基线的帧级一致率 {_fmt(100.0 * min(agr), 1)}%"
                     f"~{_fmt(100.0 * max(agr), 1)}% —— 文献判据近塌缩时，这个数≈生产在"
                     "NORMAL 档上的占比，本身不携带额外信息（看表 3 的矩阵形状）。")

    # 偏移恒定吗：逐批给差值/比值极差 + 各档差值，判读带上文件名
    off_txt, verdict = [], []
    for ds, k in ok:
        name_f = Path(ds.meta["file"]).name
        d, r = k["dspread"], k["rspread"]
        diffs = " / ".join(_fmt(o["diff"]) for o in k["off"])
        off_txt.append(f"{name_f} 各档差值 {diffs}（差值极差 {_fmt(d)}°、"
                       f"比值极差 {_fmt(r, 3)}）")
        verdict.append(f"{name_f} " + verdict_phrase(d, r))
    lines.append("[5] 偏移（proxy 中位数 − 文献各档均值 55/48/42）：" + "；".join(off_txt) + "。")
    lines.append("[6] 判读：" + "；".join(verdict) + "。")

    # 偏移再怎么"恒定"，样本也撑不起标定 —— 这行不许省
    n_person = len({ds.meta["person"] for ds, _k in ok})
    n_frame = sum(ds.stats[l]["n"] or 0 for ds, _k in ok for l in LABEL_ORDER)
    segs_txt = "；".join(
        Path(ds.meta["file"]).name + " 每档段数 " +
        "/".join(str(k["n_seg"][l] or 0) for l in LABEL_ORDER) for ds, k in ok)
    lines.append(f"[7] 样本：{n_person} 个人、{n_frame} 帧有读数（{segs_txt}）；"
                 "跨人 / 跨取景 / 跨距离的变异完全没测。")
    # 收口句也由数字决定，别写死"不能用"
    if all_lit_one:
        usable = "文献阈值不可直接使用（两套切法都把全部姿势判成同一档）"
    elif any_lit_one:
        usable = ("文献阈值不可直接使用（一套塌缩，另一套只分出极小比例，"
                  "等于把几乎全部姿势都叫正常）")
    else:
        usable = ("文献阈值在本批数据上没塌缩，但非 NORMAL 占比与存档基线差得远，"
                  "仍不足以当判据")
    lines.append(f"[8] 所以：{usable}；任何“恒定”的偏移/比例也都只是本批数据内的观察，"
                 "把它写成换算公式＝拿文献值当标定结果，本脚本不做，也不建议直接拿去用。")
    return lines[:10]


# ---------- 主流程 ----------

def analyze(paths: Sequence[Path], skip_sec: float,
            skips: Sequence[float]) -> Tuple[List[Table], List[str], List[dict], List[Frame]]:
    """跑完所有数据，返回 (表格, 结论, 假设清单, 逐帧明细)。"""
    tables: List[Table] = []
    results: List[Tuple[DatasetResult, dict]] = []
    all_frames: List[Frame] = []

    for p in paths:
        frames, meta = load_any(p)
        kept, dropped = drop_segment_heads(frames, skip_sec)
        ds = DatasetResult(meta, kept, dropped, skip_sec)
        ts, key = build_tables(ds, skips)
        tables += ts
        results.append((ds, key))
        for fr in kept:
            all_frames.append(fr)

    conclusion = build_conclusion(results)

    assumptions = [
        "数据文件：" + "；".join(
            f'{ds.meta["file"]}（{ds.meta["view"]}° / 人 {ds.meta["person"]} / 次 '
            f'{ds.meta["session"]}，带标签 {ds.meta["labeled"]} 帧）'
            for ds, _k in results),
        "每档段数：" + "；".join(
            f'{Path(ds.meta["file"]).name} = ' +
            "/".join(str(k["n_seg"][l] or 0) for l in LABEL_ORDER) +
            "（Normal/Slight/Obvious，按标签变化切段）" for ds, k in results),
        f"缺关键点的帧：置信度 < 生产 visibility_min = {VIS_MIN}（从 features.CvaProxyFeatures "
        "读，未写死）判降级，该帧无读数 → 不进统计，只在“无读数帧”那行计数；"
        "降级帧不沿用上一有效值（本脚本不看 EMA 维持值，只看本帧真算出来的读数）。",
        f"读数口径：取**平滑值**（生产判据读的那一列）。molaeifar 用 cva_proxy_raw_ema、"
        f"f45 用 cva_proxy；未平滑值另作敏感性看（molaeifar 有、f45 整列为空）。",
        # 复算核对那句由实际数字生成（写死数字换个文件就会说谎）
        "复算核对（从存的关键点像素坐标复算 CvaProxyFeatures，与录制列逐帧比对；"
        f"复算与 REPLAY_W/H 取值无关，归一化↔像素相消；容差 {REPLAY_TOL_DEG}°）："
        + "；".join(_replay_note(ds.meta) for ds, _k in results)
        + "。统计一律读录制列（生产在录制时算的），坏关键点不改变任何数字。",
        f"切段与过渡帧：段 = 同一 (标签, rec_index) 连续帧，段头 {skip_sec}s 丢弃；"
        f"rec_index 在本批数据恒为 0，所以实际按标签变化切。",
        f"文献阈值：{LIT_SOURCE}；区间按需求原文录入。两种切法（区间边缘 / 均值中点）"
        f"都算都报，不替用户选。",
        "未做的事：不拟合阈值、不给修正系数、不改生产代码、不动 FhpDecision 的阈值。",
        f"对照基线 = **存档的**自标定阈值 {_fmt(EMP_HI)} / {_fmt(EMP_LO)}（45°）："
        f"生产预设 2026-09-29 已换成文献切点，现值 = {_fmt(PROD_HI)} / {_fmt(PROD_LO)}"
        f"（与本脚本的 L1 切法重合）—— 所以 [2][3] 的对照打的是存档基线，不是现值。",
    ]
    return tables, conclusion, assumptions, all_frames


def write_outputs(tables: List[Table], conclusion: List[str], assumptions: List[str],
                  frames: List[Frame], out_csv: Path, table_md: Path,
                  report: Path) -> None:
    """写逐帧 CSV、markdown 表格、纯文本报告（都先建父目录）。"""
    for p in (out_csv, table_md, report):
        p.parent.mkdir(parents=True, exist_ok=True)

    lit_cols = [t[0] for t in LIT_CUT_SETS]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["src", "seg", "label", "t", "deg_smoothed", "deg_raw",
                    "deg_replayed", "conf", "emp_tier"] +
                   [f"tier_{c}" for c in lit_cols])
        for fr in frames:
            tiers = ["" if fr.deg is None else tier_of(fr.deg, hi, lo)
                     for _n, hi, lo in LIT_CUT_SETS]
            w.writerow([fr.src, fr.seg, fr.label, fr.t, fr.deg, fr.raw, fr.replay,
                        fr.conf,
                        "" if fr.deg is None else tier_of(fr.deg, EMP_HI, EMP_LO)] +
                       tiers)

    md = ["# Salahzadeh 文献 CVA 阈值能否直接用于 45° CVA-like proxy", "",
          f"对照基线 = 存档自标定 45° 阈值 {_fmt(EMP_HI)} / {_fmt(EMP_LO)}"
          f"（生产现值 = {_fmt(PROD_HI)} / {_fmt(PROD_LO)}，2026-09-29 起换的）；"
          f"读数列 = 生产平滑值。", ""]
    md += [t.markdown() + "\n" for t in tables]
    md += ["### 结论", ""] + [f"{i + 1}. {ln}" for i, ln in enumerate(conclusion)] + [""]
    md += ["### 假设与口径", ""] + [f"- {a}" for a in assumptions] + [""]
    with open(table_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    with open(report, "w", encoding="utf-8") as fh:
        fh.write("=== Salahzadeh 文献 CVA 阈值 vs 45° CVA-like proxy ===\n")
        fh.write(f"对照基线 = 存档自标定 45° 阈值 {_fmt(EMP_HI)} / {_fmt(EMP_LO)}"
                 f"（生产现值 = {_fmt(PROD_HI)} / {_fmt(PROD_LO)}，2026-09-29 起换的）\n\n")
        for t in tables:
            fh.write(t.text() + "\n\n")
        fh.write("=== 结论 ===\n")
        for i, ln in enumerate(conclusion):
            fh.write(f"{i + 1}. {ln}\n")
        fh.write("\n=== 假设与口径 ===\n")
        for a in assumptions:
            fh.write(f"- {a}\n")


# ---------- 自测 ----------

def selftest() -> None:
    """合成数据自测：纯函数 + 加载器 + 复算路径 + 结论分支，不碰摄像头/模型。"""
    # 分位数与描述统计
    s = sorted([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(pct(s, 0.5) - 3.0) < 1e-12
    assert abs(pct(s, 0.25) - 2.0) < 1e-12
    assert abs(pct(s, 0.75) - 4.0) < 1e-12
    assert pct([7.0], 0.25) == 7.0
    d = describe([1.0, 2.0, 3.0, 4.0])
    assert d["n"] == 4 and abs(d["mean"] - 2.5) < 1e-12
    assert abs(d["iqr"] - (3.25 - 1.75)) < 1e-12
    assert d["min"] == 1.0 and d["max"] == 4.0
    assert describe([])["n"] is None
    assert describe([5.0])["std"] == 0.0

    # 切档方向：越大越直
    assert tier_of(90.0, 68.17, 61.42) == 0
    assert tier_of(68.17, 68.17, 61.42) == 0      # 边界归更直的一档
    assert tier_of(65.0, 68.17, 61.42) == 1
    assert tier_of(61.42, 68.17, 61.42) == 1
    assert tier_of(50.0, 68.17, 61.42) == 2

    # 偏移判读措辞：擦线要标“临界”，别把 0.049 vs 0.05 说成干脆的恒定
    assert "临界" in verdict_phrase(2.25, 0.049)          # 真实 f45 那组数字
    assert verdict_phrase(1.9, 0.30).startswith("差值")
    assert "临界" in verdict_phrase(1.9, 0.30)       # 差值擦线（1.9 贴着 2.0）
    assert "临界" not in verdict_phrase(0.5, 0.30)   # 差得远就不标
    assert verdict_phrase(2.5, 0.03).startswith("比值")
    assert verdict_phrase(5.42, 0.059) == "差值与比值都不跨档恒定"   # 真实 molaeifar 那组
    assert verdict_phrase(None, None).startswith("档位不全")

    # 标签映射（1/2/3 与文本两种口径）
    assert label_of("1") == "Normal" and label_of("3") == "Obvious_FHP"
    assert label_of("Slight_FHP") == "Slight_FHP" and label_of("") is None
    assert label_of("7") is None

    tmp = Path(__file__).resolve().parent / "_salahzadeh_selftest_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    # --- 合成一批 molaeifar 形状的 CSV ---
    def synth_rows(takes: Sequence[Tuple[str, float, int]]) -> List[dict]:
        """takes 按**录制顺序**给 (档名, 目标角, 段长)。

        顺序很关键：切段靠"标签变化"，连着两段同档会被并成一段（真机采样的顺序是
        1→2→3→1→2→3，同一档不连续），所以合成数据也照这个顺序摆。
        肩中点固定，耳中点按目标角摆放 —— 复算出的角就该等于目标角。
        """
        rows: List[dict] = []
        t = 0.0
        for lab, deg, n in takes:
            th = math.radians(deg)
            sx, sy, L = 400.0, 400.0, 200.0
            ex, ey = sx + L * math.cos(th), sy - L * math.sin(th)
            for i in range(n):
                t += 1.0 / 30.0
                rows.append({
                    "posture_label": {"Normal": "1", "Slight_FHP": "2",
                                      "Obvious_FHP": "3"}[lab],
                    "view": "45", "person": "9", "session": "selftest",
                    "rec_index": "0", "rec_sec": f"{t:.6f}",
                    "label_sec": f"{i / 30.0:.6f}",
                    "cva_proxy_raw": "", "cva_proxy_raw_ema": f"{deg:.6f}",
                    "earL_x": f"{ex - 20:.6f}", "earL_y": f"{ey:.6f}",
                    "earL_conf": "0.9", "earR_x": f"{ex + 20:.6f}",
                    "earR_y": f"{ey:.6f}", "earR_conf": "0.9",
                    "shL_x": f"{sx - 120:.6f}", "shL_y": f"{sy:.6f}",
                    "shL_conf": "0.9", "shR_x": f"{sx + 120:.6f}",
                    "shR_y": f"{sy:.6f}", "shR_conf": "0.9",
                })
        return rows

    cols = ["posture_label", "view", "person", "session", "rec_index", "rec_sec",
            "label_sec", "cva_proxy_raw", "cva_proxy_raw_ema", "earL_x", "earL_y",
            "earL_conf", "earR_x", "earR_y", "earR_conf", "shL_x", "shL_y",
            "shL_conf", "shR_x", "shR_y", "shR_conf"]

    def write_csv(path: Path, rows: List[dict]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # 情形 A：读数 73/65/57 那一带（贴着生产阈值），三档各两段，按 1→2→3→1→2→3 摆
    pa = tmp / "synth_a.csv"
    write_csv(pa, synth_rows([("Normal", 73.0, 60), ("Slight_FHP", 65.0, 55),
                              ("Obvious_FHP", 57.0, 50), ("Normal", 72.0, 40),
                              ("Slight_FHP", 66.0, 45), ("Obvious_FHP", 58.0, 40)]))
    frames_a, meta_a = load_molaeifar(pa)
    assert meta_a["has_keypoints"] and meta_a["labeled"] == 290
    # 复算路径：与构造角一致（±0.05°）
    replayed = [f.replay for f in frames_a if f.replay is not None]
    assert len(replayed) == 290, len(replayed)
    assert max(abs(f.replay - f.deg) for f in frames_a) < 0.05
    kept_a, dropped_a = drop_segment_heads(frames_a, 1.0)
    assert dropped_a > 0                    # 段头确实丢了
    dsa = DatasetResult(meta_a, kept_a, dropped_a, 1.0)
    assert dsa.tier_names_seen() == 3
    # 文献阈值：全部塌缩进 NORMAL（57° 也在 53.2 之上）
    for name, hi, lo in LIT_CUT_SETS:
        c = dsa.lit_counts[name]
        assert c[0] == sum(c) and c[1] == 0 and c[2] == 0, (name, c)
    # 生产阈值：三档都占到
    assert all(v > 0 for v in dsa.emp_counts), dsa.emp_counts
    # 一致率：文献全判 NORMAL → 只有生产也判 NORMAL 的帧对齐
    assert dsa.agreement[LIT_CUT_SETS[0][0]] == \
        dsa.emp_counts[0] / sum(dsa.emp_counts)
    tabs_a, key_a = build_tables(dsa, (0.0, 1.0, 2.0))
    conc_a = build_conclusion([(dsa, key_a)])
    assert len(conc_a) <= 10, len(conc_a)
    assert any("塌缩" in ln for ln in conc_a), conc_a     # 该报的异常要报出来
    # 段级：每档 2 段（构造如此）
    assert all(key_a["n_seg"][l] == 2 for l in LABEL_ORDER), key_a["n_seg"]
    # 差值跨档不恒定（73-55=18 / 65-48=17 / 57-42=15）
    assert key_a["dspread"] is not None and key_a["dspread"] > 2.0

    # 情形 B：读数正好落在文献阈值能分开的位置（55/48/42）→ 三条结论分支都不同
    pb = tmp / "synth_b.csv"
    write_csv(pb, synth_rows([("Normal", 55.0, 30), ("Slight_FHP", 48.0, 30),
                              ("Obvious_FHP", 42.0, 30)]))
    frames_b, meta_b = load_molaeifar(pb)
    dsb = DatasetResult(meta_b, frames_b, 0, 0.0)
    assert dsb.lit_counts["L2_mean_mid_51.5_45.0"] == [30, 30, 30]
    assert sum(1 for v in dsb.lit_counts["L1_edges_53.2_43.2"] if v > 0) >= 2
    # 本例顺带证明"文献切法能分开"远不等于"和生产口径一致"：读数 55/48/42 全在
    # 生产 61.42 之下 → 生产口径判的是 90 帧全 OBVIOUS，一致率只有 1/3。
    assert dsb.emp_counts == [0, 0, 90], dsb.emp_counts
    assert abs(dsb.agreement["L2_mean_mid_51.5_45.0"] - 1.0 / 3.0) < 1e-9
    _t, key_b = build_tables(dsb, (0.0,))
    assert key_b["dspread"] is not None and key_b["dspread"] < 1e-6   # 差值恒为 0

    # 情形 C：无关键点列（fhp_csv 形状）→ 复算路径自动关闭，读数读 cva_proxy
    pc = tmp / "synth_c.csv"
    with open(pc, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "cva_proxy", "cva_proxy_raw", "valid",
                    "posture_label", "view", "person", "session"])
        for lab, deg in (("Normal", 73.0), ("Slight_FHP", 65.0),
                         ("Obvious_FHP", 57.0)):
            for i in range(5):
                w.writerow([f"{i / 30.0:.3f}", f"{deg:.3f}", "", "", lab,
                            "45", "9", "selftest"])
    frames_c, meta_c = load_fhp_csv(pc)
    assert meta_c["has_keypoints"] is False and len(frames_c) == 15
    assert all(f.replay is None and f.raw is None for f in frames_c)
    assert abs(frames_c[0].deg - 73.0) < 1e-9

    # 情形 D：全降级（没有读数）→ 结论走"没有可用读数"分支，不抛异常
    pd = tmp / "synth_d.csv"
    write_csv(pd, synth_rows([("Normal", 73.0, 10)]))
    rows_d = _read_rows(pd)
    for r in rows_d:
        r["cva_proxy_raw_ema"] = ""
        r["cva_proxy_raw"] = ""
        for k in ("earL_x", "earR_x", "shL_x", "shR_x"):
            r[k] = ""
    write_csv(pd, rows_d)
    frames_d, meta_d = load_molaeifar(pd)
    dsd = DatasetResult(meta_d, frames_d, 0, 0.0)
    assert dsd.stats["Normal"]["n"] is None
    conc_d = build_conclusion([(dsd, build_tables(dsd, (0.0,))[1])])
    assert any("没有可用读数" in ln for ln in conc_d), conc_d

    # 输出写完能读回（三种文件都建得出来）
    write_outputs(tabs_a, conc_a, ["assumption"], kept_a, tmp / "frames.csv",
                  tmp / "t.md", tmp / "t.txt")
    assert (tmp / "frames.csv").exists() and (tmp / "t.md").exists()
    assert (tmp / "t.txt").exists()
    assert len(_read_rows(tmp / "frames.csv")) == len(kept_a)

    # 清理
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    print("[selftest] 全部分组通过（统计/切档/标签/复算/塌缩分支/无读数分支/输出）")


# ---------- CLI ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="检验 Salahzadeh 文献 CVA 三档阈值能否直接用于 45° CVA-like proxy"
                    "（只测量、不改生产代码）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="*", default=None, metavar="PATH",
                   help="逐帧 CSV（默认用 data/ 下那两批 45° 数据）。"
                        "带 earL_x 等关键点列的走复算路径，否则只读录制值")
    p.add_argument("--skip-sec", type=float, default=1.0, metavar="S",
                   help="每段开头丢几秒过渡帧（默认 1.0，与 molaeifar 脚本一致）")
    p.add_argument("--sens-skip", default="0,1,2", metavar="LIST",
                   help="skip-sec 敏感性扫描用的取值（逗号分隔，默认 0,1,2）")
    p.add_argument("--out", default="data/salahzadeh_cva_frames.csv", metavar="PATH",
                   help="逐帧明细 CSV 输出路径")
    p.add_argument("--table-md", default="data/salahzadeh_cva_table.md", metavar="PATH",
                   help="markdown 表格输出路径")
    p.add_argument("--report", default="data/salahzadeh_cva_report.txt", metavar="PATH",
                   help="纯文本报告（表格 + 结论 + 假设）输出路径")
    p.add_argument("--selftest", action="store_true",
                   help="合成数据自测（不读实拍数据、不写 data/）")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.selftest:
        selftest()
        return 0

    if args.csv:
        paths = [Path(x) for x in args.csv]
    else:
        paths = [REPO_ROOT / "data" / "f45.csv",
                 REPO_ROOT / "data" / "molaeifar_frames_45.csv"]
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"[error] 找不到数据文件：{p}", file=sys.stderr)
        print("提示：实拍 CSV 不随仓库分发（个人身体数据）。"
              "用 --csv 指定你自己的录制，或先跑 tools/test_molaeifar_sternum_angle.py "
              "--view 45 --camera 0 采一批带关键点的数据。", file=sys.stderr)
        return 2

    try:
        skips = [float(x) for x in str(args.sens_skip).split(",") if x.strip() != ""]
    except ValueError:
        print("[error] --sens-skip 只能是逗号分隔的数字", file=sys.stderr)
        return 2
    if args.skip_sec not in skips:
        skips.append(args.skip_sec)

    tables, conclusion, assumptions, frames = analyze(paths, args.skip_sec, skips)

    for t in tables:
        print(t.text())
        print()
    print("=== 结论 ===")
    for i, ln in enumerate(conclusion):
        print(f"{i + 1}. {ln}")
    print("\n=== 假设与口径 ===")
    for a in assumptions:
        print(f"- {a}")

    write_outputs(tables, conclusion, assumptions, frames,
                  Path(args.out), Path(args.table_md), Path(args.report))
    print(f"\n[written] {args.out}\n[written] {args.table_md}\n[written] {args.report}")
    print("提醒：逐帧 CSV 是个人身体数据，别提交（.gitignore 已挡）。")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:      # 老解释器/重定向场景下拿不到 reconfigure 就算了
        pass
    raise SystemExit(main())
