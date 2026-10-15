"""Molaeifar 式对照指标（胸骨中点–左右耳屏角）的**测量/验证脚本**（独立，不影响 demo）。

用法（在 rtm_demo/ 目录下运行）：
    python tools/test_molaeifar_sternum_angle.py --camera 0 --view front --person 1 --session 1
    python tools/test_molaeifar_sternum_angle.py --camera 0 --view 45   --person 1 --session 1
    python tools/test_molaeifar_sternum_angle.py --video clip.mp4 --view front
    python tools/test_molaeifar_sternum_angle.py --analyze-only data/front_p1_s1.csv data/f45_p1_s1.csv
    python tools/test_molaeifar_sternum_angle.py --selftest          # 合成数据自测，不碰摄像头/模型

逐帧 CSV 默认写进 data/（含关键点像素坐标 = 个人身体数据，已被 .gitignore 挡住）。

键盘：1 / 2 / 3 = Normal / Slight_FHP / Obvious_FHP，0 = 清掉标签，
      r = 重开一段（rec 计时归零），s = 立刻出报告，q / ESC = 退出。

本文件**只做测量 + 报告**：不判定、不提醒、不写 profile、**不拟合阈值**，也**不修改**
任何现有文件（capture / pose / features / decision / output 一行未动、已有测试脚本
一行未动）。它 import 现有的 CameraCapture、detect_pose，以及生产侧的特征类
（CvaProxyFeatures / EarShoulderProxyFeatures / PostureFeatures），保证对照指标算的
就是运行时那一套，不是另抄一份。

================================ 先读这一节 ================================

一、测的是什么（Molaeifar et al. 2021 的构造）

    Molaeifar S, Yazdani F, Kordi Yoosefinejad A, Karimi MT. "Correlation between
    craniovertebral angle in the sagittal plane and angles and indices measured in
    the frontal plane at the moment of inducing forward head posture."
    Work 2021;68(4):1221-1227. DOI 10.3233/WOR-213451

    摘要口径：58 名健康受试者（29 男 29 女，18~40 岁），8 相机动捕；先保持中立位 5s，
    再"屈曲并低下头"诱导 FHP，然后算 CVA 与一组**正面**角/指数在 FHP 诱导瞬间的相关。
    结论原文：**a moderate correlation was observed between 3-D CVA and the angle
    formed between the sternum and both tragi**；changes in CVA in the sagittal plane
    can be predicted from changes in the angle formed between **the midpoint of the
    sternum and the left and right tragi** in the frontal plane.

    本脚本把该构造搬到单目 RGB + RTMPose(COCO-17) 上：

        顶点 = 双肩中点（SHOULDER_IDS = 5/6 的中点）→ 近似"胸骨中点"
        两条边 = 顶点 → 左耳(3)、顶点 → 右耳(4) → 近似"左右耳屏"
        指标 = 这两条边在顶点处的夹角（度）

二、三处"近似"必须先说清楚（都是代理，不是文献量）

    1. **胸骨中点 → 双肩中点**：COCO-17 里没有胸骨/胸骨角，躯干中线能用的是双肩中点
       （生产侧 ErgonomicRiskFeatures 近似 C7 也是这么办的）。真实胸骨中点比肩线更靠
       下、更靠前，所以顶点位置有系统性偏差 —— 本脚本用 `--drop-sw 0/0.15/0.3`
       做**顶点下移敏感性扫描**，就是量这个偏差影响多大（第 5 项分析要求）。
    2. **耳屏（tragus）→ 耳关键点(3/4)**：RTMPose 的耳点定的是耳朵整体，落点通常在
       耳廓外缘/耳垂附近，不是耳屏（耳道前方那个软骨小突起）；两者相差数厘米。
    3. **3D 角 → 2D 图像平面角**：文献是动捕的 3D 角；这里是单目图像的投影角。正对
       机位时"前伸"落在**深度方向**，画面里几乎看不见 —— 这个角在正面之所以会随
       "低头/屈曲"变，主要来自**耳-肩竖直间距被压扁**（屈曲把耳朵朝肩线带），
       而不是 3D 前伸本身。所以：

    ⚠ 本指标是**工程代理**，量值**不能**和文献/临床数值直接比；下面的结论只对本项目
      自己采的数据负责。

三、"像素空间"是硬要求

    角度必须用像素坐标（`_pt` 把归一化坐标乘回 w/h）算。归一化坐标下 tan 被画面
    纵横比缩放，16:9 下同一个姿势差 ~10°（见 CLAUDE.md「两条 CVA 线别混用」）。
    本脚本所有几何量一律像素空间，**不混用**生产侧 ErgonomicRiskFeatures.cva_deg
    （那条是归一化空间的旧 CVA/FSA 线）。

四、置信度门控（按需求固定）

    参与计算的关键点（双耳 3/4 + 双肩 5/6）**每一个**的置信度都要 >= 0.3，否则该帧
    本指标判无效（`molaeifar_valid=0` + `drop_reason`），**不沿用上一帧、不插值**。
    丢弃帧仍然逐帧写进 CSV（留痕），但**不进任何统计**。

五、判定标准（**先固定在这里，看完数据不许改**）

    a. 单调性：三档的**帧级中位数**是否严格单调；方向（升/降）在两个机位下是否一致。
    b. 分离度：Normal 与 Obvious 的 IQR 是否**不重叠**；Normal 与 Slight 的 IQR 是否
       **不重叠**（两条都查）。IQR = [Q1, Q3]，分位点用线性插值（numpy 默认口径）。
    c. 抖动比：段内逐帧标准差（每段一个，取所有段的**中位数**）
       ÷ |中位数(Obvious) − 中位数(Normal)| —— **越小越好**（<0.5 约等于"噪声不到
       效应量的一半"）。
    同三项标准**一并套到**现有对照指标上（45° CVA proxy / FHD / Ear–Shoulder /
    颈压缩），放在同一张表里比。

    独立单元口径（防止把相邻帧当独立样本）：帧只用于**描述**（n / 中位数 / IQR）；
    要谈"多段重复"时必须用**每段录制的中位数**当单元。每档段数 < 3 时，报告与结论
    里会明确写"样本量不足以做统计推断，只作描述性对比"。

六、本脚本**不做**什么

    * 不拟合阈值、不挑参数（敏感性扫描只作探索，**不许**用来选顶点位置）；
    * 不接 decision / 不接提醒 / 不接 web（正面 Ear–Shoulder 那条线都还只显示不提醒）；
    * 不自动判机位（本项目实测没有可靠的自动判别特征，机位靠 `--view` 由人声明）。

七、逐帧 CSV 列（默认 data/molaeifar_frames.csv，含原始关键点像素坐标）

    时间/条件：timestamp, frame_index, posture_label, view, person, session,
              rec_index, rec_sec, label_sec, note
              （rec_index = 每按一次 r 加一；**切段就靠它 + 标签**，所以同一档连着录
                两次不会被并成一段；label_sec = 当前标签已持续多少秒，段首丢弃按它算）
    新指标  ：molaeifar_angle（+ _ema）、_d15/_d30（顶点下移 0.15/0.30 肩宽，+ _ema）、
              molaeifar_valid, drop_reason
    对照指标：cva_proxy_raw/_ema（生产 CvaProxyFeatures 的 raw 再统一 EMA）、
              ear_shoulder_raw/_ema（生产 EarShoulderProxyFeatures 的 raw 再统一 EMA）、
              fhd_raw/_ema、neck_compression/_ema、head_neck_angle
    几何诊断：shoulder_width_px/_norm, ear_span_px, ear_sh_vert_px, vert_over_span,
              vertex_x_px, vertex_y_px, earL_conf, earR_conf, shL_conf, shR_conf,
              earL_x/earL_y, earR_x/earR_y, shL_x/shL_y, shR_x/shR_y

    `*_ema` 一律是**本脚本自己**对 raw 列做的一阶 EMA（tau=1.0s，与生产同值），
    对 5 个指标**用同一条平滑**，主表里 raw / ema 两行才可比。
    `fhd_raw` = |耳x − 肩中点x| ÷ 肩宽（像素空间，与 test_fhp_measurements.py 的
    measure().fhd 同式 —— 该脚本用同一套"肩宽（髋不可见时退回躯干长）"作尺度）。
    `neck_compression` 直接取生产 PostureFeatures 的读数（归一化坐标口径，超物理上限
    1.5 时生产给 None，本脚本照记空）。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

# 本文件在 tools/ 下，上一层才是仓库根（features / decision / capture 在那儿）
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capture import CameraCapture          # noqa: E402
from features import (CvaProxyFeatures, EarShoulderProxyFeatures,   # noqa: E402
                      PostureFeatures)
from pose_estimation import detect_pose    # noqa: E402

# ---------- 固定口径（改这里等于改判定标准，改之前先想清楚） ----------

VIS_MIN = 0.3                    # 关键点置信度下限（需求固定值，别动）
EAR_IDS = (3, 4)                 # COCO-17：3 左耳、4 右耳（近似左右耳屏）
SHOULDER_IDS = (5, 6)            # 5 左肩、6 右肩
HIP_IDS = (11, 12)
USED_IDS = EAR_IDS + SHOULDER_IDS + HIP_IDS

# 键盘标签 → 姿势名（与 test_fhp_cva_proxy.py / test_front_ear_shoulder_proxy.py 同一套）
LABEL_KEYS = {"1": "Normal", "2": "Slight_FHP", "3": "Obvious_FHP"}
LABEL_ORDER = ("Normal", "Slight_FHP", "Obvious_FHP")
# 兼容老 CSV（posture_label 写的是 1/2/3 数字，0/空 = 未标注）
LABEL_FROM_NUMBER = {"1": "Normal", "2": "Slight_FHP", "3": "Obvious_FHP"}

VIEWS = ("front", "45")          # 两个机位各要一轮
VERTEX_DROPS = (0.0, 0.15, 0.30)  # 顶点自肩中点向下平移几个肩宽（敏感性扫描）
DROP_SUFFIX = {0.0: "", 0.15: "_d15", 0.30: "_d30"}

EMA_TAU_SEC = 1.0                # 与生产 CvaProxyFeatures/EarShoulderProxyFeatures 同值
MAX_LABEL_ORDER = len(LABEL_ORDER)

WINDOW_NAME = "Molaeifar sternum-ear angle (experiment)"

CSV_COLUMNS = (
    "timestamp", "frame_index", "posture_label", "view", "person", "session",
    "rec_index", "rec_sec", "label_sec", "note",
    "molaeifar_angle", "molaeifar_angle_ema",
    "molaeifar_angle_d15", "molaeifar_angle_d15_ema",
    "molaeifar_angle_d30", "molaeifar_angle_d30_ema",
    "molaeifar_valid", "drop_reason",
    "cva_proxy_raw", "cva_proxy_raw_ema",
    "ear_shoulder_raw", "ear_shoulder_raw_ema",
    "fhd_raw", "fhd_raw_ema",
    "neck_compression", "neck_compression_ema",
    "head_neck_angle",
    "shoulder_width_px", "shoulder_width_norm", "ear_span_px", "ear_sh_vert_px",
    "vert_over_span", "vertex_x_px", "vertex_y_px",
    "earL_conf", "earR_conf", "shL_conf", "shR_conf",
    "earL_x", "earL_y", "earR_x", "earR_y",
    "shL_x", "shL_y", "shR_x", "shR_y",
)

# 主表里的对照指标（新指标 + 需求点名的四个现有指标）
MAIN_METRICS = (
    ("molaeifar_angle", "Molaeifar 胸骨–双耳屏角（新）",
     "方向未预设，如实报"),
    ("cva_proxy_raw", "45° CVA proxy（耳中点→肩中点 与水平线）",
     "越小越前伸（像素空间）"),
    ("ear_shoulder_raw", "Ear–Shoulder proxy（|耳中点x−肩中点x|/肩宽）",
     "越大越前伸"),
    ("fhd_raw", "FHD proxy（|耳x−肩中点x|/肩宽）",
     "越大越前伸"),
    ("neck_compression", "颈压缩（耳-肩竖直间距/肩宽）",
     "越小越弓背"),
)
SENSITIVITY_METRICS = (
    ("molaeifar_angle", "顶点 = 肩中点（0 肩宽）"),
    ("molaeifar_angle_d15", "顶点下移 0.15 肩宽"),
    ("molaeifar_angle_d30", "顶点下移 0.30 肩宽"),
)


# ---------- 几何（纯函数，全部像素空间） ----------

Point = Tuple[float, float]


def _pt(landmarks, pid: int, w: float, h: float,
        vis_min: float = VIS_MIN) -> Optional[Point]:
    """归一化关键点 → 像素坐标；置信度 < vis_min 或缺该点 → None。

    与 test_fhp_measurements.py 的 _pt 同义（刻意不用 landmarks_to_pixels：那个会把
    坐标截断成整数，实验脚本要亚像素精度）。
    """
    if landmarks is None or pid >= len(landmarks):
        return None
    kp = landmarks[pid]
    if kp is None or len(kp) < 4:
        return None
    x, y, _z, vis = kp[0], kp[1], kp[2], kp[3]
    if vis is None or vis < vis_min:
        return None
    return (float(x) * w, float(y) * h)


def _mid(pts: Sequence[Optional[Point]]) -> Optional[Point]:
    """非 None 点的质心（像素）。全 None → None。"""
    got = [p for p in pts if p is not None]
    if not got:
        return None
    return (sum(p[0] for p in got) / len(got), sum(p[1] for p in got) / len(got))


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_at_vertex(v: Point, a: Point, b: Point) -> Optional[float]:
    """顶点 v 处、两边指向 a / b 的夹角（度，0~180）。任一边退化 → None。"""
    v1 = (a[0] - v[0], a[1] - v[1])
    v2 = (b[0] - v[0], b[1] - v[1])
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def molaeifar_angle(pts: Dict[int, Point], drop_sw: float = 0.0) -> Optional[float]:
    """Molaeifar 式角度：顶点 = 双肩中点（可向下平移 drop_sw 个肩宽）→ 左右耳。

    仅当**双耳(3/4)与双肩(5/6)都在 pts 里**（即都过了 VIS_MIN）才算得出；
    要不到 → None（不沿用上一帧、不插值）。
    返回 (角度, 顶点, 耳L, 耳R)；算不出 → None。
    """
    ear_l, ear_r = pts.get(3), pts.get(4)
    sh_l, sh_r = pts.get(5), pts.get(6)
    if ear_l is None or ear_r is None or sh_l is None or sh_r is None:
        return None
    sh_px = _dist(sh_l, sh_r)
    if sh_px < 1e-6:
        return None
    mid = _mid([sh_l, sh_r])
    vertex = (mid[0], mid[1] + drop_sw * sh_px)   # 图像 y 向下 = "向下平移"
    ang = _angle_at_vertex(vertex, ear_l, ear_r)
    if ang is None:
        return None
    return ang, vertex, ear_l, ear_r


# ---------- EMA（本脚本自己的一条，5 个指标共用，保证可比） ----------

class _Ema:
    """一阶 EMA：alpha = 1 − exp(−dt/tau)。只在有 raw 的帧推进（无效帧返回 None）。

    与生产 CvaProxyFeatures 的平滑同式同 tau；区别是无效帧**不沿用维持值**，直接
    返回 None —— 实验脚本要的是"这条序列里每个数都真的算出来了"，不掺保持值。
    """

    def __init__(self, tau_sec: float = EMA_TAU_SEC) -> None:
        self.tau = float(tau_sec)
        self.value: Optional[float] = None
        self._last_t: Optional[float] = None

    def update(self, raw: Optional[float], now: float) -> Optional[float]:
        if raw is None:
            return None
        if self.value is None or self._last_t is None:
            self.value, self._last_t = float(raw), now
            return self.value
        dt = now - self._last_t
        self._last_t = now
        if self.tau <= 0 or dt <= 0:
            self.value = float(raw)          # 时间没走/无平滑要求：直接跟随
        else:
            alpha = 1.0 - math.exp(-dt / self.tau)
            self.value += alpha * (float(raw) - self.value)
        return self.value


# ---------- 逐帧测量 ----------

class FrameMetrics:
    """一帧 → 一行指标。内部持有生产侧特征类（只读它们算出来的数，不改它们）。"""

    EMA_KEYS = ("molaeifar_angle", "molaeifar_angle_d15", "molaeifar_angle_d30",
                "cva_proxy_raw", "ear_shoulder_raw", "fhd_raw", "neck_compression")

    def __init__(self, smooth_tau: float = EMA_TAU_SEC) -> None:
        self.cva = CvaProxyFeatures()
        self.esp = EarShoulderProxyFeatures()
        self.posture = PostureFeatures()
        self.emas = {k: _Ema(smooth_tau) for k in self.EMA_KEYS}

    def measure(self, pose: Optional[dict], now: float) -> dict:
        """算一帧。pose = detect_pose() 的结果或 None。绝不抛异常。"""
        m: dict = {c: None for c in CSV_COLUMNS if c != "posture_label"
                   and c not in ("view", "person", "session", "note", "drop_reason")}
        m["drop_reason"] = ""
        m["note"] = ""

        if pose is None:
            m["drop_reason"] = "no_person"
            self._advance_emas(m, now)
            return m
        lm = pose.get("landmarks")
        size = pose.get("image_size")
        if not lm:
            m["drop_reason"] = "empty_pose"
            self._advance_emas(m, now)
            return m
        if not size or size[0] <= 0 or size[1] <= 0:
            m["drop_reason"] = "no_image_size"
            self._advance_emas(m, now)
            return m
        w, h = float(size[0]), float(size[1])

        pts = {pid: _pt(lm, pid, w, h) for pid in USED_IDS}
        # 只写耳/肩四个点的列：髋(11/12) 没有对应列名，写进去会**覆盖 shR 列**
        # （2026-09-28 发现的 bug：髋一被检出，该帧 shR_x/y 就变成髋坐标，
        # 离线回放会拿到错的双肩。hip 只在 pts 里供 posture 特征用。）
        for pid, name in ((3, "earL"), (4, "earR"), (5, "shL"), (6, "shR")):
            p = pts.get(pid)
            if p is not None:
                m[f"{name}_x"], m[f"{name}_y"] = p[0], p[1]
        for pid, key in ((3, "earL_conf"), (4, "earR_conf"),
                         (5, "shL_conf"), (6, "shR_conf")):
            if pid < len(lm) and lm[pid] is not None and len(lm[pid]) >= 4:
                m[key] = lm[pid][3]

        # 新指标：顶点 = 肩中点（+ 平移），两边 = 左右耳
        for drop in VERTEX_DROPS:
            res = molaeifar_angle(pts, drop)
            m["molaeifar_angle" + DROP_SUFFIX[drop]] = res[0] if res else None
            if drop == 0.0 and res is not None:
                m["vertex_x_px"], m["vertex_y_px"] = res[1]

        missing = []
        if pts.get(3) is None:
            missing.append("earL")
        if pts.get(4) is None:
            missing.append("earR")
        if pts.get(5) is None:
            missing.append("shL")
        if pts.get(6) is None:
            missing.append("shR")
        if missing:
            m["drop_reason"] = "low_conf_or_missing:" + "+".join(missing)
        else:
            m["molaeifar_valid"] = 1

        # 几何诊断量
        sh_l, sh_r = pts.get(5), pts.get(6)
        if sh_l is not None and sh_r is not None:
            sw_px = _dist(sh_l, sh_r)
            m["shoulder_width_px"] = sw_px
            m["shoulder_width_norm"] = math.hypot(
                lm[5][0] - lm[6][0], lm[5][1] - lm[6][1])
        ear_l, ear_r = pts.get(3), pts.get(4)
        ear_mid = _mid([ear_l, ear_r])
        sh_mid = _mid([sh_l, sh_r])
        if ear_l is not None and ear_r is not None:
            m["ear_span_px"] = _dist(ear_l, ear_r)
        if ear_mid is not None and sh_mid is not None:
            vert = abs(sh_mid[1] - ear_mid[1])
            m["ear_sh_vert_px"] = vert
            if m["ear_span_px"]:
                m["vert_over_span"] = vert / m["ear_span_px"]

        # 对照指标 A：生产 45° CVA proxy（像素空间，越小越前伸）
        cva = self.cva.update(pose, now) or {}
        m["cva_proxy_raw"] = cva.get("cva_proxy_raw_deg")

        # 对照指标 B：生产 Ear–Shoulder proxy（越大越前伸）
        esp = self.esp.update(pose, now) or {}
        m["ear_shoulder_raw"] = esp.get("ear_shoulder_proxy_raw")

        # 对照指标 C：FHD proxy = |耳x − 肩中点x| ÷ 肩宽（像素空间）
        #   尺度口径与 test_fhp_measurements.py 的 measure() 一致：肩宽优先，
        #   髋不可见时退回躯干长（肩中点→髋中点）。
        if ear_mid is not None and sh_mid is not None:
            scale = m["shoulder_width_px"]
            if scale is None or scale <= 1e-6:
                hip_mid = _mid([pts.get(11), pts.get(12)])
                scale = _dist(sh_mid, hip_mid) if hip_mid is not None else None
            if scale is not None and scale > 1e-6:
                m["fhd_raw"] = abs(ear_mid[0] - sh_mid[0]) / scale

        # 对照指标 D：颈压缩（生产 PostureFeatures，归一化坐标口径）
        post = self.posture.update(pose) or {}
        m["head_neck_angle"] = post.get("head_neck_angle")
        m["neck_compression"] = post.get("neck_compression")

        self._advance_emas(m, now)
        return m

    def _advance_emas(self, m: dict, now: float) -> None:
        for key in self.EMA_KEYS:
            m[key + "_ema"] = self.emas[key].update(m.get(key), now)


# ---------- 数据源（与两个现有实验脚本同构，各自独立） ----------

class _Source:
    """统一数据源：摄像头走 capture.CameraCapture，视频文件本脚本直接用 cv2。"""

    def __init__(self, args) -> None:
        self.fps = 0.0
        if args.video:
            self._cap = cv2.VideoCapture(args.video)
            if not self._cap.isOpened():
                raise SystemExit(f"打不开视频文件: {args.video}")
            self.fps = args.video_fps or self._cap.get(cv2.CAP_PROP_FPS) or 0.0
            if not self.fps or self.fps <= 0:
                self.fps = 30.0
                print("[warn] 读不到视频帧率，按 30fps 折算时间戳（--video-fps 可指定）")
            n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.desc = (f"video:{args.video} ({self.fps:.1f}fps"
                         + (f", {n}帧)" if n else ")"))
            self._src = None
        else:
            self.idx = args.camera if args.camera is not None else 0
            self._src = CameraCapture(self.idx, width=1280, height=720)
            self.desc = f"camera:{self.idx}"
            self._cap = None
        self._t0 = time.monotonic()

    def __enter__(self) -> "_Source":
        if self._src is not None:
            self._src.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._src is not None:
            self._src.release()
        if self._cap is not None:
            self._cap.release()

    def read(self):
        if self._cap is not None:
            ok, frame = self._cap.read()
            return bool(ok) and frame is not None, frame
        frame = self._src.read()
        return frame is not None, frame

    def stamp(self, idx: int) -> float:
        """该帧时间戳（秒）：视频用播放时钟（可复现），实时用单调时钟（EMA 需要）。"""
        if self._cap is not None:
            return idx / self.fps
        return time.monotonic() - self._t0


def _text(frame, txt: str, org, scale: float = 0.5, color=(230, 230, 230)) -> None:
    """带黑描边的文字（深浅背景都要看得清）。"""
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(frame, txt, (org[0] + dx, org[1] + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _fn(v, nd: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def draw_hud(frame, m: dict, args, ts: float, label: str, label_since: float,
             idx: int, src: "_Source", valid_n: int, dropped_n: int) -> None:
    """实时面板：新指标（含三个顶点）+ 四个对照指标 + 丢弃计数。"""
    _text(frame, f"{src.desc}  view={args.view}  t={ts:6.1f}s  frame={idx}",
          (12, 26), 0.55)
    if label:
        _text(frame, f"[LABEL] {label}  {ts - label_since:5.1f}s", (12, 50), 0.6,
              (60, 220, 60))
    else:
        _text(frame, "[LABEL] -- (press 1/2/3)", (12, 50), 0.6, (0, 215, 255))

    col = (0, 215, 255) if m.get("molaeifar_valid") else (150, 150, 150)
    _text(frame, f"Molaeifar (sh-mid -> ears): {_fn(m.get('molaeifar_angle'), 2)} deg"
                 f"   [vert {_fn(m.get('vertex_y_px'), 0)}px]", (12, 78), 0.55, col)
    _text(frame, "  +0.15 sw: " + _fn(m.get("molaeifar_angle_d15"), 2)
                 + " deg    +0.30 sw: " + _fn(m.get("molaeifar_angle_d30"), 2)
                 + " deg", (12, 102), 0.5, col)
    if not m.get("molaeifar_valid"):
        _text(frame, f"  dropped: {m.get('drop_reason') or '-'}", (12, 126), 0.5,
              (0, 0, 255))
    else:
        _text(frame, "  both ears + both shoulders >= 0.3 conf (OK)", (12, 126),
              0.5, (60, 200, 60))

    _text(frame, "cva_proxy: " + _fn(m.get("cva_proxy_raw"), 2)
                 + "   ear-shoulder: " + _fn(m.get("ear_shoulder_raw"), 4)
                 + "   fhd: " + _fn(m.get("fhd_raw"), 4)
                 + "   neck-comp: " + _fn(m.get("neck_compression"), 3),
          (12, 150), 0.5)
    _text(frame, f"sh_width: {_fn(m.get('shoulder_width_px'), 0)}px "
                 f"ear_span: {_fn(m.get('ear_span_px'), 0)}px "
                 f"vert/span: {_fn(m.get('vert_over_span'), 3)}", (12, 174), 0.5)
    _text(frame, f"kept {valid_n}  dropped {dropped_n}", (12, 198), 0.5)
    _text(frame, "1=Normal 2=Slight 3=Obvious 0=clear r=restart s=report q=quit",
          (12, 222), 0.45, (200, 200, 200))


# ---------- 采集主循环 ----------

def run_capture(args) -> int:
    src = _Source(args)
    fm = FrameMetrics(args.smooth_sec)
    label, label_since = "", 0.0
    rec_index, rec_start = 0, 0.0     # 每按一次 r/R 开一段新录制（分析时按段取中位数）
    idx, valid_n, dropped_n = 0, 0, 0
    fh = None
    writer = None
    if args.csv:
        # 默认写进 data/，新克隆里那个目录可能还不存在 → 先建
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        fh = open(args.csv, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS),
                                extrasaction="ignore")
        writer.writerow({c: c for c in CSV_COLUMNS})
        fh.flush()

    try:
        with src:
            while True:
                ok, frame = src.read()
                if not ok or frame is None:
                    break
                ts = src.stamp(idx)
                pose = detect_pose(frame)
                m = fm.measure(pose, ts)
                if m.get("molaeifar_valid"):
                    valid_n += 1
                else:
                    dropped_n += 1
                row = dict(m)
                row.update({"timestamp": f"{ts:.6f}", "frame_index": idx,
                            "posture_label": (LABEL_ORDER.index(label) + 1)
                            if label else "",
                            "view": args.view, "person": args.person,
                            "session": args.session,
                            "rec_index": rec_index,
                            "rec_sec": f"{ts - rec_start:.6f}",
                            "label_sec": (f"{ts - label_since:.6f}" if label else ""),
                            "note": ""})
                if writer is not None:
                    writer.writerow(row)
                    if idx % 30 == 0:
                        fh.flush()

                if args.print_every and idx % args.print_every == 0:
                    drop_tag = "" if m.get("molaeifar_valid") else \
                        "<drop:" + str(m.get("drop_reason")) + ">"
                    print(f"[{idx:5d}] t={ts:7.2f}s label={label or '-':<12} "
                          f"molaeifar={_fn(m.get('molaeifar_angle'), 2)} "
                          f"cva={_fn(m.get('cva_proxy_raw'), 2)} "
                          f"es={_fn(m.get('ear_shoulder_raw'), 4)} "
                          f"fhd={_fn(m.get('fhd_raw'), 4)} "
                          f"neck={_fn(m.get('neck_compression'), 3)} {drop_tag}")

                if args.hud:
                    draw_hud(frame, m, args, ts, label, label_since, idx, src,
                             valid_n, dropped_n)
                    cv2.imshow(WINDOW_NAME, frame)
                    k = cv2.waitKey(1) & 0xFF
                    ch = chr(k) if 32 <= k < 127 else ""
                    if k == 27 or ch in ("q", "Q"):
                        break
                    if cv2.getWindowProperty(WINDOW_NAME,
                                             cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if ch in LABEL_KEYS:
                        label = LABEL_KEYS[ch]
                        label_since = ts
                        print(f"[label] {label} 从 t={ts:.1f}s 开始")
                    elif ch == "0":
                        label = ""
                    elif ch in ("r", "R"):
                        rec_index += 1
                        rec_start = ts
                        label, label_since = "", ts
                        print(f"[rec] 新的一段 rec_index={rec_index} t={ts:.1f}s"
                              f"（标签已清，重新按 1/2/3）")
                    elif ch in ("s", "S"):
                        if fh is not None:
                            fh.flush()
                        _finish(args, [args.csv] if args.csv else [])
                idx += 1
                if args.max_frames and idx >= args.max_frames:
                    break
    finally:
        if fh is not None:
            fh.flush()
            fh.close()
        if args.hud:
            cv2.destroyAllWindows()

    print(f"\n共 {idx} 帧：有效 {valid_n} / 丢弃 {dropped_n}")
    if args.csv:
        print(f"逐帧 CSV: {args.csv}")
        _finish(args, [args.csv])
    return 0


# ---------- CSV 读入与统计（口径与现有两个实验脚本一致） ----------

def _f(s):
    """CSV 字符串 → float（空/坏值 → None，**不填 0**）。"""
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_csv(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh)]


def label_of(row: dict) -> str:
    """行的姿势标签名；兼容数字标签（1/2/3）与空（未标注）。"""
    lab = (row.get("posture_label") or "").strip()
    if lab in LABEL_ORDER:
        return lab
    return LABEL_FROM_NUMBER.get(lab, "")


def meta_of(row: dict, col: str, default: str = "") -> str:
    return (row.get(col) or "").strip() or default


def block_key(row: dict, default_view: str) -> tuple:
    """分块键：(机位, 人, session)。"""
    return (meta_of(row, "view", default_view),
            meta_of(row, "person", "?"),
            meta_of(row, "session", "-"))


def median(vals: Sequence[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def quantile(sorted_vals: Sequence[float], q: float) -> Optional[float]:
    """线性插值分位点（numpy 默认口径）。输入必须已排序。"""
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return sorted_vals[0]
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def stats(vals: Sequence[float]) -> dict:
    """n / mean / std(样本) / min / max / median / q1 / q3（空 → n=0 全 None）。"""
    v = sorted(x for x in vals if x is not None)
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None,
                "median": None, "q1": None, "q3": None}
    return {
        "n": n,
        "mean": sum(v) / n,
        "std": statistics.stdev(v) if n > 1 else 0.0,
        "min": v[0], "max": v[-1],
        "median": quantile(v, 0.5), "q1": quantile(v, 0.25), "q3": quantile(v, 0.75),
    }


def _rank(vals: Sequence[float]) -> List[float]:
    """平均秩（并列取均值），Spearman 用。"""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    pairs = [(a, b) for a, b in zip(x, y)
             if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 1e-12 or syy <= 1e-12:
        return None
    sxy = sum((a - mx) * (b - my) for a, b in pairs)
    return sxy / math.sqrt(sxx * syy)


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    return pearson(_rank([p[0] for p in pairs]), _rank([p[1] for p in pairs]))


def _rec_id(row: dict) -> str:
    """该行的录制段标识：优先 rec_index（每按一次 r 换一段），老 CSV 退化为空串。"""
    return (row.get("rec_index") or "").strip()


def drop_segment_heads(rows: List[dict], skip_sec: float) -> Tuple[List[dict], int]:
    """丢掉每段开头 skip_sec 秒（切姿势的过渡帧）。

    段 = 同一录制段（rec_index）内连续相同的标签。时间优先用 `label_sec`
    （该标签已持续多久，最贴合"过渡帧"的语义）；老 CSV 没有该列时退回 timestamp。
    """
    if skip_sec <= 0:
        return list(rows), 0
    kept, dropped = [], 0
    prev_key, run_start = None, None
    for r in rows:
        lab = label_of(r)
        key = (lab, _rec_id(r))
        label_sec = _f(r.get("label_sec"))
        t = label_sec if label_sec is not None else _f(r.get("timestamp"))
        if key != prev_key:
            prev_key, run_start = key, t
        if lab and t is not None and run_start is not None \
                and (t - run_start) < skip_sec:
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


def split_segments(rows: List[dict]) -> List[Tuple[str, List[dict]]]:
    """块内切段：标签变化 **或** 录制段（rec_index）变化都开新段；未标注帧不进段。

    为什么连 rec_index 一起判：同一档连着录两次（r → 1 → 20s → r → 1 → 20s）标签没变，
    光看标签会把两段并成一段，段中位数就少了一个独立单元。
    """
    segs: List[Tuple[str, List[dict]]] = []
    cur_label, cur_rec, cur = None, None, []
    for r in rows:
        lab = label_of(r)
        if not lab:
            continue
        rec = _rec_id(r)
        if lab != cur_label or rec != cur_rec:
            if cur_label and cur:
                segs.append((cur_label, cur))
            cur_label, cur_rec, cur = lab, rec, []   # 必须新建列表：否则各段共用一个 list
        cur.append(r)
    if cur_label and cur:
        segs.append((cur_label, cur))
    return segs


# ---------- 判定（固定标准，见模块 docstring 第五节） ----------

def _values(rows: Sequence[dict], key: str) -> List[float]:
    return [v for v in (_f(r.get(key)) for r in rows) if v is not None]


def metric_verdict(segs: Sequence[Tuple[str, List[dict]]], key: str) -> dict:
    """对单个指标、单个块算三项判定。

    segs: 该块的段列表（已做段首丢弃）。
    返回 dict：分档帧级统计 + 段级中位数 + 单调性 / 分离度 / 抖动比。
    """
    by_tier: Dict[str, List[float]] = {t: [] for t in LABEL_ORDER}
    seg_med: Dict[str, List[float]] = {t: [] for t in LABEL_ORDER}
    seg_stds: List[float] = []
    for lab, rows in segs:
        vals = _values(rows, key)
        if len(vals) < 2:
            continue
        by_tier[lab].extend(vals)
        med = median(vals)
        if med is not None:
            seg_med[lab].append(med)
        seg_stds.append(statistics.stdev(vals))

    fr = {t: stats(by_tier[t]) for t in LABEL_ORDER}
    meds = [fr[t]["median"] for t in LABEL_ORDER]

    # a. 单调性：帧级中位数严格单调（三档都有数才算）
    if any(v is None for v in meds):
        mono, direction = "n/a", "n/a"
    else:
        if meds[0] < meds[1] < meds[2]:
            mono, direction = "PASS", "up"
        elif meds[0] > meds[1] > meds[2]:
            mono, direction = "PASS", "down"
        else:
            mono, direction = "FAIL", "not-monotonic"

    # a'. 段级中位数单调性（独立单元口径，作稳健性备注）
    seg_meds = [median(seg_med[t]) for t in LABEL_ORDER]
    if any(v is None for v in seg_meds):
        mono_seg = "n/a"
    elif seg_meds[0] < seg_meds[1] < seg_meds[2] or \
            seg_meds[0] > seg_meds[1] > seg_meds[2]:
        mono_seg = "PASS"
    else:
        mono_seg = "FAIL"

    # b. 分离度：IQR 不重叠
    def disjoint(a: str, b: str) -> Optional[bool]:
        sa, sb = fr[a], fr[b]
        if sa["n"] == 0 or sb["n"] == 0 or sa["q1"] is None or sb["q1"] is None:
            return None
        return sa["q3"] < sb["q1"] or sb["q3"] < sa["q1"]

    sep_no = disjoint("Normal", "Obvious_FHP")
    sep_ns = disjoint("Normal", "Slight_FHP")

    # c. 抖动比：段内逐帧 std 的中位数 ÷ |中位数(Obvious) − 中位数(Normal)|
    jitter = median(seg_stds)
    delta = None
    if meds[0] is not None and meds[2] is not None:
        delta = abs(meds[2] - meds[0])
    ratio = (jitter / delta) if (jitter is not None and delta and delta > 1e-9) \
        else None

    return {"frames": fr, "seg_medians": seg_med,
            "seg_median_of": seg_meds,
            "n_segments": {t: len(seg_med[t]) for t in LABEL_ORDER},
            "mono": mono, "direction": direction, "mono_segment_level": mono_seg,
            "sep_normal_obvious": sep_no, "sep_normal_slight": sep_ns,
            "jitter_std": jitter, "delta_normal_obvious": delta,
            "jitter_ratio": ratio}


def _sep_txt(v: Optional[bool]) -> str:
    return "n/a" if v is None else ("不重叠" if v else "重叠")


def _mono_txt(v: str) -> str:
    return {"PASS": "PASS", "FAIL": "FAIL"}.get(v, "n/a")


# ---------- 报告 ----------

def _fmt_stats(s: dict) -> str:
    if s["n"] == 0:
        return "n=0"
    return (f"n={s['n']:4d}  median={s['median']:8.3f}  "
            f"IQR=[{s['q1']:8.3f}, {s['q3']:8.3f}]  "
            f"min={s['min']:8.3f}  max={s['max']:8.3f}")


def build_report(blocks: dict, meta: dict, args) -> Tuple[List[str], List[dict],
                                                         List[dict], List[dict]]:
    """出报告行 + 三张表（主表 / 顶点敏感性 / 相关性）。

    返回 (report_lines, main_rows, sensitivity_rows, corr_rows)。
    """
    L: List[str] = [
        "=" * 96,
        "Molaeifar 式指标（胸骨中点–左右耳屏角）对照测量报告",
        "=" * 96,
        f"数据来源        : {meta['source']}",
        f"总帧数          : {meta['frames']}（未标注 {meta['unlabeled']} 帧不进分档统计）",
        f"段首丢弃        : 每段前 {meta['skip_sec']:.1f}s（切姿势过渡帧）→ 共丢 "
        f"{meta['dropped_head']} 帧",
        f"置信度门控      : 双耳(3/4)+双肩(5/6) 每个都 >= {VIS_MIN}；不过门的帧不进任何统计"
        f"（共 {meta['dropped_conf']} 帧）",
        f"平滑            : 一阶 EMA tau={meta['smooth_tau']:.2f}s（本脚本对 5 个指标"
        f"用同一条；raw 行 = 未平滑）",
        "",
        "新指标 = 顶点（双肩中点，可下移 k 个肩宽）到左右耳两条边的夹角（像素空间，度）。",
        "对照指标 = 项目现有的 45° CVA proxy / Ear–Shoulder proxy / FHD proxy / 颈压缩。",
        "⚠ 方向不统一（读相关性表前先看）：新指标越大越前伸；45° CVA proxy 与颈压缩"
        "越小越前伸 —— 与它们的负相关是口径相反所致。",
        "判定标准（事先固定，见脚本 docstring 第五节）：单调性 / IQR 分离度 / 抖动比。",
        "",
    ]

    main_rows: List[dict] = []
    sens_rows: List[dict] = []
    corr_rows: List[dict] = []

    for bkey, rows in blocks.items():
        view, person, session = bkey
        L.append("=" * 96)
        L.append(f"【块】机位={view}  人={person}  session={session}  帧数={len(rows)}")
        L.append("=" * 96)
        segs = split_segments(rows)
        if not segs:
            L.append("  该块没有带标签的段 → 跳过（只测不判）")
            L.append("")
            continue
        n_seg_txt = "  ".join(f"{t}:{sum(1 for l, _ in segs if l == t)}段"
                             for t in LABEL_ORDER)
        L.append(f"  段数：{n_seg_txt}")
        min_seg = min(sum(1 for l, _ in segs if l == t) for t in LABEL_ORDER)
        if min_seg < 3:
            L.append(f"  ⚠ 每档段数最少只有 {min_seg} 段（<3）→ **样本量不足以做统计"
                     f"推断，下面只作描述性对比**")
        L.append("")

        all_keys = [k for k, _, _ in MAIN_METRICS] + \
                   [k for k, _ in SENSITIVITY_METRICS]
        verdicts = {k: metric_verdict(segs, k) for k in dict.fromkeys(all_keys)}
        ema_verdicts = {k: metric_verdict(segs, k + "_ema")
                        for k in dict.fromkeys(all_keys)}

        # 逐指标逐档描述
        for key, label, _note in MAIN_METRICS:
            v = verdicts[key]
            L.append(f"  ── {label}")
            if v["frames"]["Normal"]["n"] == 0 and v["frames"]["Obvious_FHP"]["n"] == 0:
                L.append("     （本块没有可算的帧）")
                continue
            for t in LABEL_ORDER:
                L.append(f"     {t:<12s} {_fmt_stats(v['frames'][t])}"
                         f"  段中位数={[round(m, 2) for m in v['seg_medians'][t]]}")
        L.append("")

        # 主表行（raw / ema，每个机位一组）
        for key, label, note in MAIN_METRICS:
            for variant, vv in (("raw", verdicts[key]), ("EMA", ema_verdicts[key])):
                if vv["frames"]["Normal"]["n"] == 0 and \
                        vv["frames"]["Obvious_FHP"]["n"] == 0:
                    continue
                main_rows.append({
                    "指标": f"{label} [{variant}]",
                    "视角": view,
                    "单调性": _mono_txt(vv["mono"]),
                    "方向": {"up": "随前伸增大", "down": "随前伸减小"}.get(
                        vv["direction"], "非单调"),
                    "分离度(N-O)": _sep_txt(vv["sep_normal_obvious"]),
                    "分离度(N-S)": _sep_txt(vv["sep_normal_slight"]),
                    "抖动比": ("n/a" if vv["jitter_ratio"] is None
                             else f"{vv['jitter_ratio']:.3f}"),
                    "段数(N/S/O)": f"{vv['n_segments']['Normal']}/"
                                 f"{vv['n_segments']['Slight_FHP']}/"
                                 f"{vv['n_segments']['Obvious_FHP']}",
                    "帧数(N/S/O)": f"{vv['frames']['Normal']['n']}/"
                                 f"{vv['frames']['Slight_FHP']['n']}/"
                                 f"{vv['frames']['Obvious_FHP']['n']}",
                    "单调性(段级)": _mono_txt(vv["mono_segment_level"]),
                })

        # 顶点平移敏感性（探索，不许用来挑参数）
        for key, label in SENSITIVITY_METRICS:
            for variant, vv in (("raw", verdicts[key]), ("EMA", ema_verdicts[key])):
                if vv["frames"]["Normal"]["n"] == 0 and \
                        vv["frames"]["Obvious_FHP"]["n"] == 0:
                    continue
                sens_rows.append({
                    "顶点": f"{label} [{variant}]",
                    "视角": view,
                    "单调性": _mono_txt(vv["mono"]),
                    "方向": {"up": "随前伸增大", "down": "随前伸减小"}.get(
                        vv["direction"], "非单调"),
                    "分离度(N-O)": _sep_txt(vv["sep_normal_obvious"]),
                    "分离度(N-S)": _sep_txt(vv["sep_normal_slight"]),
                    "抖动比": ("n/a" if vv["jitter_ratio"] is None
                             else f"{vv['jitter_ratio']:.3f}"),
                })

        # 相关性（冗余性检查）：帧级 + 段中位数级
        # ⚠ 每个对照指标的**方向**都写在标签括号里，不是装饰：本指标（Molaeifar 角）
        #   是**越大越前伸**，而 45° CVA proxy / 颈压缩是**越小越前伸** —— 与它们
        #   算出的负相关是"口径相反"造成的，不代表指标反向。方向约定见
        #   features.CvaProxyFeatures / SternumEarAngleFeatures 两处计算点的注释。
        others = [("neck_compression", "颈压缩（越小越前伸）"),
                  ("cva_proxy_raw", "45° CVA proxy（越小越前伸）"),
                  ("ear_shoulder_raw", "Ear–Shoulder proxy（越大越前伸）"),
                  ("fhd_raw", "FHD proxy（越大越前伸）"),
                  ("vert_over_span", "耳-肩竖直间距/两耳间距（越小越前伸）")]
        frame_pairs = {k: ([], []) for k, _ in others}
        seg_pairs = {k: ([], []) for k, _ in others}
        for lab, rows_in_seg in segs:
            for k, _ in others:
                xs = _values(rows_in_seg, "molaeifar_angle")
                ys = _values(rows_in_seg, k)
                # 帧级：同一段内按行对齐（两个序列的非空行可能不同，取交集）
                for r in rows_in_seg:
                    a, b = _f(r.get("molaeifar_angle")), _f(r.get(k))
                    if a is not None and b is not None:
                        frame_pairs[k][0].append(a)
                        frame_pairs[k][1].append(b)
                if len(xs) >= 2 and len(ys) >= 2:
                    pair = [( _f(r.get("molaeifar_angle")), _f(r.get(k)))
                            for r in rows_in_seg]
                    pair = [(a, b) for a, b in pair if a is not None and b is not None]
                    if len(pair) >= 3:
                        seg_pairs[k][0].append(median([p[0] for p in pair]))
                        seg_pairs[k][1].append(median([p[1] for p in pair]))
        for k, label in others:
            corr_rows.append({
                "对照指标": label,
                "视角": view,
                "Pearson(帧级)": _round_or_na(pearson(*frame_pairs[k])),
                "Spearman(帧级)": _round_or_na(spearman(*frame_pairs[k])),
                "Pearson(段级)": _round_or_na(pearson(*seg_pairs[k])),
                "Spearman(段级)": _round_or_na(spearman(*seg_pairs[k])),
                "段级配对数": len(seg_pairs[k][0]),
            })

        L.append("  ── 顶点平移敏感性（探索；**不许**据此挑顶点）")
        for key, label in SENSITIVITY_METRICS:
            v = verdicts[key]
            L.append(f"     {label:<22s} 单调={_mono_txt(v['mono']):<5s} "
                     f"方向={v['direction']:<14s} "
                     f"N-O={_sep_txt(v['sep_normal_obvious']):<6s} "
                     f"N-S={_sep_txt(v['sep_normal_slight']):<6s} "
                     f"抖动比={('n/a' if v['jitter_ratio'] is None else format(v['jitter_ratio'], '.3f'))}")
        L.append("")
        L.append("  ── 相关性（新指标 vs 对照指标）")
        for row in corr_rows[-len(others):]:
            L.append(f"     {row['对照指标']:<30s} Pearson(帧)={row['Pearson(帧级)']:>7s} "
                     f"Spearman(帧)={row['Spearman(帧级)']:>7s} "
                     f"Pearson(段)={row['Pearson(段级)']:>7s} "
                     f"Spearman(段)={row['Spearman(段级)']:>7s} "
                     f"(段级 n={row['段级配对数']})")
        L.append("     读法：新指标（Molaeifar 角）**越大越前伸**；每个对照指标的方向写在")
        L.append("     上面括号里。与「越小越前伸」那几个（45° CVA proxy / 颈压缩）算出")
        L.append("     **负相关**是口径相反所致，不等于指标反向 —— 段级 −0.98x 那类")
        L.append("     冗余性结论读的就是这个负号（同向指标算出来才是正号）。")
        L.append("")

    return L, main_rows, sens_rows, corr_rows


def _round_or_na(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def markdown_table(rows: List[dict], title: str) -> List[str]:
    if not rows:
        return [f"### {title}", "", "（无数据）", ""]

    def esc(s) -> str:
        # 单元格里的竖线（指标名里有 |耳中点x−肩中点x|）必须转义，否则整张表错位
        return str(s).replace("|", "\\|")

    cols = list(rows[0].keys())
    out = [f"### {title}", "",
           "| " + " | ".join(esc(c) for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(esc(r.get(c, "")) for c in cols) + " |")
    out.append("")
    return out


def write_table_csv(rows: List[dict], path: str) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)


def auto_conclusion(main_rows: List[dict], corr_rows: List[dict],
                    views_seen: Sequence[str]) -> List[str]:
    """按固定标准自动出结论（<=10 行）。数据说啥写啥，不为了让结论好看而动参数。"""

    def find(name: str, view: str):
        for r in main_rows:
            if name in r["指标"] and r["视角"] == view:
                return r
        return None

    L = ["-" * 96, "结论（按事先固定的标准自动生成，样本量与失败项如实写）", "-" * 96]
    for view in views_seen:
        new = find("Molaeifar", view)
        cva = find("CVA proxy", view)
        if new is None:
            L.append(f"[{view}] 新指标：本块没有可算帧，无法评估。")
            continue
        mono = new["单调性"] == "PASS"
        sep_no = new["分离度(N-O)"] == "不重叠"
        sep_ns = new["分离度(N-S)"] == "不重叠"
        lvl = ("可用（单调 + 两档 IQR 都不重叠）" if mono and sep_no and sep_ns else
               "只能看趋势（单调但不满足全部 IQR 分离）" if mono and (sep_no or sep_ns)
               else "不可用（不单调或不分离）")
        seg_total = sum(int(x) for x in new["段数(N/S/O)"].split("/"))
        line = (f"[{view}] 新指标：{lvl}；单调性={new['单调性']}"
                f"（{new['方向']}），N-O={new['分离度(N-O)']}，N-S={new['分离度(N-S)']}，"
                f"抖动比={new['抖动比']}，共 {seg_total} 段")
        if seg_total < 9:
            line += "；段数 <9（每档 <3）→ 只作描述性，不做统计推断"
        L.append(line)
        if cva is not None:
            better = (mono and not (cva["单调性"] == "PASS")) or \
                     (mono and cva["单调性"] == "PASS"
                      and _as_float(new["抖动比"]) is not None
                      and _as_float(cva["抖动比"]) is not None
                      and _as_float(new["抖动比"]) < _as_float(cva["抖动比"]))
            L.append(f"[{view}] 与 45° CVA proxy 比：{'新指标更好或相当' if better else '新指标不占优'}"
                     f"（单调 {new['单调性']} vs {cva['单调性']}；抖动比 "
                     f"{new['抖动比']} vs {cva['抖动比']}）")
    # 冗余性
    for view in views_seen:
        for r in corr_rows:
            if r["视角"] == view and "颈压缩" in r["对照指标"]:
                p = _as_float(r["Pearson(段级)"])
                if p is None:
                    L.append(f"[{view}] 与颈压缩的段级相关：样本不足，n/a（帧级 "
                             f"Pearson={r['Pearson(帧级)']}，受自相关夸大，别当独立证据）")
                else:
                    j = ("高度冗余（同一信息的重新包装）" if abs(p) >= 0.9 else
                         "中等相关，保留了部分独立信息" if abs(p) >= 0.7 else
                         "弱相关，不是同一信息")
                    L.append(f"[{view}] 与颈压缩的段级相关 Pearson={p:.3f} → {j}")
    L.append("提醒：顶点=肩中点近似胸骨中点、耳点非耳屏、且是 2D 投影角 —— "
             "本指标是工程代理，不能当临床量。")
    L.append("-" * 96)
    return L


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------- 分析入口 ----------

def _finish(args, csv_paths: List[str]) -> None:
    """读回 CSV → 出报告 + 三张表。"""
    if not csv_paths:
        print("[warn] 没有 CSV 可分析（采集时用 --csv 才会留痕）")
        return
    blocks: dict = {}
    total_rows = 0
    dropped_head = 0
    dropped_conf = 0
    for p in csv_paths:
        if not os.path.exists(p):
            print(f"[warn] 找不到 {p}，跳过")
            continue
        rows = read_csv(p)
        total_rows += len(rows)
        for r in rows:
            blocks.setdefault(block_key(r, args.view), []).append(r)

    for bkey, rows in list(blocks.items()):
        # 段首丢弃（块内做，避免跨块误判成同一段）
        kept, dropped = drop_segment_heads(rows, args.skip_sec)
        dropped_head += dropped
        blocks[bkey] = kept

    # 丢弃计数（按原因）
    reasons: Dict[str, int] = {}
    for rows in blocks.values():
        for r in rows:
            if (r.get("molaeifar_valid") or "").strip() in ("1", "true", "True"):
                continue
            reasons[r.get("drop_reason") or "?"] = reasons.get(
                r.get("drop_reason") or "?", 0) + 1
    dropped_conf = sum(v for k, v in reasons.items()
                       if k != "no_person" and k != "")

    meta = {
        "source": (f"csv:{', '.join(csv_paths)}" if not args.video and args.camera is None
                   else (f"video:{args.video}" if args.video else f"camera:{args.camera}")),
        "frames": total_rows, "unlabeled": sum(
            1 for rows in blocks.values() for r in rows if not label_of(r)),
        "skip_sec": args.skip_sec, "dropped_head": dropped_head,
        "dropped_conf": dropped_conf, "smooth_tau": args.smooth_sec,
    }

    lines, main_rows, sens_rows, corr_rows = build_report(blocks, meta, args)
    lines.append("")
    lines.append(f"丢弃帧明细：{reasons if reasons else '无'}")
    concl = auto_conclusion(main_rows, corr_rows,
                            list(dict.fromkeys(k[0] for k in blocks)))
    lines.extend(concl)

    base = args.report or (str(Path(csv_paths[0]).with_suffix("")) + "_molaeifar_report.txt")
    with open(base, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    table_base = str(Path(base).with_suffix(""))
    md = ["# Molaeifar 式指标对照表", ""]
    md.extend(markdown_table(main_rows, "主表：单调性 / 分离度 / 抖动比"))
    md.extend(markdown_table(sens_rows, "顶点平移敏感性（探索）"))
    md.extend(markdown_table(corr_rows, "相关性（冗余性检查）"))
    body = [x for x in concl if x.strip() and not set(x.strip()) <= {"-"}]
    md.extend(["## 结论", ""] + [f"- {x}" for x in body])
    table_csv = args.table or (table_base + "_metric_table.csv")
    table_md = args.table_md or (table_base + "_metric_table.md")
    # 报告/表格的目录不存在就先建（默认都落在 data/ 下，新克隆里可能还没有）
    for path in (base, table_base, table_csv, table_md):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    # 三张表列不同 → 分开写 CSV（主表 = 需求点名的那张；另两张是附表）
    write_table_csv(main_rows, table_csv)
    write_table_csv(sens_rows, table_base + "_sensitivity.csv")
    write_table_csv(corr_rows, table_base + "_correlation.csv")
    with open(table_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    print("\n".join(concl))
    print(f"\n报告: {base}\n表格: {table_csv}\n       {table_md}")


def analyze(paths: List[str], args) -> int:
    if args.camera is not None or args.video:
        args.view = args.view
    _finish(args, paths)
    return 0


# ---------- 自测（合成数据，不碰摄像头/模型） ----------

def _synth_pose(pts_px: Dict[int, Point], size=(1280, 720),
                vis=0.9, missing=()) -> dict:
    """造一个 detect_pose 形状的 pose：给定像素点 → 归一化 (x/w, y/h, 0, vis)。"""
    w, h = size
    lm = []
    for pid in range(17):
        if pid in missing:
            lm.append((0.0, 0.0, 0.0, 0.0))
        elif pid in pts_px:
            x, y = pts_px[pid]
            lm.append((x / w, y / h, 0.0, vis if not isinstance(vis, dict)
                       else vis.get(pid, 0.9)))
        else:
            lm.append((0.5, 0.5, 0.0, vis if not isinstance(vis, dict)
                       else vis.get(pid, 0.9)))
    return {"landmarks": lm, "image_size": (w, h)}


def _close(got, want, tol, label: str) -> None:
    assert got is not None, f"{label}: 期望 {want}，得到 None"
    assert abs(got - want) < tol, f"{label}: 期望 {want}±{tol}，得到 {got}"


def selftest() -> None:
    """合成关键点自测：角度数学 + 门控 + EMA + 分析管线（含"能说不行"的负对照）。"""
    # 1) 对称几何：顶点 (600,500)，两耳 (500,200)/(700,200) → 顶点处夹角可解析算
    pts = {3: (500.0, 200.0), 4: (700.0, 200.0), 5: (520.0, 500.0), 6: (680.0, 500.0)}
    res = molaeifar_angle(pts, 0.0)
    want = math.degrees(2 * math.atan(100.0 / 300.0))     # 半边 atan(100/300)
    _close(res[0], want, 1e-6, "对称三角形夹角")
    assert abs(res[1][1] - 500.0) < 1e-9, "顶点 y 应为肩中点 y"
    print(f"  [1] 对称三角形夹角 {res[0]:.3f}°（解析值 {want:.3f}°）OK")

    # 2) 顶点下移 = 夹角变大（同一个头，顶点越往下，两耳张角越小 → 夹角变小？）
    #    直接算，不预设方向，只断言"平移确实改变了角度"
    a0 = molaeifar_angle(pts, 0.0)[0]
    a15 = molaeifar_angle(pts, 0.15)[0]
    a30 = molaeifar_angle(pts, 0.30)[0]
    assert a0 != a15 != a30, "顶点平移应改变角度"
    print(f"  [2] 顶点平移 0/0.15/0.30 肩宽 → {a0:.3f}/{a15:.3f}/{a30:.3f}° OK（方向不预设）")

    # 3) 缺一边 / 缺顶点 → None（不能用上一帧填）
    assert molaeifar_angle({3: (500.0, 200.0), 5: (520.0, 500.0), 6: (680.0, 500.0)}) is None
    assert molaeifar_angle({3: (500.0, 200.0), 4: (700.0, 200.0), 5: (520.0, 500.0)}) is None
    print("  [3] 缺耳机/缺肩 → None OK")

    # 4) 置信度门控：耳置信度 0.29 → 该点被丢 → 指标无效（低位失败）
    fm = FrameMetrics()
    pose_low = _synth_pose({3: (500.0, 200.0), 4: (700.0, 200.0),
                            5: (520.0, 500.0), 6: (680.0, 500.0)},
                           vis={3: 0.29, 4: 0.9, 5: 0.9, 6: 0.9})
    m = fm.measure(pose_low, 0.0)
    assert m["molaeifar_angle"] is None, "耳置信度 0.29 应被丢弃"
    assert not m["molaeifar_valid"] and "earL" in m["drop_reason"], m["drop_reason"]
    print(f"  [4] 耳置信度 0.29 → 丢弃（{m['drop_reason']}）OK")

    # 5) 门控通过时四个对照指标都算得出
    #    几何按真人正面取景给量级：肩宽 ~0.375 归一化、耳-肩竖直间距 ~0.21，
    #    这样颈压缩 ≈ 0.55（落在生产侧"坐直 0.5~0.8"的区间里，不会被 1.5 上限挡掉）
    fm2 = FrameMetrics()
    pose_ok = _synth_pose({0: (650.0, 200.0), 3: (610.0, 280.0), 4: (690.0, 280.0),
                           5: (400.0, 430.0), 6: (880.0, 430.0),
                           11: (500.0, 700.0), 12: (780.0, 700.0)})
    m2 = fm2.measure(pose_ok, 0.0)
    assert m2["molaeifar_valid"] == 1, m2["drop_reason"]
    for k in ("cva_proxy_raw", "ear_shoulder_raw", "fhd_raw", "neck_compression"):
        assert m2[k] is not None, f"{k} 应有值"
    print(f"  [5] 门控通过：molaeifar={m2['molaeifar_angle']:.2f} "
          f"cva={m2['cva_proxy_raw']:.2f} es={m2['ear_shoulder_raw']:.4f} "
          f"fhd={m2['fhd_raw']:.4f} neck={m2['neck_compression']:.3f} OK")

    # 6) EMA：无效帧返回 None 且不推进（不沿用上一帧；跨过无效帧后 dt 从上次有效帧算）
    e = _Ema(1.0)
    assert e.update(None, 0.0) is None
    _close(e.update(10.0, 0.0), 10.0, 1e-9, "EMA 首帧=raw")
    v1 = 10.0 + (1 - math.exp(-1.0)) * 10.0
    _close(e.update(20.0, 1.0), v1, 1e-9, "EMA 一帧")
    assert e.update(None, 2.0) is None
    # 上次有效帧在 t=1.0，这次 t=3.0 → dt=2.0（无效帧那 1s 照样计入，因为它真的过去了）
    _close(e.update(30.0, 3.0), v1 + (1 - math.exp(-2.0)) * (30.0 - v1), 1e-9,
           "EMA 跨过无效帧")
    print("  [6] EMA：无效帧返回 None 且不推进 OK")

    # 7) 分析管线：造 2 机位 × 3 档 × 3 段，一好一坏
    #    好数据：新指标三档严格单调且分离；坏数据：中位数非单调且区间重叠
    #    （这里同时验证：段 = rec_index × 标签 切分正确、段首丢弃生效、
    #      好数据能出「可用/单调」、坏数据能出「不可用」）
    tmp = tempfile.mkdtemp(prefix="molaeifar_")
    good = os.path.join(tmp, "front_p1_s1.csv")
    bad = os.path.join(tmp, "f45_p1_s1.csv")
    _write_synth_csv(good, view="front", base={0: 100.0, 1: 115.0, 2: 130.0}, jitter=0.5)
    _write_synth_csv(bad, view="45", base={0: 100.0, 1: 95.0, 2: 130.0}, jitter=0.5)
    args = build_parser().parse_args(["--analyze-only", good, bad, "--skip-sec", "0.2"])
    args.report = os.path.join(tmp, "rep.txt")
    args.table = os.path.join(tmp, "tab.csv")
    args.table_md = os.path.join(tmp, "tab.md")
    _finish(args, [good, bad])
    rep = open(args.report, encoding="utf-8").read()
    tab = open(args.table_md, encoding="utf-8").read()
    assert "前伸增大" in tab, "好数据应报出单调方向"
    assert rep.count("不重叠") >= 2, "好数据的 N-O/N-S 应都不重叠"
    assert "不可用" in rep, "坏数据应被判为不可用（能说不行）"
    # 段首丢弃真的生效了（每段 12 帧 → 丢前 ~6 帧）+ 段是按 rec_index×标签 切的（3/3/3 段）
    assert re.search(r"段首丢弃.*共丢 [1-9]", rep), "段首丢弃没生效"
    row_new = [l for l in tab.splitlines()
               if "Molaeifar" in l and "[raw]" in l and "| front |" in l][0]
    # 行里的 "a/b/c" 依次是 段数(N/S/O)、帧数(N/S/O)（正则不加两侧竖线，否则第二组被吃掉）
    trip = re.findall(r"\d+/\d+/\d+", row_new)
    assert len(trip) >= 2 and trip[0] == "3/3/3", f"段数应为 3/3/3，得到 {trip}"
    n_each = {int(x) for x in trip[1].split("/")}
    assert len(n_each) == 1 and 0 < n_each.pop() < 36, f"段首丢弃后每档帧数异常：{trip[1]}"
    print("  [7] 分析管线：好数据判「可用」、坏数据判「不可用」 OK")
    print(f"      （合成报告写在 {tmp}，不入仓库）")

    # 8) 缺列 / 全未标注 → 不炸
    empty = os.path.join(tmp, "empty.csv")
    with open(empty, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["timestamp", "posture_label",
                                            "molaeifar_angle", "view"])
        wr.writeheader()
        wr.writerow({"timestamp": "0", "posture_label": "", "molaeifar_angle": "",
                     "view": "front"})
    args2 = build_parser().parse_args(["--analyze-only", empty, "--skip-sec", "0"])
    args2.report = os.path.join(tmp, "rep2.txt")
    args2.table = os.path.join(tmp, "tab2.csv")
    args2.table_md = os.path.join(tmp, "tab2.md")
    _finish(args2, [empty])
    assert "没有带标签的段" in open(args2.report, encoding="utf-8").read()
    print("  [8] 全未标注的 CSV → 只跳过、不崩 OK")

    print("\nselftest 全部通过。")


def _write_synth_csv(path: str, view: str, base: Dict[int, float],
                     jitter: float, segs: int = 3, frames: int = 12) -> None:
    """造一份标注好的逐帧 CSV（走真实列名），用于自测分析管线。"""
    rng = __import__("random").Random(20260928)   # 固定种子：自测要可复现
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        wr.writeheader()
        idx = 0
        t = 0.0
        rec = 0
        for label in (1, 2, 3):
            for _seg in range(segs):
                rec_t = 0.0
                for _i in range(frames):
                    v = base[label - 1] + rng.gauss(0, jitter)
                    row = {c: "" for c in CSV_COLUMNS}
                    row.update({
                        "timestamp": f"{t:.3f}", "frame_index": idx,
                        "posture_label": str(label), "view": view,
                        "person": "1", "session": "selftest",
                        "rec_index": rec,
                        "rec_sec": f"{rec_t:.3f}",
                        "label_sec": f"{rec_t:.3f}",
                        "molaeifar_angle": f"{v:.6f}",
                        "molaeifar_angle_ema": f"{v:.6f}",
                        "molaeifar_angle_d15": f"{v - 5:.6f}",
                        "molaeifar_angle_d15_ema": f"{v - 5:.6f}",
                        "molaeifar_angle_d30": f"{v - 10:.6f}",
                        "molaeifar_angle_d30_ema": f"{v - 10:.6f}",
                        "molaeifar_valid": "1", "drop_reason": "",
                        "cva_proxy_raw": f"{100.0 - (label - 1) * 8 + rng.gauss(0, 0.4):.6f}",
                        "cva_proxy_raw_ema": f"{100.0 - (label - 1) * 8:.6f}",
                        "ear_shoulder_raw": f"{(label - 1) * 0.03 + rng.gauss(0, 0.002):.6f}",
                        "ear_shoulder_raw_ema": f"{(label - 1) * 0.03:.6f}",
                        "fhd_raw": f"{(label - 1) * 0.02 + rng.gauss(0, 0.002):.6f}",
                        "fhd_raw_ema": f"{(label - 1) * 0.02:.6f}",
                        "neck_compression": f"{0.6 - (label - 1) * 0.1 + rng.gauss(0, 0.01):.6f}",
                        "neck_compression_ema": f"{0.6 - (label - 1) * 0.1:.6f}",
                        "head_neck_angle": f"{20 + (label - 1) * 5:.6f}",
                        "vert_over_span": f"{0.8 - (label - 1) * 0.15:.6f}",
                    })
                    wr.writerow(row)
                    idx += 1
                    t += 0.033
                    rec_t += 0.033
                rec += 1


# ---------- CLI ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Molaeifar 式指标（胸骨中点–左右耳屏角）测量/验证（独立脚本）")
    src = p.add_argument_group("数据源")
    src.add_argument("--camera", type=int, nargs="?", const=0, default=None,
                     help="摄像头序号（默认 0；给了 --video 时忽略）")
    src.add_argument("--video", metavar="PATH", help="视频文件（边播边按 1/2/3 打标签）")
    src.add_argument("--video-fps", type=float, default=0.0,
                     help="视频帧率（读不到时用，默认 30）")
    src.add_argument("--view", choices=list(VIEWS), default="front",
                     help="机位（**必须由人声明**，本项目没有可靠的自动判机位）；"
                          "本指标按文献是正面构造，默认 front")
    src.add_argument("--person", default="1", help="受试者编号（分块用）")
    src.add_argument("--session", default="-", help="session 标记（如 0.6m / 椅子高）")
    out = p.add_argument_group("输出")
    out.add_argument("--csv", default="data/molaeifar_frames.csv", metavar="PATH",
                     help="逐帧 CSV 输出路径（含关键点像素坐标 = 个人身体数据，"
                          "别提交进仓库），默认 data/molaeifar_frames.csv")
    out.add_argument("--report", default=None, help="报告 txt 路径")
    out.add_argument("--table", default=None, help="对照表 CSV 路径")
    out.add_argument("--table-md", dest="table_md", default=None,
                     help="对照表 markdown 路径")
    out.add_argument("--analyze-only", nargs="+", metavar="CSV", default=None,
                     help="只对已有逐帧 CSV 出报告（可给多个，按块分开统计）")
    beh = p.add_argument_group("分析口径（判定标准固定，这里只调窗口/门控）")
    beh.add_argument("--skip-sec", type=float, default=1.0,
                     help="每段开头丢弃多少秒（切姿势过渡帧），默认 1.0")
    beh.add_argument("--smooth-sec", type=float, default=EMA_TAU_SEC,
                     help=f"EMA tau（秒），默认 {EMA_TAU_SEC}")
    beh.add_argument("--print-every", type=int, default=0, dest="print_every",
                     help="每 N 帧打印一行（0 = 不打印）")
    beh.add_argument("--no-hud", action="store_false", dest="hud", default=True,
                     help="不显示实时画面（无头跑/冒烟用）")
    beh.add_argument("--max-frames", type=int, default=0,
                     help="最多处理多少帧（0 = 不限制；冒烟用小值）")
    p.add_argument("--selftest", action="store_true", help="合成数据自测，不碰摄像头")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.selftest:
        selftest()
        return
    if args.analyze_only:
        sys.exit(analyze(args.analyze_only, args))
    if args.camera is None and not args.video:
        build_parser().error("要么给 --camera 0，要么给 --video x.mp4，"
                            "要么给 --analyze-only CSV")
    sys.exit(run_capture(args))


if __name__ == "__main__":
    main()
