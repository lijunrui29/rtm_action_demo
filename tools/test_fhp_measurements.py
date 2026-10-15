"""
FHP（Forward Head Posture，头前伸）三种测量法的**独立验证脚本**。

用法（在 rtm_demo/ 目录下运行）：
    python tools/test_fhp_measurements.py --camera 0
    python tools/test_fhp_measurements.py --camera 0 --c7-mode near --ear right
    python tools/test_fhp_measurements.py --selftest                 # 合成关键点自测数学，不碰摄像头
    python tools/test_fhp_measurements.py --camera 0 --csv data/fhp.csv   # 顺带把逐帧读数写成 CSV（写进 data/）

ESC 退出；窗口里左上角实时显示三个测量值 + 用到的关键点。

本文件只做「测量 + 显示」：不判定、不提醒、不写 profile，也**不修改**任何现有模块
（capture / pose / features / decision / output 一行未动）。它 import 现有的
CameraCapture、detect_pose、FrameRenderer 的骨架绘制、以及 ErgonomicRiskFeatures
（仅作交叉核对），保证与运行时用的是同一套推理。

================================ 先读这一节 ================================

一、关键点现实：**这个模型没有 C7**

    RTMPose 在本项目里输出 COCO 17 点（见 pose_estimation.py 的接口约定）：
        0 鼻, 1/2 眼, 3/4 耳, 5/6 肩, 7/8 肘, 9/10 腕, 11/12 髋, 13/14 膝, 15/16 踝
    **没有 C7（第 7 颈椎棘突）**。而标准 CVA 与标准 FHD 的定义都以 C7 为顶点/
    参考点。所以本文件一律用**肩点当 C7 代理（下面记作 C7proxy）**，并且画面上
    明确写 `C7: N/A (not in COCO-17)` 再把代理点坐标列出来 —— 这是 proxy，
    **不是标准 CVA / 标准 FHD**：
      - 真实 C7 在肩点**后方且更靠上**，方向偏差稳定，但会带来一个固定的量值偏移，
        因此本文件的输出**不能拿去和文献阈值（CVA < 50° 之类）直接比**；
      - 生产代码里 ErgonomicRiskFeatures.cva_deg 用的是同一个肩点近似
        （features.py 的「C7 近似（2026-09-01，思路一）」），本文件把它并列显示，
        方便交叉核对两者算的是不是同一件事。

二、视角前提：这三个量只在**侧视**下有本来的含义

    标准 CVA / FHD 定义在**矢状面**，量的是头相对躯干的**前后**位移。本文件三条
    式子全都在图像平面里算，所以：
      - **正对摄像头**：前后位移落在**深度方向**，画面里根本看不见。此时
        ear -> C7proxy 近乎竖直 → CVA 恒 ≈ 90°（永远读作"正常"）；两个 x 向位移
        量测到的是**左右侧倾**，不是前伸。**正对视角下这三个数都抓不到 FHP。**
      - **正侧方**（人相对相机转 ~90°）：图像 x 轴 ≈ 真实前后方向，三条式子才有
        本来的含义。
      - **45°**：x 位移 = sin45≈0.71 的前后分量 + cos45≈0.71 的左右分量，是
        "有信号但被污染"的中间态 —— 能看出趋势，不能当量值。
    所以**要验证 FHP，请侧对摄像头坐**，或者固定一个斜机位后**只比较同一机位下
    的相对变化**。本文件不做任何视角补偿（补偿要靠真值标定，不是这个脚本该干的
    事）；画面 `view:` 行只给一个粗判据（见下）。

三、坐标空间：用像素，不用归一化坐标

    landmarks 是按 px/w、py/h **分别**归一化的（pose_estimation.py:230）。直接在
    归一化坐标里算角度会把画面宽高比烙进结果：tan(θ_prod) = tan(θ_img) × w/h，
    1280x720（1.778）下一个 45° 的真实角度会读成 60.6°。生产
    ErgonomicRiskFeatures 正是走归一化坐标算的（_compute_frame），所以它的
    cva_deg **不是图像平面真角**；本文件把两者并列显示，差值就是这个宽高比效应。
    本文件先换算成**像素**（浮点，不用 landmarks_to_pixels 的 int 截断）再算：
    角度得到图像平面真角；两个比值两两都是 x 向像素，天然免疫宽高比。

四、尺度（除爆问题）

    两个位移量都要除以一个尺度才无量纲。取哪个尺度是先要解决的坑：
        sh_span = |p5 - p6|（欧氏，像素）
        trunk   = |sh_mid - hip_mid|（欧氏，像素；需要髋）
    正对时 sh_span/trunk ≈ 0.9~1.1；**正侧方时两肩在画面上几乎重合** → sh_span
    塌缩到 ≈0.05~0.2（这正是项目里 NECK_COMPRESSION_MAX 那条教训）。而侧视恰恰
    是这三个量唯一有意义的视角 —— 所以：
        sh_span >= SCALE_MIN_RATIO(0.35) × trunk  →  用 sh_span（画面标 "shoulder width"）
        否则（人转侧了，肩宽被投影压扁）          →  退回 trunk（标 "trunk (fallback)"）
        髋不可见 + 肩宽又塌缩                      →  三个比值全 N/A，并说明原因
    这与 features.py 里 FeatureExtractor 的「兜底尺度 = max(肩宽, 耳-肩竖直间距)」
    是同一类处理。

============================= 四条式子的定义 ===============================

    (1) CVA  = 向量 (C7proxy -> ear) 与水平线的夹角（度，0~90）。
               atan2(|dy|, |dx|)，像素空间。**越小越前伸。**
               C7proxy 由 --c7-mode 选：mid = 双肩中点（默认）/ near = 所选耳
               同侧的肩。标准定义顶点在 C7，这里是肩点代理（见第一节）。

    (2) FHD  = |x_ear - x_C7proxy| / scale           （归一化，**不是厘米**）
               标准 FHD = tragus 到**过 C7 的铅垂线**的水平距离。单摄拿不到真实
               距离，所以只用 scale 归一化，输出无量纲比值 —— 即「标准 FHD 的
               肩点代理 + 尺度归一化版」。

    (3) Ear-Shoulder = |x_ear - x_shoulder_sameside| / scale
               与 (2) 的**唯一区别是参考点**：用所选耳**同侧**的肩（acromion），
               而不是双肩中点。注意：正侧方坐下时两肩在 x 上几乎重合，所以在
               **最该用它俩的那个视角下，(2) 和 (3) 的数值会非常接近** —— 它们
               不是两个独立测量，是同一个测量取两种参考点。真正独立的那一维是
               下面这一条。

    diag: Ear->trunk = 耳到「肩中点 → 髋中点」这条躯干轴的**垂距** / scale。
               这是"头相对躯干"的另一个方向：(2)(3) 取的是水平分量，这一条取的是
               相对躯干轴的垂距。单独一行显示，供你比较哪一维更好用。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 本文件在 tools/ 下，上一层才是仓库根（features / decision / capture 在那儿）
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2                                                       # noqa: E402
import numpy as np                                               # noqa: E402

from capture import CameraCapture                                 # noqa: E402
from pose_estimation import detect_pose                           # noqa: E402
from features import ErgonomicRiskFeatures                        # noqa: E402
from output import FrameRenderer                                  # noqa: E402

# COCO-17 点编号 → 名字（编号来源见 pose_estimation.py 的接口约定）
KP_NAMES = {
    0: "nose", 1: "eyeL", 2: "eyeR", 3: "earL", 4: "earR",
    5: "shL", 6: "shR", 7: "elbL", 8: "elbR", 9: "wrL", 10: "wrR",
    11: "hipL", 12: "hipR", 13: "kneeL", 14: "kneeR", 15: "ankL", 16: "ankR",
}

EAR_IDS = (3, 4)
SH_IDS = (5, 6)
HIP_IDS = (11, 12)
USED_IDS = EAR_IDS + SH_IDS + HIP_IDS

VIS_MIN = 0.3           # 与 features.visibility_min / 骨架的最低档一致
SCALE_MIN_RATIO = 0.35  # sh_span/trunk 低于此值 = 肩宽被投影压扁 → 尺度退回 trunk

# view 粗判据的分界（两个分母各一套）。**它分不清"人转侧"和"相机俯仰"**，
# 只在同一次坐定会话内可比，绝对值没有意义、只看落在哪一档。
#   sh/trunk   ：正对 ~0.9~1.1，正侧方 ~0.05~0.2      -> 分界 0.75 / 0.45
#   sh/ear-vert：髋不可见时的退路。正对 ~2.2（肩宽归一化 0.56*1280=717px ÷ 耳肩
#                竖直 0.45*720=324px），正侧方 ~0.04~0.83（肩宽塌缩到 0.01~0.21）
#                -> 分界 1.5 / 0.9。数来自 CLAUDE.md「侧身」那批实拍读数。
VIEW_CUTS = {"sh/trunk": (0.75, 0.45), "sh/ear-vert": (1.5, 0.9)}

# OpenCV 的 Hershey 字体只有 ASCII（画 "°" 会出乱码，中文更不行）→ 画面上用 "deg"
DEG = "deg"

Point = Tuple[float, float]

# 画面颜色（BGR）
_C_OK = (60, 200, 60)      # 可算出的读数（绿）
_C_NA = (150, 150, 150)    # N/A（灰）
_C_HEAD = (255, 220, 180)  # 表头
_C_DIM = (170, 170, 170)   # 次要说明文字
_C_C7 = (255, 0, 255)      # C7 代理点（品红）
_C_EAR = (0, 215, 255)     # 耳（黄，与 output.py 骨架的耳色一致）
_C_REF = (190, 190, 190)   # 参考线（铅垂线 / 水平线 / 躯干轴）
_C_CVA_LEG = (0, 255, 0)   # CVA 的斜边（ear->C7proxy）
_C_ES_LEG = (255, 128, 0)  # ear->同侧肩 的连线
_C_FHD_GAP = (0, 140, 255)  # FHD 的水平间距


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FHP 三种测量法（CVA / FHD / Ear-Shoulder）的独立验证；只显示，不改现有逻辑")
    p.add_argument("--camera", type=int, default=0, metavar="INDEX",
                   help="本地摄像头编号，默认 0")
    p.add_argument("--c7-mode", choices=("mid", "near"), default="mid",
                   help="C7 代理取哪个肩：mid=双肩中点（默认）/ near=所选耳同侧的肩")
    p.add_argument("--ear", choices=("auto", "left", "right"), default="auto",
                   help="用哪只耳：auto=取置信度高的那只（默认；侧视时远侧耳常"
                        "掉到置信度阈值以下）/ left=3 / right=4")
    p.add_argument("--smooth-sec", type=float, default=1.0, metavar="SEC",
                   help="画面显示的读数取最近 SEC 秒的**中位数**（抗单帧抖动）；"
                        "0 = 显示瞬时值。默认 1.0")
    p.add_argument("--csv", default=None, metavar="PATH",
                   help="把逐帧读数写成 CSV（默认不写，建议写进 data/，"
                        "如 --csv data/fhp.csv）。注意：真人数据别提交进仓库")
    p.add_argument("--print", dest="console", action="store_true",
                   help="每 30 帧在控制台打一行读数（含瞬时值、方向与视角判据）")
    p.add_argument("--no-skeleton", dest="draw_skeleton", action="store_false",
                   help="不画姿态骨架（只画本测量用到的点/参考线）")
    p.set_defaults(draw_skeleton=True)
    p.add_argument("--selftest", action="store_true",
                   help="用合成关键点自测本文件的数学，不打开摄像头")
    return p


# ---------------------------------------------------------------------------
# 几何小工具（纯函数，None 安全）
# ---------------------------------------------------------------------------

def _pt(landmarks, pid: int, w: float, h: float,
        vis_min: float = VIS_MIN) -> Optional[Point]:
    """取 pid 的**像素**坐标（浮点）。越界或置信度 < vis_min → None。

    不用 pose_estimation.landmarks_to_pixels 是因为它返回 int（截断到整像素），
    而这里的信号量级偏小，保留亚像素更稳。
    """
    if pid >= len(landmarks):
        return None
    x, y, _z, vis = landmarks[pid]
    if vis < vis_min:
        return None
    return (x * w, y * h)


def _mid(pts: List[Optional[Point]]) -> Optional[Point]:
    """若中点的中点（忽略 None）；全 None 返回 None。"""
    pts = [p for p in pts if p is not None]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _median(values: List[float]) -> Optional[float]:
    """小数组的中位数（不依赖 statistics）。空列表 → None。"""
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _perp_dist(p: Point, a: Point, b: Point) -> Optional[float]:
    """点 p 到直线 a-b 的垂距；a、b 重合（退化）时返回 None。"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    if n <= 1e-9:
        return None
    return abs(dy * (p[0] - a[0]) - dx * (p[1] - a[1])) / n


def _pick_ear(pts: Dict[int, Point], vis: Dict[int, float],
              mode: str) -> Optional[int]:
    """选耳点编号。auto = 置信度高的那只（侧视时远侧耳常掉到 VIS_MIN 以下）。

    强制指定 left/right 而那只耳又不可见时返回 None → 上层显示 N/A（不编数）。
    """
    if mode == "left":
        cand = [3] if 3 in pts else []
    elif mode == "right":
        cand = [4] if 4 in pts else []
    else:
        cand = [pid for pid in EAR_IDS if pid in pts]
    if not cand:
        return None
    return max(cand, key=lambda pid: vis.get(pid, 0.0))


def _c7_proxy(pts: Dict[int, Point], ear_pid: int,
              mode: str) -> Optional[Point]:
    """C7 代理点 —— **不是真的 C7**（COCO-17 没有 C7，见模块开头第一节）。

    mid  = 双肩中点；只看得见一侧肩时退化为那一侧（features.py 同样处理）。
    near = 所选耳**同侧**的肩（earL 3 -> shL 5，earR 4 -> shR 6）。
    """
    if mode == "near":
        return pts.get(5 if ear_pid == 3 else 6)
    return _mid([pts.get(5), pts.get(6)])


# ---------------------------------------------------------------------------
# 核心：从一帧 pose 算三（四）个量
# ---------------------------------------------------------------------------

def measure(pose: Optional[dict], c7_mode: str, ear_mode: str) -> dict:
    """从一帧 pose 算三种测量 + 一条 diag。缺关键点 → 对应的值是 None（显示 N/A）。

    返回的 'pts' 只含**通过置信度过滤**的点（像素坐标，画面上会标出来核对）。
    本函数不抛异常：任何一步取不到点就留 None，由调用方显示 N/A。
    """
    m: dict = {
        "cva_deg": None, "fhd": None, "ear_shoulder": None, "ear_trunk": None,
        "scale_px": None, "scale_src": None, "ear_pid": None,
        "dx_fhd_px": None, "dx_es_px": None,
        "sh_span_px": None, "trunk_px": None, "view_ratio": None, "view_src": None,
        "c7": None, "sh_mid": None, "hip_mid": None,
        "pts": {}, "vis": {}, "note": None,
    }
    if not pose:
        m["note"] = "no person detected"
        return m
    landmarks = pose.get("landmarks")
    size = pose.get("image_size")
    if not landmarks or not size:
        m["note"] = "empty pose result"
        return m
    w, h = size

    # --- 取点：只用 COCO-17 里确实有的 3/4（耳）、5/6（肩）、11/12（髋） ---
    m["vis"] = {pid: (landmarks[pid][3] if pid < len(landmarks) else 0.0)
                for pid in USED_IDS}
    pts = {}
    for pid in USED_IDS:
        p = _pt(landmarks, pid, w, h)
        if p is not None:
            pts[pid] = p
    m["pts"] = pts

    # --- 尺度：先算肩宽与躯干长，再决定用哪个（侧身会把肩宽投影压扁） ---
    sh_span = _dist(pts[5], pts[6]) if (5 in pts and 6 in pts) else None
    sh_mid = _mid([pts.get(5), pts.get(6)])
    hip_mid = _mid([pts.get(11), pts.get(12)])
    trunk = (_dist(sh_mid, hip_mid)
             if (sh_mid is not None and hip_mid is not None) else None)
    m["sh_span_px"], m["trunk_px"] = sh_span, trunk
    m["sh_mid"], m["hip_mid"] = sh_mid, hip_mid

    if sh_span is not None and (trunk is None or sh_span >= SCALE_MIN_RATIO * trunk):
        scale, src = sh_span, "shoulder width"
    elif trunk is not None:
        scale, src = trunk, f"trunk (fallback: shoulder width collapsed)"
    elif sh_span is not None:
        scale, src = sh_span, "shoulder width (degraded: no hip for fallback)"
    else:
        scale, src = None, None
    if scale is not None and scale <= 1e-6:
        scale, src = None, None     # 退化尺度（点重合）→ 宁可 N/A，不除爆
    m["scale_px"], m["scale_src"] = scale, src

    # --- 耳 + C7 代理 ---
    ear_pid = _pick_ear(pts, m["vis"], ear_mode)
    if ear_pid is None:
        m["note"] = f"no ear point with vis >= {VIS_MIN}"
        return m
    m["ear_pid"] = ear_pid
    ear = pts[ear_pid]
    c7 = _c7_proxy(pts, ear_pid, c7_mode)      # 肩点代理，见模块开头第一节
    m["c7"] = c7

    # --- 视角粗判据：优先 sh/trunk；髋不可见（坐着常被桌子挡掉）时退路是用
    # 耳-肩竖直间距当分母（它近似垂直，侧身不塌缩，见 CLAUDE.md「侧身」）---
    if trunk is not None and trunk > 1e-6 and sh_span is not None:
        m["view_ratio"], m["view_src"] = sh_span / trunk, "sh/trunk"
    elif sh_span is not None:
        ear_sh_vert = abs(sh_mid[1] - ear[1]) if sh_mid is not None else None
        if ear_sh_vert is not None and ear_sh_vert > 1e-6:
            m["view_ratio"], m["view_src"] = sh_span / ear_sh_vert, "sh/ear-vert"

    # --- (1) CVA：ear -> C7proxy 与水平线的夹角（像素空间 = 图像平面真角）---
    if c7 is not None and _dist(ear, c7) > 1e-9:
        m["cva_deg"] = math.degrees(
            math.atan2(abs(ear[1] - c7[1]), abs(ear[0] - c7[0])))

    # --- (2)(3) 两个归一化位移 + diag（都要尺度）---
    if scale is not None:
        if c7 is not None:
            m["dx_fhd_px"] = ear[0] - c7[0]
            m["fhd"] = abs(m["dx_fhd_px"]) / scale
        sh_same = pts.get(5 if ear_pid == 3 else 6)     # 与耳同侧的肩（acromion）
        if sh_same is not None:
            m["dx_es_px"] = ear[0] - sh_same[0]
            m["ear_shoulder"] = abs(m["dx_es_px"]) / scale
        if sh_mid is not None and hip_mid is not None and trunk and trunk > 1e-9:
            perp = _perp_dist(ear, sh_mid, hip_mid)
            if perp is not None:
                m["ear_trunk"] = perp / scale

    if (m["cva_deg"] is None and m["fhd"] is None
            and m["ear_shoulder"] is None and m["note"] is None):
        m["note"] = "C7proxy or scale unavailable"
    return m


class SlidingMedian:
    """逐 key 的滑动窗口中位数（画面上显示的读数用）。

    每个 key 各自维护一个窗口；None 不进窗（缺帧不污染读数），窗口空 → None。
    seconds <= 0 时退化为「直接返回瞬时值」。
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._win: Dict[str, deque] = {}

    def update(self, now: float, values: dict) -> dict:
        if self.seconds <= 0:
            return dict(values)
        out = {}
        for key, val in values.items():
            win = self._win.setdefault(key, deque())
            if val is not None:
                win.append((now, float(val)))
            cutoff = now - self.seconds
            while win and win[0][0] < cutoff:
                win.popleft()
            out[key] = _median([v for _t, v in win])
        return out


# ---------------------------------------------------------------------------
# 画面：把测量用的点/参考线画出来（读数要能在画面上核对）
# ---------------------------------------------------------------------------

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text_outlined(frame: np.ndarray, s: str, org: Tuple[int, int],
                   scale: float = 0.45, color=_C_OK, thick: int = 1) -> None:
    """带黑色描边的文字（画在摄像画面上，背景任意，必须描边才读得清）。

    **描边和字的 thickness 必须一样**：OpenCV 的 thickness 会把字形整体撑宽，
    「粗黑底 + 细彩字」那种写法两遍的宽度对不上，长句子尾部会拖出一截重影
    （实测 "…58.3" 会渲染成 "…58.3  58.3"）。同厚度 + 四向偏移才是干净的描边。
    """
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(frame, s, (org[0] + dx, org[1] + dy), _FONT, scale,
                    (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(frame, s, org, _FONT, scale, color, thick, cv2.LINE_AA)


def draw_marks(frame: np.ndarray, m: dict) -> None:
    """画测量用到的那几个点和参考线：

        C7 代理（品红实心点 + 标签）
        ear -> C7proxy 连线（绿，CVA 的斜边）
        过 C7 代理的**局部**铅垂虚线（FHD 的参考线）与局部水平虚线（CVA 的水平边）
        ear 与 C7 代理之间的水平间距（橙，FHD 看的就是它）
        ear -> 同侧肩 连线（蓝，Ear-Shoulder 看的是它在 x 上的分量）
        肩中点 -> 髋中点 躯干轴虚线（diag 的参考线）

    参考线只画在相关点附近（不横贯整幅画面），免得把画面糊成网格。
    """
    pts = m.get("pts") or {}
    if not pts:
        return
    c7, sh_mid, hip_mid = m.get("c7"), m.get("sh_mid"), m.get("hip_mid")
    ear_pid = m.get("ear_pid")
    ear = pts.get(ear_pid) if ear_pid is not None else None

    def ip(p):
        return (int(round(p[0])), int(round(p[1])))

    # 躯干轴（diag 用）与肩/髋中点
    if sh_mid is not None and hip_mid is not None:
        FrameRenderer._dashed_line(frame, ip(sh_mid), ip(hip_mid), _C_REF)
        cv2.circle(frame, ip(sh_mid), 3, _C_REF, -1, cv2.LINE_AA)
        cv2.circle(frame, ip(hip_mid), 3, _C_REF, -1, cv2.LINE_AA)

    if ear is not None and c7 is not None:
        MARGIN = 70
        cx, cy = ip(c7)
        ex, ey = ip(ear)
        # FHD 的参考：过 C7 代理的铅垂线（只画到耳/肩附近）
        y_lo = max(ey, cy) + MARGIN
        y_hi = min(ey, cy) - MARGIN
        FrameRenderer._dashed_line(frame, (cx, y_hi), (cx, y_lo), _C_REF)
        # CVA 的水平边（也只画到耳附近）
        FrameRenderer._dashed_line(frame, (min(ex, cx) - MARGIN, cy),
                                   (max(ex, cx) + MARGIN, cy), _C_REF)
        # CVA 的斜边 + FHD 的水平间距 + Ear-Shoulder 的连线
        cv2.line(frame, (cx, cy), (ex, ey), _C_CVA_LEG, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, ey), (ex, ey), _C_FHD_GAP, 3, cv2.LINE_AA)
        sh_same = pts.get(5 if ear_pid == 3 else 6)
        if sh_same is not None:
            cv2.line(frame, (ex, ey), ip(sh_same), _C_ES_LEG, 2, cv2.LINE_AA)
        cv2.circle(frame, (ex, ey), 6, _C_EAR, -1, cv2.LINE_AA)

    if c7 is not None:
        cx, cy = ip(c7)
        cv2.circle(frame, (cx, cy), 6, _C_C7, -1, cv2.LINE_AA)
        _text_outlined(frame, "C7proxy", (cx + 9, cy - 9), 0.45, _C_C7)


def _panel_bg(frame: np.ndarray, w: int, h: int, alpha: float = 0.55) -> None:
    """在左上角压一块半透明深色底（面板底）。文字只画一遍就够，不用描边。"""
    h = max(1, min(int(h), frame.shape[0]))
    w = max(1, min(int(w), frame.shape[1]))
    roi = frame[0:h, 0:w]
    cv2.addWeighted(roi, 1.0 - alpha, np.zeros_like(roi), alpha, 0.0, roi)


def _f(v, nd: int = 2) -> str:
    return "N/A" if v is None else f"{v:.{nd}f}"


def _pt_s(p: Optional[Point]) -> str:
    return "N/A" if p is None else f"({int(round(p[0]))},{int(round(p[1]))})"


def _kp_s(pid: int, pts: dict, vis: dict) -> str:
    """`name(id) (x,y) v0.91`；点不可用（低置信度/越界）时 x,y 显示 N/A。"""
    return (f"{KP_NAMES.get(pid, pid)}({pid}) {_pt_s(pts.get(pid))} "
            f"v{vis.get(pid, 0.0):.2f}")


def _view_hint(ratio: Optional[float], src: Optional[str]) -> str:
    """视角粗判据（分档见 VIEW_CUTS）。ratio 是 None 时说明为什么算不出。"""
    if ratio is None or src not in VIEW_CUTS:
        return "unknown (need shoulder span + one of trunk/ear-shoulder)"
    front, oblique = VIEW_CUTS[src]
    if ratio >= front:
        return "frontal -> FHP is NOT visible here"
    if ratio >= oblique:
        return "oblique (~45deg) -> partial, contaminated"
    return "side-on -> FHP is visible here"


def draw_panel(frame: np.ndarray, m: dict, sm: dict, prod_cva: Optional[float],
               fps: float, args) -> None:
    """左上角面板：三个读数 + 用到的关键点 + 尺度/视角 + 交叉核对。"""
    vis = m.get("vis") or {}
    pts = m.get("pts") or {}
    lines: List[Tuple[str, tuple, float]] = [
        ("FHP measurements (COCO-17; NO C7 -> shoulder proxy)", _C_HEAD, 0.52),
        (f"CVA: {_f(sm.get('cva_deg'), 1)} {DEG}"
         "   (ear->C7proxy vs horizontal; smaller = more forward)",
         _C_OK if sm.get("cva_deg") is not None else _C_NA, 0.5),
        (f"FHD: {_f(sm.get('fhd'))}"
         "   (horizontal ear->C7proxy gap / scale; NOT cm)",
         _C_OK if sm.get("fhd") is not None else _C_NA, 0.5),
        (f"Ear-Shoulder: {_f(sm.get('ear_shoulder'))}"
         "   (same-side ear->acromion gap / scale)",
         _C_OK if sm.get("ear_shoulder") is not None else _C_NA, 0.5),
    ]
    # --- 点位明细（用户要的 debug 区）---
    ear_pid = m.get("ear_pid")
    lines += [
        (f"Ear:  {_kp_s(3, pts, vis)}   {_kp_s(4, pts, vis)}"
         + (f"   [using {KP_NAMES.get(ear_pid, '?')}]" if ear_pid else ""),
         _C_EAR, 0.45),
        (f"Left Shoulder  5: {_pt_s(pts.get(5))} v{vis.get(5, 0.0):.2f}"
         f"    Right Shoulder 6: {_pt_s(pts.get(6))} v{vis.get(6, 0.0):.2f}",
         _C_OK if (5 in pts and 6 in pts) else _C_NA, 0.45),
        (f"Hip 11: {_pt_s(pts.get(11))} v{vis.get(11, 0.0):.2f}"
         f"    Hip 12: {_pt_s(pts.get(12))} v{vis.get(12, 0.0):.2f}",
         _C_OK if (11 in pts or 12 in pts) else _C_NA, 0.45),
        (f"C7: N/A (not in COCO-17)   C7proxy({args.c7_mode}): "
         f"{_pt_s(m.get('c7'))}",
         _C_C7 if m.get("c7") is not None else _C_NA, 0.45),
        (f"scale: {_f(m.get('scale_px'), 0)}px "
         f"({m.get('scale_src') or 'none'})   "
         f"view: {m.get('view_src') or 'n/a'} {_f(m.get('view_ratio'))} -> "
         f"{_view_hint(m.get('view_ratio'), m.get('view_src'))}",
         _C_OK if m.get("scale_px") else _C_NA, 0.45),
        (f"diag  ear->trunk axis: {_f(sm.get('ear_trunk'))}"
         f"   dxFHD {_f(m.get('dx_fhd_px'), 0)}px  dxES {_f(m.get('dx_es_px'), 0)}px"
         "  (sign = which side, flips with facing)",
         _C_DIM, 0.42),
        (f"prod CVA (ErgonomicRiskFeatures, normalized-coord formula, aspect-biased):"
         f" {_f(prod_cva, 1)}",
         _C_DIM, 0.42),
    ]
    if m.get("note"):
        lines.append((f"note: {m['note']}", _C_NA, 0.45))
    lines.append((f"FPS {fps:4.1f}   ESC=quit   c7-mode={args.c7_mode} "
                  f"ear={args.ear} median={args.smooth_sec:g}s   "
                  "skeleton labels 3-16 are pid:vis", _C_DIM, 0.42))

    # 先量文字尺寸定面板大小，再压半透明底，最后每行只画一遍（不用描边）
    pad, gap = 10, 6
    y, panel_w = pad, 0
    placed = []
    for i, (s, color, sc) in enumerate(lines):
        (tw, th), _base = cv2.getTextSize(s, _FONT, sc, 1)
        y += th + gap
        panel_w = max(panel_w, tw)
        placed.append((y, s, color if i else _C_HEAD, sc))
    _panel_bg(frame, panel_w + 2 * pad, y + pad)
    for y, s, color, sc in placed:
        cv2.putText(frame, s, (pad, y), _FONT, sc, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "frame", "t", "note",
    "cva_deg", "fhd", "ear_shoulder", "ear_trunk",
    "s_cva_deg", "s_fhd", "s_ear_shoulder", "s_ear_trunk",
    "prod_cva", "scale_px", "scale_src", "sh_span_px", "trunk_px",
    "view_ratio", "view_src",
    "ear_pid", "dx_fhd_px", "dx_es_px",
    "earL_x", "earL_y", "earR_x", "earR_y",
    "shL_x", "shL_y", "shR_x", "shR_y",
    "hipL_x", "hipL_y", "hipR_x", "hipR_y",
    "c7_x", "c7_y",
]


def _csv_row(idx: int, now: float, m: dict, sm: dict,
             prod_cva: Optional[float]) -> dict:
    pts = m.get("pts") or {}
    row = {"frame": idx, "t": round(now, 4), "note": m.get("note") or "",
           "prod_cva": None if prod_cva is None else round(prod_cva, 4),
           "scale_px": None if m["scale_px"] is None else round(m["scale_px"], 3),
           "scale_src": m["scale_src"] or "",
           "sh_span_px": None if m["sh_span_px"] is None else round(m["sh_span_px"], 3),
           "trunk_px": None if m["trunk_px"] is None else round(m["trunk_px"], 3),
           "view_ratio": None if m["view_ratio"] is None else round(m["view_ratio"], 4),
           "view_src": m["view_src"] or "",
           "ear_pid": m["ear_pid"] if m["ear_pid"] is not None else "",
           "dx_fhd_px": None if m["dx_fhd_px"] is None else round(m["dx_fhd_px"], 3),
           "dx_es_px": None if m["dx_es_px"] is None else round(m["dx_es_px"], 3)}
    for key in ("cva_deg", "fhd", "ear_shoulder", "ear_trunk"):
        row[key] = None if m[key] is None else round(float(m[key]), 6)
        row["s_" + key] = (None if sm.get(key) is None
                           else round(float(sm[key]), 6))
    for pid, tag in ((3, "earL"), (4, "earR"), (5, "shL"), (6, "shR"),
                     (11, "hipL"), (12, "hipR")):
        p = pts.get(pid)
        row[tag + "_x"] = None if p is None else round(p[0], 3)
        row[tag + "_y"] = None if p is None else round(p[1], 3)
    c7 = m.get("c7")
    row["c7_x"] = None if c7 is None else round(c7[0], 3)
    row["c7_y"] = None if c7 is None else round(c7[1], 3)
    return row


def run(args) -> int:
    # 打开交给 with（FrameSource.__enter__ 里 open，失败抛 RuntimeError）——
    # 这里不先 open() 一次，否则 with 会 release 再重开一遍（多一次开关机、
    # 控制台还会多两行 capture 日志）。
    src = CameraCapture(args.camera, width=1280, height=720)

    # ErgonomicRiskFeatures 只用来在画面上并列一行生产 CVA（交叉核对），
    # 它的结果不参与本文件的任何测量。不走 decision 链路。
    ergo = ErgonomicRiskFeatures()
    # 只借 FrameRenderer 的骨架绘制（关键点 + `pid:vis` 标签 + 连线），
    # 不重写一套。它是私有方法：公开入口 draw() 需要 decision 层的 State/Movement，
    # 而本脚本刻意不跑 decision。output.py 一行未改。
    ren = FrameRenderer(draw_skeleton=args.draw_skeleton)
    smooth = SlidingMedian(args.smooth_sec)

    writer, fh = None, None
    if args.csv:
        # 建议写进 data/，那个目录可能还不存在 → 先建
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        fh = open(args.csv, "w", encoding="utf-8", newline="")
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        print(f"[fhp] 逐帧读数写入 {args.csv}（真人数据别提交进仓库）", flush=True)

    print("[fhp] 开始。三个量都只在**侧视**下有本来的含义 —— 正对摄像头时 "
          "FHP 落在深度方向，三者都看不到（见文件开头「视角前提」）。", flush=True)
    print("[fhp] 要验证 FHP：请侧对摄像头坐，然后从坐直慢慢前伸/弓背。ESC 退出。",
          flush=True)

    idx = 0
    t_start = time.monotonic()
    try:
        with src:
            while True:
                frame = src.read()
                if frame is None:
                    break
                now = time.monotonic()
                pose = detect_pose(frame)      # 与运行时同一个推理入口
                m = measure(pose, args.c7_mode, args.ear)
                sm = smooth.update(now, {k: m[k] for k in
                                         ("cva_deg", "fhd", "ear_shoulder", "ear_trunk")})
                prod = ergo.update(pose, now)
                prod_cva = prod.get("cva_deg") if prod else None

                if writer is not None:
                    writer.writerow(_csv_row(idx, now, m, sm, prod_cva))

                if args.draw_skeleton and pose is not None:
                    ren._draw_skeleton(frame, pose)
                draw_marks(frame, m)
                fps = (idx + 1) / max(now - t_start, 1e-6)
                draw_panel(frame, m, sm, prod_cva, fps, args)

                if args.console and idx % 30 == 0:
                    print(f"[fhp] cva={_f(sm.get('cva_deg'), 1)} "
                          f"fhd={_f(sm.get('fhd'))} es={_f(sm.get('ear_shoulder'))} "
                          f"trunk={_f(sm.get('ear_trunk'))} | "
                          f"raw cva={_f(m['cva_deg'], 1)} "
                          f"dxFHD={_f(m['dx_fhd_px'], 0)}px "
                          f"dxES={_f(m['dx_es_px'], 0)}px | "
                          f"scale {_f(m['scale_px'], 0)}px ({m['scale_src']}) "
                          f"view {_f(m['view_ratio'])} "
                          f"{_view_hint(m['view_ratio'], m['view_src'])} "
                          f"| prodCVA={_f(prod_cva, 1)}"
                          + (f" | {m['note']}" if m["note"] else ""), flush=True)
                idx += 1

                cv2.imshow("FHP measurements (RTMPose)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    except RuntimeError as e:              # 摄像头打不开（编号不存在/被占用）
        print(f"[fhp] 打不开摄像头 #{args.camera}：{e}", flush=True)
        return 2
    finally:
        if fh is not None:
            fh.close()                     # ESC / Ctrl+C / 异常都确保 CSV 落盘
    cv2.destroyAllWindows()
    print(f"[fhp] 结束，共 {idx} 帧", flush=True)
    return 0


# ---------------------------------------------------------------------------
# 自测：合成关键点验数学（不碰摄像头）
# ---------------------------------------------------------------------------

def _synth(pts_px: Dict[int, Point], size=(1280, 720),
           vis=0.9, missing=()) -> dict:
    """造一帧 pose。pts_px = {pid: (x_px, y_px)}；未给的点放到画面中心。"""
    w, h = size
    landmarks = []
    for pid in range(17):
        if pid in missing:
            landmarks.append((0.0, 0.0, 0.0, 0.0))
            continue
        x, y = pts_px.get(pid, (w / 2.0, h / 2.0))
        v = vis.get(pid, 0.9) if isinstance(vis, dict) else vis
        landmarks.append((x / w, y / h, 0.0, float(v)))
    return {"landmarks": landmarks, "image_size": size}


def _close(got, want, tol, label) -> None:
    assert got is not None, f"{label}: 期望 {want}，实际 None"
    assert abs(got - want) <= tol, f"{label}: 期望 {want}±{tol}，实际 {got}"


def selftest() -> None:
    """合成关键点自测。每条的假设都写在注释里，失败信息直接指出是哪条崩了。"""
    print("----- 1. CVA：耳在肩正上方 -> 90 deg -----")
    # 双肩 (500,400)-(700,400)，肩中点 (600,400)；耳 (600,200) 正上方 -> 竖直
    base = {5: (500.0, 400.0), 6: (700.0, 400.0), 11: (520.0, 700.0),
            12: (680.0, 700.0)}
    m = measure(_synth({**base, 3: (600.0, 200.0)}), "mid", "left")
    _close(m["cva_deg"], 90.0, 1e-6, "竖直耳位 CVA")
    print(f"  CVA={m['cva_deg']:.4f} OK")

    print("----- 2. CVA：耳前伸 45 deg（dx=dy）-> 45 deg -----")
    m = measure(_synth({**base, 3: (700.0, 300.0)}), "mid", "left")
    _close(m["cva_deg"], 45.0, 1e-6, "45 deg 前伸 CVA")
    print(f"  CVA={m['cva_deg']:.4f} OK")

    print("----- 3. 宽高比：像素空间给真角，归一化坐标公式给偏大的角 -----")
    # 同上一帧。像素空间：dx=100, dy=100 -> 45.0
    # 归一化坐标（生产的算法）：dx=100/1280, dy=100/720 -> atan2(dy,dx) 被拉大
    #   tan = (100/720)/(100/1280) = 1280/720 = 1.7778 -> 60.64 deg
    w, h = 1280.0, 720.0
    prod_like = math.degrees(math.atan2(100.0 / h, 100.0 / w))
    _close(prod_like, 60.6422, 1e-3, "归一化坐标公式的角度（宽高比效应）")
    assert m["cva_deg"] < prod_like, "像素空间真角应小于归一化坐标算出的角"
    print(f"  像素空间 {m['cva_deg']:.2f} vs 归一化坐标 {prod_like:.2f} "
          f"（差 {prod_like - m['cva_deg']:.2f} deg，即宽高比 {w / h:.3f} 的效应）OK")

    print("----- 4. 归一化：FHD / Ear-Shoulder 都除以尺度 -----")
    # sh_span=200（500->700）；earL(700,300)，C7proxy(mid)=(600,400)
    #   FHD        = |700-600| / 200 = 0.5
    #   Ear-Shoulder = |700-500| / 200 = 1.0   （同侧肩 = shL = 5）
    m = measure(_synth({**base, 3: (700.0, 300.0)}), "mid", "left")
    _close(m["fhd"], 0.5, 1e-9, "FHD 归一化")
    _close(m["ear_shoulder"], 1.0, 1e-9, "Ear-Shoulder 归一化")
    _close(m["scale_px"], 200.0, 1e-9, "尺度 = 肩宽")
    assert m["scale_src"] == "shoulder width", m["scale_src"]
    print(f"  FHD={m['fhd']:.4f}  Ear-Shoulder={m['ear_shoulder']:.4f}  "
          f"scale={m['scale_px']:.0f}px ({m['scale_src']}) OK")
    print("  （注意两者差一倍：FHD 的参考点是双肩中点、Ear-Shoulder 是同侧肩）")

    print("----- 5. 侧视退化：两肩重合 -> 尺度退回 trunk -----")
    # shL=shR=(600,400)（正侧方两肩在画面上重合）；hip 中点 (600,700)
    #   sh_span=0 -> 触发 fallback，trunk=|400-700|=300
    #   FHD = |700-600| / 300 = 0.3333
    side = {5: (600.0, 400.0), 6: (600.0, 400.0), 11: (600.0, 700.0),
            12: (600.0, 700.0)}
    m = measure(_synth({**side, 3: (700.0, 300.0)}), "mid", "left")
    _close(m["scale_px"], 300.0, 1e-9, "侧视兜底尺度 = trunk")
    assert "fallback" in (m["scale_src"] or ""), m["scale_src"]
    _close(m["fhd"], 1.0 / 3.0, 1e-9, "侧视 FHD")
    print(f"  scale={m['scale_px']:.0f}px ({m['scale_src']})  FHD={m['fhd']:.4f} OK")
    print("  （不做这个兜底的话会除以 ~0，读数直接爆掉 —— 项目里 "
          "NECK_COMPRESSION_MAX 是同一个坑）")

    print("----- 6. 缺关键点：单肩仍能算 mid（退化为那一侧肩），双肩都缺 -> 全 N/A -----")
    # 只剩 shR(700,400)，C7proxy(mid) 退化为它自己；耳 (700,300) 正在其上方
    #   -> dx=0, dy=-100 -> 90 deg（**不再是 45 deg**：参考点换了，读数就该变）
    m = measure(_synth({**base, 3: (700.0, 300.0)}, missing=(5,)), "mid", "left")
    _close(m["cva_deg"], 90.0, 1e-6, "只剩单肩时的 CVA（退化为那一侧肩）")
    print(f"  单肩: CVA={m['cva_deg']:.2f}（参考点变成可见的那侧肩）OK")
    # 双肩都缺 -> 没有 C7proxy、也没有躯干轴 -> 四条全 N/A，但不抛异常
    m = measure(_synth({**base, 3: (700.0, 300.0)}, missing=(5, 6)), "mid", "left")
    for key in ("cva_deg", "fhd", "ear_shoulder", "ear_trunk"):
        assert m[key] is None, f"双肩缺失时 {key} 应为 None，实际 {m[key]}"
    assert m["note"], "全 N/A 时应给出 note 说明原因"
    print(f"  双肩全缺: 四条全 N/A，note={m['note']!r} OK")

    print("----- 7. 低置信度 / 无人：全 N/A，不抛异常 -----")
    m = measure(_synth({**base, 3: (700.0, 300.0)}, vis=0.1), "mid", "auto")
    assert m["cva_deg"] is None and m["fhd"] is None, "低置信度应全部 N/A"
    assert m["note"], "低置信度应给出 note 说明原因"
    m = measure(None, "mid", "auto")
    assert m["cva_deg"] is None and m["fhd"] is None \
        and m["ear_shoulder"] is None and m["ear_trunk"] is None
    assert m["note"] == "no person detected", m["note"]
    print(f"  低置信度: note={m['note']!r}  -> 全部 N/A OK")

    print("----- 8. diag ear->trunk：躯干倾斜时与 FHD 分离 -----")
    # 躯干轴 (600,400)->(500,700)：方向 (-100,300)，长 316.23
    # 耳 (700,300) 到该轴的垂距 = |300*(700-600) - (-100)*(300-400)| / 316.23
    #                            = |30000 - 10000| / 316.23 = 63.246
    # 尺度 = sh_span = 200 -> 0.3162；而同样这帧的 FHD = 0.5（两者确实不同）
    tilt = {5: (500.0, 400.0), 6: (700.0, 400.0), 11: (450.0, 700.0),
            12: (550.0, 700.0)}
    m = measure(_synth({**tilt, 3: (700.0, 300.0)}), "mid", "left")
    _close(m["ear_trunk"], 63.24555 / 200.0, 1e-4, "ear->trunk 垂距")
    _close(m["fhd"], 0.5, 1e-9, "同一帧的 FHD")
    print(f"  ear->trunk={m['ear_trunk']:.4f}  FHD={m['fhd']:.4f} OK"
          "（躯干一倾斜两者就分开了）")

    print("----- 9. --c7-mode near 与 --ear 强制指定 -----")
    m = measure(_synth({**base, 3: (700.0, 300.0)}), "near", "left")
    # near: C7proxy = shL = (500,400)；ear (700,300) -> dx=200, dy=-100
    _close(m["cva_deg"], math.degrees(math.atan2(100.0, 200.0)), 1e-9,
           "near 模式的 CVA")
    _close(m["fhd"], 200.0 / 200.0, 1e-9, "near 模式的 FHD")
    m = measure(_synth({**base, 3: (700.0, 300.0), 4: (100.0, 300.0)},
                       vis={4: 0.95}), "mid", "auto")
    assert m["ear_pid"] == 4, f"auto 应选置信度高的 earR，实际 {m['ear_pid']}"
    m = measure(_synth({**base, 3: (700.0, 300.0)}, missing=(4,)), "mid", "right")
    assert m["ear_pid"] is None and m["cva_deg"] is None, \
        "强制 right 而 earR 不可见 -> N/A（不编数）"
    print(f"  near CVA={math.degrees(math.atan2(100.0, 200.0)):.2f} "
          "auto 选高置信度耳 / 强制不可见侧 -> N/A OK")

    print("----- 10. 视角判据：髋在时用 sh/trunk，髋不可见时退回 sh/ear-vert -----")
    # 正对（髋在）：肩宽 717px，躯干 700px -> 1.024 -> frontal
    front = {5: (300.0, 400.0), 6: (1017.0, 400.0), 11: (658.0, 1100.0),
             12: (659.0, 1100.0), 3: (658.0, 76.0)}
    m = measure(_synth(front), "mid", "left")
    assert m["view_src"] == "sh/trunk", m["view_src"]
    assert "frontal" in _view_hint(m["view_ratio"], m["view_src"]), m["view_ratio"]
    print(f"  {m['view_src']} {m['view_ratio']:.3f} -> "
          f"{_view_hint(m['view_ratio'], m['view_src'])} OK")
    # 正侧方：两肩差 50px、躯干 600px -> 0.083 -> side-on
    m = measure(_synth({**front, 5: (600.0, 400.0), 6: (650.0, 400.0)}),
                "mid", "left")
    assert "side-on" in _view_hint(m["view_ratio"], m["view_src"]), m["view_ratio"]
    print(f"  {m['view_src']} {m['view_ratio']:.3f} -> "
          f"{_view_hint(m['view_ratio'], m['view_src'])} OK")
    # 髋不可见（坐着常被桌子挡掉）-> 退回 sh/ear-vert：717/324 = 2.213 -> frontal
    m = measure(_synth(front, missing=(11, 12)), "mid", "left")
    assert m["view_src"] == "sh/ear-vert", m["view_src"]
    assert "frontal" in _view_hint(m["view_ratio"], m["view_src"]), m["view_ratio"]
    print(f"  髋不可见 -> {m['view_src']} {m['view_ratio']:.3f} -> "
          f"{_view_hint(m['view_ratio'], m['view_src'])} OK")
    # 连肩都没有 -> 判不了视角（给 unknown，不瞎猜）
    m = measure(_synth(front, missing=(5, 6)), "mid", "left")
    assert m["view_ratio"] is None, m["view_ratio"]
    assert "unknown" in _view_hint(m["view_ratio"], m["view_src"])
    print(f"  无肩 -> {_view_hint(m['view_ratio'], m['view_src'])} OK")

    print("\nFHP SELFTESTS PASSED.")


def main() -> None:
    args = build_parser().parse_args()
    if args.selftest:
        selftest()
        return
    sys.exit(run(args))


if __name__ == "__main__":
    main()
