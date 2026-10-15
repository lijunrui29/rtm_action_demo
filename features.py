"""
features 层：从一帧帧的 pose 结果里提取"移动量"这个特征。

FeatureExtractor 维护一个滑动时间窗口（默认 3 秒），跟踪"躯干锚点中心"
（肩中点与髋中点的中点；髋不可见时退化为肩中点），计算归一化的移动量：

    movement = 锚点对"窗口内锚点中位位置"的**中位**位移 / 归一化尺度

只跟踪锚点中心：五官、四肢（耳/肘/腕）的抖动基本不带动肩-髋中心，
不会因为说话、挥手、手指动就被判"在动"。位移统计用中位数而不是
RMS：偶发的一帧检测抖动/孤点不会把移动量顶上去，只有持续的整体位移
（起身、身体大幅度移动）才会让移动量变大 —— 这符合"检测是否静坐
过久"的目标：静坐期间的轻微动作不算"在动"。

其中 归一化尺度 优先 = 躯干高度 mean(‖5-11‖, ‖6-12‖)；髋部点（11/12）
全部不可见时（人只露出上半身）退化为 max(肩宽 ‖5-6‖, 耳-肩竖直间距)——侧身投影
会把肩宽压扁（正对 ~0.56 → 侧身 0.01~0.21），用塌缩肩宽当尺度会把静止移动量放大
成 MOVING；耳-肩竖直间距近似垂直、侧身不塌缩（还 ~0.45），取两者较大者兜底
（单位与 landmarks 一致，都是归一化坐标）。归一化后 movement 的量纲是
"尺度（躯干高度/肩宽/耳-肩间距）的比例"：0.05 表示平均移动了 5%，这样阈值对
摄像头距离、人体远近不敏感，可以跨场景复用。

为什么不用"相邻帧之间的平均欧氏距离"（最初想法）：
    - 慢速连续位移：每一帧之间的差都很小，人会"悄悄地移动"，相邻帧差却始终很小；
    - 原地抖动：pose 的噪声让点在原地来回抖，累计路径长度却会很大。
    两种情况下相邻帧差都会把"静止"和"在动"搞反。
相对窗口中位位置算中位位移则两种都能正确区分：抖动是对称噪声，中位位移停在噪声底；
连续漂移会让中位位移变大。

本模块是纯标准库（math/time），不 import cv2 / mediapipe / pose_estimation，
可以独立测试。

接口约定（保持稳定，别改签名）：
    FeatureExtractor(window_seconds=3.0, min_frames=5, min_window_seconds=1.0,
                     visibility_min=0.3, min_valid_points=2)
    update(pose_result, timestamp=None) -> Optional[float]  归一化移动量
    reset()

    PostureFeatures(visibility_min=0.3)                     躯干姿态角度特征
    update(pose_result) -> Optional[dict]  {'head_neck_angle', 可选的
                'torso_angle'/'back_curvature'/'neck_compression'}
                （角度单位：度；neck_compression 为无单位比值）
        取点：整侧链路优先 —— 两侧都齐用中点，否则优先完整单侧链路，
              最后退回中点法；head_neck 只要耳+肩可算即返回，髋缺失时
              torso/back 两个 key 不出现（调用方显示 N/A）；neck_compression
              需双侧肩都在（算肩宽）才出现，侧身退化（比值超物理上限）也缺失
              （N/A）；耳或肩缺失返回 None（详见类内 docstring）

    ErgonomicRiskFeatures(window_seconds=3.0, min_frames=5,
                          min_window_seconds=1.0, visibility_min=0.3)
    update(pose_result, timestamp=None) -> dict  CVA（颅椎角：耳-肩连线 vs 水平线，
                越小越前伸）平滑值 + FSA（前伸肩角：上臂 vs 水平线）平滑值与趋势；
                关键点不足/低置信时维持上一有效值（*_valid=False），不报错、不出 0
                （C7 用肩点近似，详见类内 docstring）

    CvaProxyFeatures(window_seconds=3.0, min_frames=5, min_window_seconds=1.0,
                     visibility_min=0.3, smooth_tau_sec=1.0)
    update(pose_result, timestamp=None) -> dict  CVA-like proxy（工程化代理，**非临床
                CVA**）：同一个"耳-肩连线 vs 水平线"的角度，但在**像素空间**算
                （归一化坐标会带上画面 w/h 的纵横比偏差），EMA 平滑值 + 本帧 raw
                值 + 参与点平均置信度 + 供绘制用的几何。FHP 判据用这一枚
                （见 decision.FhpDecision），ErgonomicRiskFeatures.cva_deg 那条线
                保持原语义不变（详见类内 docstring）

    EarShoulderProxyFeatures(window_seconds=3.0, min_frames=5,
                             min_window_seconds=1.0, visibility_min=0.3,
                             smooth_tau_sec=1.0,
                             min_shoulder_width_norm=0.25)
    update(pose_result, timestamp=None) -> dict  **Ear–Shoulder displacement proxy**
                （实验特征）：耳中点与肩中点的**水平像素位移**除以肩宽（像素），
                EMA 平滑值 + 本帧 raw 值 + 肩宽/中点 x/四个单点置信度（供逐帧
                CSV 审计）。**只是图像平面里的相对水平位移，不是真实的 3D 前伸
                距离，绝不能叫 Forward Head Distance / clinical FHD**。
                **2026-09-28 起正面生产入口不再装配本类**（改用下面的
                SternumEarAngleFeatures），本类只服务正面测量脚本
                tools/test_front_ear_shoulder_proxy.py（见类内 docstring）

    SternumEarAngleFeatures(window_seconds=3.0, min_frames=5,
                            min_window_seconds=1.0, visibility_min=0.3,
                            smooth_tau_sec=1.0, min_shoulder_width_norm=0.25)
    update(pose_result, timestamp=None) -> dict  **Sternum–Ear Angle**（实验特征，
                正面机位用）：顶点（双肩中点）到左右耳两条边的**夹角**（像素
                空间，度）+ EMA 平滑值 + 本帧 raw 值 + 顶点/两耳像素几何（供画
                V 形）。构造出处 Molaeifar et al. 2021，但**是工程代理、不是
                文献里验证过的那个量**（顶点≈胸骨中点、耳点非耳屏、3D 角降为
                2D 像面角）；**只供正面画面显示，无提醒、不参与任何判定**
                （见类内 docstring）

    ArmRaiseFeatures(visibility_min=0.3)                   举手角度（全身动作线）
    update(pose_result) -> dict  上臂抬升角（肩→肘 vs 向上竖直轴，像素空间，
                0=完全上举/90=平举/180=自然下垂）+ 肘角（肩-肘-腕在肘处的内角，
                180=伸直），左右各一套并给出"举得更高的那一侧"；**只算不判、
                不平滑**（刻意，见类内 docstring）

    HeadTiltFeatures(visibility_min=0.3,
                     min_dx_span_ratio=MIN_PITCH_DX_SPAN_RATIO)  仰头角度（全身动作线）
    update(pose_result) -> dict  头俯仰代理：耳中点→鼻尖 与水平线的夹角
                （像素空间，正=鼻在耳上方=仰头）；**只算不判、不平滑**。
                COCO 17 无下巴/头顶，这是 2D 代理、不是真实俯仰角，且只在
                45° 前侧机位有意义（见类内 docstring）
                横向退化门控：|Δx|/双耳间距 低于阈值（正对镜头）→ valid=False，
                不给"看着可信其实没量值"的 ±90°（见 MIN_PITCH_DX_SPAN_RATIO）

"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Dict, Optional, Tuple

# 躯干 4 个关键点（COCO 17 点标准编号，RTMPose 输出）
# 0 鼻, 1/2 眼, 3 左耳, 4 右耳, 5 左肩, 6 右肩, 7 左肘, 8 右肘,
# 9 左腕, 10 右腕, 11 左髋, 12 右髋, 13 左膝, 14 右膝, 15 左踝, 16 右踝
TAP_IDS = (5, 6, 11, 12)

# 髋部点（11/12）—— 髋部不可见时用肩宽兜底
HIP_IDS = (11, 12)

# 耳朵点（3 左耳 / 4 右耳）—— 坐姿角度特征用
EAR_IDS = (3, 4)

# 肩部点（5 左肩 / 6 右肩）—— 坐姿角度特征用
SHOULDER_IDS = (5, 6)

# 肘部点（7 左肘 / 8 右肘）—— CVA/FSA（前伸肩角）特征用
ELBOW_IDS = (7, 8)

# 腕部点（9 左腕 / 10 右腕）—— 举手特征（ArmRaiseFeatures）的肘角用；
# 抬升角只要肩+肘，腕缺失不影响抬升角（见 ArmRaiseFeatures）
WRIST_IDS = (9, 10)

# 鼻（COCO 17 的 0 号点）—— 仰头特征（HeadTiltFeatures）的头参考。
# COCO 17 没有下巴/头顶/耳屏，鼻尖是离头部旋转轴心最远的点，用它做 2D 俯仰代理
NOSE_ID = 0

# 仰头特征的横向退化门控（2026-10-12 实拍，同一人同一摄像头换机位）：
# 正对镜头时头部前后轴指着镜头，耳→鼻在像面上只剩竖直分量 → |Δx| ≈ 0 →
# atan2(-dy, |Δx|) 退化成 sign(θ)·90°：只剩"鼻子在耳中点上方还是下方"的符号、
# **没有任何量值**（实拍：Δx = -3.3~+3.3 px，读数恒定 -73~-90°，且静坐时极稳）。
# 这种错是静默的 —— -83.7°、45.0° 这类数看着完全正常，迟滞也救不了（防的是阈值
# 附近的抖动，而这里读数本身没量值）。所以必须在特征层把它作废：
# 用**双耳间距**归一化 |Δx|（抵消远近/头大小），低于此值判"几何退化" → valid=False。
# 标定依据：45° 前侧实测 0.43~0.52、正对实测 0.013（差 30 倍以上），阈值取空档中间。
# 与 NECK_COMPRESSION_MAX / MIN_SHOULDER_WIDTH_NORM 同一口径：宁可不判（N/A），不乱判。
MIN_PITCH_DX_SPAN_RATIO = 0.15

# 躯干姿态计算用到的全部关键点：耳 3/4 + 肩 5/6 + 髋 11/12
POSTURE_IDS = EAR_IDS + SHOULDER_IDS + HIP_IDS

# 腿部关键点（COCO 17 点）：13/14 膝, 15/16 踝。髋 11/12 属于躯干，不进腿特征。
LEG_IDS = (13, 14, 15, 16)
KNEE_IDS = (13, 14)
ANKLE_IDS = (15, 16)

# 侧身退化门控（2026-08-19 实拍标定，仅剩 NECK_COMPRESSION_MAX）：
# 侧身时肩宽被投影压扁（正对 ~0.56，侧身 0.01~0.21）、髋又常不可见，会产出退化值：
# neck_compression 分母是肩宽，塌缩后比值爆炸（实拍 0.5~13.5）→ 把坐姿锁死 GOOD，
# NECK_COMPRESSION_MAX 把它作废（= N/A），"视角退化时宁可不判（N/A），不乱判"。
# 移动量也曾用 SHOULDER_SCALE_MIN_RATIO 门控（肩宽塌缩→作废→UNKNOWN），但作废帧
# 不进窗口、侧身一久窗口排空 → 持续 NO PERSON；而侧身移动量只是偏高（0.045~0.143，
# 非爆炸），靠调高 still-threshold 兜底更合适 —— 该门控 2026-08-19 已移除（见 CLAUDE.md）。
NECK_COMPRESSION_MAX = 1.5       # 颈压缩比值超此物理上限 → 视角退化，作废（= N/A）

# Ear–Shoulder displacement proxy 的肩宽退化门控（2026-09-23，实验特征用）：
# 该 proxy 的分母就是肩宽（两肩像素距离），侧身时肩宽被投影压扁（正对归一化
# 肩宽 ~0.56、侧身 0.01~0.21，见 CLAUDE.md 侧身处理），比值会被放大成无意义的大数。
# 归一化肩宽低于此值的帧判"视角退化"：不出有效读数（valid=False），但 raw 与
# 肩宽仍记进 CSV 供审计 —— 与 NECK_COMPRESSION_MAX 的"视角退化时宁可不判（N/A），
# 不乱判"同一口径。阈值落在观测到的 0.56（正对）/ 0.21（侧身）之间。
MIN_SHOULDER_WIDTH_NORM = 0.25

# 一个点的二维坐标
Point = Tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class _WindowedMedianExtractor:
    """滑动窗口 + 中位统计的通用基类：只持有窗口状态与可复用原语。

    各子类的 update() 返回类型不同（躯干版 Optional[float]、腿版 dict），
    所以这里**不做模板方法**，只提供状态（window 参数 + _window deque）和
    原语（_trim / _window_ready / _median / _extract_valid / reset），
    子类各自实现 update()，避免复制窗口/中位逻辑。
    """

    def __init__(self,
                 window_seconds: float = 3.0,
                 min_frames: int = 5,
                 min_window_seconds: float = 1.0,
                 visibility_min: float = 0.3,
                 min_valid_points: int = 2) -> None:
        """
        参数:
            window_seconds:      滑动窗口时长（秒），窗口内的帧参与计算。
            min_frames:          窗口内至少要攒够这么多帧，否则数据不足。
            min_window_seconds:  窗口内首尾帧的最小时间跨度（秒），数据太稀疏
                                 时不乱下结论。
            visibility_min:      某关键点的 visibility 低于此值视为无效点。
            min_valid_points:    一帧里至少要有这么多有效点，才把该帧计入窗口。
        """
        self.window_seconds = window_seconds
        self.min_frames = min_frames
        self.min_window_seconds = min_window_seconds
        self.visibility_min = visibility_min
        self.min_valid_points = min_valid_points

        # 窗口里的每一帧：(timestamp, {点id: (x, y)}, 该帧归一化尺度)
        self._window: deque = deque()

    def reset(self) -> None:
        """清空窗口，重新开始。"""
        self._window.clear()

    def _trim(self, now: float) -> None:
        """按 window_seconds 裁剪窗口内超时的旧帧。"""
        cutoff = now - self.window_seconds
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def _window_ready(self) -> bool:
        """窗口是否已攒够帧数 + 时间跨度（首尾帧差 >= min_window_seconds）。"""
        if len(self._window) < self.min_frames:
            return False
        return self._window[-1][0] - self._window[0][0] >= self.min_window_seconds

    def _extract_valid(self, landmarks: list,
                       ids: Tuple[int, ...]) -> Dict[int, Point]:
        """从 landmarks 里提取 ids 中 visibility >= visibility_min 的点。

        返回 {点id: (x, y)}（归一化坐标），供子类做锚点/尺度计算。
        """
        valid: Dict[int, Point] = {}
        for pid in ids:
            if pid < len(landmarks):
                x, y, _z, vis = landmarks[pid]
                if vis >= self.visibility_min:
                    valid[pid] = (x, y)
        return valid

    @staticmethod
    def _median(values: list) -> float:
        """小数组的中位数（不依赖 statistics，也更快）。"""
        s = sorted(values)
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            return s[mid]
        return (s[mid - 1] + s[mid]) / 2.0


class FeatureExtractor(_WindowedMedianExtractor):
    """从 pose 序列中提取"躯干移动量"特征。

    侧身时的行为（2026-08-19 决定）：髋不可见时尺度兜底用 max(肩宽, 耳-肩竖直间距)。
    侧身投影把肩宽压扁（正对 ~0.56 → 侧身 0.01~0.21），若退回纯肩宽，塌缩尺度会把
    静止移动量放大成 MOVING（实拍侧身 0.045~0.143，几乎全是 MOVING）；耳-肩竖直间距
    近似垂直、侧身不塌缩，取大者兜底可让侧身移动量回到正常量级（偶发跳 MOVING 用
    still-threshold 兜底）。曾试过门控作废（→ UNKNOWN，侧身一久持续 NO PERSON）已移除。
    见 CLAUDE.md「侧身」。
    """

    def __init__(self,
                 window_seconds: float = 3.0,
                 min_frames: int = 5,
                 min_window_seconds: float = 1.0,
                 visibility_min: float = 0.3,
                 min_valid_points: int = 2) -> None:
        """
        参数:
            window_seconds:      滑动窗口时长（秒），窗口内的帧参与计算。
            min_frames:          窗口内至少要攒够这么多帧，否则返回 None。
            min_window_seconds:  窗口内首尾帧的最小时间跨度（秒），同样是为了
                                 数据太稀疏时不乱下结论。
            visibility_min:      某关键点的 visibility 低于此值视为无效点。
            min_valid_points:    一帧里至少要有这么多有效点，才把该帧计入窗口。
        """
        super().__init__(window_seconds=window_seconds,
                         min_frames=min_frames,
                         min_window_seconds=min_window_seconds,
                         visibility_min=visibility_min,
                         min_valid_points=min_valid_points)

    def update(self,
               pose_result: Optional[dict],
               timestamp: Optional[float] = None) -> Optional[float]:
        """喂入一帧的 pose 结果，返回当前归一化移动量。

        参数:
            pose_result: pose_estimation.detect_pose() 的返回值，结构为
                         {'landmarks': [(x, y, z, visibility) x33],
                          'image_size': (w, h)}，未检测到人体时为 None。
            timestamp:   该帧的单调时间戳（秒）。默认用 time.monotonic()。
                         —— 视频回放时请传入模拟时间戳（如 frame_idx / fps），
                            保证窗口按"真实时间"裁剪，不受播放 FPS 影响。

        返回:
            float 归一化移动量（躯干高度的比例），
            None 表示"这一帧不足以判定"（无人 / 有效点太少 / 窗口数据不足）。
        """
        now = timestamp if timestamp is not None else time.monotonic()

        if pose_result is None:
            # 没检测到人：不入窗口、返回 None。窗口保留，不重置。
            return None

        landmarks = pose_result.get('landmarks')
        if not landmarks:
            return None

        # 1) 提取本帧有效的躯干点坐标 + 耳朵（耳-肩间距参与髋缺失时的兜底尺度）
        valid = self._extract_valid(landmarks, TAP_IDS + EAR_IDS)

        # 2) 有效点太少 → 本帧作废，不入窗口
        if len(valid) < self.min_valid_points:
            return None

        # 3) 本帧归一化尺度：优先躯干高度，髋部不可见时用肩宽兜底
        scale = self._frame_scale(valid)
        if scale is None:
            return None

        # 4) 入窗口，并裁剪掉超时的旧帧
        self._window.append((now, valid, scale))
        self._trim(now)

        # 5) 数据量校验
        if not self._window_ready():
            return None

        return self._compute_normalized_movement()

    # ---------- 内部实现 ----------

    @staticmethod
    def _frame_scale(points: Dict[int, Point]) -> Optional[float]:
        """一帧的归一化尺度（长度单位）：优先躯干高度，髋不可见时用 max(肩宽, 耳-肩距)。

        躯干高度 = mean(‖5-11‖, ‖6-12‖)，只统计两点都有效的侧；
        髋部点（11/12）全部不可见时退回 max(肩宽 ‖5-6‖, 耳-肩竖直间距)：
        侧身投影把肩宽压扁（正对 ~0.56 → 侧身 0.01~0.21），塌缩尺度会把静止
        移动量放大成 MOVING；耳-肩竖直间距近似垂直、侧身不塌缩（~0.45），取大者
        兜底。这是为了兼容"只露出上半身"的场景（人坐着/画面只框到胸口），否则
        髋部永远不可见 → 每帧都被丢弃 → 永远显示"无人/数据不足"。
        """
        lengths = []
        if 5 in points and 11 in points:
            lengths.append(_dist(points[5], points[11]))
        if 6 in points and 12 in points:
            lengths.append(_dist(points[6], points[12]))
        if not lengths:
            # 髋部不可见 → 肩宽与耳-肩竖直间距取较大者（耳缺失时退回纯肩宽）
            w = None
            if 5 in points and 6 in points:
                w = _dist(points[5], points[6])
            gap = None
            s_ys = [points[pid][1] for pid in SHOULDER_IDS if pid in points]
            if s_ys:
                s_mid_y = sum(s_ys) / len(s_ys)
                gaps = [abs(points[pid][1] - s_mid_y)
                        for pid in EAR_IDS if pid in points]
                if gaps:
                    gap = max(gaps)
            candidates = [v for v in (w, gap) if v is not None]
            if not candidates:
                return None
            return max(candidates)
        return sum(lengths) / len(lengths)

    def _compute_normalized_movement(self) -> Optional[float]:
        """基于窗口算归一化移动量（供窗口数据已足时调用）。"""
        # 窗口内每一帧的躯干锚点中心（肩中点与髋中点的中点）
        anchors: list = []
        for _ts, points, _scale in self._window:
            a = self._anchor_center(points)
            if a is not None:
                anchors.append(a)
        if not anchors:
            return None

        # 锚点的中位位置，及各帧对中位的中位位移
        median_x = self._median([x for x, _y in anchors])
        median_y = self._median([y for _x, y in anchors])
        displacements = [math.hypot(x - median_x, y - median_y)
                         for x, y in anchors]
        median_disp = self._median(displacements)

        # 中位归一化尺度（躯干高度或肩宽）
        scale = self._median_torso_scale()
        if scale is None or scale < 1e-4:
            return None  # 尺度过小（离摄像头太远/退化），不作判定

        return median_disp / scale

    @staticmethod
    def _anchor_center(points: Dict[int, Point]) -> Optional[Point]:
        """一帧的躯干锚点中心：肩中点与髋中点的中点。

        髋部点（11/12）不可见时（人只露出上半身/髋在画面外）退化为肩中点，
        保证只露出上半身也能算。五官（耳/眼/鼻）和四肢（肘/腕/膝/踝）不参与，
        所以这些部位的轻微动作不会带动锚点中心 —— 移动量只反映躯干整体移动。
        """
        shoulder_mid = None
        s = [points[pid] for pid in SHOULDER_IDS if pid in points]
        if s:
            shoulder_mid = ((s[0][0] + s[-1][0]) / 2.0,
                            (s[0][1] + s[-1][1]) / 2.0)

        hip_mid = None
        h = [points[pid] for pid in HIP_IDS if pid in points]
        if h:
            hip_mid = ((h[0][0] + h[-1][0]) / 2.0,
                       (h[0][1] + h[-1][1]) / 2.0)

        if shoulder_mid is not None and hip_mid is not None:
            return ((shoulder_mid[0] + hip_mid[0]) / 2.0,
                    (shoulder_mid[1] + hip_mid[1]) / 2.0)
        if shoulder_mid is not None:
            return shoulder_mid
        if hip_mid is not None:
            return hip_mid
        return None

    def _median_torso_scale(self) -> Optional[float]:
        """窗口内各帧归一化尺度的中位数（优先躯干高度，髋部不可见时为肩宽）。"""
        scales = [scale for _ts, _points, scale in self._window]
        if not scales:
            return None
        return self._median(scales)


# ---------------------------------------------------------------------------
# PostureFeatures：坐姿角度特征（新增，纯 stdlib，不依赖 cv2/mediapipe）
# ---------------------------------------------------------------------------

# 三个角度的"定义说明"（英文，供 output 层画到画面左下角）。
# 定义跟着特征走，输出层只引用渲染，不在 output/main 里重复写语义。
# 唯一真实来源是 PostureFeatures.update() 返回的三个角度。
ANGLE_LEGEND = [
    "Ear-shoulder-hip 3-point chain:",
    "  torso   = shoulder->hip  vs vertical",
    "  neck    = ear->shoulder  vs vertical",
    "  back    = 180 - fold angle at shoulder",
    "  head    = ear-shoulder vertical gap / shoulder width",
    "           (smaller = head dropped / hunched)",
]

class PostureFeatures:
    """从单帧 pose 里计算三个坐姿角度 + 一个颈压缩比值，供坐姿识别/展示。

    三个角度（都用图片平面里的归一化坐标 (x, y) 计算，量纲无关，跨摄像头可复用）：
        torso_angle      躯干倾角：肩中→髋中向量 与竖直轴 (0,1) 的夹角。
                         0 = 身体竖直，越大越前倾/倾斜。依赖髋点。
        head_neck_angle  头颈角：耳→肩中向量 与竖直轴 (0,1) 的夹角。
                         0 = 头在肩正上方，越大头越前伸。只需耳+肩，不依赖髋。
        back_curvature   背部弯曲度：耳-肩-髋 三点折线在肩处的内角相对 180°
                         的偏折（180 − 内角）。0 = 耳肩髋基本共线（背直），
                         越大弓背越明显（低头/驼背时折线向内折叠）。依赖髋点。
        neck_compression 颈压缩：耳-肩**竖直间距** ÷ 肩宽（比值，无单位）。
                         0 = 耳与肩同高（头完全压到肩上），越大 = 头越在肩
                         正上方。只需耳+肩+双侧肩（算肩宽），不依赖髋。
                         侧身退化时（肩宽被投影压扁，比值超 NECK_COMPRESSION_MAX）
                         缺失（= N/A），不产生爆炸值把状态锁死。

    为什么有 neck_compression：正面摄像头下"耸肩+低头"式前弓背在图像平面里
    几乎不改变 x/y —— torso/back/head_neck 三个角度在正面投影都贴 0（见本模块
    开头 "局限" 段），这是现有角度抓不到前弓背的原因。弓背时唯一明显的正面
    信号是耳-肩竖直间距被压缩（肩耸起、头下沉），neck_compression 就是量化
    这个压缩的：它不依赖髋，坐着髋被桌子挡掉时依然可算 —— 恰恰是"坐着弓背"
    最需要的场景。默认阈值等起始值在 decision.py 标定（见 CLAUDE.md）。

    依赖髋点的 torso_angle / back_curvature 在髋不可见（人只露出上半身、
    髋在画面外）时算不出：update() 返回的 dict 不含这两个 key；缺失的 key
    表示 N/A（调用方显示 N/A，不要当 0）。head_neck_angle 只需耳+肩，
    neck_compression 只需耳+肩+双侧肩，髋缺失时依然能算并显示。

    取点策略（整侧链路优先）：
        两侧耳/肩/髋都可见 → 用两侧中点（最稳定，正对摄像头默认路径）；
        只有一侧三点齐 → 只用那一侧的单侧链路（人侧对摄像头时后景髋点
            常被遮挡，混用两侧中点会把角度算歪）；
        都没有完整链路但点数够 → 退回中点法（耳取可见度最高的那一只 +
            肩/髋取中点，耳部可见度门控沿用本模块风格）。
        单侧链路时无双侧肩，肩宽算不出 → neck_compression 缺失（N/A）。

    局限：正面摄像头在图像平面里主要反映左右倾斜和前伸，侧向弓背/低头只能
    部分体现；neck_compression 补上了"耸肩+低头"式前弓背，但仍有残留盲区
    （纯前倾、侧身弓背）。这些特征作为特征先落地、显示，供后续换视角更好的
    摄像头或做坐姿分类时使用。

    接口约定（保持稳定，别改签名）：
        PostureFeatures(visibility_min=0.3)
        update(pose_result: Optional[dict]) -> Optional[dict]
            返回 {'head_neck_angle':°, 可选的 'neck_compression'（比值）、
                  髋可见时另含 'torso_angle':° / 'back_curvature':°}，
                  torso/back 缺失=髋不可见、neck_compression 缺失=单侧链路
                  无肩宽（N/A）；耳或肩缺失返回 None。
    """

    def __init__(self, visibility_min: float = 0.3) -> None:
        """
        参数:
            visibility_min: 某关键点的 visibility 低于此值视为无效点。
        """
        self.visibility_min = visibility_min

    def update(self, pose_result: Optional[dict]) -> Optional[dict]:
        """喂入一帧 pose 结果，返回躯干姿态角度 + 颈压缩（度 / 比值）。

        参数:
            pose_result: pose_estimation.detect_pose() 的返回值，结构为
                         {'landmarks': [(x, y, z, visibility) x33],
                          'image_size': (w, h)}，未检测到人体时为 None。

        返回:
            dict 至少含 'head_neck_angle'（耳+肩可算即返回）；
            'neck_compression' 双侧肩都在时附加（单侧链路无肩宽则缺失）；
            髋可见时另附 'torso_angle' / 'back_curvature'（key 缺失表示该
            帧髋部不可见，调用方应显示 N/A 而不是把它当 0）；
            None 表示耳或肩缺失，任何角度都算不出（无人 / 数据不足）。
        """
        if pose_result is None:
            return None

        landmarks = pose_result.get('landmarks')
        if not landmarks:
            return None

        # 1) 按"整侧链路优先"选出耳/肩/髋三点 + 肩宽
        sel = self._select_points(landmarks)
        if sel is None:
            return None
        ear, shoulder, hip = sel['ear'], sel['shoulder'], sel['hip']

        # 2) 特征计算。head_neck 只依赖耳+肩；neck_compression 只依赖
        #    耳+肩+肩宽（不依赖髋）；torso/back 依赖髋。缺失的 key = N/A。
        result = {
            'head_neck_angle': self._angle_from_vertical(ear, shoulder),
        }
        nc = self._neck_compression(ear, shoulder, sel['shoulder_width'])
        if nc is not None:
            result['neck_compression'] = nc
        if hip is not None:
            result['torso_angle'] = self._angle_from_vertical(shoulder, hip)
            result['back_curvature'] = self._fold_angle(ear, shoulder, hip)
        return result

    # ---------- 内部实现 ----------

    def _select_points(self, landmarks: list) -> Optional[dict]:
        """从 landmarks 里提取可见的躯干姿态点，并决定用哪侧/哪个点。

        再按"整侧链路优先"选出参与角度计算的三点，同时给出肩宽
        （neck_compression 的归一化尺度，双侧肩都在才可算）。

        返回:
            {'ear': Point, 'shoulder': Point, 'hip': Point | None,
             'shoulder_width': float | None}
            或 None（数据不足）。
        """
        # 1) 提取本帧所有可见点（按 visibility_min 门控）
        pts = {}
        for pid in POSTURE_IDS:
            if pid < len(landmarks):
                x, y, _z, vis = landmarks[pid]
                if vis >= self.visibility_min:
                    pts[pid] = (x, y)
        if not pts:
            return None

        side = self._choose_chain_side(pts)
        if side is None:
            return None

        if side == "BOTH":
            ear = self._avg(pts, EAR_IDS)
            shoulder = self._avg(pts, SHOULDER_IDS)
            hip = self._avg(pts, HIP_IDS)
        elif side in ("L", "R"):
            if side == "L":
                ear, shoulder, hip = pts[3], pts[5], pts[11]
            else:
                ear, shoulder, hip = pts[4], pts[6], pts[12]
        else:  # "MID"
            ear = self._pick_best_ear(pts)
            shoulder = self._avg(pts, SHOULDER_IDS)
            hip = self._avg(pts, HIP_IDS)

        # ear / shoulder 必须可见；hip 允许缺失（人只露出上半身/髋在画面外时，
        # head_neck_angle / neck_compression 仍可算，只有 torso/back 无法算）。
        if ear is None or shoulder is None:
            return None

        # 肩宽：双侧肩（5/6）都在才可算；单侧链路（L/R）无肩宽 → None（N/A）。
        shoulder_width = None
        if 5 in pts and 6 in pts:
            shoulder_width = math.hypot(pts[6][0] - pts[5][0],
                                        pts[6][1] - pts[5][1])
        return {"ear": ear, "shoulder": shoulder, "hip": hip,
                "shoulder_width": shoulder_width}

    @staticmethod
    def _choose_chain_side(pts: dict) -> Optional[str]:
        """决定用"左右哪一侧"的耳-肩-髋链路算角度。

        返回:
            "BOTH"  两侧耳/肩/髋都齐，用两侧中点（最稳定）；
            "L"/"R" 只用这一侧完整的单侧三点链；
            "MID"   无完整链路但各点仍够算（正对摄像头、单侧略有遮挡），
                    退回"耳取可见度最高的一只 + 肩/髋取中点"；
            None    数据不够，无法计算。

        优先取整侧链路的动机：人侧对摄像头时，后景那一侧的髋点常被
        身体遮挡不可见，此时若混用两侧中点，角度会算歪。优先让
        "耳朵、肩膀、髋部都在同一侧"的链路参与计算。
        """
        def count(side: str) -> int:
            ids = (3, 5, 11) if side == "L" else (4, 6, 12)
            return sum(1 for pid in ids if pid in pts)

        c_left, c_right = count("L"), count("R")
        if c_left == 3 and c_right == 3:
            return "BOTH"
        if c_left == 3:
            return "L"
        if c_right == 3:
            return "R"
        return "MID" if pts else None

    @staticmethod
    def _avg(pts: dict, ids: Tuple[int, int]) -> Optional[Point]:
        """取 pts 中 ids 内所有可见点的中点（至少 1 个可见）。"""
        pts_l = [pts[pid] for pid in ids if pid in pts]
        if not pts_l:
            return None
        if len(pts_l) == 1:
            return pts_l[0]
        return ((pts_l[0][0] + pts_l[1][0]) / 2.0,
                (pts_l[0][1] + pts_l[1][1]) / 2.0)

    @staticmethod
    def _pick_best_ear(pts: dict) -> Optional[Point]:
        """耳只取可见的那一只（"MID" 分支专用）。"""
        for pid in EAR_IDS:
            if pid in pts:
                return pts[pid]
        return None

    @staticmethod
    def _angle_from_vertical(a: Point, b: Point) -> float:
        """向量 a→b 与竖直轴 (0,1) 的夹角（度，0~180）。"""
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return math.degrees(math.atan2(abs(dx), abs(dy)))

    @staticmethod
    def _fold_angle(a: Point, b: Point, c: Point) -> float:
        """耳(a)-肩(b)-髋(c) 折线在 b 处的内角相对 180° 的偏折（度）。

        0 = a/b/c 共线笔直；>0 越大弓背越明显。
        """
        ab = (a[0] - b[0], a[1] - b[1])
        cb = (c[0] - b[0], c[1] - b[1])
        # 内角（0~180 度）
        denom = math.hypot(*ab) * math.hypot(*cb)
        if denom < 1e-9:
            return 0.0
        cos = (ab[0] * cb[0] + ab[1] * cb[1]) / denom
        cos = max(-1.0, min(1.0, cos))  # 数值容差
        angle = math.degrees(math.acos(cos))
        return 180.0 - angle

    @staticmethod
    def _neck_compression(ear: Point, shoulder: Point,
                          shoulder_width: Optional[float]) -> Optional[float]:
        """颈压缩：耳-肩**竖直间距** ÷ 肩宽（无单位比值，越小越弓背）。

        竖直间距 = |shoulder.y − ear.y|（归一化坐标 y 向下；耳在肩上方时
        即"头高出肩多少"）。用肩宽归一化 = 以本人身体尺度为参照，正面摄像头
        下对"耸肩+低头"式前弓背敏感，且不依赖髋（坐着髋被桌子挡也能算）。
        肩宽算不出（单侧链路）→ 返回 None（= N/A）。
        间距为 0（耳与肩同高 = 头完全压到肩上）→ 返回 0.0（最大压缩，
        decision 层按超限处理，不会除零）。
        """
        if shoulder_width is None or shoulder_width < 1e-9:
            return None
        gap = abs(shoulder[1] - ear[1])
        value = gap / shoulder_width
        # 侧身退化门控（2026-08-19，实拍标定）：侧身投影把肩宽压扁，比值爆炸
        # （REPRO 段 0.5~13.5，正对才 0.5~0.8），决策层 ratio = 阈值/值 -> 0.03，
        # 把坐姿状态锁死成 GOOD。比值超 NECK_COMPRESSION_MAX（耳-肩竖直间距大于
        # 肩宽 1.5 倍，物理上不可能，只能是视角退化）-> 返回 None（= N/A），
        # 把判断让回仍有效的 head_neck_angle。
        if value > NECK_COMPRESSION_MAX:
            return None
        return value


# ---------------------------------------------------------------------------
# ErgonomicRiskFeatures：CVA（颅椎角）/ FSA（前伸肩角）姿态风险特征
# （2026-09-01 新增，纯 stdlib；文献代理指标，需按本系统取景自标定）
# ---------------------------------------------------------------------------

class ErgonomicRiskFeatures(_WindowedMedianExtractor):
    """CVA（颅椎角）/ FSA（前伸肩角）姿态风险特征：文献化的头前伸/肩前伸量化指标。

    免责声明：这里输出的角度指标反映**体表姿态模式**（耳/肩/肘关键点在图片
    平面上的投影几何），**不能替代医学影像诊断或临床评估**。CVA 分级阈值来自
    Mostafaee et al. 2022 观察性分组，仅作参考，需按本系统的取景/视角自标定。

    指标定义（图片平面，归一化坐标 (x, y)，y 向下）：
        CVA  颅椎角：耳→肩 连线 与"过肩点的水平线"的夹角（0~90°，
             竖直耳位 → 90°）。数值越小 = 头越前伸（严重）。
        FSA  前伸肩角（Lee et al. 2015 思路）：上臂（肩中点→肘中点，肱骨
             中点必在此线段上）与水平线的夹角（0~90°）。数值越小 = 上臂越
             接近水平 = 肩部越前伸。**辅助指标**：不参与判断，仅供 CVA 中
             重度及以上时附注显示（见 decision.CvaRisk / output 层）。

    C7 近似（2026-09-01，思路一）：直接用肩点（5/6，双肩可见取中点、单肩取
    可见侧）当 C7 代理，**不加偏移修正**。真实 C7（第 7 颈椎）在肩点后方且
    更靠上，方向偏差稳定但量值不可直接套用文献阈值 —— 本系统阈值必须用实拍
    样本自标定。将来接入真正的 C7 关键点或换 schema 后，这里需重做。

    ⚠ 坐标空间与读数可信度（2026-09-29 标注）：cva_deg / fsa_deg 都在**归一化坐标**
    (x, y ∈ [0,1]) 里算 —— 归一化先把 x 除以 W、y 除以 H，所以角度带上**画面宽高比**
    偏差：tan θ_归一化 = tan θ_像素 / (H/W)。16:9 取景下同一姿势系统性偏大（实测
    45° 机位同批帧 75.5° vs 像素空间 65.1°，差约 10°）。所以：
        * **cva_deg 受画面宽高比影响，仅供参考，不代表标定后的真实角度** —— 不可与
          文献的 55/50/44 直接比较，改取景 / 改分辨率 / 改裁剪后同一个人的读数会变；
        * 要"真实"角度请用**像素空间**口径（`CvaProxyFeatures.cva_proxy_deg` /
          `SternumEarAngleFeatures.sternum_ear_angle`），或先用
          `tools/test_fhp_measurements.py` 量出宽高比差值再换算（那个脚本就是把
          两种口径并列显示来量这个差）。
    消费范围（如实记录）：这条线**不参与任何判定、也不触发任何提醒**（没有任何
    Alert 读它）。唯一的消费是画面显示 `output.FrameRenderer.draw_cva_overlay` ——
    其中 CvaLevel 的**分档**（阈值同样未标定）决定那条 FSA 附注行出不出现、
    以及读数用什么颜色，所以界面上它看起来比实际可信。见 decision.CvaRisk。

    时间平滑：CVA/FSA 各自维护滑动窗口（默认 3s），每帧把可算的原始值入窗，
    输出**中位数**（对单帧检测噪声/抖动鲁棒），防止分级等级单帧跳变。
    窗口攒够 min_frames 帧 + min_window_seconds 时间跨度才输出平滑值
    （启动预热阶段返回 None，与 FeatureExtractor 的"数据不足不乱下结论"一致）。

    降级（遮挡/低置信度）：耳/肩/肘任一不足（visibility < visibility_min）时
    该指标本帧不可算 → **维持上一有效值**并标 *_valid=False（从未有效则 None），
    不报错、不出 0。CVA 与 FSA 相互独立降级（如肘被桌子挡掉不影响 CVA）。

    接口约定（保持稳定，别改签名）：
        ErgonomicRiskFeatures(window_seconds=3.0, min_frames=5,
                              min_window_seconds=1.0, visibility_min=0.3)
        update(pose_result, timestamp=None) -> dict
            {'cva_deg', 'cva_valid', 'fsa_deg', 'fsa_valid', 'fsa_trend_deg'}
        reset()
    """

    def __init__(self,
                 window_seconds: float = 3.0,
                 min_frames: int = 5,
                 min_window_seconds: float = 1.0,
                 visibility_min: float = 0.3) -> None:
        """
        参数:
            window_seconds:      滑动窗口时长（秒），CVA/FSA 中位数平滑用。
            min_frames:          窗口至少攒够多少帧才输出平滑值。
            min_window_seconds:  窗口内首尾帧的最小时间跨度（秒），数据太稀疏
                                 时不乱下结论。
            visibility_min:      某关键点 visibility 低于此值视为无效点。
        """
        super().__init__(window_seconds=window_seconds,
                         min_frames=min_frames,
                         min_window_seconds=min_window_seconds,
                         visibility_min=visibility_min,
                         min_valid_points=2)
        self._cva_win: deque = deque()   # (timestamp, cva_deg) 原始值
        self._fsa_win: deque = deque()   # (timestamp, fsa_deg) 原始值
        self._last = {                   # 上一有效值（降级时维持）
            'cva_deg': None,
            'fsa_deg': None,
            'fsa_trend_deg': None,
        }

    def reset(self) -> None:
        """清空窗口与上一有效值，重新开始。"""
        super().reset()
        self._cva_win.clear()
        self._fsa_win.clear()
        self._last = {'cva_deg': None, 'fsa_deg': None, 'fsa_trend_deg': None}

    def update(self,
               pose_result: Optional[dict],
               timestamp: Optional[float] = None) -> dict:
        """喂入一帧 pose 结果，返回 CVA / FSA 特征。

        参数:
            pose_result: pose_estimation.detect_pose() 的返回值
                         {'landmarks': [(x, y, z, visibility) x17], ...}，
                         未检测到人体时为 None。
            timestamp:   该帧的单调时间戳（秒），默认 time.monotonic()；
                         视频回放/自测请传入模拟时间戳（与 FeatureExtractor 同约定）。

        返回:
            dict，含：
                cva_deg:       平滑后的 CVA（度，0~90）。本帧可算=新值；
                               关键点不足/低置信=维持上一有效值；从未有效=None。
                cva_valid:     本帧 CVA 是否真的算了（False = 维持旧值/降级）。
                fsa_deg:       平滑后的 FSA（度，0~90），退化语义同 CVA。
                fsa_valid:     本帧 FSA 是否真的算了。
                fsa_trend_deg: FSA 相对窗口内最早帧的变化量（度；负 = 上臂
                               更接近水平 = 肩更前伸）。
        """
        now = timestamp if timestamp is not None else time.monotonic()

        cva, fsa = self._compute_frame(pose_result)

        # 能算的帧入窗 → 窗口攒够后输出中位数；否则维持上一有效值（valid=False）
        cva_out = fsa_out = None
        cva_valid = fsa_valid = False
        if cva is not None:
            self._cva_win.append((now, cva))
            self._trim_window(self._cva_win, now)
            if self._win_ready(self._cva_win):
                cva_out = self._smoothed(self._cva_win)
                cva_valid = True
                self._last['cva_deg'] = cva_out
        if fsa is not None:
            self._fsa_win.append((now, fsa))
            self._trim_window(self._fsa_win, now)
            if self._win_ready(self._fsa_win):
                fsa_out = self._smoothed(self._fsa_win)
                fsa_valid = True
                self._last['fsa_deg'] = fsa_out
                self._last['fsa_trend_deg'] = self._fsa_trend(self._fsa_win)

        return {
            'cva_deg': cva_out if cva_valid else self._last['cva_deg'],
            'cva_valid': cva_valid,
            'fsa_deg': fsa_out if fsa_valid else self._last['fsa_deg'],
            'fsa_valid': fsa_valid,
            'fsa_trend_deg': self._last['fsa_trend_deg'],
        }

    # ---------- 内部实现 ----------

    @staticmethod
    def _avg(pts: dict, ids: Tuple[int, ...]) -> Optional[Point]:
        """取 pts 中 ids 内所有可见点的中点（至少 1 个可见）。"""
        pts_l = [pts[pid] for pid in ids if pid in pts]
        if not pts_l:
            return None
        if len(pts_l) == 1:
            return pts_l[0]
        return ((pts_l[0][0] + pts_l[1][0]) / 2.0,
                (pts_l[0][1] + pts_l[1][1]) / 2.0)

    @staticmethod
    def _angle_from_horizontal(a: Point, b: Point) -> float:
        """向量 a→b 与水平线（x 轴方向）的夹角（度，0~90）。

        本原语**只给几何夹角，不带方向语义** —— "越大越前伸"还是"越小越前伸"
        由调用方定义。本文件里用它的三个量方向**一致**，都是**越小越前伸**
        （向量越接近竖直 → 越接近 90°）：
            * ErgonomicRiskFeatures.cva_deg（耳→肩）
            * ErgonomicRiskFeatures.fsa_deg（肩→肘）
            * CvaProxyFeatures.cva_proxy_deg（肩→耳；同一几何，换成像素空间）
        **本文件里唯一方向相反的是 SternumEarAngleFeatures.sternum_ear_angle
        （越大越前伸），而它不调用本函数**（自己用 _angle_at_vertex 算两条边的
        夹角）。两路要一起用（融合 / 算相关 / 共用阈值）之前必须先换算方向，
        见 CvaProxyFeatures 与 SternumEarAngleFeatures 两处计算点的注释。
        """
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return math.degrees(math.atan2(abs(dy), abs(dx)))

    def _compute_frame(self, pose_result: Optional[dict]):
        """单帧算 CVA / FSA 原始值；对应关键点不足时为 None（各自独立）。"""
        if pose_result is None:
            return None, None
        landmarks = pose_result.get('landmarks')
        if not landmarks:
            return None, None

        valid = self._extract_valid(landmarks, EAR_IDS + SHOULDER_IDS + ELBOW_IDS)
        ear = self._avg(valid, EAR_IDS)
        shoulder = self._avg(valid, SHOULDER_IDS)   # C7 近似（思路一，无偏移）
        elbow = self._avg(valid, ELBOW_IDS)

        # 两点重合（退化检测）时角度无意义，按数据不足处理
        cva = None
        if ear is not None and shoulder is not None \
                and _dist(ear, shoulder) > 1e-9:
            cva = self._angle_from_horizontal(shoulder, ear)
        fsa = None
        if shoulder is not None and elbow is not None \
                and _dist(shoulder, elbow) > 1e-9:
            fsa = self._angle_from_horizontal(shoulder, elbow)
        return cva, fsa

    def _trim_window(self, win: deque, now: float) -> None:
        """按 window_seconds 裁剪窗口内超时的旧帧（CVA/FSA 各自的窗口）。"""
        cutoff = now - self.window_seconds
        while win and win[0][0] < cutoff:
            win.popleft()

    def _win_ready(self, win: deque) -> bool:
        """窗口是否已攒够帧数 + 时间跨度（与基类 _window_ready 同语义）。"""
        if len(win) < self.min_frames:
            return False
        return win[-1][0] - win[0][0] >= self.min_window_seconds

    def _smoothed(self, win: deque) -> Optional[float]:
        """窗口内原始值的中位数（平滑用，需已通过 _win_ready）。"""
        if not win:
            return None
        return self._median([v for _t, v in win])

    def _fsa_trend(self, win: deque) -> Optional[float]:
        """FSA 相对窗口内最早帧的变化量（度）：当前平滑值 − 最早帧原始值。"""
        if not win:
            return None
        return self._smoothed(win) - win[0][1]


# ---------------------------------------------------------------------------
# CvaProxyFeatures：CVA-like proxy（像素空间）—— FHP 判据用的那一枚
# （2026-09-23 新增，纯 stdlib；**工程化代理指标，不是临床 CVA**）
# ---------------------------------------------------------------------------

class CvaProxyFeatures(ErgonomicRiskFeatures):
    """CVA-like proxy：耳-肩连线与水平线的夹角（度，像素空间），越小 = 头越前伸。

    === 是什么 / 不是什么（重要）===
    这是**工程化代理指标（proxy）**，不是临床 CVA：
        * 标准 CVA 在矢状面（纯侧面）由 C7 与耳屏（tragus）连线量；
        * 本系统是单 RGB 摄像头、机位在正前方到约 45° 前侧之间，C7 不可见
          —— C7 用双肩中点近似（真实 C7 在肩点后方且更靠上，方向偏差稳定、
          量值偏移，所以文献阈值不能直接套，见 decision.CVA_PROXY_VIEW_PRESETS）；
        * 头参考是 RTMPose 的"耳"点（3/4），不是耳屏。
    所以本指标**不等于临床 CVA**，数值**不能与文献 CVA 阈值（如 50°）比较**，
    阈值一律按本系统自己的机位自标定。
    同理：本文件里任何"耳-肩水平位移"类的量只能叫 Ear-Shoulder displacement
    proxy —— 单目图像里的 x/y 位移不等于真实的 3D 前伸距离，别叫 Forward Head
    Distance（见 CLAUDE.md / 本类注释）。

    === 数学定义（单帧，像素空间）===
        C7 代理 = 可见肩点(5/6)的中点（两肩都可见取中点，单肩取可见侧）
        头参考  = 可见耳点(3/4)的中点（同上）
        参与点先过置信度门控 visibility >= visibility_min（默认 0.3）
        cva_proxy = atan2(|Δy_px|, |Δx_px|)，单位度，范围 0~90
                    （耳在肩正上方 = 90°，头前伸越大越接近 0）

    === 为什么必须在像素空间算（不能直接用 features 里现成的 cva_deg）===
    归一化坐标是"先被画面宽高除过"的，角度会带上纵横比偏差：
        tan θ_像素 = (H/W) · tan θ_归一化
    2026-09-23 实测（front.csv / f45.csv，1280x720 取景）：两者 tan 之比的中位数
    稳定等于 0.5625 = H/W，即 16:9 取景下归一化角系统性偏大；45° 机位同一批帧的
    中位数差到约 10°（归一化 75.5° vs 像素 65.1°）。FHP 判据的阈值是在**像素
    空间**标定的（= 录制 CSV 里的 cva_proxy 列），所以判据必须用本类。
    ErgonomicRiskFeatures.cva_deg 保持原语义不动（那条线是"文献化 CVA 展示"，
    阈值待标定，见 CLAUDE.md 开放问题）。

    继承 ErgonomicRiskFeatures 只为复用取点/几何原语（_extract_valid / _avg /
    _angle_from_horizontal），语义上不是父类的特例 —— 与本模块
    _WindowedMedianExtractor"共享原语"的做法一致。

    === 时间平滑：一阶 EMA（指数滑动平均）===
        alpha = 1 - exp(-dt / smooth_tau_sec)，与帧率无关；
        只在能算出 raw 的帧上推进（丢帧期间不推进 → 恢复后按累积 dt 重新贴合，
        不会把断开前后的两段平均到一起）。
        raw 值同时保留在 cva_proxy_raw_deg（debug / 离线分析用）：判据消费平滑值，
        raw 只报不用。
        预热：沿用父类语义，凑够 min_frames 帧且跨度 >= min_window_seconds 前
        不出平滑值（刚开摄像头不报警），raw 从第一帧就有。

    === 降级（关键点缺失 / 低置信 / 取不到 image_size）===
    本帧不算 → cva_proxy_raw_deg=None、cva_proxy_valid=False；平滑值与
    conf/pts/geom 都**维持上一有效值**（从未有效则 None），不报错、不出 0。
    取不到 image_size（构造不出的 pose dict）也算降级：像素空间角度算不了，
    宁可不判也不出一个带纵横比偏差的数。

    接口约定（保持稳定，别改签名）：
        CvaProxyFeatures(window_seconds=3.0, min_frames=5,
                         min_window_seconds=1.0, visibility_min=0.3,
                         smooth_tau_sec=1.0)
        update(pose_result, timestamp=None) -> dict
            {'cva_proxy_deg':       float | None   EMA 平滑值（判据用这个）
             'cva_proxy_raw_deg':   float | None   本帧原始值（debug 用，无效帧为 None）
             'cva_proxy_valid':     bool           本帧是否真的算出了读数
                                                   （False = 预热中/降级/维持旧值）
             'cva_proxy_conf':      float | None   本帧参与点的平均置信度 0~1
             'cva_proxy_pts':       str | None     本帧用了哪些点，如 "ear_mid+sh_mid"
             'cva_proxy_geom':      dict | None    {'head': (x_px, y_px),
                                                   'c7': (x_px, y_px)}
                                                   供 output 层画辅助线用
                                                   （output 不重算几何）}
        reset()
    """

    def __init__(self,
                 window_seconds: float = 3.0,
                 min_frames: int = 5,
                 min_window_seconds: float = 1.0,
                 visibility_min: float = 0.3,
                 smooth_tau_sec: float = 1.0) -> None:
        """
        参数:
            window_seconds / min_frames / min_window_seconds: 只用于"预热"语义
                （凑够帧数 + 时间跨度前不出平滑值），与父类同约定。
            visibility_min: 某关键点的 visibility 低于此值视为无效点（置信度过滤）。
            smooth_tau_sec: EMA 时间常数（秒），越小越跟手、越大越稳。默认 1.0
                （30fps 下 alpha ≈ 0.033，约 1 秒内跟上真实变化）。
        """
        super().__init__(window_seconds=window_seconds, min_frames=min_frames,
                         min_window_seconds=min_window_seconds,
                         visibility_min=visibility_min)
        if smooth_tau_sec <= 0:
            raise ValueError("smooth_tau_sec 必须 > 0")
        self.smooth_tau_sec = smooth_tau_sec

        self._ema: Optional[float] = None       # 平滑值（只在有效帧上推进）
        self._last_ts: Optional[float] = None   # 上一有效帧时间戳（EMA 的 dt 基准）
        self._warm_start: Optional[float] = None  # 预热窗口起点
        self._warm_n = 0                        # 预热窗口内的有效帧数
        self._last = {                          # 上一有效读数（降级时维持）
            'deg': None, 'conf': None, 'pts': None, 'geom': None,
        }

    def reset(self) -> None:
        """清空平滑状态与上一有效值，重新开始。"""
        super().reset()
        self._ema = None
        self._last_ts = None
        self._warm_start = None
        self._warm_n = 0
        self._last = {'deg': None, 'conf': None, 'pts': None, 'geom': None}

    def update(self,
               pose_result: Optional[dict],
               timestamp: Optional[float] = None) -> dict:
        """喂入一帧 pose 结果，返回 CVA-like proxy 特征（见类 docstring 的返回结构）。

        参数:
            pose_result: pose_estimation.detect_pose() 的返回值
                         {'landmarks': [(x, y, z, visibility) x17],
                          'image_size': (w, h)}；未检测到人体时为 None。
                         image_size 缺失时本帧降级（像素空间角度算不了）。
            timestamp:   该帧的单调时间戳（秒），默认 time.monotonic()；
                         视频回放/自测请传入模拟时间戳（与其它特征同约定）——
                         EMA 的 dt 与预热都按它算。
        """
        now = timestamp if timestamp is not None else time.monotonic()

        raw = conf = pts = geom = None
        if pose_result is not None:
            landmarks = pose_result.get('landmarks')
            size = pose_result.get('image_size')
            if landmarks and size and size[0] > 0 and size[1] > 0:
                w, h = size
                valid = self._extract_valid(landmarks, EAR_IDS + SHOULDER_IDS)
                ear = self._avg(valid, EAR_IDS)
                shoulder = self._avg(valid, SHOULDER_IDS)
                if ear is not None and shoulder is not None:
                    # 归一化 -> 像素：这一步就是本类与父类的全部差别（纵横比）
                    ear_px = (ear[0] * w, ear[1] * h)
                    sh_px = (shoulder[0] * w, shoulder[1] * h)
                    if _dist(ear_px, sh_px) > 1e-9:
                        # 方向：**越小越前伸**（耳在肩正上方 = 90°，头前伸越大越趋 0），
                        # 与标准 CVA 同向。⚠ 正面那路 SternumEarAngleFeatures 是
                        # **越大越前伸**，与本值方向相反 —— 两路要一起用（融合 / 算
                        # 相关 / 共用阈值）前必须先把方向换算一致，别直接比大小或相加。
                        # 项目既有做法是取比值统一成"越大越差"（decision.FhpDecision
                        # 的 ratio_proxy = 阈值 / proxy）。
                        raw = self._angle_from_horizontal(sh_px, ear_px)
                        conf = self._mean_visibility(landmarks, valid)
                        pts = self._pts_label(valid)
                        geom = {'head': ear_px, 'c7': sh_px}

        if raw is not None:
            self._ema = self._ema_step(raw, now)
            self._warm_tick(now)

        deg = None
        is_valid = False
        if raw is not None and self._warm_ready():
            deg = self._ema
            is_valid = True
            self._last.update(deg=deg, conf=conf, pts=pts, geom=geom)

        return {
            'cva_proxy_deg': deg if is_valid else self._last['deg'],
            'cva_proxy_raw_deg': raw,
            'cva_proxy_valid': is_valid,
            'cva_proxy_conf': conf if is_valid else self._last['conf'],
            'cva_proxy_pts': pts if is_valid else self._last['pts'],
            'cva_proxy_geom': geom if is_valid else self._last['geom'],
        }

    # ---------- 内部实现 ----------

    def _ema_step(self, raw: float, now: float) -> float:
        """一阶 EMA 推进一步：alpha = 1 - exp(-dt / tau)（与帧率无关）。

        首帧（或 reset 后）直接取 raw 作为种子。dt 从"上一**有效**帧"算起，
        所以丢帧期间不推进；丢失较久后 dt 变大 → alpha → 1 → 直接贴合新值，
        不会把断开前后的两段平均在一起。
        """
        if self._ema is None or self._last_ts is None:
            value = raw
        else:
            dt = max(0.0, now - self._last_ts)
            alpha = 1.0 - math.exp(-dt / self.smooth_tau_sec)
            value = self._ema + alpha * (raw - self._ema)
        self._last_ts = now
        return value

    def _warm_tick(self, now: float) -> None:
        """预热窗口计数（只统计有效帧）。"""
        if self._warm_n == 0:
            self._warm_start = now
        self._warm_n += 1

    def _warm_ready(self) -> bool:
        """预热是否结束：帧数够 + 时间跨度够（与父类 _win_ready 同语义）。"""
        if self._warm_n < self.min_frames or self._warm_start is None \
                or self._last_ts is None:
            return False
        return self._last_ts - self._warm_start >= self.min_window_seconds

    @staticmethod
    def _mean_visibility(landmarks: list, valid: Dict[int, Point]) -> Optional[float]:
        """本帧参与计算的点（已过门控）的平均置信度 0~1（供显示/审计）。"""
        vals = [landmarks[pid][3] for pid in valid if pid < len(landmarks)]
        if not vals:
            return None
        return sum(vals) / len(vals)

    @staticmethod
    def _pts_label(valid: Dict[int, Point]) -> str:
        """本帧用了哪些点，如 "ear_mid+sh_mid" / "earL+sh_mid"（与录制 CSV 的
        head_ref / c7_ref 命名一致，方便离线核对）。"""
        if 3 in valid and 4 in valid:
            head = "ear_mid"
        else:
            head = "earL" if 3 in valid else "earR"
        if 5 in valid and 6 in valid:
            c7 = "sh_mid"
        else:
            c7 = "shL" if 5 in valid else "shR"
        return f"{head}+{c7}"


# ---------------------------------------------------------------------------
# EarShoulderProxyFeatures：正面机位的 Ear–Shoulder displacement proxy
# （2026-09-23 新增，纯 stdlib；**实验特征，尚未接入任何 decision**）
# ---------------------------------------------------------------------------

class EarShoulderProxyFeatures(ErgonomicRiskFeatures):
    """Ear–Shoulder displacement proxy：耳-肩水平位移 ÷ 肩宽（**实验特征**）。

    === 是什么 / 不是什么（重要）===
    这是**图像平面里的相对水平位移的代理指标（proxy）**：
        * 它量的是 2D 图像平面上"耳中点相对肩中点在水平方向偏了多少"，
          用肩宽归一化掉远近/体型；
        * 它**不是**真实的 3D 前伸距离 —— 单目图像里头的水平位移既可能来自
          头前伸，也可能来自整个人的体重塌陷、椅子/取景变化、身体旋转；
        * 所以只能叫 **Ear–Shoulder displacement proxy**，**不能**叫 Forward
          Head Distance（FHD）、**不能**声称是 3D forward displacement、
          **不能**叫 clinical FHD（与 CvaProxyFeatures 的命名纪律同源）。
    引入原因：正面机位下"耳-肩连线 vs 水平线"的**角度**会被 ~280px 的耳-肩
    竖直间距压掉（实测 Δx 19/29/53px 经 atan 只剩 3.9°/6.5°/13.9°，Normal 与
    Slight 只差 2.7° ≈ 噪声），而 **Δx/肩宽** 在正面实测单调且分得开
    （Normal/Slight/Obvious ≈ 0.041/0.061/0.109，2026-09-23 单人多机位一次观测）。
    **方向与 cva_proxy_deg 相反**：本指标**越大越前伸**（cva_proxy_deg 越小越前伸）。

    === 数学定义（单帧，像素空间）===
        参与点先过置信度门控 visibility >= visibility_min（默认 0.3）；
        双肩**都**可见才算（分母要两肩），耳至少可见一只：
            耳中点   = 可见耳点(3/4)的中点（单耳时就是那只）
            肩中点   = 双肩点(5/6)的中点（**C7 近似**，与 CvaProxyFeatures 同）
            肩宽     = |肩点5 − 肩点6| 的**欧氏像素距离**
            ear_shoulder_proxy = |耳中点x − 肩中点x| ÷ 肩宽     （无单位，>= 0）

    === 为什么在像素空间算 ===
    分子分母同为长度，大多数情况下比值与坐标空间无关，但归一化坐标下
    `tan θ` 类量会带上纵横比：肩宽用欧氏距离时，肩线一有倾斜，归一化空间里
    的 Δy 会被 H 缩放得与 Δx 不同步，比值就不再干净。固定在**像素空间**算
    （与 CvaProxyFeatures 同一理由），换分辨率不影响、换纵横比仍建议重标。

    === 视角退化（肩宽塌陷）===
    侧身时肩宽被投影压扁（正对归一化肩宽 ~0.56、侧身 0.01~0.21，见 CLAUDE.md），
    分母变小会把比值放大成无意义的大数 → 归一化肩宽 < min_shoulder_width_norm
    （默认 MIN_SHOULDER_WIDTH_NORM=0.25）的帧判**视角退化**：
    `ear_shoulder_proxy_valid=False`，不进 EMA、不计预热、不进统计；但
    raw 值与肩宽仍照常返回（CSV 里看得到），便于审计退化帧。

    === 时间平滑：一阶 EMA（与 CvaProxyFeatures 同一套）===
        alpha = 1 - exp(-dt / smooth_tau_sec)，与帧率无关；只在"有效帧"
        （非退化、能算出 raw）上推进；raw 同时保留在 ear_shoulder_proxy_raw，
        两个值都进 CSV —— 离线分析时**先看 raw 的组内离散度**，别让平滑冒充
        可重复性（做测量验证时平滑值只作参考）。
        预热：凑够 min_frames 帧且跨度 >= min_window_seconds 前不出平滑值
        （valid=False），raw 从第一帧就有。

    === 降级（关键点缺失 / 低置信 / 取不到 image_size）===
    本帧不算 → raw=None、valid=False；平滑值与 conf/pts/geom 都维持上一有效值
    （从未有效则 None），不报错、不出 0。
    `ear_shoulder_proxy_parts`（肩宽/中点 x/四个单点置信度）是**本帧**的原始量
    —— 即使本帧退化/降级也照常返回，这样 CSV 里能看出"为什么这帧不算数"。

    === 与 decision 的关系（2026-09-28 起）===
    本类**只算不判**，且**正面生产入口已不再装配它**：2026-09-28 的对照测量
    （1 人 × 2 机位 × 各 3 段，判据事先固定）里，本指标在正面**单调性 FAIL**
    （第一段 Obvious 的中位数 0.064 低于第一段 Slight 的 0.071），而
    SternumEarAngleFeatures 三段全单调、抖动更低（0.076 vs 0.252），
    于是正面那条显示行换成了后者。**但帧级准确率不是全面更好**：同一批数据、
    各自生产上界（本类 0.065 / 新角度 54.23°）、同按 EMA 口径比，本类误报
    7.6% / 漏报 24.4%，新角度 9.5% / 19.9% —— 新角度多误报、少漏报。换它的
    理由就是单调性与抖动（两档区间重叠也是后者更小），不是"更准"。
    本类现在**只服务测量脚本** tools/test_front_ear_shoulder_proxy.py（阈值也在
    那里，标明 experimental）；保留不删是因为该脚本与
    tools/test_molaeifar_sternum_angle.py 都 import 它。45° 机位继续用
    CvaProxyFeatures + FhpDecision，与本类无关。

    继承 ErgonomicRiskFeatures 与 CvaProxyFeatures 同理：只为复用取点原语
    （_extract_valid / _avg），语义上不是父类的特例 —— 父类的 CVA/FSA 语义
    与本类无关（本类不读、不写它们的字段）。

    接口约定（保持稳定，别改签名）：
        EarShoulderProxyFeatures(window_seconds=3.0, min_frames=5,
                                 min_window_seconds=1.0, visibility_min=0.3,
                                 smooth_tau_sec=1.0,
                                 min_shoulder_width_norm=0.25)
        update(pose_result, timestamp=None) -> dict
            {'ear_shoulder_proxy':        float | None  EMA 平滑值（越大越前伸）
             'ear_shoulder_proxy_raw':    float | None  本帧原始值（未平滑）
             'ear_shoulder_proxy_valid':  bool          本帧能否用于统计/判据
             'ear_shoulder_proxy_conf':   float | None  参与点平均置信度 0~1
             'ear_shoulder_proxy_pts':    str | None    如 "ear_mid+sh_mid"
             'ear_shoulder_proxy_geom':   dict | None   {'ear_mid','sh_mid',
                                                        'sh_width_px'}（像素）
             'ear_shoulder_proxy_parts':  dict          本帧原始量（CSV 列）：
                                        'shoulder_width','shoulder_width_norm',
                                        'ear_mid_x','shoulder_mid_x',
                                        'left_ear_confidence','right_ear_confidence',
                                        'left_shoulder_confidence',
                                        'right_shoulder_confidence'
                                        （取不到的量 = None，不填 0）
             'ear_shoulder_proxy_degenerate': bool      本帧是否视角退化（肩宽塌陷）}
        reset()
    """

    def __init__(self,
                 window_seconds: float = 3.0,
                 min_frames: int = 5,
                 min_window_seconds: float = 1.0,
                 visibility_min: float = 0.3,
                 smooth_tau_sec: float = 1.0,
                 min_shoulder_width_norm: float = MIN_SHOULDER_WIDTH_NORM) -> None:
        """
        参数:
            window_seconds / min_frames / min_window_seconds: 只用于"预热"语义
                （凑够帧数 + 时间跨度前不出平滑值），与其它特征同约定。
            visibility_min: 某关键点的 visibility 低于此值视为无效点（置信度过滤）。
            smooth_tau_sec: EMA 时间常数（秒），默认 1.0（与 CvaProxyFeatures 同）。
            min_shoulder_width_norm: 归一化肩宽下限，低于它判视角退化（见类 docstring）。
        """
        super().__init__(window_seconds=window_seconds, min_frames=min_frames,
                         min_window_seconds=min_window_seconds,
                         visibility_min=visibility_min)
        if smooth_tau_sec <= 0:
            raise ValueError("smooth_tau_sec 必须 > 0")
        if min_shoulder_width_norm <= 0:
            raise ValueError("min_shoulder_width_norm 必须 > 0")
        self.smooth_tau_sec = smooth_tau_sec
        self.min_shoulder_width_norm = min_shoulder_width_norm

        self._ema: Optional[float] = None         # 平滑值（只在有效帧上推进）
        self._last_ts: Optional[float] = None     # 上一有效帧时间戳（EMA 的 dt 基准）
        self._warm_start: Optional[float] = None  # 预热窗口起点
        self._warm_n = 0                          # 预热窗口内的有效帧数
        self._last = {                            # 上一有效读数（降级时维持）
            'deg': None, 'conf': None, 'pts': None, 'geom': None,
        }

    def reset(self) -> None:
        """清空平滑状态与上一有效值，重新开始。"""
        super().reset()
        self._ema = None
        self._last_ts = None
        self._warm_start = None
        self._warm_n = 0
        self._last = {'deg': None, 'conf': None, 'pts': None, 'geom': None}

    def update(self,
               pose_result: Optional[dict],
               timestamp: Optional[float] = None) -> dict:
        """喂入一帧 pose 结果，返回 Ear–Shoulder displacement proxy（见类 docstring）。"""
        now = timestamp if timestamp is not None else time.monotonic()

        raw = conf = pts = geom = None
        degenerate = False
        parts = self._empty_parts()
        if pose_result is not None:
            landmarks = pose_result.get('landmarks')
            size = pose_result.get('image_size')
            if landmarks and size and size[0] > 0 and size[1] > 0:
                w, h = size
                valid = self._extract_valid(landmarks, EAR_IDS + SHOULDER_IDS)
                parts = self._parts(landmarks, valid, w, h)
                ear = self._avg(valid, EAR_IDS)
                if ear is not None and 5 in valid and 6 in valid:
                    sh_l, sh_r = valid[5], valid[6]
                    sw_norm = _dist(sh_l, sh_r)
                    degenerate = sw_norm < self.min_shoulder_width_norm
                    ear_px = (ear[0] * w, ear[1] * h)
                    sh_px = (((sh_l[0] + sh_r[0]) / 2.0) * w,
                             ((sh_l[1] + sh_r[1]) / 2.0) * h)
                    sw_px = _dist((sh_l[0] * w, sh_l[1] * h),
                                  (sh_r[0] * w, sh_r[1] * h))
                    if sw_px > 1e-9:
                        # 全部在像素空间（见类 docstring）：水平位移 ÷ 肩宽
                        raw = abs(ear_px[0] - sh_px[0]) / sw_px
                        conf = self._mean_visibility(landmarks, valid)
                        pts = self._pts_label(valid)
                        geom = {'ear_mid': ear_px, 'sh_mid': sh_px,
                                'sh_width_px': sw_px}

        # 退化帧（肩宽塌陷）比值无意义：raw 照常返回供审计，但不推进 EMA、
        # 不计预热、不算有效读数（raw 在 CSV 里看得到，valid/degenerate 标出来）。
        if raw is not None and not degenerate:
            self._ema = self._ema_step(raw, now)
            self._warm_tick(now)

        deg = None
        is_valid = False
        if raw is not None and not degenerate and self._warm_ready():
            deg = self._ema
            is_valid = True
            self._last.update(deg=deg, conf=conf, pts=pts, geom=geom)

        return {
            'ear_shoulder_proxy': deg if is_valid else self._last['deg'],
            'ear_shoulder_proxy_raw': raw,
            'ear_shoulder_proxy_valid': is_valid,
            'ear_shoulder_proxy_conf': conf if is_valid else self._last['conf'],
            'ear_shoulder_proxy_pts': pts if is_valid else self._last['pts'],
            'ear_shoulder_proxy_geom': geom if is_valid else self._last['geom'],
            'ear_shoulder_proxy_parts': parts,
            'ear_shoulder_proxy_degenerate': degenerate,
        }

    # ---------- 内部实现 ----------

    def _ema_step(self, raw: float, now: float) -> float:
        """一阶 EMA 推进一步（与 CvaProxyFeatures 同一套：alpha = 1 - exp(-dt/tau)）。"""
        if self._ema is None or self._last_ts is None:
            value = raw
        else:
            dt = max(0.0, now - self._last_ts)
            alpha = 1.0 - math.exp(-dt / self.smooth_tau_sec)
            value = self._ema + alpha * (raw - self._ema)
        self._last_ts = now
        return value

    def _warm_tick(self, now: float) -> None:
        """预热窗口计数（只统计能算出 raw 的非退化帧）。"""
        if self._warm_n == 0:
            self._warm_start = now
        self._warm_n += 1

    def _warm_ready(self) -> bool:
        """预热是否结束：帧数够 + 时间跨度够。"""
        if self._warm_n < self.min_frames or self._warm_start is None \
                or self._last_ts is None:
            return False
        return self._last_ts - self._warm_start >= self.min_window_seconds

    @staticmethod
    def _empty_parts() -> dict:
        """本帧原始量的空壳（取不到的量为 None，不填 0）。"""
        return {
            'shoulder_width': None, 'shoulder_width_norm': None,
            'ear_mid_x': None, 'shoulder_mid_x': None,
            'left_ear_confidence': None, 'right_ear_confidence': None,
            'left_shoulder_confidence': None, 'right_shoulder_confidence': None,
        }

    def _parts(self, landmarks: list, valid: Dict[int, Point],
               w: float, h: float) -> dict:
        """本帧的原始量（全部像素 / 原始置信度），供逐帧 CSV 审计。

        中点/肩宽用**过了置信度门控**的点（与指标同一个口径，改口径时两处一起改）；
        四个单点置信度取**原始值**（含被门控掉的）—— 这样退化帧能看出是被哪只
        耳/哪侧肩拖下来的。
        """
        out = self._empty_parts()
        for pid, name in ((3, 'left_ear_confidence'), (4, 'right_ear_confidence'),
                          (5, 'left_shoulder_confidence'),
                          (6, 'right_shoulder_confidence')):
            if pid < len(landmarks):
                out[name] = float(landmarks[pid][3])
        ear = self._avg(valid, EAR_IDS)
        if ear is not None:
            out['ear_mid_x'] = ear[0] * w
        if 5 in valid and 6 in valid:
            sh_l, sh_r = valid[5], valid[6]
            out['shoulder_mid_x'] = (sh_l[0] + sh_r[0]) / 2.0 * w
            out['shoulder_width_norm'] = _dist(sh_l, sh_r)
            out['shoulder_width'] = _dist((sh_l[0] * w, sh_l[1] * h),
                                          (sh_r[0] * w, sh_r[1] * h))
        return out

    @staticmethod
    def _mean_visibility(landmarks: list, valid: Dict[int, Point]) -> Optional[float]:
        """本帧参与计算的点（已过门控）的平均置信度 0~1（供显示/审计）。"""
        vals = [landmarks[pid][3] for pid in valid if pid < len(landmarks)]
        if not vals:
            return None
        return sum(vals) / len(vals)

    @staticmethod
    def _pts_label(valid: Dict[int, Point]) -> str:
        """本帧用了哪些点，如 "ear_mid+sh_mid" / "earL+sh_mid"（双肩是硬要求）。"""
        if 3 in valid and 4 in valid:
            ear = "ear_mid"
        else:
            ear = "earL" if 3 in valid else "earR"
        return f"{ear}+sh_mid"


# ---------------------------------------------------------------------------
# SternumEarAngleFeatures：正面机位的 Sternum–Ear 角度
# （2026-09-28 新增，纯 stdlib；实验特征，只喂正面"只显示不提醒"那一行）
# ---------------------------------------------------------------------------

class SternumEarAngleFeatures(ErgonomicRiskFeatures):
    """Sternum–Ear Angle：顶点（双肩中点）到左右耳两条边的夹角（**实验特征**）。

    === 是什么 / 不是什么（重要）===
    构造出处：Molaeifar et al. 2021, Work 68(4):1221-1227 —— 该文用 8 相机
    动作捕捉（58 名健康受试者）报告"**胸骨中点与左右耳屏**三点夹的正面角"
    与 3D CVA 有中等相关（FHP 由屈颈 + 低头诱发）。本类复刻的是**那个构造**，
    **不是那个量**：
        * 顶点用**双肩中点**近似胸骨中点（真实胸骨中点在肩线下方）；
        * 两条边指向**耳关键点**（COCO 3/4），不是耳屏（tragus）；
        * 原文量的是**三维**角，本类量的是**二维像面角**（投影会压扁它）；
        * 敏感性实测（2026-09-28）：顶点下移 0 / 0.15 / 0.30 个肩宽，正面三档
          都保持单调 + 两档 IQR 分离（抖动比 0.076 → 0.083 → 0.090）——
          "肩中点≈胸骨中点"这个近似不是限制项，机位才是。
    所以它是**工程代理**，量值与文献不可比，**不是临床指标**，更**不能叫 CVA**。
    定位：正面第 4 条（半条）实验显示线，**无 Alert、不参与任何判定**。
    方向：**角度越大越前伸**（与 Ear–Shoulder 同向，与 cva_proxy_deg 相反）。

    === 数学定义（单帧，像素空间）===
        双耳(3/4) + 双肩(5/6)**四个点都**过置信度门控（visibility >=
        visibility_min，默认 0.3）且归一化肩宽 >= min_shoulder_width_norm：
            顶点 = 双肩中点（像素）
            两条边 = 顶点 → 左耳(3)、顶点 → 右耳(4)
            sternum_ear_angle = 两条边在顶点处的夹角（度，0~180）
        实测量级（正面，1 人 × 3 段，EMA）：坐直 ~44~47°、Slight ~55~59°、
        Obvious ~65~75°。**四个点缺一个就算不出**（顶点要双肩、两条边要双耳）
        —— 比 EarShoulderProxyFeatures 的"双肩+任一耳"更严，正面实测代价约
        2% 帧；45° 机位远侧耳常过不了门，故本特征只用于正面。

    === 为什么在像素空间算 ===
    归一化坐标的 x 被 W 缩放、y 被 H 缩放，夹角会带上画面纵横比偏差：同一姿势
    16:9 下差约 10°（与 CvaProxyFeatures 同一条纪律）。**阈值只对同一空间有效**。
    这也正是本类不继承父类的角度算法、自己重算一遍的原因。

    === 视角退化（肩宽塌陷）===
    与 EarShoulderProxyFeatures 同一口径（共用 MIN_SHOULDER_WIDTH_NORM=0.25）：
    归一化肩宽低于下限的帧判"视角退化"→ valid=False、不进 EMA、不计预热，
    但 raw 与几何仍照常返回供审计（CSV 里看得见这个退化帧）。

    === 阈值只对正面有效（硬约束）===
    2026-09-28 实测同口径中点：正面 **54.23°**、45° 机位 **45.02°（raw）/ 45.09°
    （EMA）**；**拿正面阈值套 45° 机位 → FHP 漏报 100%、误报 0%**（该机位读数
    整体低于正面阈值）。换机位必须重标 —— 同 CVA_PROXY_VIEW_PRESETS 那条教训
    （见 decision.py）。

    === 与颈压缩高度冗余（用它之前先知道）===
    同一批数据段级 Pearson：正面 −0.989、45° −0.935；与"耳-肩竖直间距 ÷ 两耳
    间距"的段级 Spearman 正面 −1.000。**本角度基本是那个比值的三角函数重写，
    没有引入新信息**，只是在正面实测抖动更低（0.076 vs 颈压缩 0.077）。
    `sternum_ear_angle_parts` 里的 ear_sh_vert / ear_span 就是留作这个核对用的。

    === 时间平滑：一阶 EMA（与 CvaProxyFeatures 同一套）===
        alpha = 1 - exp(-dt / smooth_tau_sec)，与帧率无关；只在有效帧上推进；
        raw 同时保留在 sternum_ear_angle_raw —— 离线分析**先看 raw 的组内
        离散度**，别让平滑冒充可重复性。预热：凑够 min_frames 帧且跨度 >=
        min_window_seconds 前不出平滑值（valid=False），raw 从第一帧就有。

    === 降级（关键点缺失 / 低置信 / 取不到 image_size）===
    本帧不算 → raw=None、valid=False；平滑值与 conf/pts/geom 都维持上一有效值
    （从未有效则 None），不报错、不出 0。`sternum_ear_angle_parts` 是**本帧**的
    原始量，退化/降级也照常返回，CSV 里能看出"为什么这帧不算数"。

    继承 ErgonomicRiskFeatures 与 EarShoulderProxyFeatures 同理：只为复用取点
    原语（_extract_valid / _avg），语义上不是父类的特例 —— 父类的 CVA/FSA 语义
    与本类无关（本类不读、不写它们的字段）。

    接口约定（保持稳定，别改签名）：
        SternumEarAngleFeatures(window_seconds=3.0, min_frames=5,
                                min_window_seconds=1.0, visibility_min=0.3,
                                smooth_tau_sec=1.0,
                                min_shoulder_width_norm=0.25)
        update(pose_result, timestamp=None) -> dict
            {'sternum_ear_angle':        float | None  EMA 平滑值（度，越大越前伸）
             'sternum_ear_angle_raw':    float | None  本帧原始角度（未平滑）
             'sternum_ear_angle_valid':  bool          本帧能否用于显示/统计
             'sternum_ear_angle_conf':   float | None  参与点（四点）平均置信度 0~1
             'sternum_ear_angle_pts':    str | None    固定 "ear_mid+sh_mid"
             'sternum_ear_angle_geom':   dict | None   {'vertex','ear_l','ear_r',
                                                        'sh_width_px'}（像素，
                                                        供 output 画 V 形）
             'sternum_ear_angle_parts':  dict          本帧原始量（CSV 列）：
                                        'shoulder_width','shoulder_width_norm',
                                        'ear_span','ear_sh_vert',
                                        'vertex_x','vertex_y',
                                        'left_ear_confidence','right_ear_confidence',
                                        'left_shoulder_confidence',
                                        'right_shoulder_confidence'
                                        （取不到的量 = None，不填 0）
             'sternum_ear_angle_degenerate': bool      本帧是否视角退化（肩宽塌陷）}
        reset()
    """

    def __init__(self,
                 window_seconds: float = 3.0,
                 min_frames: int = 5,
                 min_window_seconds: float = 1.0,
                 visibility_min: float = 0.3,
                 smooth_tau_sec: float = 1.0,
                 min_shoulder_width_norm: float = MIN_SHOULDER_WIDTH_NORM) -> None:
        """
        参数:
            window_seconds / min_frames / min_window_seconds: 只用于"预热"语义
                （凑够帧数 + 时间跨度前不出平滑值），与其它特征同约定。
            visibility_min: 某关键点的 visibility 低于此值视为无效点（置信度过滤）。
            smooth_tau_sec: EMA 时间常数（秒），默认 1.0（与 CvaProxyFeatures 同）。
            min_shoulder_width_norm: 归一化肩宽下限，低于它判视角退化（见类 docstring）。
        """
        super().__init__(window_seconds=window_seconds, min_frames=min_frames,
                         min_window_seconds=min_window_seconds,
                         visibility_min=visibility_min)
        if smooth_tau_sec <= 0:
            raise ValueError("smooth_tau_sec 必须 > 0")
        if min_shoulder_width_norm <= 0:
            raise ValueError("min_shoulder_width_norm 必须 > 0")
        self.smooth_tau_sec = smooth_tau_sec
        self.min_shoulder_width_norm = min_shoulder_width_norm

        self._ema: Optional[float] = None         # 平滑值（只在有效帧上推进）
        self._last_ts: Optional[float] = None     # 上一有效帧时间戳（EMA 的 dt 基准）
        self._warm_start: Optional[float] = None  # 预热窗口起点
        self._warm_n = 0                          # 预热窗口内的有效帧数
        self._last = {                            # 上一有效读数（降级时维持）
            'deg': None, 'conf': None, 'pts': None, 'geom': None,
        }

    def reset(self) -> None:
        """清空平滑状态与上一有效值，重新开始。"""
        super().reset()
        self._ema = None
        self._last_ts = None
        self._warm_start = None
        self._warm_n = 0
        self._last = {'deg': None, 'conf': None, 'pts': None, 'geom': None}

    def update(self,
               pose_result: Optional[dict],
               timestamp: Optional[float] = None) -> dict:
        """喂入一帧 pose 结果，返回 Sternum–Ear 角度（见类 docstring）。"""
        now = timestamp if timestamp is not None else time.monotonic()

        raw = conf = pts = geom = None
        degenerate = False
        parts = self._empty_parts()
        if pose_result is not None:
            landmarks = pose_result.get('landmarks')
            size = pose_result.get('image_size')
            if landmarks and size and size[0] > 0 and size[1] > 0:
                w, h = size
                valid = self._extract_valid(landmarks, EAR_IDS + SHOULDER_IDS)
                parts = self._parts(landmarks, valid, w, h)
                # 四个点缺一个就算不出（顶点要双肩、两条边要双耳）—— 比 ES 类严
                if all(pid in valid for pid in EAR_IDS + SHOULDER_IDS):
                    sh_l_px = (valid[5][0] * w, valid[5][1] * h)
                    sh_r_px = (valid[6][0] * w, valid[6][1] * h)
                    sw_norm = _dist(valid[5], valid[6])
                    degenerate = sw_norm < self.min_shoulder_width_norm
                    sw_px = _dist(sh_l_px, sh_r_px)
                    vertex = ((sh_l_px[0] + sh_r_px[0]) / 2.0,
                              (sh_l_px[1] + sh_r_px[1]) / 2.0)
                    ear_l_px = (valid[3][0] * w, valid[3][1] * h)
                    ear_r_px = (valid[4][0] * w, valid[4][1] * h)
                    # 全部在像素空间（见类 docstring）
                    # 方向：**越大越前伸**（顶点 = 双肩中点，两条边指向左右耳；头前伸时
                    # 两耳在像面上张开 → 夹角变大）。⚠ 与 CvaProxyFeatures.cva_proxy_deg
                    # **方向相反**（那个越小越前伸），与 Ear–Shoulder proxy 同向。
                    # 两路要一起用前必须先换算方向（见 _angle_from_horizontal 的注释）。
                    raw = self._angle_at_vertex(vertex, ear_l_px, ear_r_px)
                    if raw is not None:
                        conf = self._mean_visibility(landmarks, valid)
                        pts = "ear_mid+sh_mid"
                        geom = {'vertex': vertex, 'ear_l': ear_l_px,
                                'ear_r': ear_r_px, 'sh_width_px': sw_px}

        # 退化帧（肩宽塌陷）角度无意义：raw 照常返回供审计，但不推进 EMA、
        # 不计预热、不算有效读数（raw 在 CSV 里看得到，valid/degenerate 标出来）。
        if raw is not None and not degenerate:
            self._ema = self._ema_step(raw, now)
            self._warm_tick(now)

        deg = None
        is_valid = False
        if raw is not None and not degenerate and self._warm_ready():
            deg = self._ema
            is_valid = True
            self._last.update(deg=deg, conf=conf, pts=pts, geom=geom)

        return {
            'sternum_ear_angle': deg if is_valid else self._last['deg'],
            'sternum_ear_angle_raw': raw,
            'sternum_ear_angle_valid': is_valid,
            'sternum_ear_angle_conf': conf if is_valid else self._last['conf'],
            'sternum_ear_angle_pts': pts if is_valid else self._last['pts'],
            'sternum_ear_angle_geom': geom if is_valid else self._last['geom'],
            'sternum_ear_angle_parts': parts,
            'sternum_ear_angle_degenerate': degenerate,
        }

    # ---------- 内部实现 ----------

    @staticmethod
    def _angle_at_vertex(v: Point, a: Point, b: Point) -> Optional[float]:
        """顶点 v 处、两条边分别指向 a / b 的夹角（度，0~180）。

        与 _angle_from_horizontal 同理：本原语**只给几何夹角，不带方向语义**。
        本类用它得到的 sternum_ear_angle 是**越大越前伸**，与
        CvaProxyFeatures.cva_proxy_deg（越小越前伸）**方向相反**。

        任一条边退化（v 与端点重合，长度为 0）→ None（算不出角，别硬给 0）。
        与 PostureFeatures._fold_angle 的区别：那个返回的是"相对 180° 的偏折"，
        本式要的是夹角本身。
        """
        v1 = (a[0] - v[0], a[1] - v[1])
        v2 = (b[0] - v[0], b[1] - v[1])
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 < 1e-9 or n2 < 1e-9:
            return None
        cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

    def _ema_step(self, raw: float, now: float) -> float:
        """一阶 EMA 推进一步（与 CvaProxyFeatures 同一套：alpha = 1 - exp(-dt/tau)）。"""
        if self._ema is None or self._last_ts is None:
            value = raw
        else:
            dt = max(0.0, now - self._last_ts)
            alpha = 1.0 - math.exp(-dt / self.smooth_tau_sec)
            value = self._ema + alpha * (raw - self._ema)
        self._last_ts = now
        return value

    def _warm_tick(self, now: float) -> None:
        """预热窗口计数（只统计能算出 raw 的非退化帧）。"""
        if self._warm_n == 0:
            self._warm_start = now
        self._warm_n += 1

    def _warm_ready(self) -> bool:
        """预热是否结束：帧数够 + 时间跨度够（至少两个时间戳才谈得上"跨度"）。"""
        if self._warm_n < self.min_frames or self._warm_start is None:
            return False
        return self._last_ts is not None and \
            self._last_ts - self._warm_start >= self.min_window_seconds

    @staticmethod
    def _empty_parts() -> dict:
        """本帧原始量的空壳（取不到的量为 None，不填 0）。"""
        return {
            'shoulder_width': None, 'shoulder_width_norm': None,
            'ear_span': None, 'ear_sh_vert': None,
            'vertex_x': None, 'vertex_y': None,
            'left_ear_confidence': None, 'right_ear_confidence': None,
            'left_shoulder_confidence': None, 'right_shoulder_confidence': None,
        }

    def _parts(self, landmarks: list, valid: Dict[int, Point],
               w: float, h: float) -> dict:
        """本帧的原始量（全部像素 / 原始置信度），供逐帧 CSV 审计。

        几何用**过了置信度门控**的点（与指标同一个口径，改口径时两处一起改）；
        四个单点置信度取**原始值**（含被门控掉的）—— 这样退化帧能看出是被哪只
        耳/哪侧肩拖下来的。ear_span（两耳像素距离）与 ear_sh_vert（耳-肩竖直
        间距像素）留着核对"本角度 ≈ atan(ear_sh_vert / ear_span)"的冗余性。
        """
        out = self._empty_parts()
        for pid, name in ((3, 'left_ear_confidence'), (4, 'right_ear_confidence'),
                          (5, 'left_shoulder_confidence'),
                          (6, 'right_shoulder_confidence')):
            if pid < len(landmarks):
                out[name] = float(landmarks[pid][3])
        ear_px = None
        if 3 in valid and 4 in valid:
            ear_l_px = (valid[3][0] * w, valid[3][1] * h)
            ear_r_px = (valid[4][0] * w, valid[4][1] * h)
            ear_px = (ear_l_px, ear_r_px)
            out['ear_span'] = _dist(ear_l_px, ear_r_px)
        if 5 in valid and 6 in valid:
            sh_l_px = (valid[5][0] * w, valid[5][1] * h)
            sh_r_px = (valid[6][0] * w, valid[6][1] * h)
            out['shoulder_width_norm'] = _dist(valid[5], valid[6])
            out['shoulder_width'] = _dist(sh_l_px, sh_r_px)
            vertex = ((sh_l_px[0] + sh_r_px[0]) / 2.0,
                      (sh_l_px[1] + sh_r_px[1]) / 2.0)
            out['vertex_x'] = vertex[0]
            out['vertex_y'] = vertex[1]
            if ear_px is not None:
                ear_mid_y = (ear_px[0][1] + ear_px[1][1]) / 2.0
                out['ear_sh_vert'] = abs(vertex[1] - ear_mid_y)
        return out

    @staticmethod
    def _mean_visibility(landmarks: list, valid: Dict[int, Point]) -> Optional[float]:
        """本帧参与计算的点（已过门控，这里固定是四个）的平均置信度 0~1。"""
        vals = [landmarks[pid][3] for pid in valid if pid < len(landmarks)]
        if not vals:
            return None
        return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# ArmRaiseFeatures / HeadTiltFeatures：全身动作识别（举手 / 仰头）的角度特征
# （2026-10-12 新增，纯 stdlib）
#
# **自成一体的动作线**：与坐姿 / CVA / 头前伸那条线无关 —— 不继承
# ErgonomicRiskFeatures / CvaProxyFeatures，也不 import 任何 CVA 相关的东西。
# 唯一沿用的是"角度必须在像素空间算"这条纪律，那是画面纵横比带来的独立教训
# （tan θ_像素 = (H/W)·tan θ_归一化），与 CVA 语义无关。
#
# 两条特征都**只算不判**（不产生状态、不提醒），而且**不做任何平滑**：不 EMA、
# 不中值、不"维持上一有效值"。这是刻意的 —— 本轮的命题是"单目关键点撑不撑得住
# 全身动作"，加了平滑/维持就会把丢帧和抖动这两个最该被看到的信号掩盖掉。
# 要平滑等看清抖动幅度之后再加（加在哪里由用户定）。
# ---------------------------------------------------------------------------


def _visible_points(landmarks: list, ids: Tuple[int, ...],
                    visibility_min: float) -> Dict[int, Point]:
    """从 landmarks 里挑出 ids 中 visibility >= visibility_min 的点。

    与 _WindowedMedianExtractor._extract_valid 同语义，但**不继承那个基类**：
    本文件的两个动作特征不做窗口/中位统计（见上面"不做任何平滑"），只需要
    "取可见点"这一个原语，独立成函数比拉一条继承链干净。
    """
    out: Dict[int, Point] = {}
    for pid in ids:
        if pid < len(landmarks):
            x, y, _z, vis = landmarks[pid]
            if vis >= visibility_min:
                out[pid] = (x, y)
    return out


def _midpoint(a: Point, b: Point) -> Point:
    """两点中点。"""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _angle_from_up(v: Point) -> float:
    """向量 v（像素空间）与"向上竖直轴"的夹角（度，0~180）。

    0 = 向量正朝上，90 = 水平，180 = 正朝下。图像 y 轴向下，故"向上"= (0,-1)。

    **不能直接用 PostureFeatures._angle_from_vertical**：那个原语把 dx/dy 取了
    绝对值，只能给 0~90 的"离竖直轴多远"，分不出朝上还是朝下 —— 而举手恰恰
    要分：上举 0° 与自然下垂 180° 是同一根轴的两端（见 ArmRaiseFeatures）。
    """
    norm = math.hypot(v[0], v[1])
    if norm < 1e-9:
        return 0.0
    # dot(v, up) = -v_y
    cos = max(-1.0, min(1.0, -v[1] / norm))
    return math.degrees(math.acos(cos))


def _angle_at(joint: Point, p1: Point, p2: Point) -> Optional[float]:
    """以 joint 为顶点、两条边 joint→p1 / joint→p2 的夹角（度，0~180）。

    任一边长度 ~0（点重合）→ None：角度无意义，不返回 0 冒充读数。
    """
    u = (p1[0] - joint[0], p1[1] - joint[1])
    v = (p2[0] - joint[0], p2[1] - joint[1])
    nu = math.hypot(u[0], u[1])
    nv = math.hypot(v[0], v[1])
    if nu < 1e-9 or nv < 1e-9:
        return None
    cos = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv)))
    return math.degrees(math.acos(cos))


class ArmRaiseFeatures:
    """举手角度特征：上臂抬升角 + 肘角（像素空间，**只算不判、不平滑**）。

    === 用哪些点 ===
    左右各一套，一套三个点（COCO 17 编号）：
        L: 肩 5 → 肘 7 → 腕 9
        R: 肩 6 → 肘 8 → 腕 10
    每个点先过 visibility >= visibility_min（默认 0.3）门控；两侧**独立**计算，
    一侧丢点不影响另一侧。

    === 两个角怎么算 ===
        arm_raise_deg  上臂抬升角 = 肩→肘 向量 与 **向上竖直轴** 的夹角
                       （度，0~180；0 = 上臂完全上举，90 = 平举，180 = 自然下垂）。
                       只需要 肩+肘 两个点。
        elbow_deg      肘角 = 肩-肘-腕 在肘处的内角（度，0~180；180 = 手臂伸直）。
                       需要 肩+肘+腕 三个点。

    为什么"举手"该看抬升角而不是肘角：肘角只反映**肘弯没弯**，与手臂朝向无关
    —— 自然下垂、平举、举过头顶伸直，三种姿势的肘角都 ≈ 180°，单拿它判"举手"
    必然把下垂和平举一起误判进来。抬升角量的才是"上臂朝向"。
    肘角仍照常输出，供画面上对照（能区分"伸直举"/"屈肘举"，也便于看出腕点何时丢）。

    === 为什么在像素空间算 ===
    归一化坐标先把 x 除以 W、y 除以 H，同一个姿势算出的角度会带上画面宽高比偏差
    （16:9 下越接近水平的量偏得越多）—— 与 CvaProxyFeatures 那条纪律同源。
    image_size 缺失/非正 → 本帧不出读数（宁可不判，也不出一个带纵横比偏差的数）。

    === 为什么不平滑（与 CvaProxyFeatures 故意相反）===
    本类是"验证单目够不够用"的探针：**不 EMA、不中值、不维持上一有效值**。
    关键点丢帧就报 arm_raise_valid=False（读数 None），抖动就原样跳出来。
    加平滑会把"单目能不能撑住举手这种大动作"这个问题掩盖掉（见文件头注释）。

    === 两侧怎么合成 ===
    两条读数都算出来，再给出"**举得更高的那一侧**"（抬升角最小的一侧）：
    arm_raise_deg / arm_raise_side / arm_raise_conf / elbow_deg / arm_geom 都是
    这一侧的。判据语义是"任一侧举手即触发"（decision.ArmRaiseDecision）。
    两侧都没算出来（或角度全相同）时：arm_raise_side 取 L（确定性的平局规则）。

    === 降级（一律 N/A，不做 fallback）===
    无人 / image_size 缺失 / 某侧肩或肘不可见 → 该侧 arm_raise_deg=None；
    两侧都算不出 → arm_raise_valid=False；某侧腕不可见 → 只该侧 elbow_deg=None
    （**不影响抬升角**，也不影响另一侧）。不拿眼/鼻/对侧的点去替缺失的肘腕 ——
    换参考点等于换了一个量，混在一起会假装连续。

    接口约定（保持稳定，别改签名）：
        ArmRaiseFeatures(visibility_min=0.3)
        update(pose_result) -> dict
            {'arm_raise_deg':      float | None  举得更高那侧的抬升角（度）
             'arm_raise_side':     'L' | 'R' | None
             'arm_raise_left_deg': float | None  L 侧抬升角
             'arm_raise_right_deg': float | None R 侧抬升角
             'arm_raise_conf':     float | None  该侧参与点的平均置信度 0~1
             'arm_raise_valid':    bool          本帧是否算出了抬升角
             'elbow_deg':          float | None  同侧肘角（度，180=伸直）
             'elbow_left_deg' / 'elbow_right_deg': float | None
             'elbow_valid':        bool          同侧肘角是否可算（腕可见）
             'arm_geom': dict | None  该侧的像素几何，供 output 画线：
                             {'side': 'L'|'R', 'shoulder': (x_px,y_px),
                              'elbow': (x_px,y_px), 'wrist': (x_px,y_px)|None}}
        （无 reset：本类不持有跨帧状态）
    """

    # 左右两侧各自的 (肩, 肘, 腕) —— COCO 17 编号
    SIDES = (("L", (5, 7, 9)), ("R", (6, 8, 10)))

    # 侧别 → 返回 dict 里的全拼后缀（'L'/'R' 是画面上/几何里用的短标，
    # 返回键用 left/right，别拿短标 lower() 去拼键名）
    _SIDE_KEY = {"L": "left", "R": "right"}

    def __init__(self, visibility_min: float = 0.3) -> None:
        """
        参数:
            visibility_min: 某关键点 visibility 低于此值视为无效点（与其它特征同口径）。
        """
        self.visibility_min = visibility_min

    def update(self, pose_result: Optional[dict]) -> dict:
        """喂入一帧 pose 结果，返回两侧的抬升角/肘角（见类 docstring 的返回结构）。"""
        out = {
            'arm_raise_deg': None,
            'arm_raise_side': None,
            'arm_raise_left_deg': None,
            'arm_raise_right_deg': None,
            'arm_raise_conf': None,
            'arm_raise_valid': False,
            'elbow_deg': None,
            'elbow_left_deg': None,
            'elbow_right_deg': None,
            'elbow_valid': False,
            'arm_geom': None,
        }
        if pose_result is None:
            return out
        landmarks = pose_result.get('landmarks')
        size = pose_result.get('image_size')
        if not landmarks or not size or size[0] <= 0 or size[1] <= 0:
            return out
        w, h = size
        vis = _visible_points(landmarks, SHOULDER_IDS + ELBOW_IDS + WRIST_IDS,
                              self.visibility_min)

        per = {}
        for side, ids in self.SIDES:
            m = self._side(landmarks, vis, w, h, side, ids)
            per[side] = m
            key = self._SIDE_KEY[side]
            out[f'arm_raise_{key}_deg'] = m['raise']
            out[f'elbow_{key}_deg'] = m['elbow']

        # 举得更高 = 抬升角更小。平局（含完全对称）时按 L/R 顺序取 L（确定性）。
        cands = [s for s, _ids in self.SIDES if per[s]['raise'] is not None]
        if cands:
            best = min(cands, key=lambda s: (per[s]['raise'], s))
            out['arm_raise_deg'] = per[best]['raise']
            out['arm_raise_side'] = best
            out['arm_raise_conf'] = per[best]['conf']
            out['arm_raise_valid'] = True
            out['elbow_deg'] = per[best]['elbow']
            out['elbow_valid'] = per[best]['elbow'] is not None
            out['arm_geom'] = per[best]['geom']
        return out

    # ---------- 内部实现 ----------

    def _side(self, landmarks: list, vis: Dict[int, Point],
              w: int, h: int, side: str, ids: Tuple[int, int, int]) -> dict:
        """算单侧的两个角（点不齐就是 None，不做任何替代）。"""
        sh_id, el_id, wr_id = ids
        sh, el, wr = vis.get(sh_id), vis.get(el_id), vis.get(wr_id)
        raise_deg = elbow_deg = conf = None
        geom = None
        if sh is not None and el is not None:
            sh_px = (sh[0] * w, sh[1] * h)
            el_px = (el[0] * w, el[1] * h)
            if _dist(sh_px, el_px) > 1e-9:  # 肩肘重合 → 抬升角无意义
                raise_deg = _angle_from_up((el_px[0] - sh_px[0],
                                            el_px[1] - sh_px[1]))
                confs = [landmarks[sh_id][3], landmarks[el_id][3]]
                geom = {'side': side, 'shoulder': sh_px, 'elbow': el_px,
                        'wrist': None}
                if wr is not None:
                    wr_px = (wr[0] * w, wr[1] * h)
                    elbow_deg = _angle_at(el_px, sh_px, wr_px)
                    confs.append(landmarks[wr_id][3])
                    if elbow_deg is not None:
                        geom['wrist'] = wr_px
                conf = sum(confs) / len(confs)
        return {'raise': raise_deg, 'elbow': elbow_deg,
                'conf': conf, 'geom': geom}


class HeadTiltFeatures:
    """仰头（头后仰）角度特征：耳中点→鼻尖 与水平线的夹角（**只算不判、不平滑**）。

    === 用哪些点、怎么算 ===
        头参考 = 双耳(3/4)中点（两只都可见时）；只可见一只耳就用那一只
        鼻点   = 0（COCO 17 鼻）
        head_pitch_deg = 耳→鼻 向量 与 **水平线** 的夹角（度，-90~90）
                         正 = 鼻在耳**上方**（仰头），负 = 鼻在耳下方（低头）
    横向按 |Δx| 展开，所以人朝画面左或右都得到同一套符号约定（不用管人朝哪边）。

    === 为什么用耳→鼻，而不是现成的耳-肩（PostureFeatures.head_neck_angle）===
    耳朵几乎就在"头后仰"的旋转轴心上，**仰头时耳朵基本不动** —— 耳→肩的角度
    几乎不变（那个量量的是头相对躯干的**平移**前伸，不是**转动**俯仰）。
    鼻尖离轴心最远，头一转它摆动最大：耳→鼻是"头自身的朝向"，与肩/躯干无关，
    正好把俯仰（仰头）和头前伸分开。这两条线各量各的，不要互相替换。

    === 近似在哪（如实标注，别当真实俯仰角）===
    1) COCO 17 **没有下巴/头顶/耳屏**，用"耳点 + 鼻尖"代替头部前后轴，是经典的
       2D head-pose 代理；量值与任何文献的头屈伸角都不可比，阈值只能本项目自己定。
    2) 2D 投影：45° 机位下水平方向被压缩（≈cos45°），读数比真实俯仰角大，
       只保证**单调**（仰头越大读数越大），不是真实角度。
    3) **正面机位测不到，且已在特征层作废**：正对摄像头时仰头主要是"朝镜头方向"
       的透视缩短，耳-鼻在像面上几乎不动；更糟的是 |Δx| ≈ 0 会让读数退化成
       sign(θ)·90°（只剩符号、没有量值，还长得像正常读数）。横向退化门控
       （MIN_PITCH_DX_SPAN_RATIO）把它判为几何退化 → valid=False、读数 N/A，
       **不给出看着可信的错数**。这条线只在 45° 前侧机位有意义（与项目"机位由
       入口文件声明"的纪律一致，见 main_action_demo.py）。
    4) **不补偿转头（yaw）**：头往侧面转同样会改变耳→鼻的像面角，会串进本读数。
       画面上那条耳→鼻几何线能直接看出头有没有转（转了它会明显偏侧向）——
       所以 output 层默认把这条线画出来。

    === 为什么不平滑 / 降级 ===
    与 ArmRaiseFeatures 同一口径（见文件头）：不平滑、不维持上一有效值。
    耳或鼻缺一个 → 本帧 invalid（读数 None），**不做"退化成眼→鼻"的兜底** ——
    眼→鼻的零点与耳→鼻差一大截，换参考点等于换了一个量，混在一起会假装连续。
    几何退化（正对镜头，见上）同样直接 invalid，而不是给个假数。

    接口约定（保持稳定，别改签名）：
        HeadTiltFeatures(visibility_min=0.3, min_dx_span_ratio=MIN_PITCH_DX_SPAN_RATIO)
        update(pose_result) -> dict
            {'head_pitch_deg':  float | None  耳→鼻 与水平线夹角（度，正=仰头）
             'head_pitch_valid': bool
             'head_pitch_conf': float | None  参与点平均置信度 0~1
             'head_pitch_pts':  str | None    如 'ear_mid+nose' / 'earL+nose'
             'head_pitch_dx_ratio': float | None  诊断量：|Δx| / 双耳间距（双耳都在
                                              才算得出；**低于门控时也照记**，供画面
                                               与控制台说明"这帧为什么是 N/A"）
             'head_pitch_degenerate': bool    本帧是否因横向退化被作废（正对镜头）。
                                              画面上要能与"关键点丢了"分开显示：
                                              一个是机位问题、一个是检测问题
             'head_tilt_geom':  dict | None   {'ear_mid': (x_px,y_px),
                                               'nose': (x_px,y_px),
                                               'ear_span': float | None（像素，
                                               单耳回退时为 None）}
                                              供 output 画线（本层不重算几何）；
                                              退化帧也照给，才能把人看出来是几何不对}
        （无 reset：本类不持有跨帧状态）
    """

    def __init__(self, visibility_min: float = 0.3,
                 min_dx_span_ratio: float = MIN_PITCH_DX_SPAN_RATIO) -> None:
        """
        参数:
            visibility_min: 某关键点 visibility 低于此值视为无效点（与其它特征同口径）。
            min_dx_span_ratio: 横向退化门控阈值 = |Δx| / 双耳间距 的下限（标定依据见
                MIN_PITCH_DX_SPAN_RATIO）。低于它判"几何退化" → valid=False、读数 N/A。
        """
        self.visibility_min = visibility_min
        self.min_dx_span_ratio = min_dx_span_ratio

    def update(self, pose_result: Optional[dict]) -> dict:
        """喂入一帧 pose 结果，返回头俯仰代理角（见类 docstring 的返回结构）。"""
        out = {
            'head_pitch_deg': None,
            'head_pitch_valid': False,
            'head_pitch_conf': None,
            'head_pitch_pts': None,
            'head_pitch_dx_ratio': None,
            'head_pitch_degenerate': False,
            'head_tilt_geom': None,
        }
        if pose_result is None:
            return out
        landmarks = pose_result.get('landmarks')
        size = pose_result.get('image_size')
        if not landmarks or not size or size[0] <= 0 or size[1] <= 0:
            return out
        w, h = size
        vis = _visible_points(landmarks, EAR_IDS + (NOSE_ID,),
                              self.visibility_min)
        if NOSE_ID not in vis:
            return out

        # 头参考：双耳中点优先，退到单耳（并如实标出用了哪只 —— 两者零点不同一，
        # 画面上要能看出参考点换过）
        if 3 in vis and 4 in vis:
            ear, ref_ids, label = _midpoint(vis[3], vis[4]), (3, 4), 'ear_mid'
        elif 3 in vis:
            ear, ref_ids, label = vis[3], (3,), 'earL'
        elif 4 in vis:
            ear, ref_ids, label = vis[4], (4,), 'earR'
        else:
            return out  # 耳全丢：不做眼→鼻兜底（见类 docstring）

        ear_px = (ear[0] * w, ear[1] * h)
        nose_px = (vis[NOSE_ID][0] * w, vis[NOSE_ID][1] * h)
        dx = nose_px[0] - ear_px[0]
        dy = nose_px[1] - ear_px[1]
        if math.hypot(dx, dy) < 1e-9:  # 耳鼻重合 → 角度无意义
            return out

        # 横向退化门控（见 MIN_PITCH_DX_SPAN_RATIO）：双耳都在时用双耳间距归一化 |Δx|。
        # 单耳回退时拿不到间距 → 这一步没法做，只如实标出参考点换了（head_pitch_pts）
        span = None
        if 3 in vis and 4 in vis:
            span = _dist((vis[3][0] * w, vis[3][1] * h),
                         (vis[4][0] * w, vis[4][1] * h))
        ratio = abs(dx) / span if (span is not None and span > 1e-9) else None

        # 诊断量照记（**低于门控时也记**）：画面上那条耳→鼻线要画得出来、控制台也要
        # 说得出"为什么这帧是 N/A"，否则退化和丢点从外面看没区别
        out['head_pitch_dx_ratio'] = ratio
        out['head_pitch_conf'] = (
            sum(landmarks[pid][3] for pid in ref_ids + (NOSE_ID,))
            / float(len(ref_ids) + 1))
        out['head_pitch_pts'] = f"{label}+nose"
        out['head_tilt_geom'] = {'ear_mid': ear_px, 'nose': nose_px,
                                 'ear_span': span}

        if ratio is not None and ratio < self.min_dx_span_ratio:
            # 几何退化：deg 保持 None、valid 保持 False（宁可不判，不乱判）。
            # 由本层显式标出来，**不让 output 靠 ratio 有没有值去反推**（那是把
            # 门控逻辑复制到画图层）—— 画面上要与"关键点丢了"分开显示
            out['head_pitch_degenerate'] = True
            return out

        out['head_pitch_deg'] = math.degrees(math.atan2(-dy, abs(dx)))
        out['head_pitch_valid'] = True
        return out


# ---------------------------------------------------------------------------
# 自测：纯 stdlib 合成数据，验证各特征计算逻辑（main_demo.py --selftest 汇总调用）
# ---------------------------------------------------------------------------

def _make_pose(pts: dict, vis: float = 1.0) -> dict:
    """构造一帧合成 pose：pts 里的点可见，其余点 visibility=0。

    pts: {关键点编号: (x, y)}，x/y 为归一化坐标 0~1。
    vis: 可见点的统一 visibility（默认 1.0；传 0.1 可模拟遮挡/低置信帧）。
    返回结构与 pose_estimation.detect_pose() 一致（17 点，z/visibility 补齐）。
    """
    landmarks = []
    for pid in range(17):  # COCO 17 点（RTMPose）
        if pid in pts:
            x, y = pts[pid]
            landmarks.append((x, y, 0.0, vis))
        else:
            landmarks.append((0.0, 0.0, 0.0, 0.0))
    return {"landmarks": landmarks, "image_size": (640, 480)}


def _assert_close(actual: float, expected: float, eps: float, msg: str) -> None:
    if abs(actual - expected) > eps:
        raise AssertionError(f"{msg}: 期望 {expected:.4f}，实际 {actual:.4f}")


def _make_pose_sized(pts: dict, size: tuple,
                     vis: float = 1.0,
                     vis_map: Optional[dict] = None) -> dict:
    """同 _make_pose，但可指定画面尺寸 / 逐点置信度（CvaProxyFeatures 自测用）。

    pts:     {关键点编号: (x, y)}，归一化坐标。
    size:    (w, h) 画面尺寸（像素空间角度依赖它）。
    vis:     未在 vis_map 里单独指定的可见点的统一置信度。
    vis_map: {关键点编号: 该点置信度}，覆盖 vis。
    """
    landmarks = []
    for pid in range(17):
        if pid in pts:
            x, y = pts[pid]
            v = vis if vis_map is None else vis_map.get(pid, vis)
            landmarks.append((x, y, 0.0, v))
        else:
            landmarks.append((0.0, 0.0, 0.0, 0.0))
    return {"landmarks": landmarks, "image_size": size}


def selftest_movement() -> None:
    """合成数据自测 FeatureExtractor：静止≈0、明显位移>0.10、五官/四肢轻动仍≈0、
    无人→None、髋缺失兜底。"""
    # 躯干 4 点：肩 (0.4,0.3)/(0.6,0.3)，髋 (0.4,0.7)/(0.6,0.7) → 躯干高 0.4
    base = {5: (0.4, 0.3), 6: (0.6, 0.3), 11: (0.4, 0.7), 12: (0.6, 0.7)}

    # 1) 静止：每帧同一位置 → 移动量 ≈ 0
    fe = FeatureExtractor(window_seconds=5.0, min_frames=3, min_window_seconds=0.1)
    for i in range(5):
        fe.update(_make_pose(base), timestamp=float(i) * 0.1)
    m = fe.update(_make_pose(base), timestamp=0.5)
    assert m is not None, "静止场景不应返回 None"
    _assert_close(m, 0.0, 1e-6, "静止移动量应为 0")

    # 2) 五官/四肢轻动：耳/肘/腕小幅动，肩髋不动 → 移动量 ≈ 0（静坐期间轻微动作不算在动）
    fe = FeatureExtractor(window_seconds=5.0, min_frames=3, min_window_seconds=0.1)
    for i in range(5):
        shake = 0.02 * (1 if i % 2 == 0 else -1)  # 耳/肘/腕在肩髋之间小幅度抖
        moved = {**base,
                 3: (0.5 + shake, 0.2), 4: (0.5 + shake, 0.2),
                 7: (0.35 + shake, 0.45), 8: (0.65 + shake, 0.45),
                 9: (0.3 + shake, 0.55), 10: (0.7 + shake, 0.55)}
        fe.update(_make_pose(moved), timestamp=float(i) * 0.1)
    m = fe.update(_make_pose({**base,
                              3: (0.52, 0.2), 4: (0.52, 0.2),
                              7: (0.37, 0.45), 8: (0.67, 0.45),
                              9: (0.32, 0.55), 10: (0.72, 0.55)}),
                  timestamp=0.5)
    assert m is not None, "五官/四肢轻动场景不应返回 None"
    assert m < 0.02, f"五官/四肢轻动时移动量应很小（<0.02），实际 {m:.4f}"

    # 3) 明显位移：5 帧沿 x 匀速挪 0.2 → 移动量显著 > 0.10
    fe = FeatureExtractor(window_seconds=5.0, min_frames=3, min_window_seconds=0.1)
    for i in range(5):
        shift = 0.05 * i
        moved = {pid: (x + shift, y) for pid, (x, y) in base.items()}
        fe.update(_make_pose(moved), timestamp=float(i) * 0.1)
    m = fe.update(_make_pose({pid: (x + 0.2, y) for pid, (x, y) in base.items()}),
                  timestamp=0.5)
    assert m is not None, "移动场景不应返回 None"
    assert m > 0.10, f"明显位移应 > 0.10，实际 {m:.4f}"

    # 4) 无人 → None
    assert fe.update(None, timestamp=0.6) is None

    # 5) 窗口数据不足：首帧直接返回 None
    fe = FeatureExtractor(window_seconds=5.0, min_frames=3, min_window_seconds=0.1)
    assert fe.update(_make_pose(base), timestamp=0.0) is None

    # 6) 髋部不可见（只露出上半身）→ 肩中点兜底仍能算
    torso_upper = {5: (0.4, 0.3), 6: (0.6, 0.3)}
    fe = FeatureExtractor(window_seconds=5.0, min_frames=3, min_window_seconds=0.1)
    for i in range(5):
        fe.update(_make_pose(torso_upper), timestamp=float(i) * 0.1)
    m = fe.update(_make_pose(torso_upper), timestamp=0.5)
    assert m is not None, "髋不可见、肩中点兜底时不应返回 None"
    _assert_close(m, 0.0, 1e-6, "上半身静止移动量应为 0")

    # 7) 侧身（肩宽被投影压扁、髋不可见）：兜底尺度取 max(肩宽, 耳-肩竖直间距)。
    #    耳-肩间距侧身不塌缩 → 同一微小位移下，有耳的侧身移动量明显小于只用
    #    塌缩肩宽（不会把静止放大成 MOVING），且帧不作废（非 None）
    def side_movement(pts):
        fe = FeatureExtractor(window_seconds=5.0, min_frames=3, min_window_seconds=0.1)
        for i in range(5):
            shift = 0.002 * i
            moved = {pid: (x + shift, y) for pid, (x, y) in pts.items()}
            fe.update(_make_pose(moved), timestamp=float(i) * 0.1)
        last = {pid: (x + 0.008, y) for pid, (x, y) in pts.items()}
        return fe.update(_make_pose(last), timestamp=0.5)

    no_ear = {5: (0.52, 0.3), 6: (0.58, 0.3)}        # 无耳：兜底 = 肩宽 0.06（塌缩）
    with_ear = {3: (0.55, 0.08), 4: (0.55, 0.08),    # 有耳：兜底 = 耳-肩距 0.22
                5: (0.52, 0.3), 6: (0.58, 0.3)}
    m_no_ear = side_movement(no_ear)
    m_ear = side_movement(with_ear)
    assert m_no_ear is not None and m_ear is not None, "侧身帧应能算移动量"
    assert m_ear < m_no_ear * 0.6, \
        f"耳-肩兜底应明显压低侧身移动量（有耳 {m_ear:.4f} vs 无耳 {m_no_ear:.4f}）"

    print("  selftest_movement: OK")


def selftest_posture() -> None:
    """合成数据自测 PostureFeatures：竖直≈0、前伸>0、髋缺失→N/A、点缺失→None、
    颈压缩对弓背敏感（竖直 0.75 / 弓背 0.35）、单侧链路无肩宽→N/A。"""
    post = PostureFeatures()

    # 1) 竖直站立：耳/肩/髋同竖线 → 三个角都 ≈ 0；颈压缩 = 0.75（肩宽 0.2）
    upright = {3: (0.45, 0.2), 4: (0.55, 0.2),
               5: (0.4, 0.35), 6: (0.6, 0.35),
               11: (0.45, 0.8), 12: (0.55, 0.8)}
    r = post.update(_make_pose(upright))
    assert r is not None, "竖直场景不应返回 None"
    for key in ("head_neck_angle", "torso_angle", "back_curvature"):
        _assert_close(r[key], 0.0, 1e-6, f"{key} 竖直应为 0")
    _assert_close(r["neck_compression"], 0.75, 1e-6, "竖直颈压缩应为 0.75")

    # 2) 头前伸：耳向右平 0.15（dx=0.15, dy=0.15）→ head_neck ≈ 45°
    lean = {3: (0.65, 0.2), 4: (0.65, 0.2),
            5: (0.5, 0.35), 6: (0.5, 0.35),
            11: (0.5, 0.8), 12: (0.5, 0.8)}
    r = post.update(_make_pose(lean))
    assert r is not None
    _assert_close(r["head_neck_angle"], 45.0, 0.5, "head_neck 应为 45°")

    # 3) 单侧链路：只有左耳+左肩+左髋 → 仍能算，竖直 → 0；
    #    无双侧肩（无肩宽）→ neck_compression 缺失（= N/A）
    left_only = {3: (0.5, 0.2), 5: (0.5, 0.35), 11: (0.5, 0.8)}
    r = post.update(_make_pose(left_only))
    assert r is not None, "单侧链路不应返回 None"
    for key in ("head_neck_angle", "torso_angle", "back_curvature"):
        _assert_close(r[key], 0.0, 1e-6, f"单侧竖直 {key} 应为 0")
    assert 'neck_compression' not in r, \
        "单侧链路无双侧肩（无肩宽），neck_compression 应为 N/A（缺失）"

    # 4) 髋不可见：只剩耳+肩 → head_neck 仍可算、torso/back 缺失（= N/A）；
    #    但双侧肩在 → neck_compression 可算（正面弓背信号，不依赖髋）。
    #    注：髋不可见时走到 "MID" 分支，耳取单只（_pick_best_ear），两耳故意
    #    居中放置避免单耳的横向偏移干扰 head_neck 断言（颈压缩只看竖直间距，
    #    不受影响）。
    upper = {3: (0.5, 0.2), 4: (0.5, 0.2),
             5: (0.4, 0.35), 6: (0.6, 0.35)}
    r = post.update(_make_pose(upper))
    assert r is not None, "髋不可见时 head_neck 仍应可算"
    assert set(r.keys()) == {"head_neck_angle", "neck_compression"}, \
        f"髋不可见时应只有 head_neck+neck_compression，实际 {sorted(r.keys())}"
    _assert_close(r["head_neck_angle"], 0.0, 1e-6, "上半身竖直 head_neck 应为 0")
    _assert_close(r["neck_compression"], 0.75, 1e-6, "上半身竖直颈压缩应为 0.75")

    # 4b) 弓背（耸肩+低头，髋不可见）：头下沉、肩微抬 → 颈压缩变小
    #     （竖直 0.75 → 弓背 0.35），低于 decision 默认阈值 0.45
    hunched = {3: (0.5, 0.30), 4: (0.5, 0.30),
               5: (0.4, 0.37), 6: (0.6, 0.37)}
    r = post.update(_make_pose(hunched))
    assert r is not None
    _assert_close(r["neck_compression"], 0.35, 1e-6, "弓背颈压缩应约 0.35")
    assert r["neck_compression"] < 0.45, \
        f"弓背颈压缩应低于 decision 默认阈值 0.45，实际 {r['neck_compression']:.3f}"

    # 4c) 侧身退化：双侧肩可见但肩宽被侧身投影压扁 → neck_compression 爆炸值
    #     （> NECK_COMPRESSION_MAX）→ 视为 N/A（缺失），不把状态锁死成 GOOD
    side = {3: (0.52, 0.22), 4: (0.53, 0.22),
            5: (0.51, 0.35), 6: (0.55, 0.35)}   # 肩宽 0.04，颈压缩比值 ~3.25
    r = post.update(_make_pose(side))
    assert r is not None, "侧身退化时 head_neck 仍可算"
    assert 'neck_compression' not in r, \
        "侧身肩宽塌缩时 neck_compression 应为 N/A（缺失）"

    # 5) 耳或肩缺失（只剩髋）→ None
    assert post.update(_make_pose({11: (0.5, 0.8), 12: (0.5, 0.8)})) is None

    # 6) 无人 → None
    assert post.update(None) is None

    print("  selftest_posture: OK")


def selftest_ergonomic() -> None:
    """合成数据自测 ErgonomicRiskFeatures：CVA 竖直=90° / 前伸=45°；FSA 下垂=90° /
    前伸~26.6°；单耳/单肩可见侧仍可算；遮挡低置信→维持上一有效值、不崩溃、不出 0。"""
    # 小窗口参数：自测用显式时间戳，几帧内就能攒够窗口
    small = dict(window_seconds=5.0, min_frames=2, min_window_seconds=0.1)

    # 1) 竖直（耳在肩正上方）+ 上臂下垂 → CVA 90° / FSA 90°
    upright = {3: (0.5, 0.05), 4: (0.5, 0.05),   # 双耳同点，中点 = (0.5,0.05)
               5: (0.4, 0.35), 6: (0.6, 0.35),   # 肩中点 = (0.5,0.35)
               7: (0.4, 0.6), 8: (0.6, 0.6)}     # 肘中点 = (0.5,0.6)，上臂竖直
    fe = ErgonomicRiskFeatures(**small)
    for i in range(5):
        fe.update(_make_pose(upright), timestamp=float(i) * 0.1)
    r = fe.update(_make_pose(upright), timestamp=0.5)
    assert r['cva_valid'] and r['fsa_valid'], "竖直帧 CVA/FSA 都应可算"
    _assert_close(r['cva_deg'], 90.0, 1e-6, "竖直 CVA 应为 90°")
    _assert_close(r['fsa_deg'], 90.0, 1e-6, "上臂下垂 FSA 应为 90°")

    # 2) 头前伸（耳前移 45° 斜）+ 上臂前伸 → CVA 45° / FSA ~26.6°
    lean = {3: (0.65, 0.20), 4: (0.65, 0.20),    # 耳中点 (0.65,0.20)
            5: (0.5, 0.35), 6: (0.5, 0.35),      # 肩中点 (0.5,0.35) → dx=0.15,dy=-0.15
            7: (0.7, 0.45), 8: (0.7, 0.45)}      # 肘中点 (0.7,0.45) → dx=0.2,dy=0.1
    fe = ErgonomicRiskFeatures(**small)
    for i in range(5):
        fe.update(_make_pose(lean), timestamp=float(i) * 0.1)
    r = fe.update(_make_pose(lean), timestamp=0.5)
    _assert_close(r['cva_deg'], 45.0, 0.5, "前伸 CVA 应为 45°")
    _assert_close(r['fsa_deg'], math.degrees(math.atan(0.1 / 0.2)), 0.5,
                  "上臂前伸 FSA 应为 ~26.6°")

    # 3) 单侧可见：只有右耳(4) 或 只有左肩(5) → 取可见侧，仍能算（竖直 → 90°）
    for pts in ({4: (0.5, 0.05), 5: (0.4, 0.35), 6: (0.6, 0.35),
                 7: (0.4, 0.6), 8: (0.6, 0.6)},
                {3: (0.5, 0.05), 4: (0.5, 0.05), 5: (0.5, 0.35),
                 7: (0.5, 0.6), 8: (0.5, 0.6)}):
        fe = ErgonomicRiskFeatures(**small)
        for i in range(3):
            fe.update(_make_pose(pts), timestamp=float(i) * 0.1)
        r = fe.update(_make_pose(pts), timestamp=0.3)
        assert r['cva_deg'] is not None, "单耳/单肩可见侧 CVA 应可算"
        _assert_close(r['cva_deg'], 90.0, 1e-6, "单侧可见竖直 CVA 应为 90°")

    # 4) 降级：遮挡/低置信帧（所有关键点 vis=0.1 < 0.3）→ 维持上一有效值、
    #    valid=False；不崩溃、不出 0
    fe = ErgonomicRiskFeatures(**small)
    for i in range(5):
        fe.update(_make_pose(upright), timestamp=float(i) * 0.1)
    ok = fe.update(_make_pose(upright), timestamp=0.5)
    held = fe.update(_make_pose(upright, vis=0.1), timestamp=0.6)
    assert held['cva_deg'] == ok['cva_deg'], "低置信帧应维持上一有效 CVA"
    assert not held['cva_valid'], "低置信帧 cva_valid 应为 False"
    assert held['fsa_deg'] == ok['fsa_deg'], "低置信帧应维持上一有效 FSA"
    assert not held['fsa_valid'], "低置信帧 fsa_valid 应为 False"

    # 5) 从未有有效数据（无人 / 只剩髋）→ None + valid=False，不崩溃
    fe = ErgonomicRiskFeatures(**small)
    r = fe.update(None, timestamp=0.0)
    assert r['cva_deg'] is None and r['fsa_deg'] is None, "无人帧应为 None"
    assert not r['cva_valid'] and not r['fsa_valid'], "无人帧 valid 应为 False"
    r2 = fe.update(_make_pose({11: (0.5, 0.8), 12: (0.5, 0.8)}), timestamp=0.1)
    assert r2['cva_deg'] is None and r2['fsa_deg'] is None, \
        "耳/肩/肘都缺（只剩髋）时 CVA/FSA 应为 None"

    # 6) FSA 趋势：上臂从下垂逐渐前伸 → fsa_trend_deg < 0（负 = 更接近水平 = 更前伸）
    down = {5: (0.5, 0.35), 6: (0.5, 0.35), 7: (0.5, 0.6), 8: (0.5, 0.6)}
    fwd = {5: (0.5, 0.35), 6: (0.5, 0.35), 7: (0.7, 0.45), 8: (0.7, 0.45)}
    fe = ErgonomicRiskFeatures(**small)
    for i in range(4):
        fe.update(_make_pose(down), timestamp=float(i) * 0.1)
    for i in range(4):
        fe.update(_make_pose(fwd), timestamp=0.4 + float(i) * 0.1)
    r = fe.update(_make_pose(fwd), timestamp=0.8)
    assert r['fsa_trend_deg'] is not None and r['fsa_trend_deg'] < 0, \
        f"上臂前伸 FSA 趋势应为负，实际 {r['fsa_trend_deg']}"

    print("  selftest_ergonomic: OK")


def selftest_cva_proxy() -> None:
    """合成数据自测 CvaProxyFeatures：像素空间（纵横比）角、EMA 跟踪与 raw 保留、
    预热不出平滑值、降级维持旧值、单耳/单肩取值与 pts 标签、置信度门控、
    image_size 缺失时降级不报错。"""
    small = dict(min_frames=3, min_window_seconds=0.1, smooth_tau_sec=1.0)

    # 1) 像素空间：W=200,H=100 上 dx_px=dy_px=20 → 45°；同一组点用归一化坐标算
    #    会得到 63.43°（atan2(0.2, 0.1)）—— 证明坐标系不同、数值不可混用
    pts45 = {3: (0.6, 0.3), 4: (0.6, 0.3),      # 耳中点 (120, 30) px
             5: (0.5, 0.5), 6: (0.5, 0.5)}      # 肩中点 (100, 50) px
    fe = CvaProxyFeatures(**small)
    r = fe.update(_make_pose_sized(pts45, (200, 100)), timestamp=0.0)
    assert r['cva_proxy_raw_deg'] is not None, "第一帧应有 raw 值"
    _assert_close(r['cva_proxy_raw_deg'], 45.0, 1e-6, "像素空间角应为 45°")
    normalized = math.degrees(math.atan2(0.2, 0.1))
    assert abs(normalized - 63.4349) < 1e-3, normalized
    assert abs(normalized - r['cva_proxy_raw_deg']) > 18.0, \
        "像素角必须明显区别于归一化角（纵横比偏差），否则本类没起到作用"

    # 2) 预热：min_frames=3 且跨度 >= 0.1s 之前不出平滑值，raw 照常给
    assert r['cva_proxy_deg'] is None and not r['cva_proxy_valid'], \
        "首帧应还没预热完（不出平滑值）"
    fe.update(_make_pose_sized(pts45, (200, 100)), timestamp=0.05)
    r = fe.update(_make_pose_sized(pts45, (200, 100)), timestamp=0.2)
    assert r['cva_proxy_valid'], "攒够帧数+时间跨度后应出平滑值"
    _assert_close(r['cva_proxy_deg'], 45.0, 1e-6, "读数恒定 → EMA 应停在 45°")

    # 3) EMA：从 45° 跳到 90°（耳抬到肩正上方）→ raw 立刻是 90，平滑值只走一段
    upright = {3: (0.5, 0.3), 4: (0.5, 0.3),    # 耳中点 (100, 30)：与肩同 x → 90°
               5: (0.5, 0.5), 6: (0.5, 0.5)}
    r = fe.update(_make_pose_sized(upright, (200, 100)), timestamp=0.3)
    _assert_close(r['cva_proxy_raw_deg'], 90.0, 1e-6, "raw 应立刻到 90°")
    assert 45.0 < r['cva_proxy_deg'] < 90.0, \
        f"平滑值应只走一段（45<v<90），实际 {r['cva_proxy_deg']:.3f}"
    first_step = r['cva_proxy_deg']
    # 关键点抖动 1 帧：raw 跳一下，平滑值基本不动（这就是平滑的意义）
    r = fe.update(_make_pose_sized(pts45, (200, 100)), timestamp=0.4)
    assert abs(r['cva_proxy_deg'] - first_step) < 3.0, "单帧抖动不该把平滑值拉回去"
    # 持续喂 90° → 单调逼近 90°，不过冲
    prev = r['cva_proxy_deg']
    for i in range(1, 60):
        r = fe.update(_make_pose_sized(upright, (200, 100)),
                      timestamp=0.4 + i * 0.1)
        assert prev <= r['cva_proxy_deg'] <= 90.0, "EMA 应单调逼近且不过冲"
        prev = r['cva_proxy_deg']
    assert r['cva_proxy_deg'] > 89.0, f"持续 6s 后应贴近 90°，实际 {r['cva_proxy_deg']:.2f}"

    # 4) 降级：只剩髋点（耳/肩都不可见）→ raw=None、valid=False、平滑值维持上一有效值
    held = r['cva_proxy_deg']
    r = fe.update(_make_pose_sized({11: (0.5, 0.8), 12: (0.5, 0.8)}, (200, 100)),
                  timestamp=6.5)
    assert r['cva_proxy_raw_deg'] is None and not r['cva_proxy_valid'], \
        "关键点不足应降级（raw 为 None、valid=False）"
    _assert_close(r['cva_proxy_deg'], held, 1e-9, "降级时应维持上一有效平滑值")

    # 5) 无人 / image_size 缺失 → 降级不报错
    r = fe.update(None, timestamp=6.6)
    assert r['cva_proxy_deg'] is None or r['cva_proxy_valid'] is False
    no_size = {'landmarks': [(0.5, 0.3, 0.0, 1.0)] * 17}   # 故意不给 image_size
    r = fe.update(no_size, timestamp=6.7)
    assert r['cva_proxy_raw_deg'] is None and not r['cva_proxy_valid'], \
        "取不到 image_size 时应降级（像素空间角度算不了）"

    # 6) 单耳 + 双肩：pts 标签标出用了哪只耳；conf = 参与点平均置信度
    one_ear = {3: (0.6, 0.3), 5: (0.5, 0.5), 6: (0.5, 0.5)}
    vis_map = {3: 0.6, 5: 0.8, 6: 0.9}
    fe = CvaProxyFeatures(**small)
    for i in range(4):
        r = fe.update(_make_pose_sized(one_ear, (200, 100), vis_map=vis_map),
                      timestamp=i * 0.1)
    assert r['cva_proxy_pts'] == "earL+sh_mid", r['cva_proxy_pts']
    _assert_close(r['cva_proxy_conf'], (0.6 + 0.8 + 0.9) / 3.0, 1e-9,
                  "conf 应为参与点（耳+双肩）的平均置信度")
    # 几何：head 是耳(120,30)、c7 是肩中点(100,50)
    g = r['cva_proxy_geom']
    _assert_close(g['head'][0], 120.0, 1e-6, "geom.head.x 应为耳像素 x")
    _assert_close(g['head'][1], 30.0, 1e-6, "geom.head.y 应为耳像素 y")
    _assert_close(g['c7'][1], 50.0, 1e-6, "geom.c7.y 应为肩中点像素 y")

    # 7) 置信度门控：耳低于 visibility_min（0.3）→ 耳被丢，只剩肩 → 本帧降级
    fe = CvaProxyFeatures(**small)
    r = fe.update(_make_pose_sized({3: (0.6, 0.3), 4: (0.6, 0.3),
                                    5: (0.5, 0.5), 6: (0.5, 0.5)}, (200, 100),
                                   vis_map={3: 0.2, 4: 0.2, 5: 0.9, 6: 0.9}),
                  timestamp=0.0)
    assert r['cva_proxy_raw_deg'] is None, "低置信度的耳应被门控掉 → 算不出读数"

    print("  selftest_cva_proxy: OK")


def selftest_ear_shoulder_proxy() -> None:
    """合成数据自测 EarShoulderProxyFeatures：像素空间定义（水平位移 ÷ 肩宽）、
    尺度不变性、方向（越大越前伸）、EMA 跟踪与 raw 保留、预热不出平滑值、
    肩宽塌陷（侧身）判退化、单肩缺失降级、parts 的逐帧原始量与 pts 标签、
    参数校验。"""
    small = dict(min_frames=3, min_window_seconds=0.1, smooth_tau_sec=1.0)

    # 合成几何：肩 (0.35,0.5)/(0.65,0.5) → 归一化肩宽 0.30（>0.25，不算退化）、
    # 200x100 画面上肩宽 60px、肩中点 x=100px；耳中点 x = (0.5 + dx)*200
    base = {5: (0.35, 0.5), 6: (0.65, 0.5)}

    def _esp(dx_norm, size=(200, 100), vis=1.0, vis_map=None, override=None):
        pts = dict(base)
        pts[3] = (0.5 + dx_norm, 0.3)
        pts[4] = (0.5 + dx_norm, 0.3)
        if override:
            pts.update(override)
        return _make_pose_sized(pts, size, vis=vis, vis_map=vis_map)

    # 1) 定义：raw = |耳中点x − 肩中点x| / 肩宽（全像素空间）
    fe = EarShoulderProxyFeatures(**small)
    r = fe.update(_esp(0.0), timestamp=0.0)
    _assert_close(r['ear_shoulder_proxy_raw'], 0.0, 1e-9,
                  "耳在肩中点正上方 → 水平位移 0 → proxy 0")

    fe = EarShoulderProxyFeatures(**small)
    r = fe.update(_esp(0.1), timestamp=0.0)          # Δx = 20px / 60px 肩宽
    _assert_close(r['ear_shoulder_proxy_raw'], 20.0 / 60.0, 1e-9,
                  "Δx=20px、肩宽 60px → proxy = 1/3")

    # 2) 尺度不变：同一归一化几何换分辨率 → 比值不变（分子分母同为水平长度）
    r_big = EarShoulderProxyFeatures(**small).update(_esp(0.1, size=(400, 200)),
                                                     timestamp=0.0)
    _assert_close(r_big['ear_shoulder_proxy_raw'], 20.0 / 60.0, 1e-9,
                  "换分辨率不应改变 proxy（比值与画面尺寸无关）")

    # 3) 方向：位移越大 proxy 越大（**越大越前伸**，与 cva_proxy_deg 相反）
    vals = []
    for dx in (0.0, 0.05, 0.1, 0.2):
        vals.append(EarShoulderProxyFeatures(**small)
                    .update(_esp(dx), timestamp=0.0)['ear_shoulder_proxy_raw'])
    assert vals == sorted(vals) and vals[0] == 0.0, f"应单调递增：{vals}"
    _assert_close(vals[-1], 40.0 / 60.0, 1e-9, "Δx=40px → proxy = 2/3")

    # 4) 预热：raw 从第一帧就有，平滑值要凑够帧数 + 时间跨度才出
    fe = EarShoulderProxyFeatures(**small)
    r = fe.update(_esp(0.1), timestamp=0.0)
    assert r['ear_shoulder_proxy_raw'] is not None, "raw 第一帧就应有值"
    assert r['ear_shoulder_proxy'] is None and not r['ear_shoulder_proxy_valid'], \
        "预热未满不该出平滑值"
    for i in (1, 2):
        r = fe.update(_esp(0.1), timestamp=i * 0.05)
    assert r['ear_shoulder_proxy_valid'], "3 帧 / 0.1s 后应出平滑值"
    _assert_close(r['ear_shoulder_proxy'], 1.0 / 3.0, 1e-9,
                  "读数恒定时平滑值应收敛到该读数")

    # 5) EMA 跟踪：阶跃变化后平滑值落在新旧之间（滞后但不跳变）
    r = fe.update(_esp(0.3), timestamp=0.25)         # 阶跃到 1.0
    _assert_close(r['ear_shoulder_proxy_raw'], 1.0, 1e-9, "本帧 raw 应为阶跃后的值")
    assert 1.0 / 3.0 < r['ear_shoulder_proxy'] < 1.0, \
        f"平滑值应滞后于阶跃：{r['ear_shoulder_proxy']}"

    # 6) 视角退化（侧身肩宽塌缩）：肩宽 0.02 归一化 < 0.25 → 不算有效读数，
    #    但 raw 与肩宽照常返回（CSV 里看得见这个被放大的数）
    fe = EarShoulderProxyFeatures(**small)
    for i in range(4):
        fe.update(_esp(0.1), timestamp=i * 0.05)
    ok = fe.update(_esp(0.1), timestamp=0.2)
    held = fe.update(_esp(0.1, override={5: (0.49, 0.5), 6: (0.51, 0.5)}),
                     timestamp=0.25)
    assert held['ear_shoulder_proxy_degenerate'], "肩宽塌陷应判视角退化"
    assert not held['ear_shoulder_proxy_valid'], "退化帧不算有效读数"
    assert held['ear_shoulder_proxy'] == ok['ear_shoulder_proxy'], \
        "退化帧应维持上一有效平滑值"
    assert held['ear_shoulder_proxy_raw'] > 1.0, \
        f"退化帧 raw 仍记录（且被放大）：{held['ear_shoulder_proxy_raw']}"
    assert held['ear_shoulder_proxy_parts']['shoulder_width'] < 10.0

    # 7) 单肩不可见（置信度门控）→ 分母不齐 → 本帧降级；parts 仍给出原始置信度
    fe = EarShoulderProxyFeatures(**small)
    r = fe.update(_esp(0.1, vis_map={6: 0.2}), timestamp=0.0)
    assert r['ear_shoulder_proxy_raw'] is None, "只有单肩时算不出肩宽，应降级"
    assert not r['ear_shoulder_proxy_valid']
    _assert_close(r['ear_shoulder_proxy_parts']['right_shoulder_confidence'], 0.2,
                  1e-9, "parts 应给出被门控掉的点的原始置信度")
    assert r['ear_shoulder_proxy_parts']['shoulder_width'] is None, \
        "分母不齐时肩宽应为 None（不填 0）"

    # 8) 单耳可见：pts 标签标出用了哪只耳；conf = 参与点（耳+双肩）平均
    vis_map = {3: 0.6, 4: 0.2, 5: 0.8, 6: 0.9}
    fe = EarShoulderProxyFeatures(**small)
    for i in range(3):
        r = fe.update(_esp(0.1, vis_map=vis_map), timestamp=i * 0.05)
    assert r['ear_shoulder_proxy_pts'] == "earL+sh_mid", r['ear_shoulder_proxy_pts']
    _assert_close(r['ear_shoulder_proxy_conf'], (0.6 + 0.8 + 0.9) / 3.0, 1e-9,
                  "conf 应为参与点（耳+双肩）的平均置信度")
    _assert_close(r['ear_shoulder_proxy_geom']['ear_mid'][0], 120.0, 1e-6,
                  "geom.ear_mid.x 应为耳中点像素 x")
    _assert_close(r['ear_shoulder_proxy_geom']['sh_width_px'], 60.0, 1e-6,
                  "geom.sh_width_px 应为两肩像素距离")
    _assert_close(r['ear_shoulder_proxy_parts']['ear_mid_x'], 120.0, 1e-6,
                  "parts.ear_mid_x 应为耳中点像素 x")
    _assert_close(r['ear_shoulder_proxy_parts']['shoulder_mid_x'], 100.0, 1e-6,
                  "parts.shoulder_mid_x 应为肩中点像素 x")
    _assert_close(r['ear_shoulder_proxy_parts']['shoulder_width_norm'], 0.30, 1e-9,
                  "parts.shoulder_width_norm 应为归一化肩宽")

    # 9) 无人 / image_size 缺失 → 降级不报错，parts 全 None（不填 0）
    fe = EarShoulderProxyFeatures(**small)
    r = fe.update(None, timestamp=0.0)
    assert r['ear_shoulder_proxy_raw'] is None and not r['ear_shoulder_proxy_valid']
    assert set(r['ear_shoulder_proxy_parts'].values()) == {None}
    no_size = {'landmarks': [(0.5, 0.3, 0.0, 1.0)] * 17}   # 故意不给 image_size
    r = fe.update(no_size, timestamp=0.1)
    assert r['ear_shoulder_proxy_raw'] is None, "无 image_size 时应降级（像素空间）"

    # 10) reset() 清空平滑状态；非法参数抛 ValueError
    fe.reset()
    assert fe.update(_esp(0.1), timestamp=9.0)['ear_shoulder_proxy'] is None, \
        "reset 后应回到预热状态"
    for kw in (dict(smooth_tau_sec=0.0), dict(min_shoulder_width_norm=0.0)):
        try:
            EarShoulderProxyFeatures(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kw} 应抛 ValueError")

    print("  selftest_ear_shoulder_proxy: OK")


def selftest_sternum_ear_angle() -> None:
    """合成数据自测 SternumEarAngleFeatures：像素空间定义（顶点处夹角 = 2·atan(e/d)）、
    方向（头往肩线压低 → 角度**变大**）、像素空间的分辨率不变性、纵横比敏感性、
    四点缺一即降级（比 ES 类严）、置信度门控、肩宽塌陷判退化、EMA 与预热语义、
    parts/geom 的逐帧原始量、无人/无 image_size 降级、reset 与参数校验。"""
    small = dict(min_frames=3, min_window_seconds=0.1, smooth_tau_sec=1.0)

    # 合成几何（归一化坐标固定，size 只换画布）：
    #   肩 (200,400)/(440,400)px  → 顶点 (320,400)px、肩宽 240px / 归一化 0.375（>0.25）
    #   耳 (320±e, 400−d)px       → 顶点处夹角 = 2·atan(e/d)，e/d 为参考画布(640×480)像素
    def _sea(e_px, d_px, size=(640, 480), vis=1.0, vis_map=None, override=None):
        pts = {5: (200.0 / 640.0, 400.0 / 480.0),
               6: (440.0 / 640.0, 400.0 / 480.0),
               3: (320.0 / 640.0 - e_px / 640.0, (400.0 - d_px) / 480.0),
               4: (320.0 / 640.0 + e_px / 640.0, (400.0 - d_px) / 480.0)}
        if override:
            pts.update(override)
        return _make_pose_sized(pts, size, vis=vis, vis_map=vis_map)

    # 1) 定义：raw = 顶点处两条边的夹角 = 2·atan(e/d)（全像素空间）
    fe = SternumEarAngleFeatures(**small)
    r = fe.update(_sea(100.0, 200.0), timestamp=0.0)
    _assert_close(r['sternum_ear_angle_raw'],
                  2.0 * math.degrees(math.atan2(100.0, 200.0)), 1e-9,
                  "e=100px、d=200px → 夹角 2·atan(e/d) ≈ 53.13°")

    # 2) 方向：头往肩线压低（d 变小）→ 角度**变大**（越大越前伸）
    vals = [SternumEarAngleFeatures(**small)
            .update(_sea(100.0, d), timestamp=0.0)['sternum_ear_angle_raw']
            for d in (200.0, 160.0, 120.0, 80.0)]
    assert vals == sorted(vals), f"d 越小角度应越大：{vals}"
    _assert_close(vals[-1], 2.0 * math.degrees(math.atan2(100.0, 80.0)), 1e-9,
                  "头压到离肩线 80px → 夹角 ≈ 102.68°")

    # 3) 像素空间 ⇒ 换分辨率（等比缩放画布）角度不变
    r_big = SternumEarAngleFeatures(**small).update(_sea(100.0, 200.0, size=(1280, 960)),
                                                    timestamp=0.0)
    _assert_close(r_big['sternum_ear_angle_raw'], r['sternum_ear_angle_raw'], 1e-9,
                  "同一归一化几何、画布整体放大 → 角度不变（像素空间）")

    # 4) 纵横比敏感：归一化几何不变、画布变 4:3 外（640×360）→ 角度被缩放过的 y 改变
    r_wide = SternumEarAngleFeatures(**small).update(_sea(100.0, 200.0, size=(640, 360)),
                                                     timestamp=0.0)
    _assert_close(r_wide['sternum_ear_angle_raw'],
                  2.0 * math.degrees(math.atan2(100.0, 150.0)), 1e-9,
                  "换纵横比后 d 被压到 150px → ≈67.38°（阈值只对同一空间/纵横比有效）")

    # 5) 预热：raw 从第一帧就有，平滑值要凑够帧数 + 时间跨度才出
    fe = SternumEarAngleFeatures(**small)
    r = fe.update(_sea(100.0, 200.0), timestamp=0.0)
    assert r['sternum_ear_angle_raw'] is not None, "raw 第一帧就应有值"
    assert r['sternum_ear_angle'] is None and not r['sternum_ear_angle_valid'], \
        "预热未满不该出平滑值"
    for i in (1, 2):
        r = fe.update(_sea(100.0, 200.0), timestamp=i * 0.05)
    assert r['sternum_ear_angle_valid'], "3 帧 / 0.1s 后应出平滑值"
    _assert_close(r['sternum_ear_angle'], r['sternum_ear_angle_raw'], 1e-9,
                  "读数恒定时平滑值应收敛到该读数")

    # 6) EMA 跟踪：阶跃（头压低）后平滑值落在新旧之间（滞后但不跳变）
    r = fe.update(_sea(100.0, 120.0), timestamp=0.25)
    prev = 2.0 * math.degrees(math.atan2(100.0, 200.0))
    _assert_close(r['sternum_ear_angle_raw'],
                  2.0 * math.degrees(math.atan2(100.0, 120.0)), 1e-9,
                  "本帧 raw 应为阶跃后的值")
    assert prev < r['sternum_ear_angle'] < r['sternum_ear_angle_raw'], \
        f"平滑值应滞后于阶跃：{r['sternum_ear_angle']}"

    # 7) 视角退化（侧身肩宽塌缩）：肩宽 0.02 归一化 < 0.25 → 不算有效读数，
    #    但 raw 与几何照常返回（CSV 里看得见这个帧）
    ok = fe.update(_sea(100.0, 120.0), timestamp=0.3)
    held = fe.update(_sea(100.0, 120.0, override={5: (0.49, 0.5), 6: (0.51, 0.5)}),
                     timestamp=0.35)
    assert held['sternum_ear_angle_degenerate'], "肩宽塌陷应判视角退化"
    assert not held['sternum_ear_angle_valid'], "退化帧不算有效读数"
    assert held['sternum_ear_angle'] == ok['sternum_ear_angle'], \
        "退化帧应维持上一有效平滑值"
    assert held['sternum_ear_angle_raw'] is not None, "退化帧 raw 仍记录（供审计）"
    assert held['sternum_ear_angle_geom'] == ok['sternum_ear_angle_geom'], \
        "退化帧几何应维持上一有效值"
    assert held['sternum_ear_angle_parts']['shoulder_width_norm'] < 0.25, \
        "parts 应给出本帧塌陷的归一化肩宽"

    # 8) 四个点缺一即降级（**比 EarShoulderProxyFeatures 的"双肩+任一耳"更严**）
    for missing, why in ((4, "缺右耳"), (3, "缺左耳"), (6, "缺右肩"), (5, "缺左肩")):
        r = SternumEarAngleFeatures(**small).update(
            _sea(100.0, 200.0, vis_map={missing: 0.2}), timestamp=0.0)
        assert r['sternum_ear_angle_raw'] is None, f"{why}时应降级（四点都要）"
        assert not r['sternum_ear_angle_valid']
        _assert_close(r['sternum_ear_angle_parts']
                      [{3: 'left_ear_confidence', 4: 'right_ear_confidence',
                        5: 'left_shoulder_confidence',
                        6: 'right_shoulder_confidence'}[missing]], 0.2, 1e-9,
                      "parts 应给出被门控掉的点的原始置信度")
    r = SternumEarAngleFeatures(**small).update(_sea(100.0, 200.0, vis_map={6: 0.2}),
                                                timestamp=0.0)
    assert r['sternum_ear_angle_parts']['shoulder_width'] is None, \
        "缺肩时肩宽应为 None（不填 0）"

    # 9) parts / geom：本帧原始量（像素）+ 参与点平均置信度
    vis_map = {3: 0.6, 4: 0.7, 5: 0.8, 6: 0.9}
    fe = SternumEarAngleFeatures(**small)
    for i in range(3):
        r = fe.update(_sea(100.0, 200.0, vis_map=vis_map), timestamp=i * 0.05)
    assert r['sternum_ear_angle_pts'] == "ear_mid+sh_mid", r['sternum_ear_angle_pts']
    _assert_close(r['sternum_ear_angle_conf'], (0.6 + 0.7 + 0.8 + 0.9) / 4.0, 1e-9,
                  "conf 应为参与四点（耳+双肩）的平均置信度")
    parts = r['sternum_ear_angle_parts']
    _assert_close(parts['ear_span'], 200.0, 1e-6, "ear_span 应为两耳像素距离 2e")
    _assert_close(parts['ear_sh_vert'], 200.0, 1e-6, "ear_sh_vert 应为耳-肩竖直间距 d")
    _assert_close(parts['vertex_x'], 320.0, 1e-6, "vertex_x 应为肩中点像素 x")
    _assert_close(parts['vertex_y'], 400.0, 1e-6, "vertex_y 应为肩中点像素 y")
    _assert_close(parts['shoulder_width'], 240.0, 1e-6, "shoulder_width 应为两肩像素距离")
    _assert_close(parts['shoulder_width_norm'], 0.375, 1e-9, "shoulder_width_norm 应为归一化肩宽")
    geom = r['sternum_ear_angle_geom']
    assert geom['vertex'] == (320.0, 400.0), geom['vertex']
    assert geom['ear_l'] == (220.0, 200.0) and geom['ear_r'] == (420.0, 200.0), geom
    _assert_close(geom['sh_width_px'], 240.0, 1e-6, "geom.sh_width_px 应为两肩像素距离")

    # 10) 冗余性口径可用：角度 ≈ 2·atan(ear_span/2 ÷ ear_sh_vert)
    _assert_close(r['sternum_ear_angle_raw'],
                  2.0 * math.degrees(math.atan2(parts['ear_span'] / 2.0,
                                                parts['ear_sh_vert'])), 1e-6,
                  "本角度应等于 2·atan(span/2 ÷ vert)（与颈压缩同源，做冗余核对用）")

    # 11) 无人 / image_size 缺失 → 降级不报错，parts 全 None（不填 0）
    fe = SternumEarAngleFeatures(**small)
    r = fe.update(None, timestamp=0.0)
    assert r['sternum_ear_angle_raw'] is None and not r['sternum_ear_angle_valid']
    assert set(r['sternum_ear_angle_parts'].values()) == {None}
    no_size = {'landmarks': [(0.5, 0.3, 0.0, 1.0)] * 17}   # 故意不给 image_size
    r = fe.update(no_size, timestamp=0.1)
    assert r['sternum_ear_angle_raw'] is None, "无 image_size 时应降级（像素空间）"

    # 12) reset() 清空平滑状态；非法参数抛 ValueError
    fe.reset()
    assert fe.update(_sea(100.0, 200.0), timestamp=9.0)['sternum_ear_angle'] is None, \
        "reset 后应回到预热状态"
    for kw in (dict(smooth_tau_sec=0.0), dict(min_shoulder_width_norm=0.0)):
        try:
            SternumEarAngleFeatures(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kw} 应抛 ValueError")

    print("  selftest_sternum_ear_angle: OK")


def selftest_arm_raise() -> None:
    """合成数据自测 ArmRaiseFeatures：上举/平举/下垂的抬升角、肘角、两侧合成、
    丢点降级、像素空间（纵横比）纪律。"""
    fe = ArmRaiseFeatures()
    # 用 size=(100,100) 让归一化坐标 ×100 = 像素坐标，读数是整数角度好核对
    size = (100, 100)
    # 肩 (50,30)：肘 (50,10) = 正上方 → 0°；(70,30) = 水平 → 90°；(50,50) = 正下 → 180°
    sh = {5: (0.5, 0.3), 6: (0.5, 0.3)}

    # 1) 单侧上举：抬升角 0°，且左肘角 = 90°（腕在肘正右方）
    r = fe.update(_make_pose_sized({**sh, 7: (0.5, 0.1), 9: (0.7, 0.1)}, size))
    _assert_close(r['arm_raise_left_deg'], 0.0, 1e-6, "上举侧抬升角应为 0°")
    _assert_close(r['arm_raise_deg'], 0.0, 1e-6, "抬升角应取更举的一侧")
    assert r['arm_raise_side'] == 'L', "两侧同角度时应取 L（确定性平局规则）"
    _assert_close(r['elbow_left_deg'], 90.0, 1e-6, "肘角应为 90°")
    assert r['arm_raise_valid'] and r['elbow_valid']

    # 2) 平举 (90°) 与下垂 (180°)：方向必须分得开（这是不用 PostureFeatures
    #    那个 abs() 版 _angle_from_vertical 的原因）
    r = fe.update(_make_pose_sized({**sh, 7: (0.7, 0.3)}, size))
    _assert_close(r['arm_raise_left_deg'], 90.0, 1e-6, "平举应为 90°")
    r = fe.update(_make_pose_sized({**sh, 7: (0.5, 0.5)}, size))
    _assert_close(r['arm_raise_left_deg'], 180.0, 1e-6, "下垂应为 180°")

    # 3) 伸直的手臂（肘角 180°）在下垂和平举时**都是** 180° —— 说明肘角单独
    #    判不出举手，而抬升角分得开（本文件选择抬升角做触发依据的理由）
    r = fe.update(_make_pose_sized({**sh, 7: (0.5, 0.5), 9: (0.5, 0.7)}, size))
    _assert_close(r['elbow_left_deg'], 180.0, 1e-6, "下垂伸直：肘角 180°")
    r = fe.update(_make_pose_sized({**sh, 7: (0.7, 0.3), 9: (0.9, 0.3)}, size))
    _assert_close(r['elbow_left_deg'], 180.0, 1e-6, "平举伸直：肘角也是 180°")

    # 4) 两侧都可见：取抬升角更小（举得更高）的那一侧
    r = fe.update(_make_pose_sized(
        {5: (0.5, 0.3), 6: (0.5, 0.3), 7: (0.5, 0.5), 8: (0.5, 0.1)}, size))
    assert r['arm_raise_side'] == 'R', "应取举得更高的 R 侧"
    _assert_close(r['arm_raise_deg'], 0.0, 1e-6, "R 侧上举 → 0°")
    _assert_close(r['arm_raise_left_deg'], 180.0, 1e-6, "L 侧读数仍照报")

    # 5) 腕丢失：抬升角照算，只有该侧肘角变 N/A（腕只喂显示用的肘角）
    r = fe.update(_make_pose_sized({**sh, 7: (0.5, 0.1)}, size))
    _assert_close(r['arm_raise_deg'], 0.0, 1e-6, "腕缺失不应影响抬升角")
    assert r['elbow_left_deg'] is None and not r['elbow_valid'], \
        "腕缺失时该侧肘角应为 N/A"

    # 6) 一侧肩/肘丢失 → 该侧 N/A，另一侧照常；两侧都缺 → valid=False
    #    （不做对侧/眼鼻替代 —— 缺一个点就是缺）
    r = fe.update(_make_pose_sized({5: (0.5, 0.3), 8: (0.5, 0.5), 6: (0.5, 0.3)},
                                   size))
    assert r['arm_raise_left_deg'] is None, "左肘缺失 → 左侧 N/A"
    _assert_close(r['arm_raise_right_deg'], 180.0, 1e-6, "右侧应不受影响")
    r = fe.update(_make_pose_sized({5: (0.5, 0.3), 6: (0.5, 0.3)}, size))
    assert not r['arm_raise_valid'] and r['arm_raise_deg'] is None

    # 7) 低置信度点按不可见处理（visibility_min=0.3）
    r = fe.update(_make_pose_sized({**sh, 7: (0.5, 0.1)}, size,
                                   vis_map={7: 0.1}))
    assert not r['arm_raise_valid'], "置信度 0.1 的肘点应被门控掉"

    # 8) 像素空间纪律：同一组归一化坐标在 16:9 下读数不同（归一化空间会带
    #    纵横比偏差，本类与 CvaProxyFeatures 一样固定在像素空间算）
    #    归一化 (0.1, -0.1)：正方形 = 45°；200x100 → 像素 (20,-10) = 63.43°
    r = fe.update(_make_pose_sized({5: (0.5, 0.3), 7: (0.6, 0.2)}, (200, 100)))
    _assert_close(r['arm_raise_left_deg'],
                  math.degrees(math.atan2(20.0, 10.0)), 1e-6,
                  "16:9 下应按像素空间算（63.43°），不是归一化的 45°")

    # 9) 无人 / 无 image_size → 全 None 且 valid=False（不抛异常、不填 0）
    assert not fe.update(None)['arm_raise_valid']
    no_size = {'landmarks': [(0.5, 0.3, 0.0, 1.0)] * 17}
    assert fe.update(no_size)['arm_raise_deg'] is None, "无 image_size 应降级"

    print("  selftest_arm_raise: OK")


def selftest_head_tilt() -> None:
    """合成数据自测 HeadTiltFeatures：仰头/低头符号、单耳退化、镜像不变号、
    丢点降级、**横向退化门控**、像素空间纪律。"""
    fe = HeadTiltFeatures()
    size = (100, 100)   # 归一化 ×100 = 像素
    # 双耳分开 10px（耳中点 (50,20)，耳距 10px）。**别用两个重合的耳点**：
    # 那样耳距 = 0，横向退化门控算不出比值，就把门控本身漏测了
    ears = {3: (0.45, 0.2), 4: (0.55, 0.2)}

    # 1) 鼻与耳同高 → 0°；仰头（鼻在上方）→ 正；低头 → 负
    #    （鼻在耳中点前方 10px，耳距同为 10px → |Δx|/耳距 = 1.0，远高于门控）
    r = fe.update(_make_pose_sized({**ears, 0: (0.6, 0.2)}, size))
    _assert_close(r['head_pitch_deg'], 0.0, 1e-6, "同高应为 0°")
    r = fe.update(_make_pose_sized({**ears, 0: (0.6, 0.1)}, size))
    _assert_close(r['head_pitch_deg'], 45.0, 1e-6, "仰头 45° 应为正")
    r = fe.update(_make_pose_sized({**ears, 0: (0.6, 0.3)}, size))
    _assert_close(r['head_pitch_deg'], -45.0, 1e-6, "低头 45° 应为负")
    assert r['head_pitch_valid'] and r['head_pitch_pts'] == 'ear_mid+nose'

    # 2) 人朝画面另一边（鼻在耳的左前方）：符号约定不变
    r = fe.update(_make_pose_sized({3: (0.55, 0.2), 4: (0.45, 0.2),
                                    0: (0.4, 0.1)}, size))
    _assert_close(r['head_pitch_deg'], 45.0, 1e-6, "镜像姿势应得同一符号")

    # 3) 单耳可见 → 仍可算，并如实标出参考点换了（earL/earR 零点与 ear_mid 不同一）
    r = fe.update(_make_pose_sized({3: (0.5, 0.2), 0: (0.6, 0.1)}, size))
    assert r['head_pitch_valid'] and r['head_pitch_pts'] == 'earL+nose'
    assert r['head_tilt_geom']['ear_span'] is None, "单耳回退没有耳距可给（不许编）"
    r = fe.update(_make_pose_sized({4: (0.5, 0.2), 0: (0.6, 0.1)}, size))
    assert r['head_pitch_pts'] == 'earR+nose'

    # 4) 双耳全丢 / 鼻丢 → N/A（不做眼→鼻兜底：换参考点等于换了一个量）
    r = fe.update(_make_pose_sized({0: (0.6, 0.1)}, size))
    assert not r['head_pitch_valid'] and r['head_pitch_deg'] is None, \
        "耳全丢应为 N/A"
    r = fe.update(_make_pose_sized({**ears}, size))
    assert not r['head_pitch_valid'], "鼻丢失应为 N/A"

    # 5) 低置信度耳按不可见处理 → 退成单耳（不是直接 N/A）
    r = fe.update(_make_pose_sized({**ears, 0: (0.6, 0.1)}, size,
                                   vis_map={4: 0.1}))
    assert r['head_pitch_pts'] == 'earL+nose'

    # 6) 几何是像素坐标（供 output 画线，本层只给坐标不画）
    r = fe.update(_make_pose_sized({**ears, 0: (0.6, 0.1)}, size))
    _assert_close(r['head_tilt_geom']['ear_mid'][1], 20.0, 1e-6, "耳中点像素 y")
    _assert_close(r['head_tilt_geom']['nose'][1], 10.0, 1e-6, "鼻像素 y")
    _assert_close(r['head_tilt_geom']['ear_span'], 10.0, 1e-6, "双耳间距像素")

    # 7) 像素空间纪律：同一归一化姿势在 16:9 下读数不同（纵横比）
    r = fe.update(_make_pose_sized({**ears, 0: (0.6, 0.1)}, (200, 100)))
    _assert_close(r['head_pitch_deg'], 26.56505117707799, 1e-6,
                  "200x100 下应为 atan2(10,20)=26.57°（不是归一化的 45°）")

    # 8) **横向退化门控**：正对镜头 —— 鼻在耳中点正前方只 0.5px（耳距 10px，
    #    比值 5%，远低于 0.15）→ 读数会退化成 atan2(10,0.5)=87° 这种"看着可信
    #    其实没量值"的假数（2026-10-12 实拍：正对时 Δx = -3.3~+3.3px、读数恒定
    #    -73~-90°，静坐时还极稳）→ 必须作废
    frontal = _make_pose_sized({**ears, 0: (0.505, 0.1)}, size)
    r = fe.update(frontal)
    assert not r['head_pitch_valid'] and r['head_pitch_deg'] is None, \
        "正对镜头的帧应判几何退化（N/A），不给看似可信的 ±90°"
    _assert_close(r['head_pitch_dx_ratio'], 0.05, 1e-9,
                  "退化帧也要给 |Δx|/耳距（否则画面上说不清为什么 N/A）")
    assert r['head_tilt_geom'] is not None, "退化帧也要给几何，供画线说明"
    # 同一帧把门控关掉 → 又会给出读数：证明是门控在判断，不是这一帧算不出来
    r = HeadTiltFeatures(min_dx_span_ratio=0.0).update(frontal)
    assert r['head_pitch_valid'] and abs(r['head_pitch_deg']) > 80.0, \
        "关掉门控后该帧会给出接近 ±90° 的假读数 —— 这正是要作废它的原因"
    # 门控阈值本身可调（45° 实测 0.43~0.52、正对 0.013，默认 0.15 取在空档中间）
    # 这一帧的比值是 1.0，把阈值提到 1.1 就该被作废
    r = HeadTiltFeatures(min_dx_span_ratio=1.1).update(
        _make_pose_sized({**ears, 0: (0.6, 0.1)}, size))
    assert not r['head_pitch_valid'], "阈值调高到比值之上后，同一帧也该被作废"

    # 9) 无人 / 无 image_size → 全 None（不抛异常）
    assert not fe.update(None)['head_pitch_valid']
    no_size = {'landmarks': [(0.5, 0.3, 0.0, 1.0)] * 17}
    assert fe.update(no_size)['head_pitch_valid'] is False

    print("  selftest_head_tilt: OK")


if __name__ == "__main__":
    selftest_movement()
    selftest_posture()
    selftest_ergonomic()
    selftest_cva_proxy()
    selftest_ear_shoulder_proxy()
    selftest_sternum_ear_angle()
    selftest_arm_raise()
    selftest_head_tilt()
    print("features selftest: ALL PASSED")
