"""
features 层（全身动作线）：从一帧帧的 pose 结果里提出动作角度。**只算特征，不判状态。**

本仓库有十一个动作，每个动作一个特征类（都用 COCO 17 点、都在**像素空间**算）：

    举手 / 仰头
        ArmRaiseFeatures        上臂抬升角（0=完全上举/90=平举/180=自然下垂）
                                + 肘角（180=伸直）
        HeadTiltFeatures        耳中点→鼻尖 与水平线的夹角（仰头代理）

    躯干 / 颈 / 上肢
        TrunkFlexFeatures       肩中点→髋中点 与向上竖直轴的有符号夹角
        NeckLateralTiltFeatures 双耳连线与水平线的夹角
        ElbowFlexFeatures       肩-肘-腕在肘处的内角（左右独立）
        TrunkRotationFeatures   双肩线与双髋线的方向差

    下肢 / 骨盆 / 肩胛
        KneeFlexFeatures        髋-膝-踝在膝处的内角（左右独立）
        HipFlexFeatures         肩-髋-膝在髋处的内角（左右独立）
        ShoulderAbductionFeatures 肩→肘 与参考轴的夹角（左右独立）
        PelvicTiltFeatures      双髋连线与水平线的夹角
        ScapularSymmetryFeatures 双肩连线与水平线的夹角

所有类都**只算特征、不判状态、不做平滑**：平滑在 smoothing.AngleSmoother 里，
判定在 decision.py 里。关键点丢帧 / 几何不可信时一律给 None + valid=False，
**不做 fallback**（缺哪个点就是缺，不拿别的点替代）。

**几何门控（2026-10-18 加）**：
  * 线段短于画面对角线 `MIN_SEGMENT_RATIO` 时角度对像素噪声极敏感 → 判无效；
  * RTMPose 会把画面外的点外推出坐标（不是 None）→ 超出 `FRAME_EDGE_MARGIN`
    的点按不可见处理；
  * 无效时附 `*_invalid_reason` 短码（no_pose/no_size/no_point/out_of_frame/
    geom/degenerate），供 output 在 N/A 旁边写清原因。这些键不以 `_deg` 结尾，
    平滑层原样透传。

**为什么在像素空间算**：归一化坐标先把 x 除以 W、y 除以 H，同一个姿势算出的角度
会带上画面宽高比的偏差（16:9 下越接近水平的量偏得越多）。所以这些类都要求
`image_size`，缺失 / 非正就本帧不出读数 —— 宁可不判，也不出一个带纵横比偏差的数。

**平滑不在本层**（2026-10-18 用户实拍反馈后加在 smoothing.py）：本层仍只算原始
读数；`None` 不跨帧维持，丢帧信号原样保留。

本模块只用标准库（math），不 import cv2 / pose_estimation，可以独立测试。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

# 躯干 4 个关键点（COCO 17 点标准编号，RTMPose 输出）
# 0 鼻, 1/2 眼, 3 左耳, 4 右耳, 5 左肩, 6 右肩, 7 左肘, 8 右肘,
# 9 左腕, 10 右腕, 11 左髋, 12 右髋, 13 左膝, 14 右膝, 15 左踝, 16 右踝

# 耳朵点（3 左耳 / 4 右耳）—— 仰头特征（HeadTiltFeatures）的头参考
EAR_IDS = (3, 4)

# 肩部点（5 左肩 / 6 右肩）—— 举手特征的抬升角起点
SHOULDER_IDS = (5, 6)

# 肘部点（7 左肘 / 8 右肘）—— 举手特征抬升角的终点（抬升角 = 肩→肘 与竖直轴夹角）
ELBOW_IDS = (7, 8)

# 腕部点（9 左腕 / 10 右腕）—— 举手特征（ArmRaiseFeatures）的肘角用；
# 抬升角只要肩+肘，腕缺失不影响抬升角（见 ArmRaiseFeatures）
WRIST_IDS = (9, 10)

# 鼻（COCO 17 的 0 号点）—— 仰头特征（HeadTiltFeatures）的头参考。
# COCO 17 没有下巴/头顶/耳屏，鼻尖是离头部旋转轴心最远的点，用它做 2D 俯仰代理
NOSE_ID = 0

# 髋/膝/踝（COCO 17）—— 躯干与下肢动作特征用：
#   11/12 左/右髋、13/14 左/右膝、15/16 左/右踝
HIP_IDS = (11, 12)
KNEE_IDS = (13, 14)
ANKLE_IDS = (15, 16)

# 仰头特征的横向退化门控（2026-10-12 实拍，同一人同一摄像头换机位）：
# 正对镜头时头部前后轴指着镜头，耳→鼻在像面上只剩竖直分量 → |Δx| ≈ 0 →
# atan2(-dy, |Δx|) 退化成 sign(θ)·90°：只剩"鼻子在耳中点上方还是下方"的符号、
# **没有任何量值**（实拍：Δx = -3.3~+3.3 px，读数恒定 -73~-90°，且静坐时极稳）。
# 这种错是静默的 —— -83.7°、45.0° 这类数看着完全正常，迟滞也救不了（防的是阈值
# 附近的抖动，而这里读数本身没量值）。所以必须在特征层把它作废：
# 用**双耳间距**归一化 |Δx|（抵消远近/头大小），低于此值判"几何退化" → valid=False。
# 标定依据：45° 前侧实测 0.43~0.52、正对实测 0.013（差 30 倍以上），阈值取空档中间。
# 口径：宁可不判（N/A），不乱判。
MIN_PITCH_DX_SPAN_RATIO = 0.15

# 角度依赖的线段短于"画面对角线的 1%"时判几何不可信：角度对像素噪声极敏感，
# 远距离 / 低分辨率下几个像素的抖动就能让读数大幅摆动。
# 例：640x480 对角线 800px → 8px；1280x720 → 14.7px。
MIN_SEGMENT_RATIO = 0.01

# RTMPose 会把画面外的关节**外推**出坐标（不是 None），读数看着有效但不可信。
# 归一化坐标超出 [-2%, 102%] 的点按不可见处理（留 2% 容差给模型边缘噪声）。
FRAME_EDGE_MARGIN = 0.02

# 无效原因短码（写进 *_invalid_reason，供 output 在 N/A 旁显示）
REASON_NO_POSE = 'no_pose'
REASON_NO_SIZE = 'no_size'
REASON_NO_POINT = 'no_point'
REASON_OUT_OF_FRAME = 'out_of_frame'
REASON_GEOM = 'geom'
REASON_DEGENERATE = 'degenerate'
REASON_NO_TRUNK = 'no_trunk'

# 一个点的二维坐标
Point = Tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _min_segment_px(w: int, h: int) -> float:
    """画面尺度对应的最小可信线段长度（像素，带 2px 下限）。"""
    return max(2.0, MIN_SEGMENT_RATIO * math.hypot(w, h))


def _outside_frame(point: Point, margin: float = FRAME_EDGE_MARGIN) -> bool:
    """归一化坐标是否在画面外（含 RTMPose 的外推点）。"""
    x, y = point
    return (x < -margin or x > 1.0 + margin
            or y < -margin or y > 1.0 + margin)


def _point_reject_reason(landmarks: list, pid: int,
                         visibility_min: float,
                         margin: float = FRAME_EDGE_MARGIN) -> Optional[str]:
    """某个点为什么不可用：None = 可用，否则返回 no_point / out_of_frame。"""
    if pid >= len(landmarks):
        return REASON_NO_POINT
    x, y, _z, vis = landmarks[pid]
    if vis < visibility_min:
        return REASON_NO_POINT
    if _outside_frame((x, y), margin):
        return REASON_OUT_OF_FRAME
    return None


def _visible_points(landmarks: list, ids: Tuple[int, ...],
                    visibility_min: float,
                    edge_margin: float = FRAME_EDGE_MARGIN) -> Dict[int, Point]:
    """从 landmarks 里挑出 ids 中"可见且落在画面内"的点。

    visibility < visibility_min 或归一化坐标超出画面（含外推点）都算不可见。
    本层的特征不做窗口 / 中位统计（见模块头），只需要"取可见点"这一个原语。
    """
    out: Dict[int, Point] = {}
    for pid in ids:
        if pid < len(landmarks):
            x, y, _z, vis = landmarks[pid]
            if vis >= visibility_min and not _outside_frame((x, y), edge_margin):
                out[pid] = (x, y)
    return out


def _midpoint(a: Point, b: Point) -> Point:
    """两点中点。"""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _angle_from_up(v: Point) -> float:
    """向量 v（像素空间）与"向上竖直轴"的夹角（度，0~180）。

    0 = 向量正朝上，90 = 水平，180 = 正朝下。图像 y 轴向下，故"向上"= (0,-1)。

    这里**不能**用取绝对值的写法（像 atan(dx/dy) 那种）：那种只能给 0~90 的
    "离竖直轴多远"，分不出朝上还是朝下 —— 而举手恰恰要分：上举 0° 与自然下垂
    180° 是同一根轴的两端（见 ArmRaiseFeatures）。
    """
    norm = math.hypot(v[0], v[1])
    if norm < 1e-9:
        return 0.0
    # dot(v, up) = -v_y
    cos = max(-1.0, min(1.0, -v[1] / norm))
    return math.degrees(math.acos(cos))


def _angle_from_down(v: Point) -> float:
    """向量 v（像素空间）与"向下竖直轴"的夹角（度，0~180）。

    0 = 向量正朝下，90 = 水平，180 = 正朝上。图像 y 轴向下，故"向下"= (0,+1)。
    这是肩关节外展的参考：手臂自然垂在体侧 = 0°，平举 = 90°，举过头顶 = 180°。
    与 `_angle_from_up` 是同一根轴的两端（后者 0 = 朝上、180 = 朝下）。

    水平分量放在分子（atan2(|dx|, dy)）——外展量的是"上臂往侧面展开多少"，
    所以水平偏移是主量；abs(dx) 也意味着**分不出外展与内收**（2D 像面只能看到
    手臂离竖直轴多远，看不到它在身体前/后侧）。
    """
    dx, dy = v
    if math.hypot(dx, dy) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(abs(dx), dy))


def _angle_from_axis(v: Point, axis: Point) -> float:
    """向量 v 与参考轴 axis 的无符号夹角（度，0~180；0 = 同向）。

    axis 不必是单位向量。语义与 `_angle_from_down` 一致，但可以换任意参考轴
    （肩外展的 trunk 参考模式用肩中点→髋中点的躯干轴）。
    零向量调用方须先过门控（这里返回 0.0，不能靠它判无效）。
    """
    nv = math.hypot(v[0], v[1])
    na = math.hypot(axis[0], axis[1])
    if nv < 1e-9 or na < 1e-9:
        return 0.0
    cos = max(-1.0, min(1.0,
                        (v[0] * axis[0] + v[1] * axis[1]) / (nv * na)))
    return math.degrees(math.acos(cos))


def _angle_at(joint: Point, p1: Point, p2: Point,
              min_edge_px: float = 0.0) -> Optional[float]:
    """以 joint 为顶点、两条边 joint→p1 / joint→p2 的夹角（度，0~180）。

    任一边长度 ~0（点重合）或短于 `min_edge_px`（尺度门限）→ None：
    角度无意义 / 不可信，不返回 0 冒充读数。
    """
    u = (p1[0] - joint[0], p1[1] - joint[1])
    v = (p2[0] - joint[0], p2[1] - joint[1])
    nu = math.hypot(u[0], u[1])
    nv = math.hypot(v[0], v[1])
    floor = max(1e-9, min_edge_px)
    if nu < floor or nv < floor:
        return None
    cos = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv)))
    return math.degrees(math.acos(cos))


def _mean_point(points: list) -> Point:
    """若干点的几何平均（用于"可见点的中点"：点少了就是剩下那些点的中点）。"""
    n = float(len(points))
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


def _signed_angle_from_up(v: Point) -> float:
    """向量 v（像素空间）与"向上竖直轴"的**有符号**夹角（度，-180~180）。

    0 = 正上，正 = 偏画面右，负 = 偏画面左，±180 = 正下。
    与 `_angle_from_up`（0~180、不分左右）不同：躯干前屈/后伸要能分出往哪边倒，
    所以这里保留符号。符号是**画面左右**，不是解剖学前后 —— 45° 前侧机位下
    人朝哪个方向站，决定了正号对应前屈还是后伸，读数本身只保证单调。
    """
    return math.degrees(math.atan2(v[0], -v[1]))


def _line_angle_from_horizontal(a: Point, b: Point,
                                min_len_px: float = 0.0) -> Optional[float]:
    """a→b 连线与水平线的有符号夹角（度，-90~90）。

    先把方向规范成"图像左 → 图像右"（dx >= 0），所以：
        正 = 连线右端更高，负 = 连线左端更高。
    这样同一姿势在画面里左右镜像时，符号约定仍然一致。
    两点重合 / 连线短于 `min_len_px` → None。
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx < 0:
        dx, dy = -dx, -dy
    if math.hypot(dx, dy) < max(1e-9, min_len_px):
        return None
    return math.degrees(math.atan2(-dy, dx))


def _normalize_line_diff(deg: float) -> float:
    """两条**无向直线**的夹角差规范到 [-90, 90]（0 附近 = 两线接近平行）。"""
    return (deg + 90.0) % 180.0 - 90.0


def _side_joint_measure(landmarks: list, vis: Dict[int, Point],
                        w: int, h: int, ids: Tuple[int, int, int],
                        min_segment_px: float = 0.0,
                        visibility_min: float = 0.3) -> dict:
    """单侧三点关节角（p1 - joint - p2 在 joint 处的内角）。

    三个点必须都可见；任一点缺失 → 该侧 invalid（**不拿对侧或别的点替代**）。
    返回 {'deg','valid','conf','geom','reason'}，geom 是像素坐标
    {'p1','joint','p2'}，供 output 画三点线；无效侧 geom=None、reason 说明原因。
    """
    p1_id, joint_id, p2_id = ids
    missing = [pid for pid in ids if pid not in vis]
    if missing:
        reasons = [_point_reject_reason(landmarks, pid, visibility_min)
                   for pid in missing]
        # 画面外比"置信度低"更具体，优先报它
        reason = (REASON_OUT_OF_FRAME if REASON_OUT_OF_FRAME in reasons
                  else REASON_NO_POINT)
        return {'deg': None, 'valid': False, 'conf': None, 'geom': None,
                'reason': reason}
    p1, joint, p2 = vis[p1_id], vis[joint_id], vis[p2_id]
    p1_px = (p1[0] * w, p1[1] * h)
    joint_px = (joint[0] * w, joint[1] * h)
    p2_px = (p2[0] * w, p2[1] * h)
    deg = _angle_at(joint_px, p1_px, p2_px, min_edge_px=min_segment_px)
    if deg is None:      # 点重合 / 边太短 → 角度无意义或不可信
        return {'deg': None, 'valid': False, 'conf': None, 'geom': None,
                'reason': REASON_GEOM}
    conf = sum(landmarks[pid][3] for pid in ids) / float(len(ids))
    return {'deg': deg, 'valid': True, 'conf': conf,
            'geom': {'p1': p1_px, 'joint': joint_px, 'p2': p2_px},
            'reason': None}


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
    （16:9 下越接近水平的量偏得越多）。image_size 缺失/非正 → 本帧不出读数
    （宁可不判，也不出一个带纵横比偏差的数）。

    === 为什么不平滑 ===
    本类是"验证单目够不够用"的探针：**不 EMA、不中值、不维持上一有效值**。
    关键点丢帧就报 arm_raise_valid=False（读数 None），抖动就原样跳出来。
    加平滑会把"单目能不能撑住举手这种大动作"这个问题掩盖掉（见模块头注释）。

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
             'arm_raise_left_invalid_reason' / 'arm_raise_right_invalid_reason':
                                   str | None   该侧 N/A 的原因（no_pose /
                                   no_size / no_point / out_of_frame / geom），
                                   有效侧为 None
             'arm_geom': dict | None  该侧的像素几何，供 output 画线：
                             {'side': 'L'|'R', 'shoulder': (x_px,y_px),
                              'elbow': (x_px,y_px), 'wrist': (x_px,y_px)|None}}
        （无 reset：本类不持有跨帧状态）

    === 几何门控（尺度感知）===
    肩→肘短于画面对角线的 `MIN_SEGMENT_RATIO`（默认 1%）→ 该侧 N/A、reason
    `geom`：几个像素的抖动就能让这种短线段的角度大幅摆动，读数不可信。
    RTMPose 外推到画面外的点（归一化坐标超界）按不可见处理，reason
    `out_of_frame`。
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
            'arm_raise_left_invalid_reason': None,
            'arm_raise_right_invalid_reason': None,
            'elbow_deg': None,
            'elbow_left_deg': None,
            'elbow_right_deg': None,
            'elbow_valid': False,
            'arm_geom': None,
        }
        if pose_result is None:
            out['arm_raise_left_invalid_reason'] = REASON_NO_POSE
            out['arm_raise_right_invalid_reason'] = REASON_NO_POSE
            return out
        landmarks = pose_result.get('landmarks')
        size = pose_result.get('image_size')
        if not landmarks or not size or size[0] <= 0 or size[1] <= 0:
            reason = REASON_NO_POSE if not landmarks else REASON_NO_SIZE
            out['arm_raise_left_invalid_reason'] = reason
            out['arm_raise_right_invalid_reason'] = reason
            return out
        w, h = size
        vis = _visible_points(landmarks, SHOULDER_IDS + ELBOW_IDS + WRIST_IDS,
                              self.visibility_min)
        min_len = _min_segment_px(w, h)

        per = {}
        for side, ids in self.SIDES:
            m = self._side(landmarks, vis, w, h, side, ids, min_len)
            per[side] = m
            key = self._SIDE_KEY[side]
            out[f'arm_raise_{key}_deg'] = m['raise']
            out[f'elbow_{key}_deg'] = m['elbow']
            out[f'arm_raise_{key}_invalid_reason'] = m['reason']

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
              w: int, h: int, side: str, ids: Tuple[int, int, int],
              min_len_px: float) -> dict:
        """算单侧的两个角（点不齐 / 线段太短就是 None，不做任何替代）。"""
        sh_id, el_id, wr_id = ids
        sh, el, wr = vis.get(sh_id), vis.get(el_id), vis.get(wr_id)
        raise_deg = elbow_deg = conf = None
        geom = None
        reason = None
        if sh is None or el is None:
            missing = [pid for pid, p in ((sh_id, sh), (el_id, el))
                       if p is None]
            reasons = [_point_reject_reason(landmarks, pid, self.visibility_min)
                       for pid in missing]
            reason = (REASON_OUT_OF_FRAME if REASON_OUT_OF_FRAME in reasons
                      else REASON_NO_POINT)
        else:
            sh_px = (sh[0] * w, sh[1] * h)
            el_px = (el[0] * w, el[1] * h)
            if _dist(sh_px, el_px) < min_len_px:
                # 肩肘重合 / 太短 → 抬升角无意义或不可信
                reason = REASON_GEOM
            else:
                raise_deg = _angle_from_up((el_px[0] - sh_px[0],
                                            el_px[1] - sh_px[1]))
                confs = [landmarks[sh_id][3], landmarks[el_id][3]]
                geom = {'side': side, 'shoulder': sh_px, 'elbow': el_px,
                        'wrist': None}
                if wr is not None:
                    wr_px = (wr[0] * w, wr[1] * h)
                    elbow_deg = _angle_at(el_px, sh_px, wr_px,
                                          min_edge_px=min_len_px)
                    confs.append(landmarks[wr_id][3])
                    if elbow_deg is not None:
                        geom['wrist'] = wr_px
                conf = sum(confs) / len(confs)
        return {'raise': raise_deg, 'elbow': elbow_deg,
                'conf': conf, 'geom': geom, 'reason': reason}


class HeadTiltFeatures:
    """仰头（头后仰）角度特征：耳中点→鼻尖 与水平线的夹角（**只算不判、不平滑**）。

    === 用哪些点、怎么算 ===
        头参考 = 双耳(3/4)中点（两只都可见时）；只可见一只耳就用那一只
        鼻点   = 0（COCO 17 鼻）
        head_pitch_deg = 耳→鼻 向量 与 **水平线** 的夹角（度，-90~90）
                         正 = 鼻在耳**上方**（仰头），负 = 鼻在耳下方（低头）
    横向按 |Δx| 展开，所以人朝画面左或右都得到同一套符号约定（不用管人朝哪边）。

    === 为什么用耳→鼻（而不是耳-肩那类角度）===
    耳朵几乎就在"头后仰"的旋转轴心上，**仰头时耳朵基本不动** —— 耳→肩的角度
    几乎不变（那种量量的是头相对躯干的**平移**前伸，不是**转动**俯仰）。
    鼻尖离轴心最远，头一转它摆动最大：耳→鼻是"头自身的朝向"，与肩/躯干无关，
    正好把俯仰（仰头）和头前伸分开。

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
    与 ArmRaiseFeatures 同一口径（见模块头）：不平滑、不维持上一有效值。
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
             'head_pitch_invalid_reason': str | None  本帧 N/A 的原因（no_pose /
                                              no_size / no_point / out_of_frame /
                                              geom / degenerate）；有效帧为 None
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
            'head_pitch_invalid_reason': None,
            'head_tilt_geom': None,
        }
        if pose_result is None:
            out['head_pitch_invalid_reason'] = REASON_NO_POSE
            return out
        landmarks = pose_result.get('landmarks')
        size = pose_result.get('image_size')
        if not landmarks or not size or size[0] <= 0 or size[1] <= 0:
            out['head_pitch_invalid_reason'] = (
                REASON_NO_POSE if not landmarks else REASON_NO_SIZE)
            return out
        w, h = size
        min_len = _min_segment_px(w, h)
        vis = _visible_points(landmarks, EAR_IDS + (NOSE_ID,),
                              self.visibility_min)
        if NOSE_ID not in vis:
            out['head_pitch_invalid_reason'] = (
                _point_reject_reason(landmarks, NOSE_ID, self.visibility_min)
                or REASON_NO_POINT)
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
            # 耳全丢：不做眼→鼻兜底（见类 docstring）
            reasons = [_point_reject_reason(landmarks, pid, self.visibility_min)
                       for pid in EAR_IDS]
            out['head_pitch_invalid_reason'] = (
                REASON_OUT_OF_FRAME if REASON_OUT_OF_FRAME in reasons
                else REASON_NO_POINT)
            return out

        ear_px = (ear[0] * w, ear[1] * h)
        nose_px = (vis[NOSE_ID][0] * w, vis[NOSE_ID][1] * h)
        dx = nose_px[0] - ear_px[0]
        dy = nose_px[1] - ear_px[1]

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

        # **退化先判**：正对镜头时耳→鼻本身就很短，若先过下面的长度门控，会被
        # 报成"线段太短"，把"机位测不了"和"点没检测好"混为一谈（还丢掉既有的
        # head_pitch_degenerate 标记）。长度门控只在退化门控不适用时兜底
        # （单耳参考 / 双耳间距≈0）。
        if ratio is not None and ratio < self.min_dx_span_ratio:
            # 几何退化：deg 保持 None、valid 保持 False（宁可不判，不乱判）。
            # 由本层显式标出来，**不让 output 靠 ratio 有没有值去反推**（那是把
            # 门控逻辑复制到画图层）—— 画面上要与"关键点丢了"分开显示
            out['head_pitch_degenerate'] = True
            out['head_pitch_invalid_reason'] = REASON_DEGENERATE
            return out

        if math.hypot(dx, dy) < min_len:  # 耳鼻重合 / 太短 → 角度无意义
            out['head_pitch_invalid_reason'] = REASON_GEOM
            return out

        out['head_pitch_deg'] = math.degrees(math.atan2(-dy, abs(dx)))
        out['head_pitch_valid'] = True
        return out


class TrunkFlexFeatures:
    """躯干前屈/后伸特征：肩中点 → 髋中点 与竖直轴的夹角（**只算不判、不平滑**）。

    === 用哪些点、怎么算 ===
        肩中点 = 可见肩点(5/6)的几何平均
        髋中点 = 可见髋点(11/12)的几何平均
        trunk_flex_deg = 髋中点→肩中点 向量 与 **向上竖直轴** 的有符号夹角
                         （度，-180~180；0 = 躯干竖直，正 = 肩中点偏向画面右，
                          负 = 偏画面左）

    这是 2D 像面内的躯干倾斜，**不是真实的三维躯干屈伸角**：45° 机位下前后方向
    被投影压缩，读数只保证单调（往同一方向弯得越多，绝对值越大）。符号是画面
    左右，不是解剖学前/后 —— 人朝哪个方向站决定了正号对应前屈还是后伸。

    === 中点门控（按需求刻意允许"可见点少于 2 个"）===
    肩或髋任一侧的关键点 visibility < visibility_min → 该点不参与中点计算；
    只有一个肩点可见时，中点就是那一个肩点（不是 N/A）；两个肩点都不可见 →
    读数无效。髋同理。这是本动作明确的降级规则，**不要拿对侧点或别的关节替代**。

    === 降级 ===
    无人 / image_size 缺失或非正 / 双肩都不可见 / 双髋都不可见 / 两个中点太近
    → trunk_flex_deg=None、valid=False。**不做平滑、不维持上一有效值**。

    接口约定：
        TrunkFlexFeatures(visibility_min=0.3)
        update(pose_result) -> dict
            {'trunk_flex_deg':   float | None
             'trunk_flex_valid': bool
             'trunk_flex_conf':  float | None  参与点平均置信度
             'trunk_flex_pts':   str | None    如 'sh5+6+hip11+12'，标明用了哪些点
             'trunk_flex_invalid_reason': str | None  本帧 N/A 的原因（no_pose /
                                    no_size / no_point / out_of_frame / geom）
             'trunk_geom':       dict | None   {'shoulder_mid': (x_px,y_px),
                                                 'hip_mid': (x_px,y_px)}
                                               供 output 画肩髋连线
        （无 reset：本类不持有跨帧状态）
    """

    def __init__(self, visibility_min: float = 0.3) -> None:
        self.visibility_min = visibility_min

    def update(self, pose_result: Optional[dict]) -> dict:
        out = {
            'trunk_flex_deg': None,
            'trunk_flex_valid': False,
            'trunk_flex_conf': None,
            'trunk_flex_pts': None,
            'trunk_flex_invalid_reason': None,
            'trunk_geom': None,
        }
        if pose_result is None:
            out['trunk_flex_invalid_reason'] = REASON_NO_POSE
            return out
        landmarks = pose_result.get('landmarks')
        size = pose_result.get('image_size')
        if not landmarks or not size or size[0] <= 0 or size[1] <= 0:
            out['trunk_flex_invalid_reason'] = (
                REASON_NO_POSE if not landmarks else REASON_NO_SIZE)
            return out
        w, h = size
        min_len = _min_segment_px(w, h)
        vis = _visible_points(landmarks, SHOULDER_IDS + HIP_IDS,
                              self.visibility_min)
        sh_ids = [pid for pid in SHOULDER_IDS if pid in vis]
        hip_ids = [pid for pid in HIP_IDS if pid in vis]
        if not sh_ids or not hip_ids:
            missing = [pid for pid in (SHOULDER_IDS if not sh_ids else HIP_IDS)
                       if pid not in vis]
            reasons = [_point_reject_reason(landmarks, pid, self.visibility_min)
                       for pid in missing]
            out['trunk_flex_invalid_reason'] = (
                REASON_OUT_OF_FRAME if REASON_OUT_OF_FRAME in reasons
                else REASON_NO_POINT)
            return out

        sh_mid = _mean_point([vis[pid] for pid in sh_ids])
        hip_mid = _mean_point([vis[pid] for pid in hip_ids])
        sh_px = (sh_mid[0] * w, sh_mid[1] * h)
        hip_px = (hip_mid[0] * w, hip_mid[1] * h)
        v = (sh_px[0] - hip_px[0], sh_px[1] - hip_px[1])
        if math.hypot(v[0], v[1]) < min_len:   # 两个中点重合 / 太短 → 没有方向
            out['trunk_flex_invalid_reason'] = REASON_GEOM
            return out

        used = sh_ids + hip_ids
        out['trunk_flex_deg'] = _signed_angle_from_up(v)
        out['trunk_flex_valid'] = True
        out['trunk_flex_conf'] = (
            sum(landmarks[pid][3] for pid in used) / float(len(used)))
        out['trunk_flex_pts'] = (
            f"sh{'+'.join(str(i) for i in sh_ids)}"
            f"+hip{'+'.join(str(i) for i in hip_ids)}")
        out['trunk_geom'] = {'shoulder_mid': sh_px, 'hip_mid': hip_px}
        return out


class NeckLateralTiltFeatures:
    """颈部侧屈特征：双耳连线与水平线的夹角（**只算不判、不平滑**）。

    === 用哪些点、怎么算 ===
        3 左耳、4 右耳（COCO 17）
        neck_lateral_deg = 双耳连线 与 **水平线** 的有符号夹角
                           （度，-90~90；方向统一成图像左→右，
                            正 = 画面右侧的耳更高，负 = 左侧的耳更高）

    这是 2D 像面内的头侧倾量，不是真实颈椎侧屈角；45° 前侧机位下只保证单调。
    它和 `HeadTiltFeatures` 的仰头（耳中点→鼻）是两个独立的量，别互相替代：
    一个看**头侧倾**（双耳连线倾斜），一个看**头俯仰**（耳→鼻）。

    === 门控（按需求刻意不做单耳降级）===
    双耳必须都可见；缺任意一只耳 → 本帧 invalid、读数 N/A。
    不像 HeadTiltFeatures 那样允许单耳回退 —— 单耳根本构不成"双耳连线"。
    低置信度耳按不可见处理。

    接口约定：
        NeckLateralTiltFeatures(visibility_min=0.3)
        update(pose_result) -> dict
            {'neck_lateral_deg':   float | None
             'neck_lateral_valid': bool
             'neck_lateral_conf':  float | None  双耳平均置信度
             'neck_lateral_invalid_reason': str | None  本帧 N/A 的原因
             'neck_geom':          dict | None   {'earL': (x_px,y_px),
                                                  'earR': (x_px,y_px)}
        （无 reset：本类不持有跨帧状态）
    """

    def __init__(self, visibility_min: float = 0.3) -> None:
        self.visibility_min = visibility_min

    def update(self, pose_result: Optional[dict]) -> dict:
        out = {
            'neck_lateral_deg': None,
            'neck_lateral_valid': False,
            'neck_lateral_conf': None,
            'neck_lateral_invalid_reason': None,
            'neck_geom': None,
        }
        if pose_result is None:
            out['neck_lateral_invalid_reason'] = REASON_NO_POSE
            return out
        landmarks = pose_result.get('landmarks')
        size = pose_result.get('image_size')
        if not landmarks or not size or size[0] <= 0 or size[1] <= 0:
            out['neck_lateral_invalid_reason'] = (
                REASON_NO_POSE if not landmarks else REASON_NO_SIZE)
            return out
        w, h = size
        min_len = _min_segment_px(w, h)
        vis = _visible_points(landmarks, EAR_IDS, self.visibility_min)
        if 3 not in vis or 4 not in vis:    # 缺一只耳就不做连线（不降级）
            missing = [pid for pid in EAR_IDS if pid not in vis]
            reasons = [_point_reject_reason(landmarks, pid, self.visibility_min)
                       for pid in missing]
            out['neck_lateral_invalid_reason'] = (
                REASON_OUT_OF_FRAME if REASON_OUT_OF_FRAME in reasons
                else REASON_NO_POINT)
            return out
        ear_l = (vis[3][0] * w, vis[3][1] * h)
        ear_r = (vis[4][0] * w, vis[4][1] * h)
        deg = _line_angle_from_horizontal(ear_l, ear_r, min_len_px=min_len)
        if deg is None:     # 两点重合 / 双耳太近 → 角度不可信
            out['neck_lateral_invalid_reason'] = REASON_GEOM
            return out

        out['neck_lateral_deg'] = deg
        out['neck_lateral_valid'] = True
        out['neck_lateral_conf'] = (
            landmarks[3][3] + landmarks[4][3]) / 2.0
        out['neck_geom'] = {'earL': ear_l, 'earR': ear_r}
        return out


class _JointFlexFeaturesBase:
    """三点关节角特征的公共实现（肘/膝/髋共用）。

    子类只声明 `PREFIX`（返回键前缀）与 `SIDES`（左右各自的三个点编号）。
    每侧三个点必须**全部可见**才计算；任一点缺失 → 该侧 invalid，另一侧照算。
    角度是三点在中间关节处的内角（180 = 伸直），左右各自输出、**不合并**。
    两条边任一短于画面对角线的 `MIN_SEGMENT_RATIO` → 该侧 N/A、reason `geom`
    （短边角度对像素噪声极敏感）。
    """

    PREFIX = ""
    SIDES: Tuple[Tuple[str, Tuple[int, int, int]], ...] = ()
    # 侧别 → 返回键里的全拼后缀（'L'/'R' 是画面上/几何里用的短标，
    # 返回键用 left/right，别拿短标 lower() 去拼键名）
    _SIDE_KEY = {"L": "left", "R": "right"}

    def __init__(self, visibility_min: float = 0.3) -> None:
        self.visibility_min = visibility_min

    def update(self, pose_result: Optional[dict]) -> dict:
        out = {}
        for side, _ids in self.SIDES:
            key = self._SIDE_KEY[side]
            out[f'{self.PREFIX}_{key}_deg'] = None
            out[f'{self.PREFIX}_{key}_valid'] = False
            out[f'{self.PREFIX}_{key}_conf'] = None
            out[f'{self.PREFIX}_{key}_invalid_reason'] = None
        out[f'{self.PREFIX}_geom'] = {}

        if pose_result is None:
            for side, _ids in self.SIDES:
                out[f'{self.PREFIX}_{self._SIDE_KEY[side]}_invalid_reason'] = \
                    REASON_NO_POSE
            return out
        landmarks = pose_result.get('landmarks')
        size = pose_result.get('image_size')
        if not landmarks or not size or size[0] <= 0 or size[1] <= 0:
            reason = REASON_NO_POSE if not landmarks else REASON_NO_SIZE
            for side, _ids in self.SIDES:
                out[f'{self.PREFIX}_{self._SIDE_KEY[side]}_invalid_reason'] = reason
            return out
        w, h = size
        min_len = _min_segment_px(w, h)
        all_ids = tuple(pid for _side, ids in self.SIDES for pid in ids)
        vis = _visible_points(landmarks, all_ids, self.visibility_min)

        geom = {}
        for side, ids in self.SIDES:
            key = self._SIDE_KEY[side]
            m = _side_joint_measure(landmarks, vis, w, h, ids,
                                    min_segment_px=min_len,
                                    visibility_min=self.visibility_min)
            out[f'{self.PREFIX}_{key}_deg'] = m['deg']
            out[f'{self.PREFIX}_{key}_valid'] = m['valid']
            out[f'{self.PREFIX}_{key}_conf'] = m['conf']
            out[f'{self.PREFIX}_{key}_invalid_reason'] = m['reason']
            if m['geom'] is not None:
                geom[side] = m['geom']
        out[f'{self.PREFIX}_geom'] = geom
        return out


class ElbowFlexFeatures(_JointFlexFeaturesBase):
    """肘关节屈伸特征：肩-肘-腕在肘处的内角（左右独立，**只算不判、不平滑**）。

        L: 肩 5 → 肘 7 → 腕 9
        R: 肩 6 → 肘 8 → 腕 10
        角度 = 在肘处的内角（度，0~180；180 = 手臂伸直，越小 = 屈肘越明显）

    与 `ArmRaiseFeatures` 里的肘角是同一个几何量，但本类把它提升为**独立动作**：
    左右各自输出，没有"举得更高那一侧"的合并；某一侧缺任一肩/肘/腕点 →
    该侧 invalid、读数 N/A，另一侧不受影响。

    接口约定：
        ElbowFlexFeatures(visibility_min=0.3)
        update(pose_result) -> dict
            {'elbow_flex_left_deg':   float | None
             'elbow_flex_right_deg':  float | None
             'elbow_flex_left_valid' / 'elbow_flex_right_valid': bool
             'elbow_flex_left_conf' / 'elbow_flex_right_conf': float | None
             'elbow_flex_left_invalid_reason' / '..._right_...': str | None
             'elbow_flex_geom': {'L': {'p1','joint','p2'}, 'R': ...}  # 像素坐标，
                                只有该侧有效时才出现
        （无 reset：本类不持有跨帧状态）
    """

    PREFIX = 'elbow_flex'
    SIDES = (("L", (5, 7, 9)), ("R", (6, 8, 10)))


class TrunkRotationFeatures:
    """躯干旋转特征：双肩连线与双髋连线的方向差（**只算不判、不平滑**）。

    === 用哪些点、怎么算 ===
        双肩连线：5 左肩 → 6 右肩
        双髋连线：11 左髋 → 12 右髋
        两条线各自与水平线的有符号夹角之差（规范到 -90~90）
        trunk_rotation_deg 正/负 = 肩线相对髋线向某一侧转（画面方向，见下）

    **这是 2D 像面内的肩髋线夹角，不是真实的三维躯干轴向旋转（yaw）**。在
    45° 前侧机位下它是投影量，只能评估"肩髋是否同步转动/有没有相对扭转"，
    不能当解剖学旋转角用。用途是看动作代偿：例如转身时肩和髋应该同步，
    差值持续变大说明某一节段在代偿。

    === 门控（严格：两组都要完整）===
    双肩、双髋四个点**全部可见**才算；任一点不可见 → invalid、读数 N/A。
    （这里不做"可见点中点"降级 —— 两条线各少一个端点就没有方向了。）

    接口约定：
        TrunkRotationFeatures(visibility_min=0.3)
        update(pose_result) -> dict
            {'trunk_rotation_deg': float | None  肩线角 - 髋线角（-90~90）
             'trunk_rotation_valid': bool
             'trunk_rotation_conf': float | None 四个点平均置信度
             'trunk_rotation_shoulder_deg': float | None 肩线相对水平线角度（诊断）
             'trunk_rotation_hip_deg': float | None      髋线相对水平线角度（诊断）
             'trunk_rotation_invalid_reason': str | None 本帧 N/A 的原因
             'trunk_rotation_geom': dict | None
                 {'shoulderL','shoulderR','hipL','hipR'}  像素坐标
        （无 reset：本类不持有跨帧状态）
    """

    def __init__(self, visibility_min: float = 0.3) -> None:
        self.visibility_min = visibility_min

    def update(self, pose_result: Optional[dict]) -> dict:
        out = {
            'trunk_rotation_deg': None,
            'trunk_rotation_valid': False,
            'trunk_rotation_conf': None,
            'trunk_rotation_shoulder_deg': None,
            'trunk_rotation_hip_deg': None,
            'trunk_rotation_invalid_reason': None,
            'trunk_rotation_geom': None,
        }
        if pose_result is None:
            out['trunk_rotation_invalid_reason'] = REASON_NO_POSE
            return out
        landmarks = pose_result.get('landmarks')
        size = pose_result.get('image_size')
        if not landmarks or not size or size[0] <= 0 or size[1] <= 0:
            out['trunk_rotation_invalid_reason'] = (
                REASON_NO_POSE if not landmarks else REASON_NO_SIZE)
            return out
        w, h = size
        min_len = _min_segment_px(w, h)
        vis = _visible_points(
            landmarks, SHOULDER_IDS + HIP_IDS, self.visibility_min)
        required = (5, 6, 11, 12)
        missing = [pid for pid in required if pid not in vis]
        if missing:
            reasons = [_point_reject_reason(landmarks, pid, self.visibility_min)
                       for pid in missing]
            out['trunk_rotation_invalid_reason'] = (
                REASON_OUT_OF_FRAME if REASON_OUT_OF_FRAME in reasons
                else REASON_NO_POINT)
            return out

        sh_l = (vis[5][0] * w, vis[5][1] * h)
        sh_r = (vis[6][0] * w, vis[6][1] * h)
        hip_l = (vis[11][0] * w, vis[11][1] * h)
        hip_r = (vis[12][0] * w, vis[12][1] * h)
        sh_angle = _line_angle_from_horizontal(sh_l, sh_r, min_len_px=min_len)
        hip_angle = _line_angle_from_horizontal(hip_l, hip_r, min_len_px=min_len)
        if sh_angle is None or hip_angle is None:   # 肩线/髋线太短 → 不可信
            out['trunk_rotation_invalid_reason'] = REASON_GEOM
            return out

        out['trunk_rotation_deg'] = _normalize_line_diff(sh_angle - hip_angle)
        out['trunk_rotation_valid'] = True
        out['trunk_rotation_shoulder_deg'] = sh_angle
        out['trunk_rotation_hip_deg'] = hip_angle
        out['trunk_rotation_conf'] = (
            sum(landmarks[pid][3] for pid in (5, 6, 11, 12)) / 4.0)
        out['trunk_rotation_geom'] = {
            'shoulderL': sh_l, 'shoulderR': sh_r,
            'hipL': hip_l, 'hipR': hip_r,
        }
        return out


class KneeFlexFeatures(_JointFlexFeaturesBase):
    """膝关节屈伸特征：髋-膝-踝在膝处的内角（左右独立，**只算不判、不平滑**）。

        L: 髋 11 → 膝 13 → 踝 15
        R: 髋 12 → 膝 14 → 踝 16
        角度 = 在膝处的内角（度，0~180；180 = 腿伸直，越小 = 屈膝越明显）

    左右各自输出、不合并；某侧缺任一髋/膝/踝点 → 该侧 invalid、读数 N/A，
    另一侧不受影响。下半身必须真的在画面里 —— RTMPose 对画面外的关节会给
    "外推坐标"（不是 None、置信度也可能很高），这些点现在按 `FRAME_EDGE_MARGIN`
    的边界门控判为 `out_of_frame`、读数 N/A，不再出一个看着有效的错数。

    接口约定：
        KneeFlexFeatures(visibility_min=0.3)
        update(pose_result) -> dict
            {'knee_flex_left_deg' / 'knee_flex_right_deg': float | None
             'knee_flex_left_valid' / 'knee_flex_right_valid': bool
             'knee_flex_left_conf' / 'knee_flex_right_conf': float | None
             'knee_flex_left_invalid_reason' / '..._right_...': str | None
             'knee_flex_geom': {'L': {'p1','joint','p2'}, 'R': ...}
        （无 reset：本类不持有跨帧状态）
    """

    PREFIX = 'knee_flex'
    SIDES = (("L", (11, 13, 15)), ("R", (12, 14, 16)))


class HipFlexFeatures(_JointFlexFeaturesBase):
    """髋关节前屈特征：肩-髋-膝在髋处的内角（左右独立，**只算不判、不平滑**）。

        L: 肩 5 → 髋 11 → 膝 13
        R: 肩 6 → 髋 12 → 膝 14
        角度 = 在髋处的内角（度，0~180；180 = 躯干与大腿成一直线，
               越小 = 屈髋/抬腿越明显）

    左右各自输出、不合并；某侧缺任一肩/髋/膝点 → 该侧 invalid、读数 N/A，
    另一侧不受影响。这是 2D 像面内的大腿相对躯干夹角，不含深度，只保证单调。

    接口约定：
        HipFlexFeatures(visibility_min=0.3)
        update(pose_result) -> dict
            {'hip_flex_left_deg' / 'hip_flex_right_deg': float | None
             'hip_flex_left_valid' / 'hip_flex_right_valid': bool
             'hip_flex_left_conf' / 'hip_flex_right_conf': float | None
             'hip_flex_left_invalid_reason' / '..._right_...': str | None
             'hip_flex_geom': {'L': {'p1','joint','p2'}, 'R': ...}
        （无 reset：本类不持有跨帧状态）
    """

    PREFIX = 'hip_flex'
    SIDES = (("L", (5, 11, 13)), ("R", (6, 12, 14)))


class ShoulderAbductionFeatures:
    """肩关节外展特征：肩→肘与竖直轴的夹角，左右独立（**只算不判、不平滑**）。

        L: 肩 5 → 肘 7      R: 肩 6 → 肘 8
        shoulder_abduction_deg = 肩→肘 与 **向下竖直轴** 的夹角
                                 （度，0~180；0 = 手臂垂在体侧，
                                  90 = 平举/水平，180 = 举过头顶）

    === 和举手（ArmRaiseFeatures）差在哪 ===
    两类的点、公式结构相同，差别在**参考方向**：举手用"向上竖直轴"（0 = 上举、
    180 = 下垂），外展用"向下竖直轴"（0 = 垂在体侧、180 = 上举）——外展量的
    是"手臂从体侧往侧面展开多少"，所以 0 在体侧。水平分量放在 atan2 的分子，
    即"上臂离竖直轴的水平偏移"为主量。

    **2D 像面测不到运动平面**：单目画面里"额状面外展"和"矢状面前抬"都表现为
    上臂偏离竖直轴，本读数**分不开两者**，也分不出外展/内收（取 |dx|）。
    要真正区分得靠 3D 姿态或多机位；这里只保证"上臂离体侧竖直轴多远"单调。

    === 门控 ===
    每侧肩、肘必须都可见；缺任一点 → 该侧 invalid、读数 N/A，另一侧照算。
    **不拿对侧点或别的关节替代**。

    === 参考轴 reference（默认 'image'，保持旧行为）===
    - `'image'`：参考画面向下竖直轴 —— 与"垂在体侧 = 0°"一致，但躯干侧倾时会
      把躯干倾斜算进外展读数。
    - `'trunk'`：参考**肩中点→髋中点**的躯干轴（"躯干竖直轴"的原始定义）。
      5/6/11/12 缺任一点、或肩髋中点太近 → **两侧都 invalid**（reason
      `no_trunk` / `geom`），**不静默退回 image** —— 换了参考就是换了量。

    接口约定：
        ShoulderAbductionFeatures(visibility_min=0.3, reference='image')
        update(pose_result) -> dict
            {'shoulder_abduction_left_deg' / 'shoulder_abduction_right_deg':
                 float | None
             'shoulder_abduction_left_valid' / '..._right_valid': bool
             'shoulder_abduction_left_conf' / '..._right_conf': float | None
             'shoulder_abduction_left_invalid_reason' / '..._right_...': str | None
             'shoulder_abduction_ref': 'image' | 'trunk'
             'shoulder_abduction_geom': {'L': {'shoulder': (x_px,y_px),
                                               'elbow': (x_px,y_px)},
                                          'R': ...,
                                          'axis': {'shoulder_mid': ...,
                                                   'hip_mid': ...}}
                 # 每侧只有该侧有效时才出现；axis 只在 trunk 模式且轴可用时出现
        （无 reset：本类不持有跨帧状态）
    """

    _SIDE_KEY = {"L": "left", "R": "right"}
    _SIDES = (("L", (5, 7)), ("R", (6, 8)))
    _VALID_REFERENCES = ('image', 'trunk')

    def __init__(self, visibility_min: float = 0.3,
                 reference: str = 'image') -> None:
        if reference not in self._VALID_REFERENCES:
            raise ValueError(
                f"reference 必须是 {self._VALID_REFERENCES} 之一，"
                f"收到 {reference!r}")
        self.visibility_min = visibility_min
        self.reference = reference

    def update(self, pose_result: Optional[dict]) -> dict:
        out = {}
        for side, _ids in self._SIDES:
            key = self._SIDE_KEY[side]
            out[f'shoulder_abduction_{key}_deg'] = None
            out[f'shoulder_abduction_{key}_valid'] = False
            out[f'shoulder_abduction_{key}_conf'] = None
            out[f'shoulder_abduction_{key}_invalid_reason'] = None
        out['shoulder_abduction_ref'] = self.reference
        out['shoulder_abduction_geom'] = {}

        if pose_result is None:
            self._mark_both(out, REASON_NO_POSE)
            return out
        landmarks = pose_result.get('landmarks')
        size = pose_result.get('image_size')
        if not landmarks or not size or size[0] <= 0 or size[1] <= 0:
            self._mark_both(out, REASON_NO_POSE if not landmarks
                            else REASON_NO_SIZE)
            return out
        w, h = size
        min_len = _min_segment_px(w, h)

        # 参考轴：image 模式 = 画面向下；trunk 模式 = 肩中点→髋中点
        axis = (0.0, 1.0)
        axis_geom = None
        if self.reference == 'trunk':
            need = SHOULDER_IDS + HIP_IDS
            tvis = _visible_points(landmarks, need, self.visibility_min)
            missing = [pid for pid in need if pid not in tvis]
            if missing:
                # 躯干点共用：缺任一点就没有躯干轴，两侧都不能算
                reasons = [_point_reject_reason(landmarks, pid,
                                                self.visibility_min)
                           for pid in missing]
                reason = (REASON_OUT_OF_FRAME
                          if REASON_OUT_OF_FRAME in reasons
                          else REASON_NO_TRUNK)
                self._mark_both(out, reason)
                return out
            sh_mid = _mean_point([tvis[pid] for pid in SHOULDER_IDS])
            hip_mid = _mean_point([tvis[pid] for pid in HIP_IDS])
            sh_mid_px = (sh_mid[0] * w, sh_mid[1] * h)
            hip_mid_px = (hip_mid[0] * w, hip_mid[1] * h)
            axis = (hip_mid_px[0] - sh_mid_px[0],
                    hip_mid_px[1] - sh_mid_px[1])
            if math.hypot(axis[0], axis[1]) < min_len:   # 躯干轴太短
                self._mark_both(out, REASON_GEOM)
                return out
            axis_geom = {'shoulder_mid': sh_mid_px, 'hip_mid': hip_mid_px}

        all_ids = tuple(pid for _side, ids in self._SIDES for pid in ids)
        vis = _visible_points(landmarks, all_ids, self.visibility_min)

        geom = {}
        for side, (sh_id, el_id) in self._SIDES:
            key = self._SIDE_KEY[side]
            sh, el = vis.get(sh_id), vis.get(el_id)
            if sh is None or el is None:
                missing = [pid for pid in (sh_id, el_id) if pid not in vis]
                reasons = [_point_reject_reason(landmarks, pid,
                                                self.visibility_min)
                           for pid in missing]
                out[f'shoulder_abduction_{key}_invalid_reason'] = (
                    REASON_OUT_OF_FRAME if REASON_OUT_OF_FRAME in reasons
                    else REASON_NO_POINT)
                continue
            sh_px = (sh[0] * w, sh[1] * h)
            el_px = (el[0] * w, el[1] * h)
            v = (el_px[0] - sh_px[0], el_px[1] - sh_px[1])
            if math.hypot(v[0], v[1]) < min_len:   # 肩肘重合/太短 → 没有方向
                out[f'shoulder_abduction_{key}_invalid_reason'] = REASON_GEOM
                continue
            if self.reference == 'trunk':
                out[f'shoulder_abduction_{key}_deg'] = _angle_from_axis(v, axis)
            else:
                out[f'shoulder_abduction_{key}_deg'] = _angle_from_down(v)
            out[f'shoulder_abduction_{key}_valid'] = True
            out[f'shoulder_abduction_{key}_conf'] = (
                landmarks[sh_id][3] + landmarks[el_id][3]) / 2.0
            geom[side] = {'shoulder': sh_px, 'elbow': el_px}
        if axis_geom is not None:
            geom['axis'] = axis_geom
        out['shoulder_abduction_geom'] = geom
        return out

    def _mark_both(self, out: dict, reason: str) -> None:
        """两侧一起置无效原因（参考轴不可用 / 整帧无效时用）。"""
        for side, _ids in self._SIDES:
            out[f'shoulder_abduction_{self._SIDE_KEY[side]}'
                f'_invalid_reason'] = reason


class _BilateralLineFeaturesBase:
    """两点连线与水平线的夹角（双侧合并、缺一点即无效）的公共实现。

    骨盆倾斜 / 肩胛对称性共用：两个点必须**都可见**才算 —— 单点构不成连线，
    不做单侧降级（与 NeckLateralTiltFeatures 的门控逻辑一致）。
    角度方向统一成"图像左 → 图像右"（见 `_line_angle_from_horizontal`）；
    连线短于画面对角线的 `MIN_SEGMENT_RATIO` → N/A、reason `geom`。
    附带 `{PREFIX}_higher`（'L'/'R'/None）= 哪一侧更高，由角度符号直接映射，
    只供显示用，不引入新阈值。
    """

    PREFIX = ""
    IDS: Tuple[int, int] = (0, 0)
    GEOM_KEYS: Tuple[str, str] = ("a", "b")

    def __init__(self, visibility_min: float = 0.3) -> None:
        self.visibility_min = visibility_min

    def update(self, pose_result: Optional[dict]) -> dict:
        out = {
            f'{self.PREFIX}_deg': None,
            f'{self.PREFIX}_valid': False,
            f'{self.PREFIX}_conf': None,
            f'{self.PREFIX}_invalid_reason': None,
            f'{self.PREFIX}_higher': None,     # 'L' / 'R'：哪一侧更高（显示用）
            f'{self.PREFIX}_geom': None,
        }
        if pose_result is None:
            out[f'{self.PREFIX}_invalid_reason'] = REASON_NO_POSE
            return out
        landmarks = pose_result.get('landmarks')
        size = pose_result.get('image_size')
        if not landmarks or not size or size[0] <= 0 or size[1] <= 0:
            out[f'{self.PREFIX}_invalid_reason'] = (
                REASON_NO_POSE if not landmarks else REASON_NO_SIZE)
            return out
        w, h = size
        min_len = _min_segment_px(w, h)
        vis = _visible_points(landmarks, self.IDS, self.visibility_min)
        missing = [pid for pid in self.IDS if pid not in vis]
        if missing:       # 缺一个就不做连线
            reasons = [_point_reject_reason(landmarks, pid, self.visibility_min)
                       for pid in missing]
            out[f'{self.PREFIX}_invalid_reason'] = (
                REASON_OUT_OF_FRAME if REASON_OUT_OF_FRAME in reasons
                else REASON_NO_POINT)
            return out
        a_id, b_id = self.IDS
        a = (vis[a_id][0] * w, vis[a_id][1] * h)
        b = (vis[b_id][0] * w, vis[b_id][1] * h)
        deg = _line_angle_from_horizontal(a, b, min_len_px=min_len)
        if deg is None:       # 两点重合 / 太近 → 没有方向
            out[f'{self.PREFIX}_invalid_reason'] = REASON_GEOM
            return out

        out[f'{self.PREFIX}_deg'] = deg
        out[f'{self.PREFIX}_valid'] = True
        # 正 = 画面右侧（b，图像右侧点）更高；负 = 左侧（a）更高；0 = 等高
        out[f'{self.PREFIX}_higher'] = 'R' if deg > 0 else ('L' if deg < 0
                                                            else None)
        out[f'{self.PREFIX}_conf'] = (
            landmarks[a_id][3] + landmarks[b_id][3]) / 2.0
        out[f'{self.PREFIX}_geom'] = {
            self.GEOM_KEYS[0]: a, self.GEOM_KEYS[1]: b}
        return out


class PelvicTiltFeatures(_BilateralLineFeaturesBase):
    """骨盆倾斜特征：双髋连线与水平线的夹角（**只算不判、不平滑**）。

        11 左髋 → 12 右髋
        pelvic_tilt_deg = 双髋连线 与 **水平线** 的有符号夹角
                          （度，-90~90；方向统一成图像左→右，
                           正 = 画面右侧髋更高，负 = 左侧髋更高）

    这是 2D 像面内的骨盆侧倾量，不是真实的三维骨盆倾斜/旋转；45° 前侧机位下
    前后方向被压缩，只保证单调。与 `TrunkRotationFeatures` 的髋线分量是同一个
    几何量，但本类把它当独立动作报告。

    === 门控 ===
    双髋必须都可见；缺一个 → invalid、读数 N/A，**不做单侧降级**。

    接口约定：
        PelvicTiltFeatures(visibility_min=0.3)
        update(pose_result) -> dict
            {'pelvic_tilt_deg': float | None
             'pelvic_tilt_valid': bool
             'pelvic_tilt_conf': float | None
             'pelvic_tilt_invalid_reason': str | None
             'pelvic_tilt_higher': 'L' | 'R' | None  哪侧髋更高（显示用）
             'pelvic_tilt_geom': {'hipL': (x_px,y_px),
                                  'hipR': (x_px,y_px)} | None}
        （无 reset：本类不持有跨帧状态）
    """

    PREFIX = 'pelvic_tilt'
    IDS = (11, 12)
    GEOM_KEYS = ('hipL', 'hipR')


class ScapularSymmetryFeatures(_BilateralLineFeaturesBase):
    """肩胛对称性特征：双肩连线与水平线的夹角（**只算不判、不平滑**）。

        5 左肩 → 6 右肩
        scapular_symmetry_deg = 双肩连线 与 **水平线** 的有符号夹角
                                （度，-90~90；方向统一成图像左→右，
                                 正 = 画面右侧肩更高，负 = 左侧肩更高）

    双肩等高 = 0°（对称）；读数越大说明一侧肩越高（耸肩/肩胛不对称）。
    这是 2D 像面内的肩线倾斜，不是肩胛骨三维对称性的完整评估；肩线倾斜也可能
    来自侧弯/机位倾斜，不能直接等同肩胛问题。

    === 门控 ===
    双肩必须都可见；缺一个 → invalid、读数 N/A，**不做单侧降级**。

    接口约定：
        ScapularSymmetryFeatures(visibility_min=0.3)
        update(pose_result) -> dict
            {'scapular_symmetry_deg': float | None
             'scapular_symmetry_valid': bool
             'scapular_symmetry_conf': float | None
             'scapular_symmetry_invalid_reason': str | None
             'scapular_symmetry_higher': 'L' | 'R' | None  哪侧肩更高（显示用）
             'scapular_symmetry_geom': {'shoulderL': (x_px,y_px),
                                        'shoulderR': (x_px,y_px)} | None}
        （无 reset：本类不持有跨帧状态）
    """

    PREFIX = 'scapular_symmetry'
    IDS = (5, 6)
    GEOM_KEYS = ('shoulderL', 'shoulderR')


# ---------------------------------------------------------------------------
# 自测：纯 stdlib 合成数据（main_action_demo.py --selftest 汇总调用）
# ---------------------------------------------------------------------------

def _assert_close(actual: float, expected: float, eps: float, msg: str) -> None:
    if abs(actual - expected) > eps:
        raise AssertionError(f"{msg}: 期望 {expected:.4f}，实际 {actual:.4f}")


def _make_pose_sized(pts: dict, size: tuple,
                     vis: float = 1.0,
                     vis_map: Optional[dict] = None) -> dict:
    """构造一帧合成 pose：可指定画面尺寸 / 逐点置信度。

    pts:     {关键点编号: (x, y)}，归一化坐标。
    size:    (w, h) 画面尺寸（像素空间角度依赖它）。
    vis:     未在 vis_map 里单独指定的可见点的统一置信度。
    vis_map: {关键点编号: 该点置信度}，覆盖 vis。
    返回结构与 pose_estimation.detect_pose() 一致（17 点，z/visibility 补齐）。
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

    # 2) 平举 (90°) 与下垂 (180°)：方向必须分得开（这是不用取绝对值那个写法的原因）
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
    #    纵横比偏差，本类固定在像素空间算）
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


def selftest_trunk_flex() -> None:
    """合成数据自测 TrunkFlexFeatures：竖直/左右倾符号、单侧中点降级、
    双肩/双髋全丢 N/A、低置信度门控、像素空间纪律。"""
    fe = TrunkFlexFeatures()
    size = (100, 100)

    # 1) 躯干竖直：肩中点 (50,30)、髋中点 (50,70) → 0°
    upright = {5: (0.4, 0.3), 6: (0.6, 0.3),
               11: (0.4, 0.7), 12: (0.6, 0.7)}
    r = fe.update(_make_pose_sized(upright, size))
    _assert_close(r['trunk_flex_deg'], 0.0, 1e-6, "竖直躯干应为 0°")
    assert r['trunk_flex_valid']
    _assert_close(r['trunk_geom']['shoulder_mid'][0], 50.0, 1e-6,
                  "肩中点像素 x")
    _assert_close(r['trunk_geom']['hip_mid'][1], 70.0, 1e-6, "髋中点像素 y")

    # 2) 肩中点偏画面右 → 正；偏左 → 负（atan2(10,40)=14.04°）
    lean_r = {5: (0.5, 0.3), 6: (0.7, 0.3),
              11: (0.4, 0.7), 12: (0.6, 0.7)}
    r = fe.update(_make_pose_sized(lean_r, size))
    _assert_close(r['trunk_flex_deg'], math.degrees(math.atan2(10, 40)),
                  1e-6, "向右倾应为正")
    lean_l = {5: (0.3, 0.3), 6: (0.5, 0.3),
              11: (0.4, 0.7), 12: (0.6, 0.7)}
    r = fe.update(_make_pose_sized(lean_l, size))
    _assert_close(r['trunk_flex_deg'], -math.degrees(math.atan2(10, 40)),
                  1e-6, "向左倾应为负")

    # 3) 单侧肩可见：该肩点就是中点（需求明确允许，不是 N/A）
    r = fe.update(_make_pose_sized(
        {5: (0.6, 0.3), 11: (0.4, 0.7), 12: (0.6, 0.7)}, size))
    assert r['trunk_flex_valid']
    _assert_close(r['trunk_flex_deg'], math.degrees(math.atan2(10, 40)),
                  1e-6, "单肩中点应取该肩点")
    assert r['trunk_flex_pts'] == 'sh5+hip11+12', r['trunk_flex_pts']
    # 单侧髋同理
    r = fe.update(_make_pose_sized(
        {5: (0.4, 0.3), 6: (0.6, 0.3), 11: (0.4, 0.7)}, size))
    assert r['trunk_flex_valid']
    _assert_close(r['trunk_flex_deg'], math.degrees(math.atan2(10, 40)),
                  1e-6, "单髋中点应取该髋点")

    # 4) 双肩都不可见 / 双髋都不可见 → N/A
    assert not fe.update(_make_pose_sized(
        {11: (0.4, 0.7), 12: (0.6, 0.7)}, size))['trunk_flex_valid']
    assert not fe.update(_make_pose_sized(
        {5: (0.4, 0.3), 6: (0.6, 0.3)}, size))['trunk_flex_valid']

    # 5) 低置信度点不参与中点（6 号肩 0.1 → 只剩 5 号肩）
    r = fe.update(_make_pose_sized(
        {5: (0.4, 0.3), 6: (0.6, 0.3),
         11: (0.4, 0.7), 12: (0.6, 0.7)}, size,
        vis_map={6: 0.1}))
    assert r['trunk_flex_valid'] and r['trunk_flex_pts'] == 'sh5+hip11+12'
    _assert_close(r['trunk_flex_deg'], -math.degrees(math.atan2(10, 40)),
                  1e-6, "剔除低置信度肩点后应只剩 5 号肩")

    # 6) 像素空间纪律：同一归一化姿势在 200x100 下角度不同
    r = fe.update(_make_pose_sized(lean_r, (200, 100)))
    _assert_close(r['trunk_flex_deg'], math.degrees(math.atan2(20, 40)),
                  1e-6, "200x100 下应按像素空间算")

    # 7) 无人 / 无 image_size → invalid（不抛异常）
    assert not fe.update(None)['trunk_flex_valid']
    assert not fe.update({'landmarks': [(0.5, 0.3, 0.0, 1.0)] * 17}
                         )['trunk_flex_valid']

    print("  selftest_trunk_flex: OK")


def selftest_neck_lateral() -> None:
    """合成数据自测 NeckLateralTiltFeatures：水平/左右倾符号、缺一耳 N/A
    （不做单耳降级）、低置信度门控、像素空间纪律。"""
    fe = NeckLateralTiltFeatures()
    size = (100, 100)

    # 1) 双耳水平 → 0°
    r = fe.update(_make_pose_sized(
        {3: (0.45, 0.2), 4: (0.55, 0.2)}, size))
    _assert_close(r['neck_lateral_deg'], 0.0, 1e-6, "双耳水平应为 0°")
    assert r['neck_lateral_valid']

    # 2) 画面右侧的耳更高 → 正；左侧更高 → 负
    r = fe.update(_make_pose_sized(
        {3: (0.45, 0.2), 4: (0.55, 0.1)}, size))
    _assert_close(r['neck_lateral_deg'], 45.0, 1e-6, "右耳更高应为正 45°")
    r = fe.update(_make_pose_sized(
        {3: (0.45, 0.1), 4: (0.55, 0.2)}, size))
    _assert_close(r['neck_lateral_deg'], -45.0, 1e-6, "左耳更高应为负 45°")

    # 3) 缺任意一只耳 → N/A（单耳构不成双耳连线，不做单耳降级）
    assert not fe.update(_make_pose_sized(
        {3: (0.45, 0.2)}, size))['neck_lateral_valid']
    assert not fe.update(_make_pose_sized(
        {4: (0.55, 0.2)}, size))['neck_lateral_valid']

    # 4) 低置信度耳按不可见处理 → N/A
    assert not fe.update(_make_pose_sized(
        {3: (0.45, 0.2), 4: (0.55, 0.1)}, size,
        vis_map={4: 0.1}))['neck_lateral_valid']

    # 5) 像素空间纪律：200x100 下 dx=20, dy=-10 → atan2(10,20)
    r = fe.update(_make_pose_sized(
        {3: (0.45, 0.2), 4: (0.55, 0.1)}, (200, 100)))
    _assert_close(r['neck_lateral_deg'],
                  math.degrees(math.atan2(10, 20)), 1e-6,
                  "200x100 下应按像素空间算")

    # 6) 无人 / 无 image_size → invalid
    assert not fe.update(None)['neck_lateral_valid']
    assert not fe.update({'landmarks': [(0.5, 0.3, 0.0, 1.0)] * 17}
                         )['neck_lateral_valid']

    print("  selftest_neck_lateral: OK")


def selftest_elbow_flex() -> None:
    """合成数据自测 ElbowFlexFeatures：伸直/屈肘读数、左右独立、单侧缺点 N/A、
    低置信度门控。"""
    fe = ElbowFlexFeatures()
    size = (100, 100)
    # L 伸直（肘角 180°）；R 屈肘 90°（上臂竖直、前臂水平）
    both = {5: (0.3, 0.3), 7: (0.3, 0.5), 9: (0.3, 0.7),
            6: (0.7, 0.3), 8: (0.7, 0.5), 10: (0.9, 0.5)}
    r = fe.update(_make_pose_sized(both, size))
    _assert_close(r['elbow_flex_left_deg'], 180.0, 1e-6, "左肘伸直应为 180°")
    _assert_close(r['elbow_flex_right_deg'], 90.0, 1e-6, "右肘屈曲应为 90°")
    assert r['elbow_flex_left_valid'] and r['elbow_flex_right_valid']
    assert set(r['elbow_flex_geom']) == {'L', 'R'}

    # 左腕丢失 → 左侧 N/A，右侧照算（左右独立，不合并）
    r = fe.update(_make_pose_sized(
        {5: (0.3, 0.3), 7: (0.3, 0.5),
         6: (0.7, 0.3), 8: (0.7, 0.5), 10: (0.9, 0.5)}, size))
    assert not r['elbow_flex_left_valid'] and r['elbow_flex_left_deg'] is None
    _assert_close(r['elbow_flex_right_deg'], 90.0, 1e-6, "右肘不应受影响")

    # 右肩丢失 → 右侧 N/A，左侧照算
    r = fe.update(_make_pose_sized(
        {5: (0.3, 0.3), 7: (0.3, 0.5), 9: (0.3, 0.7),
         8: (0.7, 0.5), 10: (0.9, 0.5)}, size))
    assert not r['elbow_flex_right_valid']
    _assert_close(r['elbow_flex_left_deg'], 180.0, 1e-6, "左肘不应受影响")

    # 低置信度肘点 → 该侧 N/A
    r = fe.update(_make_pose_sized(both, size, vis_map={7: 0.1}))
    assert not r['elbow_flex_left_valid'] and r['elbow_flex_right_valid']

    # 两侧都缺点 → 两侧都 N/A
    r = fe.update(_make_pose_sized({5: (0.3, 0.3), 6: (0.7, 0.3)}, size))
    assert not r['elbow_flex_left_valid'] and not r['elbow_flex_right_valid']

    # 无人 / 无 image_size → invalid
    assert not fe.update(None)['elbow_flex_left_valid']
    assert not fe.update({'landmarks': [(0.5, 0.3, 0.0, 1.0)] * 17}
                         )['elbow_flex_left_valid']

    print("  selftest_elbow_flex: OK")


def selftest_trunk_rotation() -> None:
    """合成数据自测 TrunkRotationFeatures：肩髋平行/相对倾斜、四点缺一 N/A、
    低置信度门控、几何坐标。"""
    fe = TrunkRotationFeatures()
    size = (100, 100)
    flat = {5: (0.4, 0.3), 6: (0.6, 0.3),
            11: (0.4, 0.7), 12: (0.6, 0.7)}

    # 1) 肩线与髋线都水平 → 0°
    r = fe.update(_make_pose_sized(flat, size))
    _assert_close(r['trunk_rotation_deg'], 0.0, 1e-6, "肩髋平行应为 0°")
    assert r['trunk_rotation_valid']
    _assert_close(r['trunk_rotation_shoulder_deg'], 0.0, 1e-6,
                  "水平肩线角应为 0°")
    _assert_close(r['trunk_rotation_hip_deg'], 0.0, 1e-6,
                  "水平髋线角应为 0°")

    # 2) 肩线右端高一格 → 正 5.71°；髋线反向倾 → 差值加倍
    sh_tilt = {5: (0.4, 0.3), 6: (0.6, 0.28),
               11: (0.4, 0.7), 12: (0.6, 0.7)}
    r = fe.update(_make_pose_sized(sh_tilt, size))
    one = math.degrees(math.atan2(2, 20))
    _assert_close(r['trunk_rotation_deg'], one, 1e-6, "肩线倾斜应为正")
    hip_opposite = {5: (0.4, 0.3), 6: (0.6, 0.28),
                    11: (0.4, 0.7), 12: (0.6, 0.72)}
    r = fe.update(_make_pose_sized(hip_opposite, size))
    _assert_close(r['trunk_rotation_deg'], 2 * one, 1e-6,
                  "肩髋反向倾斜时差值应为两倍")

    # 3) 四个点缺任意一个 → N/A（严格，不做中点降级）
    for missing in (5, 6, 11, 12):
        pts = {k: v for k, v in flat.items() if k != missing}
        assert not fe.update(_make_pose_sized(pts, size))[
            'trunk_rotation_valid'], f"缺 {missing} 号点应 N/A"

    # 4) 低置信度点 → N/A
    assert not fe.update(_make_pose_sized(flat, size, vis_map={12: 0.1}))[
        'trunk_rotation_valid']

    # 5) 几何是像素坐标，四个端点都在
    r = fe.update(_make_pose_sized(flat, size))
    _assert_close(r['trunk_rotation_geom']['shoulderL'][0], 40.0, 1e-6,
                  "左肩像素 x")
    _assert_close(r['trunk_rotation_geom']['hipR'][1], 70.0, 1e-6,
                  "右髋像素 y")

    # 6) 无人 / 无 image_size → invalid
    assert not fe.update(None)['trunk_rotation_valid']
    assert not fe.update({'landmarks': [(0.5, 0.3, 0.0, 1.0)] * 17}
                         )['trunk_rotation_valid']

    print("  selftest_trunk_rotation: OK")


def selftest_knee_flex() -> None:
    """合成数据自测 KneeFlexFeatures：伸直/屈膝读数、左右独立、单侧缺点 N/A。"""
    fe = KneeFlexFeatures()
    size = (100, 100)
    both = {11: (0.3, 0.3), 13: (0.3, 0.6), 15: (0.3, 0.9),
            12: (0.7, 0.3), 14: (0.7, 0.6), 16: (0.9, 0.6)}
    r = fe.update(_make_pose_sized(both, size))
    _assert_close(r['knee_flex_left_deg'], 180.0, 1e-6, "左膝伸直应为 180°")
    _assert_close(r['knee_flex_right_deg'], 90.0, 1e-6, "右膝屈曲应为 90°")
    assert r['knee_flex_left_valid'] and r['knee_flex_right_valid']

    # 右踝丢失 → 右侧 N/A，左侧照算
    r = fe.update(_make_pose_sized(
        {11: (0.3, 0.3), 13: (0.3, 0.6), 15: (0.3, 0.9),
         12: (0.7, 0.3), 14: (0.7, 0.6)}, size))
    assert not r['knee_flex_right_valid'] and r['knee_flex_right_deg'] is None
    _assert_close(r['knee_flex_left_deg'], 180.0, 1e-6, "左膝不应受影响")

    # 低置信度左膝 → 左侧 N/A
    r = fe.update(_make_pose_sized(both, size, vis_map={13: 0.1}))
    assert not r['knee_flex_left_valid'] and r['knee_flex_right_valid']

    # 两侧都缺点 → 两侧 N/A
    r = fe.update(_make_pose_sized({11: (0.3, 0.3), 12: (0.7, 0.3)}, size))
    assert not r['knee_flex_left_valid'] and not r['knee_flex_right_valid']

    assert not fe.update(None)['knee_flex_left_valid']
    assert not fe.update({'landmarks': [(0.5, 0.3, 0.0, 1.0)] * 17}
                         )['knee_flex_left_valid']

    print("  selftest_knee_flex: OK")


def selftest_hip_flex() -> None:
    """合成数据自测 HipFlexFeatures：伸直/屈髋读数、左右独立、单侧缺点 N/A。"""
    fe = HipFlexFeatures()
    size = (100, 100)
    both = {5: (0.3, 0.3), 11: (0.3, 0.6), 13: (0.3, 0.9),
            6: (0.7, 0.3), 12: (0.7, 0.6), 14: (0.9, 0.6)}
    r = fe.update(_make_pose_sized(both, size))
    _assert_close(r['hip_flex_left_deg'], 180.0, 1e-6, "左髋伸直应为 180°")
    _assert_close(r['hip_flex_right_deg'], 90.0, 1e-6, "右髋屈曲应为 90°")
    assert r['hip_flex_left_valid'] and r['hip_flex_right_valid']

    # 右肩丢失 → 右侧 N/A，左侧照算
    r = fe.update(_make_pose_sized(
        {5: (0.3, 0.3), 11: (0.3, 0.6), 13: (0.3, 0.9),
         12: (0.7, 0.6), 14: (0.9, 0.6)}, size))
    assert not r['hip_flex_right_valid'] and r['hip_flex_right_deg'] is None
    _assert_close(r['hip_flex_left_deg'], 180.0, 1e-6, "左髋不应受影响")

    # 低置信度右髋 → 右侧 N/A
    r = fe.update(_make_pose_sized(both, size, vis_map={12: 0.1}))
    assert not r['hip_flex_right_valid'] and r['hip_flex_left_valid']

    # 两侧都缺点 → 两侧 N/A
    r = fe.update(_make_pose_sized({11: (0.3, 0.6), 12: (0.7, 0.6)}, size))
    assert not r['hip_flex_left_valid'] and not r['hip_flex_right_valid']

    assert not fe.update(None)['hip_flex_left_valid']
    assert not fe.update({'landmarks': [(0.5, 0.3, 0.0, 1.0)] * 17}
                         )['hip_flex_left_valid']

    print("  selftest_hip_flex: OK")


def selftest_shoulder_abduction() -> None:
    """合成数据自测 ShoulderAbductionFeatures：体侧/平举/上举、左右独立、
    丢点降级、像素空间纪律。"""
    fe = ShoulderAbductionFeatures()
    size = (100, 100)   # 归一化 ×100 = 像素
    sh = {5: (0.5, 0.3), 6: (0.5, 0.3)}

    # 1) 左肘在肩正下方 = 手臂垂在体侧 → 0°；水平 → 90°；正上方 → 180°
    r = fe.update(_make_pose_sized({**sh, 7: (0.5, 0.5)}, size))
    _assert_close(r['shoulder_abduction_left_deg'], 0.0, 1e-6,
                  "手臂垂在体侧应为 0°")
    r = fe.update(_make_pose_sized({**sh, 7: (0.7, 0.3)}, size))
    _assert_close(r['shoulder_abduction_left_deg'], 90.0, 1e-6, "平举应为 90°")
    r = fe.update(_make_pose_sized({**sh, 7: (0.5, 0.1)}, size))
    _assert_close(r['shoulder_abduction_left_deg'], 180.0, 1e-6,
                  "举过头顶应为 180°")

    # 2) 水平偏移取绝对值：往画面左展开与往右展开读数相同（不分外展/内收）
    r = fe.update(_make_pose_sized({**sh, 7: (0.3, 0.3)}, size))
    _assert_close(r['shoulder_abduction_left_deg'], 90.0, 1e-6,
                  "往另一侧展开也应得 90°（2D 分不出外展/内收）")

    # 3) 左右独立：左平举、右垂在体侧
    r = fe.update(_make_pose_sized(
        {5: (0.5, 0.3), 6: (0.5, 0.3), 7: (0.7, 0.3), 8: (0.5, 0.5)}, size))
    _assert_close(r['shoulder_abduction_left_deg'], 90.0, 1e-6, "左平举 90°")
    _assert_close(r['shoulder_abduction_right_deg'], 0.0, 1e-6, "右垂在体侧 0°")
    assert r['shoulder_abduction_left_valid'] and \
        r['shoulder_abduction_right_valid']

    # 4) 单侧丢点：只作废该侧，另一侧照常（不拿对侧点替代）
    r = fe.update(_make_pose_sized({**sh, 8: (0.5, 0.5)}, size))
    assert not r['shoulder_abduction_left_valid'] and \
        r['shoulder_abduction_left_deg'] is None, "左肘缺失 → 左侧 N/A"
    _assert_close(r['shoulder_abduction_right_deg'], 0.0, 1e-6,
                  "右侧不应受影响")
    r = fe.update(_make_pose_sized({**sh, 7: (0.5, 0.5)}, size, vis_map={5: 0.1}))
    assert r['shoulder_abduction_left_deg'] is None, "低置信度肩点应被门控掉"

    # 5) 像素空间纪律：同一组归一化坐标在宽画面下读数不同
    #    归一化 (0.1, -0.1) 正方形 = 135°；200x100 → 像素 (20,-10)
    r = fe.update(_make_pose_sized({5: (0.5, 0.5), 7: (0.6, 0.4)}, (200, 100)))
    _assert_close(r['shoulder_abduction_left_deg'],
                  math.degrees(math.atan2(20.0, -10.0)), 1e-6,
                  "16:9 下应按像素空间算，不是归一化的 135°")

    # 6) geom 是像素坐标；无效侧不出现
    r = fe.update(_make_pose_sized({**sh, 7: (0.7, 0.3)}, size))
    _assert_close(r['shoulder_abduction_geom']['L']['shoulder'][0], 50.0,
                  1e-6, "肩点像素 x")
    _assert_close(r['shoulder_abduction_geom']['L']['elbow'][0], 70.0,
                  1e-6, "肘点像素 x")
    assert 'R' not in r['shoulder_abduction_geom'], "无效侧不该给 geom"

    # 7) 无人 / 无 image_size → 两侧 invalid
    assert not fe.update(None)['shoulder_abduction_left_valid']
    assert not fe.update({'landmarks': [(0.5, 0.3, 0.0, 1.0)] * 17}
                         )['shoulder_abduction_left_valid']

    print("  selftest_shoulder_abduction: OK")


def selftest_pelvic_tilt() -> None:
    """合成数据自测 PelvicTiltFeatures：水平/左右倾斜符号、缺一髋 N/A、
    像素空间纪律。"""
    fe = PelvicTiltFeatures()
    size = (100, 100)

    # 1) 双髋等高 → 0°；右侧髋更高 → 正；左侧更高 → 负
    r = fe.update(_make_pose_sized({11: (0.4, 0.6), 12: (0.6, 0.6)}, size))
    _assert_close(r['pelvic_tilt_deg'], 0.0, 1e-6, "双髋水平应为 0°")
    r = fe.update(_make_pose_sized({11: (0.4, 0.6), 12: (0.6, 0.5)}, size))
    _assert_close(r['pelvic_tilt_deg'], 26.565051, 1e-4,
                  "画面右侧髋更高应为正 26.57°")
    r = fe.update(_make_pose_sized({11: (0.4, 0.5), 12: (0.6, 0.6)}, size))
    _assert_close(r['pelvic_tilt_deg'], -26.565051, 1e-4,
                  "画面左侧髋更高应为负 26.57°")
    assert r['pelvic_tilt_valid'] and r['pelvic_tilt_conf'] == 1.0

    # 2) 缺一个髋 → 整条 invalid（不做单侧降级）
    r = fe.update(_make_pose_sized({11: (0.4, 0.6)}, size))
    assert not r['pelvic_tilt_valid'] and r['pelvic_tilt_deg'] is None
    assert r['pelvic_tilt_geom'] is None
    r = fe.update(_make_pose_sized(
        {11: (0.4, 0.6), 12: (0.6, 0.6)}, size, vis_map={12: 0.1}))
    assert not r['pelvic_tilt_valid'], "低置信度髋点应被门控掉"

    # 3) 两点重合 → 没有方向，invalid
    r = fe.update(_make_pose_sized({11: (0.5, 0.6), 12: (0.5, 0.6)}, size))
    assert not r['pelvic_tilt_valid']

    # 4) 像素空间：wide 画面下同一归一化倾斜读数不同
    r = fe.update(_make_pose_sized({11: (0.4, 0.6), 12: (0.6, 0.5)}, (200, 100)))
    _assert_close(r['pelvic_tilt_deg'],
                  math.degrees(math.atan2(10.0, 40.0)), 1e-6,
                  "应按像素空间算（14.04°），不是归一化的 26.57°")

    # 5) geom 是像素坐标
    r = fe.update(_make_pose_sized({11: (0.4, 0.6), 12: (0.6, 0.6)}, size))
    _assert_close(r['pelvic_tilt_geom']['hipL'][0], 40.0, 1e-6, "左髋像素 x")
    _assert_close(r['pelvic_tilt_geom']['hipR'][1], 60.0, 1e-6, "右髋像素 y")

    # 6) 无人 / 无 image_size → invalid
    assert not fe.update(None)['pelvic_tilt_valid']
    assert not fe.update({'landmarks': [(0.5, 0.3, 0.0, 1.0)] * 17}
                         )['pelvic_tilt_valid']

    print("  selftest_pelvic_tilt: OK")


def selftest_scapular_symmetry() -> None:
    """合成数据自测 ScapularSymmetryFeatures：水平/左右倾斜符号、缺一肩 N/A、
    像素空间纪律。"""
    fe = ScapularSymmetryFeatures()
    size = (100, 100)

    # 1) 双肩等高 → 0°；右肩更高 → 正；左肩更高 → 负
    r = fe.update(_make_pose_sized({5: (0.4, 0.4), 6: (0.6, 0.4)}, size))
    _assert_close(r['scapular_symmetry_deg'], 0.0, 1e-6, "双肩水平应为 0°")
    r = fe.update(_make_pose_sized({5: (0.4, 0.4), 6: (0.6, 0.3)}, size))
    _assert_close(r['scapular_symmetry_deg'], 26.565051, 1e-4,
                  "画面右侧肩更高应为正 26.57°")
    r = fe.update(_make_pose_sized({5: (0.4, 0.3), 6: (0.6, 0.4)}, size))
    _assert_close(r['scapular_symmetry_deg'], -26.565051, 1e-4,
                  "画面左侧肩更高应为负 26.57°")
    assert r['scapular_symmetry_valid'] and r['scapular_symmetry_conf'] == 1.0

    # 2) 缺一个肩 → 整条 invalid（不做单侧降级）
    r = fe.update(_make_pose_sized({6: (0.6, 0.4)}, size))
    assert not r['scapular_symmetry_valid'] and \
        r['scapular_symmetry_deg'] is None
    assert r['scapular_symmetry_geom'] is None
    r = fe.update(_make_pose_sized(
        {5: (0.4, 0.4), 6: (0.6, 0.4)}, size, vis_map={5: 0.1}))
    assert not r['scapular_symmetry_valid'], "低置信度肩点应被门控掉"

    # 3) 两点重合 → 没有方向，invalid
    r = fe.update(_make_pose_sized({5: (0.5, 0.4), 6: (0.5, 0.4)}, size))
    assert not r['scapular_symmetry_valid']

    # 4) 像素空间：wide 画面下同一归一化倾斜读数不同
    r = fe.update(_make_pose_sized({5: (0.4, 0.4), 6: (0.6, 0.3)}, (200, 100)))
    _assert_close(r['scapular_symmetry_deg'],
                  math.degrees(math.atan2(10.0, 40.0)), 1e-6,
                  "应按像素空间算（14.04°），不是归一化的 26.57°")

    # 5) geom 是像素坐标（供 output 画双肩连线 + 水平参考线）
    r = fe.update(_make_pose_sized({5: (0.4, 0.4), 6: (0.6, 0.4)}, size))
    _assert_close(r['scapular_symmetry_geom']['shoulderL'][0], 40.0, 1e-6,
                  "左肩像素 x")
    _assert_close(r['scapular_symmetry_geom']['shoulderR'][1], 40.0, 1e-6,
                  "右肩像素 y")

    # 6) 无人 / 无 image_size → invalid
    assert not fe.update(None)['scapular_symmetry_valid']
    assert not fe.update({'landmarks': [(0.5, 0.3, 0.0, 1.0)] * 17}
                         )['scapular_symmetry_valid']

    print("  selftest_scapular_symmetry: OK")


def selftest_geometry_gates() -> None:
    """尺度感知门控自测：短线段 → N/A + reason='geom'；画面外 → 'out_of_frame'；
    有效帧 reason=None；骨盆/肩胛 higher 键；肩外展 reference 选项。"""
    size = (100, 100)
    # 100x100 对角线 141.4px → 最小可信线段 = max(2, 1.41) = 2px
    tiny = 0.01        # 归一化 0.01 = 1px（短于门限）
    off = 1.2          # 归一化超界 → RTMPose 外推点

    def check(feat, key, expected, pose, msg):
        r = feat.update(pose)
        got = r.get(key)
        assert got == expected, f"{msg}: 期望 reason={expected!r}，实际 {got!r}"
        return r

    # 1) 仰头：**单耳**参考下耳→鼻 1px → geom（双耳都在且横向退化时先报
    #    degenerate，那是"机位测不了"，不是"点太近"）；鼻在画面外 → out_of_frame
    head = HeadTiltFeatures()
    r = check(head, 'head_pitch_invalid_reason', 'geom',
              _make_pose_sized({3: (0.4, 0.2),
                                0: (0.4, 0.2 + tiny)}, size), "仰头耳鼻过短")
    assert not r['head_pitch_valid']
    r = check(head, 'head_pitch_invalid_reason', 'degenerate',
              _make_pose_sized({3: (0.4, 0.2), 4: (0.6, 0.2),
                                0: (0.5, 0.2 + tiny)}, size),
              "双耳横向退化应报 degenerate（优先于 geom）")
    assert r['head_pitch_degenerate'] is True
    check(head, 'head_pitch_invalid_reason', 'out_of_frame',
          _make_pose_sized({3: (0.4, 0.2), 4: (0.6, 0.2), 0: (0.5, off)}, size),
          "仰头鼻点在画面外")

    # 2) 举手：肩→肘 1px → geom；肘在画面外 → out_of_frame
    arm = ArmRaiseFeatures()
    r = check(arm, 'arm_raise_left_invalid_reason', 'geom',
              _make_pose_sized({5: (0.5, 0.3), 7: (0.5 + tiny, 0.3)}, size),
              "举手肩肘过短")
    assert not r['arm_raise_valid'] and r['arm_raise_left_deg'] is None
    check(arm, 'arm_raise_left_invalid_reason', 'out_of_frame',
          _make_pose_sized({5: (0.5, 0.3), 7: (0.5, off)}, size),
          "举手肘在画面外")

    # 3) 躯干前屈：肩中点→髋中点 1px → geom；双肩都在画面外 → out_of_frame
    trunk = TrunkFlexFeatures()
    r = check(trunk, 'trunk_flex_invalid_reason', 'geom',
              _make_pose_sized({5: (0.4, 0.3), 6: (0.6, 0.3),
                                11: (0.4, 0.3 + tiny),
                                12: (0.6, 0.3 + tiny)}, size), "躯干中点过短")
    assert not r['trunk_flex_valid']
    check(trunk, 'trunk_flex_invalid_reason', 'out_of_frame',
          _make_pose_sized({5: (0.4, off), 6: (0.6, off),
                            11: (0.4, 0.6), 12: (0.6, 0.6)}, size),
          "躯干双肩都在画面外")

    # 4) 颈部侧屈：双耳 1px → geom；一耳在画面外 → out_of_frame
    neck = NeckLateralTiltFeatures()
    check(neck, 'neck_lateral_invalid_reason', 'geom',
          _make_pose_sized({3: (0.5, 0.2), 4: (0.5 + tiny, 0.2)}, size),
          "颈部双耳过短")
    check(neck, 'neck_lateral_invalid_reason', 'out_of_frame',
          _make_pose_sized({3: (0.5, 0.2), 4: (off, 0.2)}, size),
          "颈部耳在画面外")

    # 5) 肘屈伸：短边 → 该侧 geom；另一侧缺点是 no_point；腕在画面外
    elbow = ElbowFlexFeatures()
    r = check(elbow, 'elbow_flex_left_invalid_reason', 'geom',
              _make_pose_sized({5: (0.5, 0.3), 7: (0.5 + tiny, 0.3),
                                9: (0.7, 0.3)}, size), "肘角短边")
    assert r['elbow_flex_right_invalid_reason'] == 'no_point'
    check(elbow, 'elbow_flex_left_invalid_reason', 'out_of_frame',
          _make_pose_sized({5: (0.5, 0.3), 7: (0.6, 0.4),
                            9: (0.7, off)}, size), "肘腕在画面外")

    # 6) 膝屈伸 / 髋前屈：同一套三点门控
    knee = KneeFlexFeatures()
    check(knee, 'knee_flex_left_invalid_reason', 'geom',
          _make_pose_sized({11: (0.5, 0.6), 13: (0.5 + tiny, 0.6),
                            15: (0.5, 0.9)}, size), "膝角短边")
    check(knee, 'knee_flex_left_invalid_reason', 'out_of_frame',
          _make_pose_sized({11: (0.5, 0.6), 13: (0.5, 0.8),
                            15: (0.5, off)}, size), "膝踝在画面外")
    hip = HipFlexFeatures()
    check(hip, 'hip_flex_left_invalid_reason', 'geom',
          _make_pose_sized({5: (0.5, 0.3), 11: (0.5, 0.6),
                            13: (0.5 + tiny, 0.6)}, size), "髋角短边")

    # 7) 躯干旋转：肩线 1px → geom；髋在画面外 → out_of_frame
    rot = TrunkRotationFeatures()
    check(rot, 'trunk_rotation_invalid_reason', 'geom',
          _make_pose_sized({5: (0.5, 0.3), 6: (0.5 + tiny, 0.3),
                            11: (0.4, 0.6), 12: (0.6, 0.6)}, size),
          "旋转肩线过短")
    check(rot, 'trunk_rotation_invalid_reason', 'out_of_frame',
          _make_pose_sized({5: (0.4, 0.3), 6: (0.6, 0.3),
                            11: (0.4, 0.6), 12: (0.6, off)}, size),
          "旋转髋在画面外")

    # 8) 肩外展：肩→肘 1px → geom；trunk 参考缺躯干点 → no_trunk（不退回 image）
    abd = ShoulderAbductionFeatures()
    r = check(abd, 'shoulder_abduction_left_invalid_reason', 'geom',
              _make_pose_sized({5: (0.5, 0.3), 7: (0.5 + tiny, 0.3)}, size),
              "外展肩肘过短")
    assert r['shoulder_abduction_ref'] == 'image'
    abd_trunk = ShoulderAbductionFeatures(reference='trunk')
    r = check(abd_trunk, 'shoulder_abduction_left_invalid_reason', 'no_trunk',
              _make_pose_sized({5: (0.4, 0.3), 6: (0.6, 0.3),
                                7: (0.3, 0.3), 8: (0.7, 0.3)}, size),
              "trunk 参考缺髋点")
    assert r['shoulder_abduction_right_invalid_reason'] == 'no_trunk'
    # trunk 参考可用：肩中点(50,30)→髋中点(50,60) 轴向下；左肩 (40,30)、
    # 左肘 (60,60) → 肩→肘 (20,30)，与轴夹角 atan2(20,30)
    r = abd_trunk.update(_make_pose_sized(
        {5: (0.4, 0.3), 6: (0.6, 0.3), 7: (0.6, 0.6), 8: (0.6, 0.3),
         11: (0.4, 0.6), 12: (0.6, 0.6)}, size))
    _assert_close(r['shoulder_abduction_left_deg'],
                  math.degrees(math.atan2(20.0, 30.0)), 1e-6,
                  "trunk 参考应按肩髋轴算")
    assert r['shoulder_abduction_ref'] == 'trunk'
    assert 'axis' in r['shoulder_abduction_geom']
    try:
        ShoulderAbductionFeatures(reference='bogus')
    except ValueError:
        pass
    else:
        raise AssertionError("非法 reference 应报错")

    # 9) 骨盆倾斜 / 肩胛对称性：1px → geom；画面外；higher 键
    pelvis = PelvicTiltFeatures()
    check(pelvis, 'pelvic_tilt_invalid_reason', 'geom',
          _make_pose_sized({11: (0.5, 0.6), 12: (0.5 + tiny, 0.6)}, size),
          "骨盆双髋过短")
    check(pelvis, 'pelvic_tilt_invalid_reason', 'out_of_frame',
          _make_pose_sized({11: (0.4, 0.6), 12: (off, 0.6)}, size),
          "骨盆髋在画面外")
    r = pelvis.update(_make_pose_sized({11: (0.4, 0.6), 12: (0.6, 0.5)}, size))
    assert r['pelvic_tilt_higher'] == 'R' and \
        r['pelvic_tilt_invalid_reason'] is None
    r = pelvis.update(_make_pose_sized({11: (0.4, 0.5), 12: (0.6, 0.6)}, size))
    assert r['pelvic_tilt_higher'] == 'L'

    scap = ScapularSymmetryFeatures()
    check(scap, 'scapular_symmetry_invalid_reason', 'geom',
          _make_pose_sized({5: (0.5, 0.4), 6: (0.5 + tiny, 0.4)}, size),
          "肩胛双肩过短")
    r = scap.update(_make_pose_sized({5: (0.4, 0.4), 6: (0.6, 0.3)}, size))
    assert r['scapular_symmetry_higher'] == 'R' and \
        r['scapular_symmetry_invalid_reason'] is None
    r = scap.update(_make_pose_sized({5: (0.4, 0.3), 6: (0.6, 0.4)}, size))
    assert r['scapular_symmetry_higher'] == 'L'

    print("  selftest_geometry_gates: OK")


if __name__ == "__main__":
    selftest_arm_raise()
    selftest_head_tilt()
    selftest_trunk_flex()
    selftest_neck_lateral()
    selftest_elbow_flex()
    selftest_trunk_rotation()
    selftest_knee_flex()
    selftest_hip_flex()
    selftest_shoulder_abduction()
    selftest_pelvic_tilt()
    selftest_scapular_symmetry()
    selftest_geometry_gates()
    print("features selftest: ALL PASSED")
