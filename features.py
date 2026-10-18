"""
features 层（全身动作线）：从一帧帧的 pose 结果里提出动作角度。**只算特征，不判状态。**

本仓库有两个动作，各一个特征类（都用 COCO 17 点、都在**像素空间**算）：

    ArmRaiseFeatures(visibility_min=0.3)                   举手角度
    update(pose_result) -> dict  上臂抬升角（肩→肘 vs 向上竖直轴，像素空间，
                0=完全上举/90=平举/180=自然下垂）+ 肘角（肩-肘-腕在肘处的内角，
                180=伸直），左右各一套并给出"举得更高的那一侧"；**只算不判、
                不平滑**（刻意，见类内 docstring）

    HeadTiltFeatures(visibility_min=0.3,
                     min_dx_span_ratio=MIN_PITCH_DX_SPAN_RATIO)  仰头角度
    update(pose_result) -> dict  头俯仰代理：耳中点→鼻尖 与水平线的夹角
                （像素空间，正=鼻在耳上方=仰头）；**只算不判、不平滑**。
                COCO 17 无下巴/头顶，这是 2D 代理、不是真实俯仰角，且只在
                45° 前侧机位有意义（见类内 docstring）
                横向退化门控：|Δx|/双耳间距 低于阈值（正对镜头）→ valid=False，
                不给"看着可信其实没量值"的 ±90°（见 MIN_PITCH_DX_SPAN_RATIO）

两个类共用下面这几个原语：`_visible_points`（取可见点）、`_midpoint`（两点中点）、
`_angle_from_up`（向量与向上竖直轴的夹角）、`_angle_at`（以某点为顶点的夹角），
以及 COCO 点的编号常量。

**为什么在像素空间算**：归一化坐标先把 x 除以 W、y 除以 H，同一个姿势算出的角度
会带上画面宽高比的偏差（16:9 下越接近水平的量偏得越多）。所以两个类都要求
`image_size`，缺失 / 非正就本帧不出读数 —— 宁可不判，也不出一个带纵横比偏差的数。

**v1 刻意不做任何平滑**（别悄悄改）：不 EMA、不中值、不"维持上一有效值"。
关键点丢帧就标 invalid、读数写 N/A，抖动就原样跳出来。本轮的命题是
"**单目关键点撑不撑得住全身动作**"，加平滑 / 维持恰好会把这两个最该被看到的
信号掩盖掉。降级一律是 N/A，**不做 fallback**：缺哪个点就是缺，不拿别的点去替
（换参考点等于换了一个量，混在一起会假装连续）。

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

# 一个点的二维坐标
Point = Tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _visible_points(landmarks: list, ids: Tuple[int, ...],
                    visibility_min: float) -> Dict[int, Point]:
    """从 landmarks 里挑出 ids 中 visibility >= visibility_min 的点。

    本文件的两个动作特征不做窗口 / 中位统计（见模块头"不做任何平滑"），只需要
    "取可见点"这一个原语，独立成函数即可。
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


if __name__ == "__main__":
    selftest_arm_raise()
    selftest_head_tilt()
    print("features selftest: ALL PASSED")
