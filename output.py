"""
output 层（全身动作线）：把 features 的读数与 decision 的状态"呈现"在画面上。

两个出口：
    FrameRenderer.draw_skeleton_frame(frame, pose)   叠骨架（关键点 + 连线 + 标签）
    FrameRenderer.draw_action_overlay(frame, ...)    叠举手 / 仰头两行读数 + 两行状态
                                                     +（可选的）角度几何辅助线

**本层只呈现**：读数是 features 算好的、状态是 decision 判好的、几何坐标是 features
给的像素值 —— 这里既不算特征、不判状态、也不重算几何（分层约束）。缺读数一律写
N/A，**不让整行消失**："这一帧算不出来"本身就是要看的信息。

为什么全部用英文：cv2.putText 自带 Hershey 字体不支持中文，画出来是方块；
统一英文让画面和控制台一致。

画面上四行（y 依次 30 / 56 / 82 / 108，见 _ACTION_Y / _ACTION_LINE_H）：
    Arm raise deg:  L 38.2  R 152.4   Elbow deg:  L 175.1  R --
    Arm State: RAISED  (enter <= 45.0 deg)
    Head pitch deg: 12.4  (ear_mid+nose, conf 0.91, dx/span 0.47)
    Head State: NEUTRAL  (enter >= 20.0 deg)

举手那条线 v1 默认"只报数不判"（main_action_demo.py 的 --arm-rule off）：此时
arm_state 传 None，状态行照样画，但写 `not decided (report only)` —— 不假装判过
（**不写 N/A**：N/A 会被读成"没读到数"，而这里读数是有、只是没做判定）。

仰头那条线的"几何退化"（正对镜头，见 features.MIN_PITCH_DX_SPAN_RATIO）在读数行
尾标 `[geometry degenerate]`：它必须和"关键点丢了"在画面上分得开 —— 一个是机位
问题、一个是检测问题。标记由 features 层给（本层不反推门控逻辑）。

几何辅助线（--no-geometry 关掉）：举手画肩→肘（抬升角的一条边）+ 肘→腕 + 从肩
向上的竖直参考虚线；仰头画耳中点→鼻（+ 过耳中点的水平参考虚线 + 鼻到水平线的
竖直虚线，夹角一眼能对上读数）。那条耳→鼻线同时暴露**转头（yaw）**：头往侧面
转时它会明显偏向侧向，而 features 不补偿 yaw（见其 docstring），画出来才看得见。

本模块只消费 features 的读数 dict + decision 的两个状态枚举 + pose dict，
不 import features 的任何类或函数。
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from decision import (ArmRaiseState, HeadTiltState,
                      TrunkFlexState, NeckLateralState, ElbowFlexState,
                      TrunkRotationState, KneeFlexState, HipFlexState,
                      ShoulderAbductionState, PelvicTiltState,
                      ScapularSymmetryState)

# 骨架：17 个 COCO 关键点之间怎么连线（RTMPose 输出的 COCO 17 点）。
# 只保留躯干 + 四肢连线；头部/面部 0-4 不画（脸只用于角度计算）。
POSE_CONNECTIONS = frozenset([
    (5, 6), (5, 7), (7, 9),         # 肩 + 左臂
    (6, 8), (8, 10),                # 右臂
    (5, 11), (6, 12), (11, 12),     # 躯干
    (11, 13), (13, 15),             # 左腿
    (12, 14), (14, 16),             # 右腿
])

# 动作线（举手 / 仰头）的标签/颜色（draw_action_overlay 用，BGR）。
# 颜色只用"触发/未触发/数据不足"三分：琥珀=触发（要一眼看见）、绿=未触发、
# 灰=N/A。触发不是"错误"（这是个动作识别 demo，不是告警），用琥珀只是让它醒目。
_ACTION_TRIGGER_COLOR = (0, 140, 255)   # 触发（琥珀）
_ACTION_IDLE_COLOR = (60, 200, 60)      # 未触发（绿）
_ACTION_NA_COLOR = (180, 180, 180)      # 数据不足（灰）
_ACTION_STATE_LABEL = {
    ArmRaiseState.RAISED: "RAISED",
    ArmRaiseState.DOWN: "DOWN",
    HeadTiltState.TILTED_BACK: "TILTED_BACK",
    HeadTiltState.NEUTRAL: "NEUTRAL",
    TrunkFlexState.FLEXED: "FLEXED",
    TrunkFlexState.NEUTRAL: "NEUTRAL",
    NeckLateralState.TILTED: "TILTED",
    NeckLateralState.NEUTRAL: "NEUTRAL",
    ElbowFlexState.FLEXED: "FLEXED",
    ElbowFlexState.EXTENDED: "EXTENDED",
    TrunkRotationState.ROTATED: "ROTATED",
    TrunkRotationState.NEUTRAL: "NEUTRAL",
    KneeFlexState.FLEXED: "FLEXED",
    KneeFlexState.EXTENDED: "EXTENDED",
    HipFlexState.FLEXED: "FLEXED",
    HipFlexState.EXTENDED: "EXTENDED",
    ShoulderAbductionState.ABDUCTED: "ABDUCTED",
    ShoulderAbductionState.ADDUCTED: "ADDUCTED",
    PelvicTiltState.TILTED: "TILTED",
    PelvicTiltState.NEUTRAL: "NEUTRAL",
    ScapularSymmetryState.ASYMMETRIC: "ASYMMETRIC",
    ScapularSymmetryState.SYMMETRIC: "SYMMETRIC",
}
_ACTION_STATE_COLOR = {
    ArmRaiseState.RAISED: _ACTION_TRIGGER_COLOR,
    ArmRaiseState.DOWN: _ACTION_IDLE_COLOR,
    HeadTiltState.TILTED_BACK: _ACTION_TRIGGER_COLOR,
    HeadTiltState.NEUTRAL: _ACTION_IDLE_COLOR,
    TrunkFlexState.FLEXED: _ACTION_TRIGGER_COLOR,
    TrunkFlexState.NEUTRAL: _ACTION_IDLE_COLOR,
    NeckLateralState.TILTED: _ACTION_TRIGGER_COLOR,
    NeckLateralState.NEUTRAL: _ACTION_IDLE_COLOR,
    ElbowFlexState.FLEXED: _ACTION_TRIGGER_COLOR,
    ElbowFlexState.EXTENDED: _ACTION_IDLE_COLOR,
    TrunkRotationState.ROTATED: _ACTION_TRIGGER_COLOR,
    TrunkRotationState.NEUTRAL: _ACTION_IDLE_COLOR,
    KneeFlexState.FLEXED: _ACTION_TRIGGER_COLOR,
    KneeFlexState.EXTENDED: _ACTION_IDLE_COLOR,
    HipFlexState.FLEXED: _ACTION_TRIGGER_COLOR,
    HipFlexState.EXTENDED: _ACTION_IDLE_COLOR,
    ShoulderAbductionState.ABDUCTED: _ACTION_TRIGGER_COLOR,
    ShoulderAbductionState.ADDUCTED: _ACTION_IDLE_COLOR,
    PelvicTiltState.TILTED: _ACTION_TRIGGER_COLOR,
    PelvicTiltState.NEUTRAL: _ACTION_IDLE_COLOR,
    ScapularSymmetryState.ASYMMETRIC: _ACTION_TRIGGER_COLOR,
    ScapularSymmetryState.SYMMETRIC: _ACTION_IDLE_COLOR,
}

# 动作显示顺序：入口的 --actions 按这个顺序取子集，画面行序也按它。
# arm/head 是原有两条线；其余按 2026-10-18 两批新增的顺序排在下面。
ACTION_ORDER = ("arm", "head", "trunk", "neck", "elbow", "rotation", "knee", "hip",
                "abduction", "pelvic", "scapular")
# 状态未判时画面上的写法：**不写状态词**，明说"只报数不判"，免得看画的人以为
# 已经有个规则在跑。举手与九个新动作默认就是这一档（--*-rule off）；双摄入口
# 所有动作都走这一档（两台都只报数）。所有块共用同一句话，别各写一份。
_ACTION_NOT_DECIDED = "not decided (report only)"

# features 层给的无效原因短码 → 画面上的一句话（英文短文本，cv2 只能画 ASCII）。
# 缺键 = 老 payload / 原因未知：不追加任何后缀，保持旧画面不变。
_INVALID_REASON_TEXT = {
    'no_pose': 'no pose',
    'no_size': 'no frame size',
    'no_point': 'no point',
    'out_of_frame': 'out of frame',
    'geom': 'geom too small',
    'degenerate': 'geometry degenerate',
    'no_trunk': 'no trunk ref',
}

# 动作叠加的 y 行位（像素）：举手读数 30 / 举手状态 56 / 仰头读数 82 / 仰头状态 108。
_ACTION_Y = 30
_ACTION_LINE_H = 26


def overlay_layout(frame_height: int, n_actions: int,
                   top: int = _ACTION_Y, min_line_h: int = 14) -> tuple:
    """按画面高度/动作数算 (line_h, text_scale)，让每块两行都留在画面内。

    单机入口 720p 全开 8 个动作时 avail//16 = 42 > 26，**取默认行高/字号**
    （画面上和原来一致）；双摄每半幅可能只有 640x360，22 行按 26px 会出画，
    这里按可用高度收紧行高，字号等比缩（不低到 0.35，再小就看不清了）。
    """
    n_rows = 2 * max(1, n_actions)
    avail = max(1, frame_height - top - 6)
    line_h = max(min_line_h, min(_ACTION_LINE_H, avail // n_rows))
    text_scale = 0.50
    if line_h < _ACTION_LINE_H:
        text_scale = max(0.35, 0.50 * line_h / _ACTION_LINE_H)
    return line_h, text_scale


# 动作几何用色（BGR）：举手（肩-肘-腕）蓝、仰头（耳-鼻）黄、参考虚线灰
_ACTION_ARM_GEOM_COLOR = (255, 160, 0)
_ACTION_HEAD_GEOM_COLOR = (0, 220, 255)
_ACTION_REF_COLOR = (110, 110, 110)     # 几何参考虚线（竖直/水平参考轴）

# 新增动作的几何用色（BGR）：每个动作一种，画面上一眼能分清是哪条线
_ACTION_TRUNK_GEOM_COLOR = (255, 80, 80)      # 躯干前屈：肩髋连线
_ACTION_NECK_GEOM_COLOR = (0, 255, 180)       # 颈部侧屈：双耳连线
_ACTION_ELBOW_GEOM_COLOR = (0, 140, 255)      # 肘屈伸：肩肘腕三点线
_ACTION_ROTATION_GEOM_COLOR = (200, 0, 255)   # 躯干旋转：肩线 + 髋线
_ACTION_KNEE_GEOM_COLOR = (0, 200, 0)         # 膝屈伸：髋膝踝三点线
_ACTION_HIP_GEOM_COLOR = (255, 0, 200)        # 髋前屈：肩髋膝三点线
_ACTION_ABDUCTION_GEOM_COLOR = (0, 180, 255)  # 肩外展：肩肘连线 + 参考轴
_ACTION_PELVIC_GEOM_COLOR = (255, 120, 0)     # 骨盆倾斜：双髋连线
_ACTION_SCAPULAR_GEOM_COLOR = (0, 255, 120)   # 肩胛对称：双肩连线


class FrameRenderer:
    """画面输出器（骨架 + 动作读数/状态/几何）。"""

    def __init__(self, draw_skeleton: bool = True) -> None:
        """
        参数:
            draw_skeleton: 是否把姿态识别的骨架（关键点 + 连线 + 标签）叠到画面上，
                         默认 True。想看纯数字/需要省 CPU 时用 --no-skeleton 关闭。
        """
        self.draw_skeleton = draw_skeleton

    # ---------- 全身动作识别（举手 / 仰头）叠加 ----------

    def draw_action_overlay(self, frame: np.ndarray,
                            arm: Optional[dict] = None,
                            arm_state: Optional[ArmRaiseState] = None,
                            head: Optional[dict] = None,
                            head_state: Optional[HeadTiltState] = None,
                            arm_enter_deg: Optional[float] = None,
                            head_enter_deg: Optional[float] = None,
                            show_geometry: bool = True) -> np.ndarray:
        """叠加举手 / 仰头两行读数 + 两行状态 +（可选的）几何辅助线。

        **本层只呈现**：读数是 features 算好的，状态是 decision 判好的，这里
        既不算特征也不判状态（分层约束）。

        参数:
            arm: features.ArmRaiseFeatures.update() 的返回值，含
                 'arm_raise_deg'（举得更高那侧，度）、'arm_raise_side'、
                 'arm_raise_left_deg'/'arm_raise_right_deg'、
                 'elbow_deg'/'elbow_left_deg'/'elbow_right_deg'、
                 'arm_raise_conf'、'arm_geom'（像素坐标
                 {'side','shoulder','elbow','wrist'}）。None 时整个举手块不画。
            arm_state: decision.ArmRaiseState 枚举。**None 有专门的含义**：
                 举手这条线 v1 默认「只报数不判」（--arm-rule off），此时照样画
                 两行读数，状态处写 `not decided (report only)`，不假装判过。
            head: features.HeadTiltFeatures.update() 的返回值，含
                 'head_pitch_deg'、'head_pitch_valid'、'head_pitch_conf'、
                 'head_pitch_pts'、'head_tilt_geom'（{'ear_mid','nose'}，像素）。
                 None 时整个仰头块不画。
            head_state: decision.HeadTiltState 枚举（None = 未判，同 arm_state）。
            arm_enter_deg / head_enter_deg: 两条线当前的触发阈值（度），只在状态行
                 附一句 `enter <= x` / `enter >= x` 供对照 —— 阈值是占位值，
                 画面上写出来才知道现在用的是多少。
            show_geometry: 是否画几何辅助线（举手的肩-肘-腕 + 向上竖直参考线、
                 仰头的耳中点→鼻 + 水平参考线）。几何用 features 给的像素坐标，
                 本层不重算。

        返回: 同一帧（原地绘制，返回只为链式调用方便）。
        """
        if show_geometry:
            if arm is not None:
                self._draw_arm_geometry(frame, arm.get('arm_geom'))
            if head is not None:
                self._draw_head_tilt_geometry(frame, head.get('head_tilt_geom'))

        y = _ACTION_Y
        if arm is not None:
            y = self._draw_arm_block(frame, arm, arm_state, arm_enter_deg, y)
        if head is not None:
            self._draw_head_block(frame, head, head_state, head_enter_deg, y)
        return frame

    # ---------- 多动作统一叠加（原两条线 + 九个新动作） ----------

    def draw_all_action_overlay(self, frame: np.ndarray,
                                actions: list,
                                show_geometry: bool = True,
                                line_h: Optional[int] = None,
                                text_scale: Optional[float] = None) -> np.ndarray:
        """按顺序叠加本次运行的各动作读数/状态/几何（**只呈现，不计算**）。

        参数:
            actions: [(kind, payload), ...]，顺序就是画面的行序。kind 取
                'arm' / 'head' / 'trunk' / 'neck' / 'elbow' / 'rotation' /
                'knee' / 'hip' / 'abduction' / 'pelvic' / 'scapular'；
                payload 是 dict：
                    'features': features.update() 的返回值；None 表示本次不显示
                    'state':    单动作状态（arm/head/trunk/neck/rotation/
                                pelvic/scapular）
                    'states':   左右独立动作的状态 {'L': ..., 'R': ...}
                                （elbow/knee/hip/abduction）
                    'enter_deg': 当前阈值，仅用于状态行后面的对照文字
                **state/states 传 None 的含义与 draw_action_overlay 一致**：
                只报数、不判状态，状态行写 `not decided (report only)`。
            show_geometry: 是否画各动作的几何辅助线（--no-geometry 统一控制）。

        画面行序由调用方决定（入口按 output.ACTION_ORDER 取 --actions 子集）。
        每个动作占两行：单动作 = 读数行 + 状态行；左右独立动作 = 左、右各一行
        （每行同时带该侧的读数和状态，左右各自一套状态机，不合并）。

        本方法不 import / 调用 features 或 decision 的计算逻辑；几何坐标全部来自
        features 返回的 *_geom dict。

        line_h/text_scale: 可选的行高与字号。双摄入口每半幅可能要缩到 640 宽
        （高 ~360），十一个动作 22 行按默认 26px 会出画 —— 调用方按半幅高度收紧，
        本层只负责画。None = 用模块默认值（单机入口不需要传）。
        """
        line_h = _ACTION_LINE_H if line_h is None else int(line_h)
        text_scale = 0.50 if text_scale is None else float(text_scale)
        # 先画所有几何，再画所有文字：后面的几何线不会盖住前面已经画好的读数行
        if show_geometry:
            for kind, payload in actions:
                feats = payload.get('features')
                if feats is not None:
                    self._draw_action_geometry(kind, frame, feats)

        y = _ACTION_Y
        for kind, payload in actions:
            feats = payload.get('features')
            if feats is None:
                continue
            y = self._draw_action_block(kind, frame, feats, payload, y,
                                        line_h, text_scale)
        return frame

    def _draw_action_block(self, kind: str, frame: np.ndarray,
                           feats: dict, payload: dict, y: int,
                           line_h: int = _ACTION_LINE_H,
                           text_scale: float = 0.50) -> int:
        """按 kind 画一个动作的两行，返回下一块的起始 y。"""
        state = payload.get('state')
        enter_deg = payload.get('enter_deg')
        if kind == 'arm':
            return self._draw_arm_block(frame, feats, state, enter_deg, y,
                                        line_h, text_scale)
        if kind == 'head':
            self._draw_head_block(frame, feats, state, enter_deg, y,
                                  line_h, text_scale)
            return y + 2 * line_h
        if kind == 'trunk':
            return self._draw_single_action_block(
                frame, feats, state, enter_deg, y,
                'Trunk flex', 'trunk_flex_deg', '>=', line_h, text_scale)
        if kind == 'neck':
            return self._draw_single_action_block(
                frame, feats, state, enter_deg, y,
                'Neck lateral', 'neck_lateral_deg', '>=', line_h, text_scale)
        if kind == 'rotation':
            return self._draw_single_action_block(
                frame, feats, state, enter_deg, y,
                'Trunk rotation', 'trunk_rotation_deg', '>=', line_h,
                text_scale)
        if kind == 'elbow':
            return self._draw_bilateral_action_block(
                frame, feats, payload.get('states') or {}, y,
                'Elbow flex', 'elbow_flex', line_h, text_scale)
        if kind == 'knee':
            return self._draw_bilateral_action_block(
                frame, feats, payload.get('states') or {}, y,
                'Knee flex', 'knee_flex', line_h, text_scale)
        if kind == 'hip':
            return self._draw_bilateral_action_block(
                frame, feats, payload.get('states') or {}, y,
                'Hip flex', 'hip_flex', line_h, text_scale)
        if kind == 'abduction':
            return self._draw_bilateral_action_block(
                frame, feats, payload.get('states') or {}, y,
                'Shoulder abduction', 'shoulder_abduction', line_h, text_scale)
        if kind == 'pelvic':
            return self._draw_single_action_block(
                frame, feats, state, enter_deg, y,
                'Pelvic tilt', 'pelvic_tilt_deg', '>=', line_h, text_scale)
        if kind == 'scapular':
            return self._draw_single_action_block(
                frame, feats, state, enter_deg, y,
                'Scapular sym', 'scapular_symmetry_deg', '>=', line_h,
                text_scale)
        raise ValueError(f"未知动作 kind: {kind!r}")

    @staticmethod
    def _state_color(state) -> tuple:
        """状态对应的显示色；None（未判）与 UNKNOWN 都用灰色。"""
        if state is None:
            return _ACTION_NA_COLOR
        return _ACTION_STATE_COLOR.get(state, _ACTION_NA_COLOR)

    def _draw_single_action_block(self, frame: np.ndarray, feats: dict,
                                  state, enter_deg: Optional[float], y: int,
                                  label: str, deg_key: str,
                                  enter_op: str,
                                  line_h: int = _ACTION_LINE_H,
                                  text_scale: float = 0.50) -> int:
        """单动作两行：读数行 + 状态行（未判时写 report only）。"""
        color = self._state_color(state)
        deg = feats.get(deg_key)
        line1 = f"{label} deg: {self._deg_text(deg)}"
        # 读数 N/A 时把 features 给的原因写出来（缺键 = 老 payload，不追加）
        if deg is None:
            stem = deg_key[:-len('_deg')] if deg_key.endswith('_deg') else deg_key
            line1 += self._reason_suffix(feats.get(f'{stem}_invalid_reason'))
        # 骨盆/肩胛：哪一侧更高（键缺失或 None 时不加；不引入新阈值）
        higher = feats.get(f'{deg_key[:-len("_deg")]}_higher') \
            if deg_key.endswith('_deg') else None
        if higher:
            line1 += f"  ({higher} higher)"
        if state is None:
            line2 = f"{label} State: {_ACTION_NOT_DECIDED}"
        else:
            line2 = f"{label} State: {_ACTION_STATE_LABEL.get(state, 'N/A')}"
            if enter_deg is not None:
                line2 += f"  (enter {enter_op} {enter_deg:.1f} deg)"
        _put_text_outlined(frame, line1, (12, y), text_scale, color)
        _put_text_outlined(frame, line2, (12, y + line_h), text_scale, color)
        return y + 2 * line_h

    def _draw_bilateral_action_block(self, frame: np.ndarray, feats: dict,
                                     states: dict, y: int,
                                     label: str, prefix: str,
                                     line_h: int = _ACTION_LINE_H,
                                     text_scale: float = 0.50) -> int:
        """左右独立动作：左、右各一行（每行带该侧读数与该侧状态）。"""
        for side, key in (("L", "left"), ("R", "right")):
            state = states.get(side)
            color = self._state_color(state)
            deg = feats.get(f'{prefix}_{key}_deg')
            state_text = (_ACTION_NOT_DECIDED if state is None
                          else _ACTION_STATE_LABEL.get(state, 'N/A'))
            line = (f"{label} {side}: {self._deg_text(deg)} deg")
            if deg is None:
                line += self._reason_suffix(
                    feats.get(f'{prefix}_{key}_invalid_reason'))
            line += f"  State: {state_text}"
            _put_text_outlined(frame, line, (12, y), text_scale, color)
            y += line_h
        return y

    @staticmethod
    def _deg_text(v: Optional[float]) -> str:
        """角度读数统一格式：算不出写 N/A（不写 0，0 是个有效读数）。

        `-0.0`（合成/对称构型下 atan2 会给出 -0.0）归一成 `0.0`：读数是 0 时
        画面上出现「-0.0」会被当成 bug，而它确实就是 0。
        """
        if v is None:
            return "N/A"
        return f"{0.0 if v == 0 else v:.1f}"

    @staticmethod
    def _reason_suffix(reason: Optional[str]) -> str:
        """把 features 的无效原因短码变成画面后缀；未知/缺失 → 空串。"""
        if not reason:
            return ""
        text = _INVALID_REASON_TEXT.get(reason)
        return f"  [{text}]" if text else ""

    def _draw_arm_block(self, frame: np.ndarray, arm: dict,
                        arm_state: Optional[ArmRaiseState],
                        enter_deg: Optional[float], y: int,
                        line_h: int = _ACTION_LINE_H,
                        text_scale: float = 0.50) -> int:
        """画举手两行（读数行 + 状态行），返回下一块的起始 y。

        line_h/text_scale 只在多动作叠加时由调用方按画面高度收紧（双摄半幅）；
        默认值就是原来的行高/字号，`draw_action_overlay` 的行为不变。
        """
        color = _ACTION_STATE_COLOR.get(arm_state, _ACTION_NA_COLOR) \
            if arm_state is not None else _ACTION_NA_COLOR
        side = arm.get('arm_raise_side')
        side_s = side if side is not None else "--"

        def side_text(deg_key: str, reason_key: str) -> str:
            """一侧读数；N/A 时带上该侧原因（两侧原因可能不同，分开写）。"""
            txt = self._deg_text(arm.get(deg_key))
            if arm.get(deg_key) is None:
                txt += self._reason_suffix(arm.get(reason_key))
            return txt

        line1 = (f"Arm raise deg:  {side_s} {self._deg_text(arm.get('arm_raise_deg'))}"
                 f"  (L {side_text('arm_raise_left_deg', 'arm_raise_left_invalid_reason')}"
                 f"  R {side_text('arm_raise_right_deg', 'arm_raise_right_invalid_reason')})"
                 f"   Elbow deg:  L {self._deg_text(arm.get('elbow_left_deg'))}"
                 f"  R {self._deg_text(arm.get('elbow_right_deg'))}")
        if arm.get('arm_raise_deg') is None:
            # 合并读数 N/A：把两侧原因并成一段（原因相同只写一次）
            l_reason = arm.get('arm_raise_left_invalid_reason')
            r_reason = arm.get('arm_raise_right_invalid_reason')
            l_text = _INVALID_REASON_TEXT.get(l_reason) if l_reason else None
            r_text = _INVALID_REASON_TEXT.get(r_reason) if r_reason else None
            if l_text and r_text:
                line1 += (f"  [{l_text}]" if l_text == r_text
                          else f"  [L {l_text} / R {r_text}]")
            elif l_text or r_text:
                line1 += f"  [{l_text or r_text}]"
        if arm_state is None:
            # v1 默认：只报数不判（用户要先看数据再定触发用哪个角）
            line2 = (f"Arm State: {_ACTION_NOT_DECIDED}")
        else:
            line2 = f"Arm State: {_ACTION_STATE_LABEL.get(arm_state, 'N/A')}"
            if enter_deg is not None:
                line2 += f"  (enter <= {enter_deg:.1f} deg)"
        _put_text_outlined(frame, line1, (12, y), text_scale, color)
        _put_text_outlined(frame, line2, (12, y + line_h), text_scale, color)
        return y + 2 * line_h

    def _draw_head_block(self, frame: np.ndarray, head: dict,
                         head_state: Optional[HeadTiltState],
                         enter_deg: Optional[float], y: int,
                         line_h: int = _ACTION_LINE_H,
                         text_scale: float = 0.50) -> None:
        """画仰头两行（读数行 + 状态行）。"""
        color = _ACTION_STATE_COLOR.get(head_state, _ACTION_NA_COLOR) \
            if head_state is not None else _ACTION_NA_COLOR
        conf = head.get('head_pitch_conf')
        pts = head.get('head_pitch_pts')
        ratio = head.get('head_pitch_dx_ratio')
        line1 = f"Head pitch deg: {self._deg_text(head.get('head_pitch_deg'))}"
        bits = []
        if pts:
            bits.append(str(pts))
        if conf is not None:
            bits.append(f"conf {conf:.2f}")
        if ratio is not None:
            bits.append(f"dx/span {ratio:.2f}")
        if bits:
            line1 += f"  ({', '.join(bits)})"
        # 几何退化（正对镜头）和"关键点丢了"在画面上必须能分开：前者是这套
        # 几何在这个机位没有量值，后者是没读到点 —— 一个是机位问题、一个是检测问题。
        # 标记由 features 层给（本层不反推门控逻辑，见 features 的接口约定）
        if head.get('head_pitch_degenerate'):
            line1 += "  [geometry degenerate]"
        elif head.get('head_pitch_deg') is None:
            # 其它 N/A 原因（丢点 / 画面外 / 线段太短）；degenerate 已单独写出，
            # 这里不重复。缺键（老 payload）→ 不追加。
            line1 += self._reason_suffix(head.get('head_pitch_invalid_reason'))
        if head_state is None:
            line2 = f"Head State: {_ACTION_NOT_DECIDED}"
        else:
            line2 = f"Head State: {_ACTION_STATE_LABEL.get(head_state, 'N/A')}"
            if enter_deg is not None:
                line2 += f"  (enter >= {enter_deg:.1f} deg)"
        _put_text_outlined(frame, line1, (12, y), text_scale, color)
        _put_text_outlined(frame, line2, (12, y + line_h), text_scale, color)

    def _draw_arm_geometry(self, frame: np.ndarray,
                           geom: Optional[dict]) -> None:
        """画举手几何：肩→肘（抬升角的两条边之一）+ 肘→腕 + 向上竖直参考虚线。

        geom 是 features 给的**像素坐标**（本层只画，不重算）。
        抬升角 = 肩→肘 与"向上竖直轴"的夹角，所以补一条从肩点向上的竖直虚线，
        画面上的夹角一眼能对上读数。腕缺失（None）只画到肘 —— 丢点要在画面上
        看得出来，不能悄悄画成有腕。
        """
        if not geom or not geom.get('shoulder') or not geom.get('elbow'):
            return
        sh = (int(geom['shoulder'][0]), int(geom['shoulder'][1]))
        el = (int(geom['elbow'][0]), int(geom['elbow'][1]))
        cv2.line(frame, sh, el, _ACTION_ARM_GEOM_COLOR, 2, cv2.LINE_AA)
        cv2.circle(frame, sh, 5, _ACTION_ARM_GEOM_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, el, 5, _ACTION_ARM_GEOM_COLOR, -1, cv2.LINE_AA)
        # 向上竖直参考虚线：从肩点向上，长度取肩→肘距离（画多长都行，夹角一样）
        up_len = max(20, int(math.hypot(el[0] - sh[0], el[1] - sh[1])))
        self._dashed_line(frame, sh, (sh[0], sh[1] - up_len),
                          _ACTION_REF_COLOR)
        if geom.get('wrist'):   # 腕可能不可见（None）—— 缺就不画，别补一个假的
            wr = (int(geom['wrist'][0]), int(geom['wrist'][1]))
            cv2.line(frame, el, wr, _ACTION_ARM_GEOM_COLOR, 1, cv2.LINE_AA)
            cv2.circle(frame, wr, 5, _ACTION_ARM_GEOM_COLOR, -1, cv2.LINE_AA)
        side = geom.get('side', '')
        _put_text_outlined(frame, f"arm {side}", (sh[0] + 8, sh[1] + 18), 0.40,
                           _ACTION_ARM_GEOM_COLOR)

    def _draw_head_tilt_geometry(self, frame: np.ndarray,
                                 geom: Optional[dict]) -> None:
        """画仰头几何：耳中点→鼻 连线 + 过耳中点的水平参考虚线 + 竖直投影虚线。

        geom 是 features 给的**像素坐标**（本层只画，不重算）。读数是这条连线与
        水平线的夹角，所以补水平参考线与"鼻到水平线"的竖直虚线，夹角一眼可对。
        这条线同时暴露**转头（yaw）**：头往侧面转时它会明显偏向侧向，而不是
        只反映仰俯 —— features 层不补偿 yaw，画出来才看得见（见其 docstring）。
        """
        if not geom or not geom.get('ear_mid') or not geom.get('nose'):
            return
        ex, ey = int(geom['ear_mid'][0]), int(geom['ear_mid'][1])
        nx, ny = int(geom['nose'][0]), int(geom['nose'][1])
        cv2.line(frame, (ex, ey), (nx, ny), _ACTION_HEAD_GEOM_COLOR, 2, cv2.LINE_AA)
        span = max(abs(nx - ex), 12)
        x_end = ex + int(span * 1.25) * (1 if nx >= ex else -1)
        self._dashed_line(frame, (ex, ey), (x_end, ey), _ACTION_REF_COLOR)
        self._dashed_line(frame, (nx, ny), (nx, ey), _ACTION_REF_COLOR)
        cv2.circle(frame, (ex, ey), 5, _ACTION_HEAD_GEOM_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, (nx, ny), 5, _ACTION_HEAD_GEOM_COLOR, -1, cv2.LINE_AA)
        _put_text_outlined(frame, "ear_mid", (ex + 8, ey + 18), 0.40,
                           _ACTION_HEAD_GEOM_COLOR)
        _put_text_outlined(frame, "nose", (nx + 8, ny - 6), 0.40,
                           _ACTION_HEAD_GEOM_COLOR)

    # ---------- 九个新动作的几何辅助线 ----------

    def _draw_action_geometry(self, kind: str, frame: np.ndarray,
                              feats: dict) -> None:
        """按 kind 分派几何绘制。坐标全部来自 features 的 *_geom（本层只画）。"""
        if kind == 'arm':
            self._draw_arm_geometry(frame, feats.get('arm_geom'))
        elif kind == 'head':
            self._draw_head_tilt_geometry(frame, feats.get('head_tilt_geom'))
        elif kind == 'trunk':
            self._draw_trunk_flex_geometry(frame, feats.get('trunk_geom'))
        elif kind == 'neck':
            self._draw_neck_lateral_geometry(frame, feats.get('neck_geom'))
        elif kind == 'rotation':
            self._draw_trunk_rotation_geometry(
                frame, feats.get('trunk_rotation_geom'))
        elif kind == 'elbow':
            self._draw_joint_chain_geometry(
                frame, feats.get('elbow_flex_geom'),
                _ACTION_ELBOW_GEOM_COLOR, 'elbow')
        elif kind == 'knee':
            self._draw_joint_chain_geometry(
                frame, feats.get('knee_flex_geom'),
                _ACTION_KNEE_GEOM_COLOR, 'knee')
        elif kind == 'hip':
            self._draw_joint_chain_geometry(
                frame, feats.get('hip_flex_geom'),
                _ACTION_HIP_GEOM_COLOR, 'hip')
        elif kind == 'abduction':
            self._draw_abduction_geometry(
                frame, feats.get('shoulder_abduction_geom'))
        elif kind == 'pelvic':
            self._draw_pair_line_geometry(
                frame, feats.get('pelvic_tilt_geom'),
                ('hipL', 'hipR'), _ACTION_PELVIC_GEOM_COLOR, 'pelvis')
        elif kind == 'scapular':
            self._draw_pair_line_geometry(
                frame, feats.get('scapular_symmetry_geom'),
                ('shoulderL', 'shoulderR'), _ACTION_SCAPULAR_GEOM_COLOR,
                'shoulders')
        else:
            raise ValueError(f"未知动作 kind: {kind!r}")

    def _draw_trunk_flex_geometry(self, frame: np.ndarray,
                                  geom: Optional[dict]) -> None:
        """画躯干前屈几何：髋中点→肩中点连线 + 过髋中点的向上竖直参考虚线。

        读数是这条连线与竖直轴的夹角，所以竖直参考线从**髋中点**（角的顶点）
        向上画，画面上的夹角一眼能对上读数。
        """
        if not geom or not geom.get('shoulder_mid') or not geom.get('hip_mid'):
            return
        hip = (int(geom['hip_mid'][0]), int(geom['hip_mid'][1]))
        sh = (int(geom['shoulder_mid'][0]), int(geom['shoulder_mid'][1]))
        cv2.line(frame, hip, sh, _ACTION_TRUNK_GEOM_COLOR, 2, cv2.LINE_AA)
        up_len = max(20, int(math.hypot(sh[0] - hip[0], sh[1] - hip[1])))
        self._dashed_line(frame, hip, (hip[0], hip[1] - up_len),
                          _ACTION_REF_COLOR)
        cv2.circle(frame, hip, 5, _ACTION_TRUNK_GEOM_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, sh, 5, _ACTION_TRUNK_GEOM_COLOR, -1, cv2.LINE_AA)
        _put_text_outlined(frame, "trunk", (sh[0] + 8, sh[1] + 18), 0.40,
                           _ACTION_TRUNK_GEOM_COLOR)

    def _draw_neck_lateral_geometry(self, frame: np.ndarray,
                                    geom: Optional[dict]) -> None:
        """画颈部侧屈几何：双耳连线 + 过一只耳的水平参考虚线。

        features 只给"两只耳都可见"时的 geom（单耳不做降级），所以这里拿到的
        一定是两耳；缺一个时整块不画，与读数 N/A 同步。
        """
        if not geom or not geom.get('earL') or not geom.get('earR'):
            return
        el = (int(geom['earL'][0]), int(geom['earL'][1]))
        er = (int(geom['earR'][0]), int(geom['earR'][1]))
        cv2.line(frame, el, er, _ACTION_NECK_GEOM_COLOR, 2, cv2.LINE_AA)
        self._dashed_line(frame, el, (er[0], el[1]), _ACTION_REF_COLOR)
        cv2.circle(frame, el, 5, _ACTION_NECK_GEOM_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, er, 5, _ACTION_NECK_GEOM_COLOR, -1, cv2.LINE_AA)
        _put_text_outlined(frame, "ears", (el[0] - 46, el[1] - 8), 0.40,
                           _ACTION_NECK_GEOM_COLOR)

    def _draw_trunk_rotation_geometry(self, frame: np.ndarray,
                                      geom: Optional[dict]) -> None:
        """画躯干旋转几何：双肩连线 + 双髋连线（+ 各自过左端的水平参考虚线）。

        读数是两条连线方向之差，两条线都画出来才看得出"肩转没转、髋跟没跟"。
        四点缺任一 → 整块不画（features 已经按这条件置无效）。
        """
        if not geom:
            return
        keys = ('shoulderL', 'shoulderR', 'hipL', 'hipR')
        if any(not geom.get(k) for k in keys):
            return
        sl = (int(geom['shoulderL'][0]), int(geom['shoulderL'][1]))
        sr = (int(geom['shoulderR'][0]), int(geom['shoulderR'][1]))
        hl = (int(geom['hipL'][0]), int(geom['hipL'][1]))
        hr = (int(geom['hipR'][0]), int(geom['hipR'][1]))
        for p, q in ((sl, sr), (hl, hr)):
            cv2.line(frame, p, q, _ACTION_ROTATION_GEOM_COLOR, 2, cv2.LINE_AA)
            self._dashed_line(frame, p, (q[0], p[1]), _ACTION_REF_COLOR)
            cv2.circle(frame, p, 5, _ACTION_ROTATION_GEOM_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, q, 5, _ACTION_ROTATION_GEOM_COLOR, -1, cv2.LINE_AA)
        _put_text_outlined(frame, "shoulder", (sr[0] + 8, sr[1] - 6), 0.40,
                           _ACTION_ROTATION_GEOM_COLOR)
        _put_text_outlined(frame, "hip", (hr[0] + 8, hr[1] - 6), 0.40,
                           _ACTION_ROTATION_GEOM_COLOR)

    def _draw_joint_chain_geometry(self, frame: np.ndarray,
                                   geom: Optional[dict],
                                   color: tuple, label: str) -> None:
        """画三点关节链（肩-肘-腕 / 髋-膝-踝 / 肩-髋-膝），左右各一条。

        geom: {'L': {'p1','joint','p2'}, 'R': {...}}（像素坐标）。
        某侧缺任一关键点时该侧没有 geom（features 已置无效），这里就不画 ——
        与读数 N/A 同步，不拿别的点补。
        """
        if not geom:
            return
        for side in ('L', 'R'):
            chain = geom.get(side)
            if not chain or not all(chain.get(k) for k in ('p1', 'joint', 'p2')):
                continue
            p1 = (int(chain['p1'][0]), int(chain['p1'][1]))
            joint = (int(chain['joint'][0]), int(chain['joint'][1]))
            p2 = (int(chain['p2'][0]), int(chain['p2'][1]))
            cv2.line(frame, p1, joint, color, 2, cv2.LINE_AA)
            cv2.line(frame, joint, p2, color, 2, cv2.LINE_AA)
            for p in (p1, joint, p2):
                cv2.circle(frame, p, 5, color, -1, cv2.LINE_AA)
            _put_text_outlined(frame, f"{label} {side}",
                               (joint[0] + 8, joint[1] + 18), 0.40, color)

    def _draw_abduction_geometry(self, frame: np.ndarray,
                                 geom: Optional[dict]) -> None:
        """画肩外展几何：肩→肘连线 + 参考轴虚线。

        image 参考（默认）：参考轴 = 过肩点的向下竖直轴（0 = 手臂垂在体侧、
        90 = 平举），另补一条水平参考线显示"水平分量展开到哪"。
        trunk 参考：参考轴 = 肩中点→髋中点的躯干轴（geom['axis']），只画这一根
        并标 `ref trunk` —— 画的必须是**读数真正用的那根轴**。
        某侧缺肩或肘点时该侧没有 geom（features 已置无效），这里就不画 ——
        与读数 N/A 同步，不拿别的点补。
        """
        if not geom:
            return
        for side in ('L', 'R'):
            side_geom = geom.get(side)
            if not side_geom or not side_geom.get('shoulder') \
                    or not side_geom.get('elbow'):
                continue
            sh = (int(side_geom['shoulder'][0]), int(side_geom['shoulder'][1]))
            el = (int(side_geom['elbow'][0]), int(side_geom['elbow'][1]))
            arm_len = max(20, int(math.hypot(el[0] - sh[0], el[1] - sh[1])))
            cv2.line(frame, sh, el, _ACTION_ABDUCTION_GEOM_COLOR, 2, cv2.LINE_AA)
            axis = geom.get('axis')
            if axis and axis.get('shoulder_mid') and axis.get('hip_mid'):
                # trunk 参考：沿肩中点→髋中点的躯干轴方向画虚线（真正用的那根轴）
                ax = axis['hip_mid'][0] - axis['shoulder_mid'][0]
                ay = axis['hip_mid'][1] - axis['shoulder_mid'][1]
                norm = math.hypot(ax, ay)
                if norm > 1e-9:
                    self._dashed_line(
                        frame, sh,
                        (int(sh[0] + arm_len * ax / norm),
                         int(sh[1] + arm_len * ay / norm)),
                        _ACTION_REF_COLOR)
                    _put_text_outlined(frame, "ref trunk",
                                       (sh[0] + 8, sh[1] - 8), 0.38,
                                       _ACTION_REF_COLOR)
            else:
                # 向下竖直轴 = 手臂垂在体侧时的方向（外展 0° 的参考）
                self._dashed_line(frame, sh, (sh[0], sh[1] + arm_len),
                                  _ACTION_REF_COLOR)
                # 水平参考线：朝肘所在的那一侧画
                direction = 1 if el[0] >= sh[0] else -1
                self._dashed_line(frame, sh,
                                  (sh[0] + direction * arm_len, sh[1]),
                                  _ACTION_REF_COLOR)
            cv2.circle(frame, sh, 5, _ACTION_ABDUCTION_GEOM_COLOR, -1,
                       cv2.LINE_AA)
            cv2.circle(frame, el, 5, _ACTION_ABDUCTION_GEOM_COLOR, -1,
                       cv2.LINE_AA)
            _put_text_outlined(frame, f"abduction {side}",
                               (el[0] + 8, el[1] + 18), 0.40,
                               _ACTION_ABDUCTION_GEOM_COLOR)

    def _draw_pair_line_geometry(self, frame: np.ndarray,
                                 geom: Optional[dict],
                                 keys: tuple, color: tuple,
                                 label: str) -> None:
        """画"两点连线 + 水平参考线"（骨盆倾斜 / 肩胛对称性）。

        geom 是 features 给的像素坐标，keys 是两端点的键名。读数是这条连线与
        水平线的夹角，所以补一条过左侧点的水平虚线，夹角一眼能对上读数。
        两点缺任一 → 整块不画（features 已经按这条件置无效）。
        """
        if not geom or any(not geom.get(k) for k in keys):
            return
        a = (int(geom[keys[0]][0]), int(geom[keys[0]][1]))
        b = (int(geom[keys[1]][0]), int(geom[keys[1]][1]))
        cv2.line(frame, a, b, color, 2, cv2.LINE_AA)
        self._dashed_line(frame, a, (b[0], a[1]), _ACTION_REF_COLOR)
        cv2.circle(frame, a, 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, b, 5, color, -1, cv2.LINE_AA)
        _put_text_outlined(frame, label, (a[0] + 8, a[1] + 18), 0.40, color)

    # ---------- 数据源标签 ----------

    def draw_stream_label(self, frame: np.ndarray, text: str,
                          org: tuple = (12, 24),
                          color: tuple = (255, 255, 255)) -> None:
        """在画面上画一个数据源标签（如 `cam A`），带描边。

        给多路画面用：哪半幅是哪个摄像头必须一眼看出来，否则两边的读数分不清
        谁是谁。位置由调用方给（org = 文字左下角的像素坐标）—— 本层不猜
        别的东西画在哪，也不去动 draw_action_overlay 那四行的固定行位。
        """
        _put_text_outlined(frame, text, org, 0.7, color, 2)

    # ---------- 骨架叠加 ----------

    def draw_skeleton_frame(self, frame: np.ndarray, pose: Optional[dict]) -> None:
        """只叠骨架（关键点 + 连线 + pid/置信度标签），不画读数/状态/几何。

        不想走 draw_action_overlay（比如只想看关键点对不对）时用这个入口。
        受 draw_skeleton 开关控制（--no-skeleton 时什么都不画）。
        """
        if self.draw_skeleton and pose is not None:
            self._draw_skeleton(frame, pose)

    def _draw_skeleton(self, frame: np.ndarray, pose: dict) -> None:
        """把姿态识别的骨架（关键点 + 连接线 + pid/置信度标签）叠到画面上。

        pose: {'landmarks': [(x,y,z,vis) x17], 'image_size': (w,h)}
        画 3-16 号点（能看到编号+置信度）；面部点 0-2（鼻/眼）不画 —— 这两个动作
        不用鼻/眼做**骨架**展示（鼻是仰头特征的参考点，它的数值在读数行里，
        不在骨架层重复画）。
            点颜色按部位——躯干/四肢 5-16 绿、耳 3/4 黄（仰头特征依赖耳点）；
            置信度 <0.3 的"没检测到"点画灰色空心小圈示意位置。
            标签字色按置信度分档：>=0.5 绿、0.3~0.5 黄（低置信度）、<0.3 灰（没检测到）。
        连线仍只连躯干/四肢，且两端都可见(>=0.3)才画，避免噪点乱连。
        """
        landmarks = pose.get('landmarks')
        image_size = pose.get('image_size')
        if not landmarks or image_size is None:
            return

        w, h = image_size
        VIS_LOW = 0.3    # < 此值 = 没检测到（灰）
        VIS_OK = 0.5     # >= 此值 = 置信度足够（绿）

        def px(pid: int) -> Optional[tuple]:
            """关键点归一化坐标 -> 像素坐标（不过滤可见度，分级用颜色表达）。"""
            if pid >= len(landmarks):
                return None
            x, y, _z, _vis = landmarks[pid]
            return int(x * w), int(y * h)

        def vis_of(pid: int) -> float:
            return landmarks[pid][3] if pid < len(landmarks) else 0.0

        # 关键点：画 3-16 号（连线见下）。面部 0-2（鼻/眼）不画。
        # 躯干/四肢 5-16 绿；耳 3/4 黄（仰头特征的参考点，单独标出）。
        # 没检测到(<0.3)的点画灰色空心小圈——示意模型认为的位置，方便定位丢点。
        for pid in range(3, 17):
            p = px(pid)
            if p is None:
                continue
            if vis_of(pid) < VIS_LOW:
                cv2.circle(frame, p, 3, (150, 150, 150), 1, cv2.LINE_AA)
            elif pid <= 4:
                cv2.circle(frame, p, 4, (0, 215, 255), -1, cv2.LINE_AA)  # 黄：耳
            else:
                cv2.circle(frame, p, 4, (0, 255, 0), -1, cv2.LINE_AA)    # 绿：躯干/四肢

        # 标签：每个画出的点（3-16）右下角标 `pid:置信度`，字色按置信度分档（黑描边保证可读）。
        for pid in range(3, 17):
            p = px(pid)
            if p is None:
                continue
            vis = vis_of(pid)
            if vis >= VIS_OK:
                color = (60, 200, 60)      # 绿：置信度够
            elif vis >= VIS_LOW:
                color = (0, 215, 255)      # 黄：低置信度
            else:
                color = (150, 150, 150)    # 灰：没检测到
            label = f"{pid}:{vis:.2f}"
            tx = min(p[0] + 6, w - 42)     # 不超出画面右缘（约 40px 宽的标签）
            ty = p[1] + 12
            cv2.putText(frame, label, (tx, ty + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # 连接线（两端都可见才画，防止线条插到残缺点）
        for a, b in POSE_CONNECTIONS:
            if vis_of(a) < VIS_LOW or vis_of(b) < VIS_LOW:
                continue
            pa, pb = px(a), px(b)
            if pa is not None and pb is not None:
                cv2.line(frame, pa, pb, (255, 0, 0), 2, cv2.LINE_AA)

    @staticmethod
    def _dashed_line(img, pt1, pt2, color, dash=5, gap=3) -> None:
        """在图上画一条虚线（cv2 没有内建虚线）。"""
        x1, y1 = pt1
        x2, y2 = pt2
        dist = int(np.hypot(x2 - x1, y2 - y1))
        if dist == 0:
            return
        dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
        i = 0
        while i < dist:
            s = i
            e = min(i + dash, dist)
            cv2.line(img,
                     (int(x1 + dx * s), int(y1 + dy * s)),
                     (int(x1 + dx * e), int(y1 + dy * e)),
                     color, 1)
            i += dash + gap


# ---------- 绘制小工具 ----------

def _put_text_outlined(frame: np.ndarray, text: str, org: tuple,
                       scale: float = 0.5, color: tuple = (230, 230, 230),
                       thickness: int = 1) -> None:
    """画一行带黑色描边的文字（视频画面上比纯色文字好认，深色/浅色背景都清楚）。

    描边用**上下左右各偏移 1px 的 4 次细笔画**，不用「同一位置先粗后细画两遍」：
    后者在 OpenCV 5.0（LINE_AA）上会把最后一个字重复画在字符串末尾（画面右侧
    多出一个淡淡的字影），实测粗笔画（thickness>=2）重叠同一个 baseline 就触发。
    4 次偏移的都是 thickness=1，不触发该问题（2026-09-23 实拍对比确认）。
    """
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(frame, text, (org[0] + dx, org[1] + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness,
                    cv2.LINE_AA)
    cv2.putText(frame, text, org,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                cv2.LINE_AA)


# ---------- 自测辅助 ----------

def _summary_row_pixels(frame: np.ndarray, near: Optional[tuple] = None,
                        y: int = _ACTION_Y) -> int:
    """文字行（y 上下 12px 带）里的像素数：near=None 时数所有非黑像素；
    给了 BGR 颜色则数"接近该色"的像素（抗锯齿会把边缘混色，所以按距离判定，
    不要求精确相等）。给 selftest_action_overlay 数行位/配色用。
    """
    band = frame[y - 12:y + 6]
    if near is None:
        return int(np.count_nonzero(band.any(axis=2)))
    # int32 起步：差值平方最大 255²=65025，int16 会溢出成负数
    diff = band.astype(np.int32) - np.array(near, dtype=np.int32)
    return int(np.count_nonzero(np.sqrt((diff ** 2).sum(axis=2)) < 40))


def selftest_action_overlay() -> None:
    """自测 draw_action_overlay（举手 / 仰头两行读数 + 两行状态 + 几何）：
    缺特征时整块不画、状态未判（--arm-rule off）时明说"只报数不判"、颜色随状态、
    无读数写 N/A 不让整行消失、几何开关生效、几何缺端点/缺腕不崩。
    用黑底帧数像素，不碰摄像头。"""
    def arm(raise_deg=38.0, side='L', l=38.0, r=152.0,
            elbow_l=175.0, elbow_r=None, valid=True, geom=None):
        return {'arm_raise_deg': raise_deg if valid else None,
                'arm_raise_side': side if valid else None,
                'arm_raise_left_deg': l, 'arm_raise_right_deg': r,
                'arm_raise_conf': 0.91, 'arm_raise_valid': valid,
                'elbow_deg': elbow_l, 'elbow_left_deg': elbow_l,
                'elbow_right_deg': elbow_r,
                'elbow_valid': elbow_l is not None, 'arm_geom': geom}

    def head(pitch=12.0, valid=True, geom=None):
        return {'head_pitch_deg': pitch if valid else None,
                'head_pitch_valid': valid, 'head_pitch_conf': 0.93,
                'head_pitch_pts': 'ear_mid+nose' if valid else None,
                'head_tilt_geom': geom}

    # 举手几何画在 y≈120~200、x≈120~260（避开文字行 [18,36)/[44,62)/[70,88)/[96,114)）
    arm_geom = {'side': 'L', 'shoulder': (200.0, 200.0), 'elbow': (140.0, 130.0),
                'wrist': (120.0, 90.0)}
    arm_box = lambda f: int(np.count_nonzero(f[200:300, 100:300].any(axis=2)))
    # 仰头几何：耳中点 (420,300) → 鼻 (500,220)，水平参考线在 y=300、x 到 520
    head_geom = {'ear_mid': (420.0, 300.0), 'nose': (500.0, 220.0)}
    head_box = lambda f: int(np.count_nonzero(f[260:340, 460:560].any(axis=2)))

    def render(a, a_state, h, h_state, **kw):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        FrameRenderer(draw_skeleton=False).draw_action_overlay(
            frame, a, a_state, h, h_state, **kw)
        return frame

    # 1) 两个特征都不给 → 什么都不画
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    FrameRenderer(draw_skeleton=False).draw_action_overlay(blank)
    assert int(np.count_nonzero(blank)) == 0, "不给特征时不该画任何像素"

    # 2) 举手块：状态未判（v1 默认）也照画两行，且状态行不写状态词
    f = render(arm(), None, None, None)
    assert _summary_row_pixels(f, y=_ACTION_Y) > 0, "举手读数行没画出来"
    assert _summary_row_pixels(f, y=_ACTION_Y + _ACTION_LINE_H) > 0, \
        "举手状态行没画出来（未判也要写 report only）"
    assert _summary_row_pixels(f, y=_ACTION_Y + 2 * _ACTION_LINE_H) == 0, \
        "没有仰头特征时不该画仰头块"

    # 3) 状态行着色随状态（RAISED/DOWN 各取规定色）；未判用灰
    for st in (ArmRaiseState.RAISED, ArmRaiseState.DOWN):
        f = render(arm(), st, None, None, arm_enter_deg=45.0)
        assert _summary_row_pixels(f, near=_ACTION_STATE_COLOR[st],
                                   y=_ACTION_Y) > 0, f"{st} 颜色不对"
        assert _summary_row_pixels(f, near=_ACTION_STATE_COLOR[st],
                                   y=_ACTION_Y + _ACTION_LINE_H) > 0
    f = render(arm(), None, None, None)
    assert _summary_row_pixels(f, near=_ACTION_NA_COLOR, y=_ACTION_Y) > 0

    # 4) 仰头块：状态着色 + 位置在举手块之下（同一 x 栏、往下排四行）
    for st in (HeadTiltState.TILTED_BACK, HeadTiltState.NEUTRAL):
        f = render(None, None, head(), st)
        assert _summary_row_pixels(f, near=_ACTION_STATE_COLOR[st],
                                   y=_ACTION_Y) > 0, f"{st} 颜色不对"
    f = render(arm(), ArmRaiseState.DOWN, head(), HeadTiltState.NEUTRAL)
    assert _summary_row_pixels(f, y=_ACTION_Y + 2 * _ACTION_LINE_H) > 0, \
        "仰头读数行应排在举手块之下"

    # 5) 无读数（丢点帧）照画 N/A —— 不让整行消失（"算不出来"本身就是要看的信息）
    for a, h in ((arm(valid=False), head(valid=False)),
                 (arm(raise_deg=None, elbow_l=None), head(pitch=None)),
                 (None, None)):
        f = render(a, ArmRaiseState.UNKNOWN, h, HeadTiltState.UNKNOWN)
        if a is not None:
            assert _summary_row_pixels(f, y=_ACTION_Y) > 0
        if h is not None:
            assert _summary_row_pixels(f, y=_ACTION_Y + 2 * _ACTION_LINE_H) > 0

    # 6) 几何开关：开→有像素，关→没有；缺端点/缺腕不崩（腕丢了就别画腕）
    assert arm_box(render(arm(geom=arm_geom), None, None, None,
                          show_geometry=True)) > 0
    assert arm_box(render(arm(geom=arm_geom), None, None, None,
                          show_geometry=False)) == 0
    assert head_box(render(None, None, head(geom=head_geom), None,
                           show_geometry=True)) > 0
    assert head_box(render(None, None, head(geom=head_geom), None,
                           show_geometry=False)) == 0
    for bad in (None, {}, {'side': 'L', 'shoulder': (200.0, 200.0)},
                {'side': 'L', 'elbow': (140.0, 130.0)}):
        assert arm_box(render(arm(geom=bad), None, None, None)) == 0, \
            f"举手几何缺端点时不该画：{bad}"
    # 腕缺失（None）：仍然画肩→肘与参考线，只是不画肘→腕那一段
    no_wrist = {'side': 'L', 'shoulder': (200.0, 200.0),
                'elbow': (140.0, 130.0), 'wrist': None}
    assert arm_box(render(arm(geom=no_wrist), None, None, None)) > 0, \
        "腕缺失时肩→肘与参考线仍应画出来"
    for bad in (None, {}, {'ear_mid': (420.0, 300.0)}):
        assert head_box(render(None, None, head(geom=bad), None)) == 0, \
            f"仰头几何缺端点时不该画：{bad}"

    # 7) 只给一侧特征时，行位不串（仰头块不会画到举手那块槽位上）
    f = render(arm(), ArmRaiseState.DOWN, None, None)
    assert _summary_row_pixels(f, y=_ACTION_Y + 2 * _ACTION_LINE_H) == 0
    assert _summary_row_pixels(f, y=_ACTION_Y) > 0

    print("  selftest_action_overlay: OK")


def selftest_extended_action_overlay() -> None:
    """自测 draw_all_action_overlay：十一个动作按序排 22 行、左右独立动作左右分离、
    无效读数写 N/A 不让行消失、状态未判写 report only、几何开关、未知 kind 报错。
    用黑底帧数像素，不碰摄像头。"""
    def trunk(deg=10.0, valid=True, geom=None):
        return {'trunk_flex_deg': deg if valid else None,
                'trunk_flex_valid': valid, 'trunk_flex_conf': 0.90,
                'trunk_flex_pts': 'shoulder_mid+hip_mid' if valid else None,
                'trunk_geom': geom}

    def neck(deg=4.0, valid=True, geom=None):
        return {'neck_lateral_deg': deg if valid else None,
                'neck_lateral_valid': valid, 'neck_lateral_conf': 0.88,
                'neck_geom': geom}

    def rotation(deg=6.0, valid=True, geom=None):
        return {'trunk_rotation_deg': deg if valid else None,
                'trunk_rotation_valid': valid, 'trunk_rotation_conf': 0.86,
                'trunk_rotation_shoulder_deg': 3.0 if valid else None,
                'trunk_rotation_hip_deg': -3.0 if valid else None,
                'trunk_rotation_geom': geom}

    def bilateral(prefix: str, l=165.0, r=155.0, geom=None):
        return {f'{prefix}_left_deg': l, f'{prefix}_right_deg': r,
                f'{prefix}_left_valid': l is not None,
                f'{prefix}_right_valid': r is not None,
                f'{prefix}_left_conf': 0.92, f'{prefix}_right_conf': 0.91,
                f'{prefix}_geom': geom or {}}

    def arm(deg=38.0):
        return {'arm_raise_deg': deg, 'arm_raise_side': 'L',
                'arm_raise_left_deg': deg, 'arm_raise_right_deg': 152.0,
                'arm_raise_conf': 0.91, 'arm_raise_valid': True,
                'elbow_deg': 175.0, 'elbow_left_deg': 175.0,
                'elbow_right_deg': None, 'elbow_valid': True,
                'arm_geom': None}

    def head(pitch=12.0):
        return {'head_pitch_deg': pitch, 'head_pitch_valid': True,
                'head_pitch_conf': 0.93, 'head_pitch_pts': 'ear_mid+nose',
                'head_tilt_geom': None}

    def abduction(l=90.0, r=15.0, geom=None):
        return {'shoulder_abduction_left_deg': l,
                'shoulder_abduction_right_deg': r,
                'shoulder_abduction_left_valid': l is not None,
                'shoulder_abduction_right_valid': r is not None,
                'shoulder_abduction_left_conf': 0.90,
                'shoulder_abduction_right_conf': 0.89,
                'shoulder_abduction_geom': geom or {}}

    def pelvic(deg=6.0, valid=True, geom=None):
        return {'pelvic_tilt_deg': deg if valid else None,
                'pelvic_tilt_valid': valid, 'pelvic_tilt_conf': 0.87,
                'pelvic_tilt_geom': geom}

    def scapular(deg=-4.0, valid=True, geom=None):
        return {'scapular_symmetry_deg': deg if valid else None,
                'scapular_symmetry_valid': valid,
                'scapular_symmetry_conf': 0.88,
                'scapular_symmetry_geom': geom}

    # 几何画在右侧（x>=560），文字行都在左边，两边互不干扰
    geoms = {
        'trunk': {'shoulder_mid': (600.0, 200.0), 'hip_mid': (620.0, 300.0)},
        'neck': {'earL': (560.0, 350.0), 'earR': (620.0, 360.0)},
        'rotation': {'shoulderL': (560.0, 120.0), 'shoulderR': (620.0, 140.0),
                     'hipL': (560.0, 180.0), 'hipR': (620.0, 200.0)},
        'elbow': {'L': {'p1': (580.0, 200.0), 'joint': (600.0, 240.0),
                        'p2': (620.0, 280.0)}},
        'knee': {'L': {'p1': (580.0, 300.0), 'joint': (600.0, 340.0),
                       'p2': (620.0, 380.0)}},
        'hip': {'R': {'p1': (580.0, 220.0), 'joint': (600.0, 260.0),
                      'p2': (620.0, 300.0)}},
        'abduction': {'L': {'shoulder': (560.0, 160.0),
                            'elbow': (640.0, 160.0)}},
        'pelvic': {'hipL': (560.0, 320.0), 'hipR': (620.0, 300.0)},
        'scapular': {'shoulderL': (560.0, 100.0),
                     'shoulderR': (620.0, 90.0)},
    }

    def payloads(with_geom=False):
        g = geoms if with_geom else {}
        return [
            ('arm', {'features': arm(), 'state': ArmRaiseState.DOWN,
                     'enter_deg': 45.0}),
            ('head', {'features': head(), 'state': HeadTiltState.NEUTRAL,
                      'enter_deg': 20.0}),
            ('trunk', {'features': trunk(geom=g.get('trunk')),
                       'state': TrunkFlexState.NEUTRAL, 'enter_deg': 25.0}),
            ('neck', {'features': neck(geom=g.get('neck')),
                      'state': NeckLateralState.TILTED, 'enter_deg': 15.0}),
            ('elbow', {'features': bilateral('elbow_flex', geom=g.get('elbow')),
                       'states': {'L': ElbowFlexState.FLEXED,
                                  'R': ElbowFlexState.EXTENDED}}),
            ('rotation', {'features': rotation(geom=g.get('rotation')),
                          'state': TrunkRotationState.NEUTRAL,
                          'enter_deg': 15.0}),
            ('knee', {'features': bilateral('knee_flex', geom=g.get('knee')),
                      'states': {'L': KneeFlexState.EXTENDED,
                                 'R': KneeFlexState.FLEXED}}),
            ('hip', {'features': bilateral('hip_flex', geom=g.get('hip')),
                     'states': {'L': HipFlexState.FLEXED,
                                'R': HipFlexState.EXTENDED}}),
            ('abduction', {'features': abduction(geom=g.get('abduction')),
                           'states': {'L': ShoulderAbductionState.ABDUCTED,
                                      'R': ShoulderAbductionState.ADDUCTED}}),
            ('pelvic', {'features': pelvic(geom=g.get('pelvic')),
                        'state': PelvicTiltState.NEUTRAL, 'enter_deg': 10.0}),
            ('scapular', {'features': scapular(geom=g.get('scapular')),
                          'state': ScapularSymmetryState.SYMMETRIC,
                          'enter_deg': 10.0}),
        ]

    def render(actions, **kw):
        frame = np.zeros((720, 640, 3), dtype=np.uint8)
        FrameRenderer(draw_skeleton=False).draw_all_action_overlay(
            frame, actions, **kw)
        return frame

    row_y = lambda i: _ACTION_Y + i * _ACTION_LINE_H

    # 1) 十一个动作按 ACTION_ORDER 排满 22 行（每块两行），行位都落在画面内
    f = render(payloads())
    assert len(ACTION_ORDER) == 11
    assert row_y(21) < 720, "22 行必须放得进 720p"
    for i in range(22):
        assert _summary_row_pixels(f, y=row_y(i)) > 0, f"第 {i} 行没画出来"

    # 2) 左右独立动作：左/右各用各的状态色（不合并成一个状态）
    assert _summary_row_pixels(f, near=_ACTION_STATE_COLOR[ElbowFlexState.FLEXED],
                               y=row_y(8)) > 0, "肘左行颜色不对"
    assert _summary_row_pixels(f, near=_ACTION_STATE_COLOR[ElbowFlexState.EXTENDED],
                               y=row_y(9)) > 0, "肘右行颜色不对"
    assert _summary_row_pixels(f, near=_ACTION_STATE_COLOR[KneeFlexState.FLEXED],
                               y=row_y(13)) > 0, "膝右行颜色不对"
    assert _summary_row_pixels(f, near=_ACTION_STATE_COLOR[HipFlexState.FLEXED],
                               y=row_y(14)) > 0, "髋左行颜色不对"
    assert _summary_row_pixels(
        f, near=_ACTION_STATE_COLOR[ShoulderAbductionState.ABDUCTED],
        y=row_y(16)) > 0, "肩外展左行颜色不对"
    assert _summary_row_pixels(
        f, near=_ACTION_STATE_COLOR[ShoulderAbductionState.ADDUCTED],
        y=row_y(17)) > 0, "肩外展右行颜色不对"

    # 3) 未判（None）→ 灰 + report only；无效读数 → N/A，但行照画
    off = [(kind, {'features': p['features'], 'state': None,
                   'states': {k: None for k in ('L', 'R')}})
           for kind, p in payloads()]
    f = render(off)
    for i in range(22):
        assert _summary_row_pixels(f, near=_ACTION_NA_COLOR, y=row_y(i)) > 0, \
            f"第 {i} 行未判时应为灰色 report only"
    invalid = [
        ('trunk', {'features': trunk(deg=None, valid=False),
                   'state': TrunkFlexState.UNKNOWN, 'enter_deg': 25.0}),
        ('neck', {'features': neck(deg=None, valid=False),
                  'state': NeckLateralState.UNKNOWN, 'enter_deg': 15.0}),
        ('elbow', {'features': bilateral('elbow_flex', l=None, r=155.0),
                   'states': {'L': ElbowFlexState.UNKNOWN,
                              'R': ElbowFlexState.EXTENDED}}),
        ('rotation', {'features': rotation(deg=None, valid=False),
                      'state': TrunkRotationState.UNKNOWN, 'enter_deg': 15.0}),
        ('knee', {'features': bilateral('knee_flex', l=None, r=None),
                  'states': {'L': KneeFlexState.UNKNOWN,
                             'R': KneeFlexState.UNKNOWN}}),
        ('hip', {'features': bilateral('hip_flex', l=None, r=160.0),
                 'states': {'L': HipFlexState.UNKNOWN,
                            'R': HipFlexState.EXTENDED}}),
        ('abduction', {'features': abduction(l=None, r=15.0),
                       'states': {'L': ShoulderAbductionState.UNKNOWN,
                                  'R': ShoulderAbductionState.ADDUCTED}}),
        ('pelvic', {'features': pelvic(deg=None, valid=False),
                    'state': PelvicTiltState.UNKNOWN, 'enter_deg': 10.0}),
        ('scapular', {'features': scapular(deg=None, valid=False),
                      'state': ScapularSymmetryState.UNKNOWN,
                      'enter_deg': 10.0}),
    ]
    f = render(invalid)
    for i in range(18):
        assert _summary_row_pixels(f, y=row_y(i)) > 0, \
            f"读数无效时第 {i} 行也要写 N/A，不能消失"

    # 4) 几何开关：开→多出像素；关→与文字画面一致；缺端点不崩
    base = render(payloads())
    with_geom = render(payloads(with_geom=True), show_geometry=True)
    assert int(np.count_nonzero(with_geom)) > int(np.count_nonzero(base)), \
        "show_geometry=True 时几何线应额外画出像素"
    assert int(np.count_nonzero(render(payloads(with_geom=True),
                                       show_geometry=False))) == \
        int(np.count_nonzero(base)), "show_geometry=False 时不该画任何几何"
    for kind, bad in (
            ('trunk', None), ('trunk', {}),
            ('trunk', {'shoulder_mid': (600.0, 200.0)}),
            ('neck', {'earL': (560.0, 350.0)}),
            ('rotation', {'shoulderL': (560.0, 120.0),
                          'shoulderR': (620.0, 140.0),
                          'hipL': (560.0, 180.0)}),
            ('elbow', {'L': {'p1': (580.0, 200.0),
                             'joint': (600.0, 240.0)}})):
        feats = {'trunk': trunk, 'neck': neck, 'rotation': rotation}.get(kind)
        if feats is None:
            feats = {'elbow_flex_left_deg': 165.0, 'elbow_flex_right_deg': None,
                     'elbow_flex_geom': bad}
        else:
            feats = feats(geom=bad)
        render([(kind, {'features': feats, 'state': None})])  # 不崩即可
    # 新增几何的"缺端点不崩"：肩外展缺肘、双髋缺一个、双肩缺一个
    render([('abduction', {'features': abduction(
        geom={'L': {'shoulder': (560.0, 160.0)}}), 'states': {}})])
    render([('pelvic', {'features': pelvic(
        geom={'hipL': (560.0, 320.0)}), 'state': None})])
    render([('scapular', {'features': scapular(
        geom={'shoulderL': (560.0, 100.0)}), 'state': None})])

    # 5) 只给子集时，行从顶上开始排、不出现空槽位
    f = render(payloads()[2:4])
    assert _summary_row_pixels(f, y=row_y(0)) > 0
    assert _summary_row_pixels(f, y=row_y(3)) > 0
    assert _summary_row_pixels(f, y=row_y(4)) == 0

    # 6) 未知 kind 必须报错（防止入口拼错名字后静默少画一块）
    try:
        render([('balabala', {'features': trunk()})])
    except ValueError:
        pass
    else:
        raise AssertionError("未知动作 kind 应抛 ValueError")

    # 7) 行高自适应：720p 全开用默认行高；360 高的半幅要收紧且仍放得下
    assert overlay_layout(720, len(ACTION_ORDER)) == (_ACTION_LINE_H, 0.50)
    lh, sc = overlay_layout(360, len(ACTION_ORDER))
    assert lh < _ACTION_LINE_H and 0.35 <= sc < 0.50, \
        f"矮画面应收紧行高/字号，实际 {lh}/{sc}"
    assert _ACTION_Y + 2 * len(ACTION_ORDER) * lh <= 360, "收紧后 22 行仍须在画面内"

    # 8) 优化轮新键：无效原因后缀 / 高侧标签 / trunk 参考几何。
    #    只要求"带新键不崩、行仍画得出来"；具体文案不比对（cv2 字体渲染不可读）。
    head_bad = {'head_pitch_deg': None, 'head_pitch_valid': False,
                'head_pitch_conf': None, 'head_pitch_pts': None,
                'head_pitch_invalid_reason': 'geom', 'head_tilt_geom': None}
    head_deg = {'head_pitch_deg': None, 'head_pitch_valid': False,
                'head_pitch_conf': None, 'head_pitch_pts': None,
                'head_pitch_degenerate': True,
                'head_pitch_invalid_reason': 'degenerate',
                'head_tilt_geom': None}
    arm_bad = {'arm_raise_deg': None, 'arm_raise_side': None,
               'arm_raise_left_deg': None, 'arm_raise_right_deg': None,
               'arm_raise_conf': None, 'arm_raise_valid': False,
               'arm_raise_left_invalid_reason': 'no_point',
               'arm_raise_right_invalid_reason': 'geom',
               'elbow_deg': None, 'elbow_left_deg': None,
               'elbow_right_deg': None, 'elbow_valid': False,
               'arm_geom': None}
    new_key_actions = [
        ('arm', {'features': arm_bad, 'state': None}),
        ('head', {'features': head_bad, 'state': HeadTiltState.UNKNOWN,
                  'enter_deg': 20.0}),
        ('head', {'features': head_deg, 'state': HeadTiltState.UNKNOWN,
                  'enter_deg': 20.0}),
        ('trunk', {'features': {**trunk(deg=None, valid=False),
                                'trunk_flex_invalid_reason': 'geom'},
                   'state': TrunkFlexState.UNKNOWN, 'enter_deg': 25.0}),
        ('elbow', {'features': {**bilateral('elbow_flex', l=None, r=155.0),
                                'elbow_flex_left_invalid_reason':
                                    'out_of_frame'},
                   'states': {'L': ElbowFlexState.UNKNOWN,
                              'R': ElbowFlexState.EXTENDED}}),
        ('abduction', {'features': {
            **abduction(l=None, r=15.0),
            'shoulder_abduction_left_invalid_reason': 'no_trunk'},
            'states': {'L': ShoulderAbductionState.UNKNOWN,
                       'R': ShoulderAbductionState.ADDUCTED}}),
        ('pelvic', {'features': {**pelvic(deg=12.0),
                                 'pelvic_tilt_higher': 'R'},
                    'state': PelvicTiltState.TILTED, 'enter_deg': 10.0}),
        ('scapular', {'features': {**scapular(deg=-8.0),
                                   'scapular_symmetry_higher': 'L'},
                      'state': ScapularSymmetryState.ASYMMETRIC,
                      'enter_deg': 10.0}),
    ]
    f = render(new_key_actions)
    for i in range(2 * len(new_key_actions)):
        assert _summary_row_pixels(f, y=row_y(i)) > 0, \
            f"带无效原因/高侧键时第 {i} 行也要画出来"

    # trunk 参考几何：geom['axis'] 存在时沿躯干轴画虚线（额外像素）
    abd_trunk = abduction(geom={
        'L': {'shoulder': (560.0, 160.0), 'elbow': (640.0, 160.0)},
        'axis': {'shoulder_mid': (600.0, 200.0), 'hip_mid': (600.0, 300.0)}})
    acts = [('abduction', {'features': abd_trunk,
                           'states': {'L': ShoulderAbductionState.ABDUCTED,
                                      'R': ShoulderAbductionState.ADDUCTED}})]
    assert int(np.count_nonzero(render(acts, show_geometry=True))) > \
        int(np.count_nonzero(render(acts, show_geometry=False))), \
        "trunk 参考轴应额外画出虚线"

    print("  selftest_extended_action_overlay: OK")


if __name__ == "__main__":
    # 自测（不需要摄像头；绘制部分用黑底帧数像素，其余靠人工跑 demo 看）
    selftest_action_overlay()
    selftest_extended_action_overlay()
    print("output selftest: ALL PASSED")
