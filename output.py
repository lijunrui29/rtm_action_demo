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

from decision import ArmRaiseState, HeadTiltState

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
}
_ACTION_STATE_COLOR = {
    ArmRaiseState.RAISED: _ACTION_TRIGGER_COLOR,
    ArmRaiseState.DOWN: _ACTION_IDLE_COLOR,
    HeadTiltState.TILTED_BACK: _ACTION_TRIGGER_COLOR,
    HeadTiltState.NEUTRAL: _ACTION_IDLE_COLOR,
}
# 状态未判时画面上的写法：**不写状态词**，明说"只报数不判"，免得看画的人以为
# 已经有个规则在跑。举手那条线 v1 默认就是这一档（--arm-rule off）；双摄入口
# 两条线都走这一档（两台都只报数）。两个块共用同一句话，别各写一份。
_ACTION_NOT_DECIDED = "not decided (report only)"

# 动作叠加的 y 行位（像素）：举手读数 30 / 举手状态 56 / 仰头读数 82 / 仰头状态 108。
_ACTION_Y = 30
_ACTION_LINE_H = 26

# 动作几何用色（BGR）：举手（肩-肘-腕）蓝、仰头（耳-鼻）黄、参考虚线灰
_ACTION_ARM_GEOM_COLOR = (255, 160, 0)
_ACTION_HEAD_GEOM_COLOR = (0, 220, 255)
_ACTION_REF_COLOR = (110, 110, 110)     # 几何参考虚线（竖直/水平参考轴）


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

    @staticmethod
    def _deg_text(v: Optional[float]) -> str:
        """角度读数统一格式：算不出写 N/A（不写 0，0 是个有效读数）。

        `-0.0`（合成/对称构型下 atan2 会给出 -0.0）归一成 `0.0`：读数是 0 时
        画面上出现「-0.0」会被当成 bug，而它确实就是 0。
        """
        if v is None:
            return "N/A"
        return f"{0.0 if v == 0 else v:.1f}"

    def _draw_arm_block(self, frame: np.ndarray, arm: dict,
                        arm_state: Optional[ArmRaiseState],
                        enter_deg: Optional[float], y: int) -> int:
        """画举手两行（读数行 + 状态行），返回下一块的起始 y。"""
        color = _ACTION_STATE_COLOR.get(arm_state, _ACTION_NA_COLOR) \
            if arm_state is not None else _ACTION_NA_COLOR
        side = arm.get('arm_raise_side')
        side_s = side if side is not None else "--"
        line1 = (f"Arm raise deg:  {side_s} {self._deg_text(arm.get('arm_raise_deg'))}"
                 f"  (L {self._deg_text(arm.get('arm_raise_left_deg'))}"
                 f"  R {self._deg_text(arm.get('arm_raise_right_deg'))})"
                 f"   Elbow deg:  L {self._deg_text(arm.get('elbow_left_deg'))}"
                 f"  R {self._deg_text(arm.get('elbow_right_deg'))}")
        if arm_state is None:
            # v1 默认：只报数不判（用户要先看数据再定触发用哪个角）
            line2 = (f"Arm State: {_ACTION_NOT_DECIDED}")
        else:
            line2 = f"Arm State: {_ACTION_STATE_LABEL.get(arm_state, 'N/A')}"
            if enter_deg is not None:
                line2 += f"  (enter <= {enter_deg:.1f} deg)"
        _put_text_outlined(frame, line1, (12, y), 0.50, color)
        _put_text_outlined(frame, line2, (12, y + _ACTION_LINE_H), 0.50, color)
        return y + 2 * _ACTION_LINE_H

    def _draw_head_block(self, frame: np.ndarray, head: dict,
                         head_state: Optional[HeadTiltState],
                         enter_deg: Optional[float], y: int) -> None:
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
        if head_state is None:
            line2 = f"Head State: {_ACTION_NOT_DECIDED}"
        else:
            line2 = f"Head State: {_ACTION_STATE_LABEL.get(head_state, 'N/A')}"
            if enter_deg is not None:
                line2 += f"  (enter >= {enter_deg:.1f} deg)"
        _put_text_outlined(frame, line1, (12, y), 0.50, color)
        _put_text_outlined(frame, line2, (12, y + _ACTION_LINE_H), 0.50, color)

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


if __name__ == "__main__":
    # 自测（不需要摄像头；绘制部分用黑底帧数像素，其余靠人工跑 demo 看）
    selftest_action_overlay()
    print("output selftest: ALL PASSED")
