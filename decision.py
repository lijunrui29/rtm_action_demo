"""
decision 层（全身动作线）：把 features 层给出的动作特征映射成状态。

本仓库有十一个动作，每个动作一个双阈值迟滞状态机（肘/膝/髋/肩外展按左右
各建一个实例，不合并）：

1) 举手（ArmRaiseFeatures 的抬升角，**越小越举**）：<= enter → RAISED，
   >= exit → DOWN；
2) 仰头（HeadTiltFeatures 的俯仰代理角，**越大越仰头**）：>= enter →
   TILTED_BACK，<= exit → NEUTRAL；
3) 躯干前屈 / 颈部侧屈 / 躯干旋转：**|deg| >= enter** 触发，<= exit 回中性；
4) 肘 / 膝 / 髋屈伸：内角 **<= enter** 触发（FLEXED），>= exit 回 EXTENDED；
5) 肩关节外展：**|deg| >= enter** 触发（ABDUCTED），<= exit 回 ADDUCTED。

所有状态机都只有"未触发 / 触发"两档 + UNKNOWN，都**不产生提醒** —— 本仓库
目前没有任何提醒，也没有任何时长判断。UNKNOWN 的含义一律是"这一帧没有证据"
（无人 / 关键点丢了 / 特征被判几何退化）：此时兜底"未触发"，即**没证据就不判
动作**，而不是沿用上一帧的状态（不维持上一有效值）。

每个状态机都用**双阈值迟滞带**：进入看 enter 阈值、退出要越过 exit 阈值、
两者之间保持当前状态（防阈值附近抖动来回跳）。

2026-10-18 优化轮新增 **多帧确认**（`confirm_frames`，默认 1 = 旧行为）：
目标状态与当前不同时要连续保持 N 帧才提交，压住单帧噪声打出的假状态。
丢帧（无证据）仍**立即**回到 UNKNOWN 并清掉确认计数。

接口约定（保持稳定，别改签名；新增参数只加带默认值的关键字参数）：
    ArmRaiseState 枚举：DOWN / RAISED / UNKNOWN
    ArmRaiseDecision(enter_deg=45.0, exit_deg=60.0, confirm_frames=1)
    update(arm: Optional[dict]) -> ArmRaiseState
    state（只读属性）

    HeadTiltState 枚举：NEUTRAL / TILTED_BACK / UNKNOWN
    HeadTiltDecision(enter_deg=20.0, exit_deg=10.0, confirm_frames=1)
    update(head: Optional[dict]) -> HeadTiltState
    state（只读属性）

⚠ 所有 enter/exit 默认值都是**占位值**、不是标定结果：读数的中立基线随摄像头
高度、机位、人体比例变化，先跑起来看画面上的数再定（main_action_demo.py 里
都有对应的 CLI 参数可现场覆盖）。
"""

from __future__ import annotations

import enum
from typing import Optional


class ArmRaiseState(enum.Enum):
    """举手状态。"""
    DOWN = "DOWN"
    RAISED = "RAISED"
    UNKNOWN = "UNKNOWN"


class ArmRaiseDecision:
    """举手判断：上臂抬升角 <= enter_deg 进 RAISED，>= exit_deg 回 DOWN。

    角度方向：features.ArmRaiseFeatures.arm_raise_deg，**越小越举**
    （0 = 完全上举、90 = 平举、180 = 自然下垂）。
        arm_raise_deg <= enter_deg       -> RAISED
        arm_raise_deg >= exit_deg        -> DOWN
        enter_deg < 角度 < exit_deg      -> 迟滞带，保持当前状态
                                            （UNKNOWN 时兜底 DOWN：没有充分证据
                                             就不判"举手"）

    两侧怎么合成：features 层已经把"举得更高的那一侧"（抬升角最小的一侧）选好
    放进 arm_raise_deg，本类直接用 —— 判据语义 = **任一侧举手即触发**。
    本类不分开判左右（demo 阶段不做"哪只手"的区分；画面上有两侧读数可看）。

    ⚠ v1 默认不装配：main_action_demo.py 的 --arm-rule 默认 off，举手那条线只
    显示两个角度、不判状态 —— 用户要先看几眼数据，再决定触发用抬升角还是肘角。
    本类是"规则就绪、随时可开"的那一档（--arm-rule raise 即可）。

    接口约定（保持稳定，别改签名；新增参数只加带默认值的关键字参数）：
        ArmRaiseDecision(enter_deg=45.0, exit_deg=60.0, confirm_frames=1)
        update(arm: Optional[dict]) -> ArmRaiseState
        state（只读属性）
    """

    def __init__(self, enter_deg: float = 45.0, exit_deg: float = 60.0,
                 confirm_frames: int = 1) -> None:
        """
        参数:
            enter_deg: 抬升角 <= 此值判 RAISED（度；越小越举）。
            exit_deg:  抬升角 >= 此值判 DOWN（度）。必须大于 enter_deg，
                       两者之间就是迟滞带。
            confirm_frames: 目标状态需要连续保持多少帧才提交（>= 1；
                       默认 1 = 旧行为，无额外延迟）。
        """
        if exit_deg <= enter_deg:
            raise ValueError("exit_deg 必须大于 enter_deg（否则没有迟滞带）")
        if confirm_frames < 1:
            raise ValueError("confirm_frames 必须 >= 1")
        self.enter_deg = enter_deg
        self.exit_deg = exit_deg
        self.confirm_frames = int(confirm_frames)
        self._state = ArmRaiseState.UNKNOWN
        self._pending_state = None
        self._pending_count = 0

    @property
    def state(self) -> ArmRaiseState:
        return self._state

    def _clear_pending(self) -> None:
        self._pending_state = None
        self._pending_count = 0

    def _commit(self, target: ArmRaiseState) -> ArmRaiseState:
        """目标状态与当前不同时做多帧确认；确认够了才真正切换。"""
        if target is self._state:
            self._clear_pending()
            return self._state
        if self._pending_state is not target:
            self._pending_state = target
            self._pending_count = 0
        self._pending_count += 1
        if self._pending_count >= self.confirm_frames:
            self._state = target
            self._clear_pending()
        return self._state

    def update(self, arm: Optional[dict]) -> ArmRaiseState:
        """喂入一帧的 ArmRaiseFeatures 输出，返回举手状态。

        参数:
            arm: features.ArmRaiseFeatures.update() 的返回值（读
                 'arm_raise_deg' / 'arm_raise_valid'）；None 表示无人/无数据。

        返回:
            ArmRaiseState：DOWN / RAISED / UNKNOWN。
        """
        if arm is None or not arm.get('arm_raise_valid'):
            self._state = ArmRaiseState.UNKNOWN
            self._clear_pending()
            return self._state
        deg = arm.get('arm_raise_deg')
        if deg is None:
            self._state = ArmRaiseState.UNKNOWN
            self._clear_pending()
            return self._state

        if deg <= self.enter_deg:
            return self._commit(ArmRaiseState.RAISED)
        if deg >= self.exit_deg:
            return self._commit(ArmRaiseState.DOWN)
        # 迟滞带：保持当前状态；还没有历史状态 → 兜底 DOWN（没证据不判举手）
        self._clear_pending()
        if self._state is ArmRaiseState.UNKNOWN:
            self._state = ArmRaiseState.DOWN
        return self._state


class HeadTiltState(enum.Enum):
    """仰头状态。"""
    NEUTRAL = "NEUTRAL"
    TILTED_BACK = "TILTED_BACK"
    UNKNOWN = "UNKNOWN"


class HeadTiltDecision:
    """仰头判断：头俯仰代理角 >= enter_deg 进 TILTED_BACK，<= exit_deg 回 NEUTRAL。

    角度方向：features.HeadTiltFeatures.head_pitch_deg，**越大越仰头**
    （正 = 鼻在耳之上；负 = 低头）。迟滞带语义与 ArmRaiseDecision 完全相同
    （UNKNOWN 时兜底 NEUTRAL）。
    ⚠ enter/exit 默认值同样是**占位值**：中立基线取决于机位俯仰角与个人头型，
    先跑起来看数再定（--head-tilt-enter / --head-tilt-exit 可现场覆盖）。

    局限随 features 层一起成立，别在这里当独立结论用：只在 45° 前侧机位有意义
    （正面机位测不到），且不补偿转头（yaw）—— 见 features.HeadTiltFeatures 的
    "近似在哪"一节。

    接口约定（保持稳定，别改签名）：
        HeadTiltDecision(enter_deg=20.0, exit_deg=10.0, confirm_frames=1)
        update(head: Optional[dict]) -> HeadTiltState
        state（只读属性）
    """

    def __init__(self, enter_deg: float = 20.0, exit_deg: float = 10.0,
                 confirm_frames: int = 1) -> None:
        """
        参数:
            enter_deg: 俯仰角 >= 此值判 TILTED_BACK（度；越大越仰头）。
            exit_deg:  俯仰角 <= 此值判 NEUTRAL（度）。必须小于 enter_deg，
                       两者之间就是迟滞带。
            confirm_frames: 目标状态需要连续保持多少帧才提交（>= 1；
                       默认 1 = 旧行为，无额外延迟）。
        """
        if exit_deg >= enter_deg:
            raise ValueError("exit_deg 必须小于 enter_deg（否则没有迟滞带）")
        if confirm_frames < 1:
            raise ValueError("confirm_frames 必须 >= 1")
        self.enter_deg = enter_deg
        self.exit_deg = exit_deg
        self.confirm_frames = int(confirm_frames)
        self._state = HeadTiltState.UNKNOWN
        self._pending_state = None
        self._pending_count = 0

    @property
    def state(self) -> HeadTiltState:
        return self._state

    def _clear_pending(self) -> None:
        self._pending_state = None
        self._pending_count = 0

    def _commit(self, target: HeadTiltState) -> HeadTiltState:
        """目标状态与当前不同时做多帧确认；确认够了才真正切换。"""
        if target is self._state:
            self._clear_pending()
            return self._state
        if self._pending_state is not target:
            self._pending_state = target
            self._pending_count = 0
        self._pending_count += 1
        if self._pending_count >= self.confirm_frames:
            self._state = target
            self._clear_pending()
        return self._state

    def update(self, head: Optional[dict]) -> HeadTiltState:
        """喂入一帧的 HeadTiltFeatures 输出，返回仰头状态。

        参数:
            head: features.HeadTiltFeatures.update() 的返回值（读
                  'head_pitch_deg' / 'head_pitch_valid'）；None 表示无人/无数据。

        返回:
            HeadTiltState：NEUTRAL / TILTED_BACK / UNKNOWN。
        """
        if head is None or not head.get('head_pitch_valid'):
            self._state = HeadTiltState.UNKNOWN
            self._clear_pending()
            return self._state
        deg = head.get('head_pitch_deg')
        if deg is None:
            self._state = HeadTiltState.UNKNOWN
            self._clear_pending()
            return self._state

        if deg >= self.enter_deg:
            return self._commit(HeadTiltState.TILTED_BACK)
        if deg <= self.exit_deg:
            return self._commit(HeadTiltState.NEUTRAL)
        # 迟滞带：保持当前状态；还没有历史状态 → 兜底 NEUTRAL（没证据不判仰头）
        self._clear_pending()
        if self._state is HeadTiltState.UNKNOWN:
            self._state = HeadTiltState.NEUTRAL
        return self._state


# ---------------------------------------------------------------------------
# 2026-10-18 两批新增动作（共九个）的状态机
#
# 全部沿用 ArmRaiseDecision / HeadTiltDecision 的迟滞双阈值模式：
#   进入看 enter、退出要越过 exit、中间保持当前状态；
#   读数无效（valid=False / deg=None / features=None）→ UNKNOWN；
#   UNKNOWN 落在迟滞带里时兜底到"未触发"（没证据不判动作）。
#
# 方向分两类：
#   * 低角度触发（肘/膝/髋屈伸）：内角越小越屈曲 → deg <= enter 触发，>= exit 回伸直；
#   * 偏离量触发（躯干前屈、颈部侧屈、躯干旋转）：abs(deg) >= enter 触发，
#     abs(deg) <= exit 回中立（符号只表示往哪边，不参与触发）。
#
# ⚠ 所有 enter/exit 默认值都是**占位值，未标定**，量级参照举手/仰头现有占位值；
#   先跑起来看画面上的中立/动作读数，再把边界定到两档之间。
# ---------------------------------------------------------------------------

class TrunkFlexState(enum.Enum):
    """躯干前屈/后伸状态。"""
    NEUTRAL = "NEUTRAL"
    FLEXED = "FLEXED"
    UNKNOWN = "UNKNOWN"


class NeckLateralState(enum.Enum):
    """颈部侧屈状态。"""
    NEUTRAL = "NEUTRAL"
    TILTED = "TILTED"
    UNKNOWN = "UNKNOWN"


class ElbowFlexState(enum.Enum):
    """肘关节屈伸状态（EXTENDED=伸直，FLEXED=屈曲）。"""
    EXTENDED = "EXTENDED"
    FLEXED = "FLEXED"
    UNKNOWN = "UNKNOWN"


class TrunkRotationState(enum.Enum):
    """躯干旋转（肩髋线相对扭转）状态。"""
    NEUTRAL = "NEUTRAL"
    ROTATED = "ROTATED"
    UNKNOWN = "UNKNOWN"


class KneeFlexState(enum.Enum):
    """膝关节屈伸状态（EXTENDED=伸直，FLEXED=屈曲）。"""
    EXTENDED = "EXTENDED"
    FLEXED = "FLEXED"
    UNKNOWN = "UNKNOWN"


class HipFlexState(enum.Enum):
    """髋关节前屈状态（EXTENDED=伸直，FLEXED=屈曲）。"""
    EXTENDED = "EXTENDED"
    FLEXED = "FLEXED"
    UNKNOWN = "UNKNOWN"


class ShoulderAbductionState(enum.Enum):
    """肩关节外展状态（ADDUCTED=手臂垂在体侧，ABDUCTED=向侧面展开）。"""
    ADDUCTED = "ADDUCTED"
    ABDUCTED = "ABDUCTED"
    UNKNOWN = "UNKNOWN"


class PelvicTiltState(enum.Enum):
    """骨盆倾斜状态（NEUTRAL=双髋水平，TILTED=一侧髋抬高）。"""
    NEUTRAL = "NEUTRAL"
    TILTED = "TILTED"
    UNKNOWN = "UNKNOWN"


class ScapularSymmetryState(enum.Enum):
    """肩胛对称性状态（SYMMETRIC=双肩水平，ASYMMETRIC=一侧肩抬高）。"""
    SYMMETRIC = "SYMMETRIC"
    ASYMMETRIC = "ASYMMETRIC"
    UNKNOWN = "UNKNOWN"


def _read_valid_deg(features: Optional[dict], deg_key: str,
                    valid_key: str) -> Optional[float]:
    """从特征 dict 里读一个"有效才返回"的角度；无效/缺键/None → None。"""
    if features is None or not features.get(valid_key):
        return None
    return features.get(deg_key)


class _HysteresisDecisionBase:
    """两档迟滞状态机的公共骨架（子类只需声明状态成员与读数方向）。"""

    ACTIVE = None
    INACTIVE = None
    UNKNOWN = None

    def __init__(self, enter_deg: float, exit_deg: float,
                 confirm_frames: int = 1) -> None:
        if confirm_frames < 1:
            raise ValueError("confirm_frames 必须 >= 1")
        self.enter_deg = enter_deg
        self.exit_deg = exit_deg
        self.confirm_frames = int(confirm_frames)
        self._state = self.UNKNOWN
        self._pending_state = None
        self._pending_count = 0

    @property
    def state(self):
        return self._state

    def _read(self, features: Optional[dict]) -> Optional[float]:
        raise NotImplementedError

    def _is_active(self, deg: float) -> bool:
        raise NotImplementedError

    def _is_inactive(self, deg: float) -> bool:
        raise NotImplementedError

    def _clear_pending(self) -> None:
        self._pending_state = None
        self._pending_count = 0

    def _commit(self, target):
        """目标状态与当前不同时做多帧确认；确认够了才真正切换。"""
        if target is self._state:
            self._clear_pending()
            return self._state
        if self._pending_state is not target:
            self._pending_state = target
            self._pending_count = 0
        self._pending_count += 1
        if self._pending_count >= self.confirm_frames:
            self._state = target
            self._clear_pending()
        return self._state

    def update(self, features: Optional[dict]):
        """喂入一帧特征 dict，返回本动作状态（UNKNOWN = 本帧没有证据）。"""
        deg = self._read(features)
        if deg is None:
            # 丢帧立即失去证据：回到 UNKNOWN，并清掉确认计数（不跨丢帧累计）
            self._state = self.UNKNOWN
            self._clear_pending()
            return self._state
        if self._is_active(deg):
            return self._commit(self.ACTIVE)
        if self._is_inactive(deg):
            return self._commit(self.INACTIVE)
        # 迟滞带：保持当前状态，并清掉未完成的确认
        self._clear_pending()
        if self._state is self.UNKNOWN:
            # 迟滞带 + 没有历史 → 兜底"未触发"（没证据不判动作）
            self._state = self.INACTIVE
        return self._state


class _MagnitudeDecisionBase(_HysteresisDecisionBase):
    """按**绝对值**触发的迟滞状态机（躯干前屈 / 颈部侧屈 / 躯干旋转）。

    支持有符号读数：符号只表示往哪边，触发只看 abs(deg)。
        abs(deg) >= enter_deg -> ACTIVE
        abs(deg) <= exit_deg  -> INACTIVE
        中间                  -> 迟滞带，保持当前状态（UNKNOWN 兜底 INACTIVE）
    """

    def __init__(self, enter_deg: float, exit_deg: float,
                 confirm_frames: int = 1) -> None:
        if exit_deg >= enter_deg:
            raise ValueError("exit_deg 必须小于 enter_deg（否则没有迟滞带）")
        super().__init__(enter_deg, exit_deg, confirm_frames)

    def _is_active(self, deg: float) -> bool:
        return abs(deg) >= self.enter_deg

    def _is_inactive(self, deg: float) -> bool:
        return abs(deg) <= self.exit_deg


class _LowTriggerDecisionBase(_HysteresisDecisionBase):
    """按**低角度**触发的迟滞状态机（肘 / 膝 / 髋屈伸）。

    内角越小越屈曲：
        deg <= enter_deg -> ACTIVE（FLEXED）
        deg >= exit_deg  -> INACTIVE（EXTENDED）
        中间             -> 迟滞带，保持当前状态（UNKNOWN 兜底 INACTIVE）
    """

    def __init__(self, enter_deg: float, exit_deg: float,
                 confirm_frames: int = 1) -> None:
        if exit_deg <= enter_deg:
            raise ValueError("exit_deg 必须大于 enter_deg（否则没有迟滞带）")
        super().__init__(enter_deg, exit_deg, confirm_frames)

    def _is_active(self, deg: float) -> bool:
        return deg <= self.enter_deg

    def _is_inactive(self, deg: float) -> bool:
        return deg >= self.exit_deg


class TrunkFlexDecision(_MagnitudeDecisionBase):
    """躯干前屈/后伸判断：|trunk_flex_deg| >= enter_deg 进 FLEXED，<= exit_deg 回 NEUTRAL。

    消费 features.TrunkFlexFeatures 的有符号读数；符号（往画面左/右）只用于
    显示，触发只看偏离竖直轴的大小。⚠ enter=25 / exit=15 是**占位值，未标定**；
    45° 机位下中立基线不一定是 0，先看画面读数再定。
    """

    ACTIVE = TrunkFlexState.FLEXED
    INACTIVE = TrunkFlexState.NEUTRAL
    UNKNOWN = TrunkFlexState.UNKNOWN

    def __init__(self, enter_deg: float = 25.0, exit_deg: float = 15.0,
                 confirm_frames: int = 1) -> None:
        super().__init__(enter_deg, exit_deg, confirm_frames)

    def _read(self, features: Optional[dict]) -> Optional[float]:
        return _read_valid_deg(
            features, 'trunk_flex_deg', 'trunk_flex_valid')


class NeckLateralTiltDecision(_MagnitudeDecisionBase):
    """颈部侧屈判断：|neck_lateral_deg| >= enter_deg 进 TILTED，<= exit_deg 回 NEUTRAL。

    消费 features.NeckLateralTiltFeatures 的双耳连线角；触发只看侧倾大小。
    ⚠ enter=15 / exit=8 是**占位值，未标定**，中立基线可能不在 0。
    """

    ACTIVE = NeckLateralState.TILTED
    INACTIVE = NeckLateralState.NEUTRAL
    UNKNOWN = NeckLateralState.UNKNOWN

    def __init__(self, enter_deg: float = 15.0, exit_deg: float = 8.0,
                 confirm_frames: int = 1) -> None:
        super().__init__(enter_deg, exit_deg, confirm_frames)

    def _read(self, features: Optional[dict]) -> Optional[float]:
        return _read_valid_deg(
            features, 'neck_lateral_deg', 'neck_lateral_valid')


class _SideLowTriggerDecision(_LowTriggerDecisionBase):
    """单侧屈伸状态机公共骨架：按 side 读 `<prefix>_left/right_deg` 与 valid。"""

    PREFIX = ""

    _SIDE_KEY = {"L": "left", "R": "right"}

    def __init__(self, side: str = "L",
                 enter_deg: float = 100.0, exit_deg: float = 130.0,
                 confirm_frames: int = 1) -> None:
        if side not in self._SIDE_KEY:
            raise ValueError("side 必须是 'L' 或 'R'")
        self.side = side
        super().__init__(enter_deg, exit_deg, confirm_frames)

    def _read(self, features: Optional[dict]) -> Optional[float]:
        key = self._SIDE_KEY[self.side]
        return _read_valid_deg(
            features, f'{self.PREFIX}_{key}_deg',
            f'{self.PREFIX}_{key}_valid')


class _SideHighTriggerDecision(_MagnitudeDecisionBase):
    """单侧高角度触发状态机公共骨架（肩关节外展）。

    按 side 读 `<prefix>_left/right_deg` 与 `<prefix>_left/right_valid`；
    外展角 0 = 体侧、90 = 水平、180 = 过顶：
        |deg| >= enter_deg -> ACTIVE（ABDUCTED）
        |deg| <= exit_deg  -> INACTIVE（ADDUCTED）
        中间               -> 迟滞带，保持当前状态（UNKNOWN 兜底 INACTIVE）
    """

    PREFIX = ""

    _SIDE_KEY = {"L": "left", "R": "right"}

    def __init__(self, side: str = "L",
                 enter_deg: float = 60.0, exit_deg: float = 40.0,
                 confirm_frames: int = 1) -> None:
        if side not in self._SIDE_KEY:
            raise ValueError("side 必须是 'L' 或 'R'")
        self.side = side
        super().__init__(enter_deg, exit_deg, confirm_frames)

    def _read(self, features: Optional[dict]) -> Optional[float]:
        key = self._SIDE_KEY[self.side]
        return _read_valid_deg(
            features, f'{self.PREFIX}_{key}_deg',
            f'{self.PREFIX}_{key}_valid')


class ElbowFlexDecision(_SideLowTriggerDecision):
    """单侧肘关节屈伸判断：肘角 <= enter_deg 进 FLEXED，>= exit_deg 回 EXTENDED。

    左右各建一个实例（side='L'/'R'），**不合并**。⚠ enter=100 / exit=130 是
    **占位值，未标定**；直立/自然下垂时肘角接近 180，屈肘到 90 左右才应触发。
    """

    ACTIVE = ElbowFlexState.FLEXED
    INACTIVE = ElbowFlexState.EXTENDED
    UNKNOWN = ElbowFlexState.UNKNOWN

    PREFIX = 'elbow_flex'

    def __init__(self, side: str = "L",
                 enter_deg: float = 100.0, exit_deg: float = 130.0,
                 confirm_frames: int = 1) -> None:
        super().__init__(side, enter_deg, exit_deg, confirm_frames)


class TrunkRotationDecision(_MagnitudeDecisionBase):
    """躯干旋转判断：|trunk_rotation_deg| >= enter_deg 进 ROTATED，<= exit_deg 回 NEUTRAL。

    消费 features.TrunkRotationFeatures 的肩髋线夹角差；它量的是 2D 像面内的
    肩髋相对扭转，不是真实 yaw。⚠ enter=15 / exit=8 是**占位值，未标定**。
    """

    ACTIVE = TrunkRotationState.ROTATED
    INACTIVE = TrunkRotationState.NEUTRAL
    UNKNOWN = TrunkRotationState.UNKNOWN

    def __init__(self, enter_deg: float = 15.0, exit_deg: float = 8.0,
                 confirm_frames: int = 1) -> None:
        super().__init__(enter_deg, exit_deg, confirm_frames)

    def _read(self, features: Optional[dict]) -> Optional[float]:
        return _read_valid_deg(
            features, 'trunk_rotation_deg', 'trunk_rotation_valid')


class KneeFlexDecision(_SideLowTriggerDecision):
    """单侧膝关节屈伸判断：膝角 <= enter_deg 进 FLEXED，>= exit_deg 回 EXTENDED。

    左右各建一个实例（side='L'/'R'），**不合并**。⚠ enter=120 / exit=150 是
    **占位值，未标定**；站立伸直时膝角接近 180。
    """

    ACTIVE = KneeFlexState.FLEXED
    INACTIVE = KneeFlexState.EXTENDED
    UNKNOWN = KneeFlexState.UNKNOWN

    PREFIX = 'knee_flex'

    def __init__(self, side: str = "L",
                 enter_deg: float = 120.0, exit_deg: float = 150.0,
                 confirm_frames: int = 1) -> None:
        super().__init__(side, enter_deg, exit_deg, confirm_frames)


class HipFlexDecision(_SideLowTriggerDecision):
    """单侧髋关节前屈判断：髋角 <= enter_deg 进 FLEXED，>= exit_deg 回 EXTENDED。

    左右各建一个实例（side='L'/'R'），**不合并**。⚠ enter=120 / exit=150 是
    **占位值，未标定**；直立时肩-髋-膝接近一直线（180）。
    """

    ACTIVE = HipFlexState.FLEXED
    INACTIVE = HipFlexState.EXTENDED
    UNKNOWN = HipFlexState.UNKNOWN

    PREFIX = 'hip_flex'

    def __init__(self, side: str = "L",
                 enter_deg: float = 120.0, exit_deg: float = 150.0,
                 confirm_frames: int = 1) -> None:
        super().__init__(side, enter_deg, exit_deg, confirm_frames)


class ShoulderAbductionDecision(_SideHighTriggerDecision):
    """单侧肩关节外展判断：外展角 >= enter_deg 进 ABDUCTED，<= exit_deg 回 ADDUCTED。

    左右各建一个实例（side='L'/'R'），**不合并**。⚠ enter=60 / exit=40 是
    **占位值，未标定**；手臂自然下垂时接近 0°，平举约 90°。
    """

    ACTIVE = ShoulderAbductionState.ABDUCTED
    INACTIVE = ShoulderAbductionState.ADDUCTED
    UNKNOWN = ShoulderAbductionState.UNKNOWN

    PREFIX = 'shoulder_abduction'

    def __init__(self, side: str = "L",
                 enter_deg: float = 60.0, exit_deg: float = 40.0,
                 confirm_frames: int = 1) -> None:
        super().__init__(side, enter_deg, exit_deg, confirm_frames)


class PelvicTiltDecision(_MagnitudeDecisionBase):
    """骨盆倾斜判断：|pelvic_tilt_deg| >= enter_deg 进 TILTED，<= exit_deg 回 NEUTRAL。

    消费 features.PelvicTiltFeatures 的双髋连线角；符号只表示哪侧髋更高，触发
    只看倾斜大小。⚠ enter=10 / exit=5 是**占位值，未标定**；中立站立时不一定
    正好 0°，先看画面读数再定。
    """

    ACTIVE = PelvicTiltState.TILTED
    INACTIVE = PelvicTiltState.NEUTRAL
    UNKNOWN = PelvicTiltState.UNKNOWN

    def __init__(self, enter_deg: float = 10.0, exit_deg: float = 5.0,
                 confirm_frames: int = 1) -> None:
        super().__init__(enter_deg, exit_deg, confirm_frames)

    def _read(self, features: Optional[dict]) -> Optional[float]:
        return _read_valid_deg(
            features, 'pelvic_tilt_deg', 'pelvic_tilt_valid')


class ScapularSymmetryDecision(_MagnitudeDecisionBase):
    """肩胛对称性判断：|scapular_symmetry_deg| >= enter_deg 进 ASYMMETRIC，
    <= exit_deg 回 SYMMETRIC。

    消费 features.ScapularSymmetryFeatures 的双肩连线角；触发只看肩线倾斜大小。
    ⚠ enter=10 / exit=5 是**占位值，未标定**；肩线倾斜也可能来自侧弯/机位倾斜，
    不能直接当肩胛问题。
    """

    ACTIVE = ScapularSymmetryState.ASYMMETRIC
    INACTIVE = ScapularSymmetryState.SYMMETRIC
    UNKNOWN = ScapularSymmetryState.UNKNOWN

    def __init__(self, enter_deg: float = 10.0, exit_deg: float = 5.0,
                 confirm_frames: int = 1) -> None:
        super().__init__(enter_deg, exit_deg, confirm_frames)

    def _read(self, features: Optional[dict]) -> Optional[float]:
        return _read_valid_deg(
            features, 'scapular_symmetry_deg', 'scapular_symmetry_valid')


# ---------------------------------------------------------------------------
# 自测：纯 stdlib 合成数据（main_action_demo.py --selftest 汇总调用）
# ---------------------------------------------------------------------------

def _arm(deg, valid=True):
    """构造 ArmRaiseFeatures.update() 的最小返回值（只带判定要读的两个键）。"""
    return {'arm_raise_deg': deg, 'arm_raise_valid': valid}


def _head(deg, valid=True):
    """构造 HeadTiltFeatures.update() 的最小返回值（只带判定要读的两个键）。"""
    return {'head_pitch_deg': deg, 'head_pitch_valid': valid}


def selftest_arm_raise_decision() -> None:
    """合成数据自测 ArmRaiseDecision：两档、边界含等号、迟滞带保持、边界不抖、
    UNKNOWN 兜底、非法参数。"""
    # 1) 边界：<= enter 判 RAISED、>= exit 判 DOWN（都含等号）
    d = ArmRaiseDecision(enter_deg=45.0, exit_deg=60.0)
    assert d.update(_arm(45.0)) is ArmRaiseState.RAISED, "=enter 应判 RAISED"
    assert d.update(_arm(60.0)) is ArmRaiseState.DOWN, "=exit 应判 DOWN"
    assert d.update(_arm(44.9)) is ArmRaiseState.RAISED
    assert d.update(_arm(60.1)) is ArmRaiseState.DOWN

    # 2) 迟滞带保持：进档后落到带内不掉档，越过 exit 才掉；反向同理
    d = ArmRaiseDecision(enter_deg=45.0, exit_deg=60.0)
    for deg in (30.0, 50.0, 52.0, 58.0, 59.9):
        assert d.update(_arm(deg)) is ArmRaiseState.RAISED, \
            f"{deg}° 应在迟滞带内保持 RAISED"
    assert d.update(_arm(60.0)) is ArmRaiseState.DOWN
    for deg in (120.0, 58.0, 50.0, 46.0):
        assert d.update(_arm(deg)) is ArmRaiseState.DOWN, \
            f"{deg}° 应保持 DOWN"

    # 3) 阈值带内来回摆动不抖出状态（迟滞的目的）
    d = ArmRaiseDecision(enter_deg=45.0, exit_deg=60.0)
    states = [d.update(_arm(v)).value
              for v in (40.0, 55.0, 42.0, 58.0, 44.0, 57.0)]
    assert set(states) == {"RAISED"}, f"带内来回不应抖出 RAISED：{states}"

    # 4) 带内 + 无历史 → 兜底 DOWN（没证据不判举手）
    d = ArmRaiseDecision(enter_deg=45.0, exit_deg=60.0)
    assert d.update(_arm(55.0)) is ArmRaiseState.DOWN

    # 5) UNKNOWN：无人 / 该帧没算出抬升角 / valid=False，且**清掉历史状态**
    #    （丢帧后回来要重新建立，不沿用丢帧前的 RAISED）
    d = ArmRaiseDecision()
    assert d.update(_arm(20.0)) is ArmRaiseState.RAISED
    assert d.update(_arm(None)) is ArmRaiseState.UNKNOWN
    assert d.update(None) is ArmRaiseState.UNKNOWN
    assert d.update(_arm(30.0, valid=False)) is ArmRaiseState.UNKNOWN
    assert d.update(_arm(50.0)) is ArmRaiseState.DOWN, "丢帧后带内应兜底 DOWN"

    # 6) 非法参数（exit 必须大于 enter）
    for kwargs in (dict(enter_deg=60.0, exit_deg=45.0),
                   dict(enter_deg=45.0, exit_deg=45.0)):
        try:
            ArmRaiseDecision(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kwargs} 应抛 ValueError")

    print("  selftest_arm_raise_decision: OK")


def selftest_head_tilt_decision() -> None:
    """合成数据自测 HeadTiltDecision：两档、边界含等号、迟滞带保持、UNKNOWN、
    低头（负角）不误报、非法参数。"""
    # 1) 边界：>= enter 判仰头、<= exit 判 NEUTRAL（都含等号）
    d = HeadTiltDecision(enter_deg=20.0, exit_deg=10.0)
    assert d.update(_head(20.0)) is HeadTiltState.TILTED_BACK, "=enter 应判仰头"
    assert d.update(_head(10.0)) is HeadTiltState.NEUTRAL, "=exit 应判 NEUTRAL"
    assert d.update(_head(25.0)) is HeadTiltState.TILTED_BACK
    assert d.update(_head(9.9)) is HeadTiltState.NEUTRAL

    # 2) 迟滞带保持（两个方向都要保持住）
    d = HeadTiltDecision(enter_deg=20.0, exit_deg=10.0)
    for deg in (20.0, 15.0, 12.0, 10.1):
        assert d.update(_head(deg)) is HeadTiltState.TILTED_BACK, \
            f"{deg}° 应在迟滞带内保持仰头"
    assert d.update(_head(10.0)) is HeadTiltState.NEUTRAL
    for deg in (0.0, 10.1, 15.0, 19.9):
        assert d.update(_head(deg)) is HeadTiltState.NEUTRAL, \
            f"{deg}° 应保持 NEUTRAL"

    # 3) 低头（负角）不触发；带内 + 无历史 → 兜底 NEUTRAL
    d = HeadTiltDecision(enter_deg=20.0, exit_deg=10.0)
    assert [d.update(_head(v)).value for v in (-30.0, 0.0, 12.0, 19.0)] == \
        ["NEUTRAL"] * 4

    # 4) UNKNOWN：无人 / 没算出角度 / valid=False
    d = HeadTiltDecision()
    assert d.update(_head(30.0)) is HeadTiltState.TILTED_BACK
    assert d.update(_head(None)) is HeadTiltState.UNKNOWN
    assert d.update(None) is HeadTiltState.UNKNOWN
    assert d.update(_head(30.0, valid=False)) is HeadTiltState.UNKNOWN

    # 5) 非法参数（exit 必须小于 enter）
    for kwargs in (dict(enter_deg=10.0, exit_deg=20.0),
                   dict(enter_deg=15.0, exit_deg=15.0)):
        try:
            HeadTiltDecision(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kwargs} 应抛 ValueError")

    print("  selftest_head_tilt_decision: OK")


def _side_feats(prefix: str, l_deg, r_deg,
                l_valid: bool = True, r_valid: bool = True) -> dict:
    """构造左右独立的特征 dict（只带判定要读的键，够用即可）。"""
    return {f'{prefix}_left_deg': l_deg,
            f'{prefix}_left_valid': l_valid,
            f'{prefix}_right_deg': r_deg,
            f'{prefix}_right_valid': r_valid}


def _assert_raises_value_error(fn, msg: str) -> None:
    try:
        fn()
    except ValueError:
        pass
    else:
        raise AssertionError(msg)


def _selftest_magnitude_decision(name: str, cls, active, inactive, unknown,
                                 deg_key: str, valid_key: str) -> None:
    """绝对值触发状态机的通用自测（躯干前屈 / 颈部侧屈 / 躯干旋转）。"""
    # 1) 边界含等号：|deg| >= enter 触发、<= exit 回中立
    d = cls(enter_deg=20.0, exit_deg=10.0)
    assert d.update({deg_key: 20.0, valid_key: True}) is active
    assert d.update({deg_key: 10.0, valid_key: True}) is inactive
    assert d.update({deg_key: 25.0, valid_key: True}) is active
    assert d.update({deg_key: 9.9, valid_key: True}) is inactive

    # 2) 有符号读数：负值同样按绝对值触发（符号只表示往哪边）
    d = cls(enter_deg=20.0, exit_deg=10.0)
    assert d.update({deg_key: -20.0, valid_key: True}) is active
    assert d.update({deg_key: -10.0, valid_key: True}) is inactive

    # 3) 迟滞带保持（两个方向）
    d = cls(enter_deg=20.0, exit_deg=10.0)
    for v in (20.0, 15.0, 12.0, 10.1):
        assert d.update({deg_key: v, valid_key: True}) is active, \
            f"{name}: {v}° 应在迟滞带内保持触发"
    assert d.update({deg_key: 10.0, valid_key: True}) is inactive
    for v in (0.0, 10.1, 15.0, 19.9):
        assert d.update({deg_key: v, valid_key: True}) is inactive, \
            f"{name}: {v}° 应保持未触发"

    # 4) 带内 + 无历史 → 兜底未触发
    d = cls(enter_deg=20.0, exit_deg=10.0)
    assert d.update({deg_key: 15.0, valid_key: True}) is inactive

    # 5) UNKNOWN：无人 / valid=False / deg=None
    d = cls(enter_deg=20.0, exit_deg=10.0)
    assert d.update({deg_key: 30.0, valid_key: True}) is active
    assert d.update(None) is unknown
    assert d.update({deg_key: 30.0, valid_key: False}) is unknown
    assert d.update({deg_key: None, valid_key: True}) is unknown

    # 6) 非法参数（exit 必须小于 enter）
    _assert_raises_value_error(
        lambda: cls(enter_deg=10.0, exit_deg=20.0),
        f"{name}: exit > enter 应抛 ValueError")
    _assert_raises_value_error(
        lambda: cls(enter_deg=15.0, exit_deg=15.0),
        f"{name}: exit == enter 应抛 ValueError")


def _selftest_low_trigger_side_decision(
        name: str, cls, prefix: str, active, inactive, unknown,
        enter_deg: float = 100.0, exit_deg: float = 130.0) -> None:
    """单侧低角度触发状态机的通用自测（肘 / 膝 / 髋屈伸）。"""
    # 1) 边界含等号：deg <= enter 触发、>= exit 回伸直
    d = cls(side='L', enter_deg=enter_deg, exit_deg=exit_deg)
    assert d.update(_side_feats(prefix, enter_deg, 180.0)) is active
    assert d.update(_side_feats(prefix, exit_deg, 180.0)) is inactive
    assert d.update(_side_feats(prefix, enter_deg - 5, 180.0)) is active
    assert d.update(_side_feats(prefix, exit_deg + 5, 180.0)) is inactive

    # 2) 迟滞带保持（两个方向）
    d = cls(side='L', enter_deg=enter_deg, exit_deg=exit_deg)
    mid = (enter_deg + exit_deg) / 2.0
    for v in (enter_deg, enter_deg + 5, mid, exit_deg - 1):
        assert d.update(_side_feats(prefix, v, 180.0)) is active, \
            f"{name}: {v}° 应在迟滞带内保持 FLEXED"
    assert d.update(_side_feats(prefix, exit_deg, 180.0)) is inactive
    for v in (180.0, exit_deg - 1, mid, enter_deg + 1):
        assert d.update(_side_feats(prefix, v, 180.0)) is inactive, \
            f"{name}: {v}° 应保持 EXTENDED"

    # 3) 带内 + 无历史 → 兜底未触发（EXTENDED）
    d = cls(side='L', enter_deg=enter_deg, exit_deg=exit_deg)
    assert d.update(_side_feats(prefix, mid, 180.0)) is inactive

    # 4) UNKNOWN：无人 / 该侧 valid=False / deg=None
    d = cls(side='L', enter_deg=enter_deg, exit_deg=exit_deg)
    assert d.update(_side_feats(prefix, enter_deg - 5, 180.0)) is active
    assert d.update(None) is unknown
    assert d.update(_side_feats(prefix, enter_deg - 5, 180.0,
                                l_valid=False)) is unknown
    assert d.update(_side_feats(prefix, None, 180.0)) is unknown

    # 5) 左右两套状态机互不影响：同一帧 L 屈曲、R 伸直
    dl = cls(side='L', enter_deg=enter_deg, exit_deg=exit_deg)
    dr = cls(side='R', enter_deg=enter_deg, exit_deg=exit_deg)
    f = _side_feats(prefix, enter_deg - 5, 180.0)
    assert dl.update(f) is active and dr.update(f) is inactive, \
        f"{name}: 左右状态机不应互相牵连"

    # 6) 非法参数
    _assert_raises_value_error(
        lambda: cls(side='X'), f"{name}: 非法 side 应抛 ValueError")
    _assert_raises_value_error(
        lambda: cls(side='L', enter_deg=exit_deg + 1,
                    exit_deg=exit_deg),
        f"{name}: exit <= enter 应抛 ValueError")


def _selftest_side_high_trigger_decision(
        name: str, cls, prefix: str, active, inactive, unknown,
        enter_deg: float = 60.0, exit_deg: float = 40.0) -> None:
    """单侧高角度触发状态机的通用自测（肩关节外展）。"""
    # 1) 边界含等号：deg >= enter 触发、<= exit 回未触发
    d = cls(side='L', enter_deg=enter_deg, exit_deg=exit_deg)
    assert d.update(_side_feats(prefix, enter_deg, 0.0)) is active
    assert d.update(_side_feats(prefix, exit_deg, 0.0)) is inactive
    assert d.update(_side_feats(prefix, enter_deg + 5, 0.0)) is active
    assert d.update(_side_feats(prefix, exit_deg - 5, 0.0)) is inactive

    # 2) 迟滞带保持（两个方向）
    d = cls(side='L', enter_deg=enter_deg, exit_deg=exit_deg)
    mid = (enter_deg + exit_deg) / 2.0
    for v in (enter_deg, enter_deg - 5, mid, exit_deg + 1):
        assert d.update(_side_feats(prefix, v, 0.0)) is active, \
            f"{name}: {v}° 应在迟滞带内保持 ABDUCTED"
    assert d.update(_side_feats(prefix, exit_deg, 0.0)) is inactive
    for v in (0.0, exit_deg + 1, mid, enter_deg - 1):
        assert d.update(_side_feats(prefix, v, 0.0)) is inactive, \
            f"{name}: {v}° 应保持 ADDUCTED"

    # 3) 带内 + 无历史 → 兜底未触发（ADDUCTED）
    d = cls(side='L', enter_deg=enter_deg, exit_deg=exit_deg)
    assert d.update(_side_feats(prefix, mid, 0.0)) is inactive

    # 4) UNKNOWN：无人 / 该侧 valid=False / deg=None
    d = cls(side='L', enter_deg=enter_deg, exit_deg=exit_deg)
    assert d.update(_side_feats(prefix, enter_deg + 5, 0.0)) is active
    assert d.update(None) is unknown
    assert d.update(_side_feats(prefix, enter_deg + 5, 0.0,
                                l_valid=False)) is unknown
    assert d.update(_side_feats(prefix, None, 0.0)) is unknown

    # 5) 左右两套状态机互不影响
    dl = cls(side='L', enter_deg=enter_deg, exit_deg=exit_deg)
    dr = cls(side='R', enter_deg=enter_deg, exit_deg=exit_deg)
    f = _side_feats(prefix, enter_deg + 5, 0.0)
    assert dl.update(f) is active and dr.update(f) is inactive, \
        f"{name}: 左右状态机不应互相牵连"

    # 6) 非法参数（exit 必须小于 enter；side 只能是 L/R）
    _assert_raises_value_error(
        lambda: cls(side='X'), f"{name}: 非法 side 应抛 ValueError")
    _assert_raises_value_error(
        lambda: cls(side='L', enter_deg=exit_deg, exit_deg=enter_deg),
        f"{name}: exit >= enter 应抛 ValueError")


def selftest_trunk_flex_decision() -> None:
    """自测 TrunkFlexDecision：绝对值触发、迟滞带、UNKNOWN、非法参数。"""
    _selftest_magnitude_decision(
        "trunk_flex", TrunkFlexDecision,
        TrunkFlexState.FLEXED, TrunkFlexState.NEUTRAL,
        TrunkFlexState.UNKNOWN, 'trunk_flex_deg', 'trunk_flex_valid')
    print("  selftest_trunk_flex_decision: OK")


def selftest_neck_lateral_decision() -> None:
    """自测 NeckLateralTiltDecision：绝对值触发、迟滞带、UNKNOWN、非法参数。"""
    _selftest_magnitude_decision(
        "neck_lateral", NeckLateralTiltDecision,
        NeckLateralState.TILTED, NeckLateralState.NEUTRAL,
        NeckLateralState.UNKNOWN, 'neck_lateral_deg', 'neck_lateral_valid')
    print("  selftest_neck_lateral_decision: OK")


def selftest_elbow_flex_decision() -> None:
    """自测 ElbowFlexDecision：低角度触发、迟滞带、左右独立、UNKNOWN。"""
    _selftest_low_trigger_side_decision(
        "elbow_flex", ElbowFlexDecision, 'elbow_flex',
        ElbowFlexState.FLEXED, ElbowFlexState.EXTENDED,
        ElbowFlexState.UNKNOWN)
    print("  selftest_elbow_flex_decision: OK")


def selftest_trunk_rotation_decision() -> None:
    """自测 TrunkRotationDecision：绝对值触发、迟滞带、UNKNOWN、非法参数。"""
    _selftest_magnitude_decision(
        "trunk_rotation", TrunkRotationDecision,
        TrunkRotationState.ROTATED, TrunkRotationState.NEUTRAL,
        TrunkRotationState.UNKNOWN,
        'trunk_rotation_deg', 'trunk_rotation_valid')
    print("  selftest_trunk_rotation_decision: OK")


def selftest_knee_flex_decision() -> None:
    """自测 KneeFlexDecision：低角度触发、迟滞带、左右独立、UNKNOWN。"""
    _selftest_low_trigger_side_decision(
        "knee_flex", KneeFlexDecision, 'knee_flex',
        KneeFlexState.FLEXED, KneeFlexState.EXTENDED,
        KneeFlexState.UNKNOWN, enter_deg=120.0, exit_deg=150.0)
    print("  selftest_knee_flex_decision: OK")


def selftest_hip_flex_decision() -> None:
    """自测 HipFlexDecision：低角度触发、迟滞带、左右独立、UNKNOWN。"""
    _selftest_low_trigger_side_decision(
        "hip_flex", HipFlexDecision, 'hip_flex',
        HipFlexState.FLEXED, HipFlexState.EXTENDED,
        HipFlexState.UNKNOWN, enter_deg=120.0, exit_deg=150.0)
    print("  selftest_hip_flex_decision: OK")


def selftest_shoulder_abduction_decision() -> None:
    """自测 ShoulderAbductionDecision：高角度触发、迟滞带、左右独立、UNKNOWN。"""
    _selftest_side_high_trigger_decision(
        "shoulder_abduction", ShoulderAbductionDecision, 'shoulder_abduction',
        ShoulderAbductionState.ABDUCTED, ShoulderAbductionState.ADDUCTED,
        ShoulderAbductionState.UNKNOWN, enter_deg=60.0, exit_deg=40.0)
    print("  selftest_shoulder_abduction_decision: OK")


def selftest_pelvic_tilt_decision() -> None:
    """自测 PelvicTiltDecision：绝对值触发、迟滞带、UNKNOWN、非法参数。"""
    _selftest_magnitude_decision(
        "pelvic_tilt", PelvicTiltDecision,
        PelvicTiltState.TILTED, PelvicTiltState.NEUTRAL,
        PelvicTiltState.UNKNOWN, 'pelvic_tilt_deg', 'pelvic_tilt_valid')
    print("  selftest_pelvic_tilt_decision: OK")


def selftest_scapular_symmetry_decision() -> None:
    """自测 ScapularSymmetryDecision：绝对值触发、迟滞带、UNKNOWN、非法参数。"""
    _selftest_magnitude_decision(
        "scapular_symmetry", ScapularSymmetryDecision,
        ScapularSymmetryState.ASYMMETRIC, ScapularSymmetryState.SYMMETRIC,
        ScapularSymmetryState.UNKNOWN,
        'scapular_symmetry_deg', 'scapular_symmetry_valid')
    print("  selftest_scapular_symmetry_decision: OK")


def selftest_confirm_frames() -> None:
    """自测多帧确认：默认 1 帧 = 旧行为；2 帧首帧不翻、第二帧才翻；
    迟滞带/丢帧打断计数；左右独立类与基类骨架都生效；非法参数报错。"""
    # 1) confirm_frames=1 与旧行为逐帧一致
    d = ArmRaiseDecision(enter_deg=45.0, exit_deg=60.0, confirm_frames=1)
    assert d.update(_arm(30.0)) is ArmRaiseState.RAISED
    assert d.update(_arm(60.0)) is ArmRaiseState.DOWN

    # 2) confirm_frames=2：首帧返回旧状态（UNKNOWN），第二帧才提交
    d = ArmRaiseDecision(enter_deg=45.0, exit_deg=60.0, confirm_frames=2)
    assert d.update(_arm(30.0)) is ArmRaiseState.UNKNOWN, "单帧不应翻状态"
    assert d.update(_arm(30.0)) is ArmRaiseState.RAISED, "连续两帧才提交"
    assert d.update(_arm(30.0)) is ArmRaiseState.RAISED, "保持后不再延迟"
    # 退出方向同样要两帧
    assert d.update(_arm(120.0)) is ArmRaiseState.RAISED, "首帧退出不生效"
    assert d.update(_arm(120.0)) is ArmRaiseState.DOWN

    # 3) 迟滞带打断未完成的确认（计数重新开始）
    d = ArmRaiseDecision(enter_deg=45.0, exit_deg=60.0, confirm_frames=2)
    assert d.update(_arm(30.0)) is ArmRaiseState.UNKNOWN
    assert d.update(_arm(55.0)) is ArmRaiseState.DOWN, "带内 UNKNOWN 兜底 DOWN"
    assert d.update(_arm(30.0)) is ArmRaiseState.DOWN, "计数已清，重新数"
    assert d.update(_arm(30.0)) is ArmRaiseState.RAISED

    # 4) 丢帧立即 UNKNOWN，并清掉计数
    d = ArmRaiseDecision(enter_deg=45.0, exit_deg=60.0, confirm_frames=2)
    d.update(_arm(30.0))
    assert d.update(_arm(None)) is ArmRaiseState.UNKNOWN
    assert d.update(_arm(30.0)) is ArmRaiseState.UNKNOWN, "丢帧后确认重新计数"
    assert d.update(_arm(30.0)) is ArmRaiseState.RAISED

    # 5) 基类骨架（绝对值触发）同样支持
    d = TrunkFlexDecision(enter_deg=25.0, exit_deg=15.0, confirm_frames=2)
    f = {'trunk_flex_deg': 30.0, 'trunk_flex_valid': True}
    assert d.update(f) is TrunkFlexState.UNKNOWN
    assert d.update(f) is TrunkFlexState.FLEXED
    f0 = {'trunk_flex_deg': 0.0, 'trunk_flex_valid': True}
    assert d.update(f0) is TrunkFlexState.FLEXED
    assert d.update(f0) is TrunkFlexState.NEUTRAL

    # 6) 单侧类（side）同样支持
    d = ElbowFlexDecision(side='L', confirm_frames=2)
    f = {'elbow_flex_left_deg': 80.0, 'elbow_flex_left_valid': True,
         'elbow_flex_right_deg': 180.0, 'elbow_flex_right_valid': True}
    assert d.update(f) is ElbowFlexState.UNKNOWN
    assert d.update(f) is ElbowFlexState.FLEXED

    # 7) 仰头独立实现同样支持
    d = HeadTiltDecision(confirm_frames=2)
    assert d.update(_head(30.0)) is HeadTiltState.UNKNOWN
    assert d.update(_head(30.0)) is HeadTiltState.TILTED_BACK

    # 8) 非法 confirm_frames
    for bad in (0, -1):
        for cls, kwargs in ((ArmRaiseDecision, {}), (HeadTiltDecision, {}),
                            (TrunkFlexDecision, {}),
                            (ElbowFlexDecision, {'side': 'L'})):
            _assert_raises_value_error(
                lambda c=cls, k=kwargs, b=bad: c(confirm_frames=b, **k),
                f"confirm_frames={bad} 应抛 ValueError")

    print("  selftest_confirm_frames: OK")


if __name__ == "__main__":
    selftest_arm_raise_decision()
    selftest_head_tilt_decision()
    selftest_trunk_flex_decision()
    selftest_neck_lateral_decision()
    selftest_elbow_flex_decision()
    selftest_trunk_rotation_decision()
    selftest_knee_flex_decision()
    selftest_hip_flex_decision()
    selftest_shoulder_abduction_decision()
    selftest_pelvic_tilt_decision()
    selftest_scapular_symmetry_decision()
    selftest_confirm_frames()
    print("decision selftest: ALL PASSED")
