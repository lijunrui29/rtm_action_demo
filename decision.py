"""
decision 层（全身动作线）：把 features 层给出的动作特征映射成状态。

本仓库只有两个动作，各一个状态机：

1) 举手（消费 ArmRaiseFeatures 的上臂抬升角，**越小越举**）：
    arm_raise_deg <= enter_deg  -> RAISED
    arm_raise_deg >= exit_deg   -> DOWN
    enter_deg < 角度 < exit_deg -> 迟滞带，保持当前状态
                                    （UNKNOWN 时兜底 DOWN：没有充分证据
                                     就不判"举手"）

2) 仰头（消费 HeadTiltFeatures 的头俯仰代理角，**越大越仰头**）：
    head_pitch_deg >= enter_deg -> TILTED_BACK
    head_pitch_deg <= exit_deg  -> NEUTRAL
    中间 -> 迟滞带，保持当前状态（UNKNOWN 时兜底 NEUTRAL）

两个状态机都只有"未触发 / 触发"两档 + UNKNOWN，都**不产生提醒** —— 本仓库
目前没有任何提醒，也没有任何时长判断。UNKNOWN 的含义一律是"这一帧没有证据"
（无人 / 关键点丢了 / 特征被判几何退化）：此时分别兜底 DOWN / NEUTRAL，即
**没证据就不判动作**，而不是沿用上一帧的状态（v1 刻意不平滑、不维持上一有效值，
理由见 features.py 的模块说明）。

两者都用**双阈值迟滞带**：进入看 enter 阈值、退出要越过 exit 阈值、两者之间
保持当前状态（防阈值附近抖动来回跳）。用绝对量纲的双阈值而不用乘性迟滞，
是因为两个阈值都落在观测区间中段，不存在"阈值贴着区间边缘、恢复点跑到区间外"
的问题。

接口约定（保持稳定，别改签名；新增参数只加带默认值的关键字参数）：
    ArmRaiseState 枚举：DOWN / RAISED / UNKNOWN
    ArmRaiseDecision(enter_deg=45.0, exit_deg=60.0)
    update(arm: Optional[dict]) -> ArmRaiseState
    state（只读属性）

    HeadTiltState 枚举：NEUTRAL / TILTED_BACK / UNKNOWN
    HeadTiltDecision(enter_deg=20.0, exit_deg=10.0)
    update(head: Optional[dict]) -> HeadTiltState
    state（只读属性）

⚠ 两个 enter/exit 默认值都是**占位值**、不是标定结果：读数的中立基线随摄像头
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
        ArmRaiseDecision(enter_deg=45.0, exit_deg=60.0)
        update(arm: Optional[dict]) -> ArmRaiseState
        state（只读属性）
    """

    def __init__(self, enter_deg: float = 45.0, exit_deg: float = 60.0) -> None:
        """
        参数:
            enter_deg: 抬升角 <= 此值判 RAISED（度；越小越举）。
            exit_deg:  抬升角 >= 此值判 DOWN（度）。必须大于 enter_deg，
                       两者之间就是迟滞带。
        """
        if exit_deg <= enter_deg:
            raise ValueError("exit_deg 必须大于 enter_deg（否则没有迟滞带）")
        self.enter_deg = enter_deg
        self.exit_deg = exit_deg
        self._state = ArmRaiseState.UNKNOWN

    @property
    def state(self) -> ArmRaiseState:
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
            return self._state
        deg = arm.get('arm_raise_deg')
        if deg is None:
            self._state = ArmRaiseState.UNKNOWN
            return self._state

        if deg <= self.enter_deg:
            self._state = ArmRaiseState.RAISED
        elif deg >= self.exit_deg:
            self._state = ArmRaiseState.DOWN
        elif self._state is ArmRaiseState.UNKNOWN:
            # 迟滞带 + 还没有历史状态 → 兜底 DOWN（没证据不判举手）
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
        HeadTiltDecision(enter_deg=20.0, exit_deg=10.0)
        update(head: Optional[dict]) -> HeadTiltState
        state（只读属性）
    """

    def __init__(self, enter_deg: float = 20.0, exit_deg: float = 10.0) -> None:
        """
        参数:
            enter_deg: 俯仰角 >= 此值判 TILTED_BACK（度；越大越仰头）。
            exit_deg:  俯仰角 <= 此值判 NEUTRAL（度）。必须小于 enter_deg，
                       两者之间就是迟滞带。
        """
        if exit_deg >= enter_deg:
            raise ValueError("exit_deg 必须小于 enter_deg（否则没有迟滞带）")
        self.enter_deg = enter_deg
        self.exit_deg = exit_deg
        self._state = HeadTiltState.UNKNOWN

    @property
    def state(self) -> HeadTiltState:
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
            return self._state
        deg = head.get('head_pitch_deg')
        if deg is None:
            self._state = HeadTiltState.UNKNOWN
            return self._state

        if deg >= self.enter_deg:
            self._state = HeadTiltState.TILTED_BACK
        elif deg <= self.exit_deg:
            self._state = HeadTiltState.NEUTRAL
        elif self._state is HeadTiltState.UNKNOWN:
            # 迟滞带 + 还没有历史状态 → 兜底 NEUTRAL（没证据不判仰头）
            self._state = HeadTiltState.NEUTRAL
        return self._state


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


if __name__ == "__main__":
    selftest_arm_raise_decision()
    selftest_head_tilt_decision()
    print("decision selftest: ALL PASSED")
