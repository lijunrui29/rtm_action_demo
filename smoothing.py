"""
角度读数平滑（features 与 decision/output 之间的一层）。

背景：v1 **刻意不做平滑**（当时的命题是"先看清单目关键点撑不撑得住"）。
2026-10-18 用户实拍反馈"人没动、读数却一直在跳"，因此加这一层。改的只是
**有效读数之间的抖动**：

  * 用 EMA（指数滑动平均）按 key 独立平滑所有以 `_deg` 结尾的角度读数；
  * 2026-10-18 优化轮新增两个**可选**开关（模块默认关闭，入口默认开启）：
    - `adaptive`：大跳变时把 alpha 临时推向 1，真动作跟得上、小抖动照压；
    - `deadband_deg`：原始读数与上一输出值之差在死区内就沿用上一输出，
      压住固定噪声（**不比较相邻两帧原始值**，否则慢速漂移会被永久冻住）；
  * **仍然不"维持上一有效值"**：关键点丢帧时读数照旧是 None / N/A，
    下一帧有效时从原始值重新起步（不把丢帧前后的数据混在一起）；
  * `*_valid` / `*_conf` / `*_geom` 原样透传 —— 几何辅助线仍画原始关键点，
    这一层不动像素坐标；
  * 有跨帧状态，**每路数据源必须各建一个实例**（双摄入口两台不能共用）。

数据流因此变成：
    capture → pose_estimation → features → smoothing → decision → output
"""

from __future__ import annotations

from typing import Dict, Optional


class AngleSmoother:
    """按 key 独立做 EMA 的角度平滑器（有跨帧状态）。

    参数:
        alpha: EMA 系数，取值 (0, 1]。
               **alpha = 1.0 = 原样输出（等于关掉平滑）**；
               alpha 越小越平滑，但跟随真实动作也越滞后。
               默认 0.3（约 3~4 帧的时间常数，60fps 下 ~0.05s、30fps 下 ~0.1s）。
        adaptive: 自适应 EMA（默认 False = 与旧版逐位一致）。
               本帧原始值与上一帧原始值之差为 d 时：
                   a_eff = alpha + (1 - alpha) * min(1, |d| / fast_delta_deg)
               小抖动 → a_eff 接近 alpha（继续压噪）；真动作 → a_eff = 1
               （当帧跟上，减少滞后）。
        fast_delta_deg: adaptive 的"算大跳变"的差值（度），必须 > 0。
        deadband_deg: 死区（度，>= 0，默认 0 = 关闭）。原始读数与**上一输出值**
               之差小于它时沿用上一输出值。与"相邻两帧原始值"比较不同：慢速
               连续漂移会累积到超过死区后更新，不会被永久冻住。
    """

    def __init__(self, alpha: float = 0.3, adaptive: bool = False,
                 fast_delta_deg: float = 10.0,
                 deadband_deg: float = 0.0) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha 必须满足 0 < alpha <= 1")
        if fast_delta_deg <= 0.0:
            raise ValueError("fast_delta_deg 必须 > 0")
        if deadband_deg < 0.0:
            raise ValueError("deadband_deg 必须 >= 0")
        self.alpha = float(alpha)
        self.adaptive = bool(adaptive)
        self.fast_delta_deg = float(fast_delta_deg)
        self.deadband_deg = float(deadband_deg)
        self._prev: Dict[str, float] = {}       # key → 上一输出（平滑值）
        self._prev_raw: Dict[str, float] = {}   # key → 上一帧原始值（adaptive 用）

    def update(self, features: Optional[dict]) -> Optional[dict]:
        """返回平滑后的 features **副本**（不修改传入的 dict）。

        规则：
          * key 以 `_deg` 结尾且值是数值 → 做 EMA（旧版）/ 自适应 EMA /
            死区保持；
          * 该 key 本帧是 None（算不出 / 丢点）→ 输出 None，并**丢掉这个 key 的
            历史状态**，下一帧有效时从原始值重新起步（不跨丢帧沿用旧值）；
          * 其它 key（valid/conf/geom/文本）原样带过。
        """
        if features is None:
            return None

        out = dict(features)
        for key, value in features.items():
            if not key.endswith('_deg'):
                continue
            if value is None:
                out[key] = None
                self._forget(key)
                continue
            if not isinstance(value, (int, float)):
                # 约定里角度键要么是数值、要么是 None；出现别的就原样带过
                out[key] = value
                self._forget(key)
                continue
            if self.alpha >= 1.0:       # 关掉平滑：不保留状态，逐帧等于原始值
                out[key] = float(value)
                continue

            raw = float(value)
            prev = self._prev.get(key)
            raw_prev = self._prev_raw.get(key)
            if prev is None:
                smoothed = raw          # 首帧（或丢帧后重启）从原始值起步
            elif self.deadband_deg > 0.0 and \
                    abs(raw - prev) < self.deadband_deg:
                smoothed = prev         # 死区内：沿用上一输出
            else:
                a_eff = self.alpha
                if self.adaptive and raw_prev is not None:
                    delta = abs(raw - raw_prev) / self.fast_delta_deg
                    a_eff = self.alpha + (1.0 - self.alpha) * min(1.0, delta)
                smoothed = a_eff * raw + (1.0 - a_eff) * prev
            self._prev[key] = smoothed
            self._prev_raw[key] = raw
            out[key] = smoothed
        return out

    def _forget(self, key: str) -> None:
        self._prev.pop(key, None)
        self._prev_raw.pop(key, None)

    def reset(self) -> None:
        """清空全部跨帧状态（换人、重开一路画面时可显式调用）。"""
        self._prev.clear()
        self._prev_raw.clear()


# ---------------------------------------------------------------------------
# 自测：合成数据，不碰摄像头
# ---------------------------------------------------------------------------

def selftest_angle_smoother() -> None:
    """自测 EMA 数学、左右/各 key 独立、丢帧后重启、alpha=1 原样、参数校验。"""
    # 1) alpha=0.5：0 → 10 → 10 应得到 0 → 5 → 7.5
    s = AngleSmoother(0.5)
    raw1 = {'arm_raise_deg': 0.0, 'arm_raise_valid': True,
            'arm_geom': {'side': 'L'}}
    out1 = s.update(raw1)
    assert out1['arm_raise_deg'] == 0.0
    assert out1 is not raw1, "必须返回副本，不能改调用方的 dict"
    assert raw1['arm_raise_deg'] == 0.0, "原 dict 不能被改"
    out2 = s.update({'arm_raise_deg': 10.0})
    assert out2['arm_raise_deg'] == 5.0, f"EMA 第二帧应为 5，实际 {out2['arm_raise_deg']}"
    out3 = s.update({'arm_raise_deg': 10.0})
    assert out3['arm_raise_deg'] == 7.5, f"EMA 第三帧应为 7.5，实际 {out3['arm_raise_deg']}"
    # 非角度键原样透传（valid/geom 不参与平滑，也不被改成 float）
    assert out1['arm_raise_valid'] is True
    assert out1['arm_geom'] is raw1['arm_geom']

    # 2) 左右独立、不同动作的 key 互不影响
    s = AngleSmoother(0.5)
    s.update({'elbow_flex_left_deg': 100.0, 'elbow_flex_right_deg': 160.0})
    o = s.update({'elbow_flex_left_deg': 120.0, 'elbow_flex_right_deg': 160.0})
    assert o['elbow_flex_left_deg'] == 110.0
    assert o['elbow_flex_right_deg'] == 160.0

    # 3) 丢帧 → None，且历史状态被丢掉；下一帧有效时从原始值重新起步
    s = AngleSmoother(0.5)
    s.update({'knee_flex_left_deg': 100.0})
    mid = s.update({'knee_flex_left_deg': None})
    assert mid['knee_flex_left_deg'] is None, "丢帧必须如实 N/A，不维持旧值"
    back = s.update({'knee_flex_left_deg': 40.0})
    assert back['knee_flex_left_deg'] == 40.0, \
        f"丢帧后首帧应从原值起步，实际 {back['knee_flex_left_deg']}"

    # 4) alpha=1 = 关掉平滑，逐帧原样
    s = AngleSmoother(1.0)
    assert s.update({'head_pitch_deg': 10.0})['head_pitch_deg'] == 10.0
    assert s.update({'head_pitch_deg': 30.0})['head_pitch_deg'] == 30.0

    # 5) 输入 None 原样返回；非法 alpha 报错
    assert AngleSmoother().update(None) is None
    for bad in (0.0, -0.1, 1.01):
        try:
            AngleSmoother(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法 alpha 应报错：{bad}")

    # 6) 默认参数与旧版逐位一致（adaptive/deadband 都关着）
    s = AngleSmoother(0.5)
    assert s.update({'a_deg': 10.0})['a_deg'] == 10.0
    assert s.update({'a_deg': 20.0})['a_deg'] == 15.0
    assert s.update({'a_deg': 20.0})['a_deg'] == 17.5

    # 7) adaptive：小抖动按 alpha 压，大跳变当帧跟上
    s = AngleSmoother(0.3, adaptive=True, fast_delta_deg=10.0)
    s.update({'a_deg': 100.0})
    # |101-100|/10 = 0.1 → a_eff = 0.3 + 0.7*0.1 = 0.37
    o = s.update({'a_deg': 101.0})
    assert abs(o['a_deg'] - (0.37 * 101.0 + 0.63 * 100.0)) < 1e-9, \
        f"小抖动应按自适应 alpha 平滑，实际 {o['a_deg']}"
    # |120-101|/10 > 1 → a_eff = 1 → 当帧等于原始值
    o = s.update({'a_deg': 120.0})
    assert o['a_deg'] == 120.0, \
        f"大跳变应立刻跟上，实际 {o['a_deg']}"

    # 8) deadband：与**上一输出值**比较；慢速漂移仍能跟上（不永久冻住）
    s = AngleSmoother(0.5, deadband_deg=2.0)
    s.update({'a_deg': 100.0})
    assert s.update({'a_deg': 101.0})['a_deg'] == 100.0, "死区内应保持"
    assert s.update({'a_deg': 100.5})['a_deg'] == 100.0, "死区内应保持"
    # |104-100| = 4 >= 2 → 走 EMA：0.5*104 + 0.5*100 = 102
    assert s.update({'a_deg': 104.0})['a_deg'] == 102.0, "越死区应更新"
    # 慢速连续漂移：每帧 +1°，最终必须跟上
    s = AngleSmoother(0.5, deadband_deg=2.0)
    s.update({'a_deg': 100.0})
    outs = [s.update({'a_deg': 100.0 + i})['a_deg'] for i in range(1, 11)]
    assert outs[-1] > 100.5, f"慢速漂移被冻住了：{outs}"

    # 9) alpha=1 优先：adaptive/deadband 都原样输出
    s = AngleSmoother(1.0, adaptive=True, deadband_deg=5.0)
    assert s.update({'a_deg': 10.0})['a_deg'] == 10.0
    assert s.update({'a_deg': 10.5})['a_deg'] == 10.5

    # 10) reset 清两套状态：下一帧从原始值重新起步
    s = AngleSmoother(0.5, adaptive=True)
    s.update({'a_deg': 100.0})
    s.reset()
    assert s.update({'a_deg': 40.0})['a_deg'] == 40.0

    # 11) 丢帧同时清 _prev_raw
    s = AngleSmoother(0.5, adaptive=True)
    s.update({'a_deg': 100.0})
    s.update({'a_deg': None})
    assert s.update({'a_deg': 40.0})['a_deg'] == 40.0

    # 12) 非法 adaptive 参数报错
    for kwargs in ({'fast_delta_deg': 0.0}, {'fast_delta_deg': -1.0},
                   {'deadband_deg': -0.1}):
        try:
            AngleSmoother(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法参数应报错：{kwargs}")

    print("  selftest_angle_smoother: OK")


if __name__ == "__main__":
    selftest_angle_smoother()
    print("smoothing selftest: ALL PASSED")
