"""新动作 + 优化项的独立冒烟测试。

覆盖：肩关节外展 / 骨盆倾斜 / 肩胛对称性，以及 2026-10-18 优化轮的自适应
EMA / 死区 / 多帧确认 / 几何门控与 N/A 原因显示。

不碰摄像头、不碰模型：用合成 COCO-17 全身姿态，把
    features → smoothing → decision → output
整条链路各走一遍，检查读数、门控、左右独立、迟滞状态机和画面叠加都能跑。

运行：
    python test_new_actions.py

通过时最后打印 `test_new_actions: ALL PASSED`；任一断言失败会直接退非 0。
"""

from __future__ import annotations

import math
import statistics

import numpy as np

from decision import (
    PelvicTiltDecision,
    PelvicTiltState,
    ScapularSymmetryDecision,
    ScapularSymmetryState,
    ShoulderAbductionDecision,
    ShoulderAbductionState,
)
from features import (
    PelvicTiltFeatures,
    ScapularSymmetryFeatures,
    ShoulderAbductionFeatures,
)
from output import FrameRenderer
from smoothing import AngleSmoother

SIZE = (640, 480)


# ---------------------------------------------------------------------------
# 合成姿态
# ---------------------------------------------------------------------------

def _neutral_landmarks() -> dict:
    """正面直立的中立姿态（归一化坐标，y 向下）。"""
    return {
        0: (0.50, 0.15),   # 鼻
        1: (0.47, 0.13), 2: (0.53, 0.13),   # 眼
        3: (0.44, 0.14), 4: (0.56, 0.14),   # 耳
        5: (0.40, 0.30), 6: (0.60, 0.30),   # 肩
        7: (0.40, 0.45), 8: (0.60, 0.45),   # 肘（自然下垂）
        9: (0.40, 0.58), 10: (0.60, 0.58),  # 腕
        11: (0.42, 0.60), 12: (0.58, 0.60),  # 髋
        13: (0.42, 0.78), 14: (0.58, 0.78),  # 膝
        15: (0.41, 0.95), 16: (0.59, 0.95),  # 踝
    }


def _make_pose(pts: dict, vis_map: dict | None = None) -> dict:
    """构造与 pose_estimation.detect_pose() 同结构的合成帧。"""
    landmarks = []
    for pid in range(17):
        if pid in pts:
            x, y = pts[pid]
            v = 1.0 if vis_map is None else vis_map.get(pid, 1.0)
            landmarks.append((x, y, 0.0, v))
        else:
            landmarks.append((0.0, 0.0, 0.0, 0.0))
    return {"landmarks": landmarks, "image_size": SIZE}


def _pose_with(changes: dict | None = None, drop=()) -> dict:
    """中立姿态上改若干关键点；`drop=(7,)` 用来模拟该点丢失。"""
    pts = _neutral_landmarks()
    if changes:
        pts.update(changes)
    for pid in drop:
        pts.pop(pid, None)
    return _make_pose(pts)


def _close(actual: float, expected: float, eps: float = 1e-6) -> None:
    assert actual is not None, f"读数不应为 None（期望 {expected:.2f}°）"
    assert abs(actual - expected) <= eps, \
        f"期望 {expected:.4f}°，实际 {actual:.4f}°"


def _fmt(deg) -> str:
    return "N/A" if deg is None else f"{deg:6.1f}°"


# ---------------------------------------------------------------------------
# 1. 肩关节外展
# ---------------------------------------------------------------------------

def test_shoulder_abduction() -> None:
    print("\n[1] 肩关节外展 ShoulderAbductionFeatures")
    fe = ShoulderAbductionFeatures()
    dec_l = ShoulderAbductionDecision(side='L')
    dec_r = ShoulderAbductionDecision(side='R')

    # 中立：两臂垂在体侧 → 0°
    f = fe.update(_pose_with())
    _close(f['shoulder_abduction_left_deg'], 0.0)
    _close(f['shoulder_abduction_right_deg'], 0.0)
    s_l, s_r = dec_l.update(f), dec_r.update(f)
    assert s_l is ShoulderAbductionState.ADDUCTED
    assert s_r is ShoulderAbductionState.ADDUCTED
    print(f"  中立      L={_fmt(f['shoulder_abduction_left_deg'])} "
          f"R={_fmt(f['shoulder_abduction_right_deg'])}  "
          f"状态 L={s_l.value} R={s_r.value}")

    # 左臂平举（90°）、右臂过顶（180°）：左右各自独立判定
    f = fe.update(_pose_with({7: (0.60, 0.30), 8: (0.60, 0.10)}))
    _close(f['shoulder_abduction_left_deg'], 90.0)
    _close(f['shoulder_abduction_right_deg'], 180.0)
    s_l, s_r = dec_l.update(f), dec_r.update(f)
    assert s_l is ShoulderAbductionState.ABDUCTED
    assert s_r is ShoulderAbductionState.ABDUCTED
    print(f"  左平举/右过顶 L={_fmt(f['shoulder_abduction_left_deg'])} "
          f"R={_fmt(f['shoulder_abduction_right_deg'])}  "
          f"状态 L={s_l.value} R={s_r.value}")

    # 左右独立：左平举、右垂下 → 左 ABDUCTED，右仍 ADDUCTED
    f = fe.update(_pose_with({7: (0.60, 0.30)}))
    s_l = dec_l.update(f)
    s_r = dec_r.update(f)
    assert s_l is ShoulderAbductionState.ABDUCTED
    assert s_r is ShoulderAbductionState.ADDUCTED
    print(f"  仅左平举  L={_fmt(f['shoulder_abduction_left_deg'])} "
          f"R={_fmt(f['shoulder_abduction_right_deg'])}  "
          f"状态 L={s_l.value} R={s_r.value}")

    # 左侧肘点丢失 → 只作废左侧，右侧照常；原因如实标 no_point
    f = fe.update(_pose_with(drop=(7,)))
    assert not f['shoulder_abduction_left_valid']
    assert f['shoulder_abduction_left_deg'] is None
    assert f['shoulder_abduction_left_invalid_reason'] == 'no_point'
    assert f['shoulder_abduction_right_valid']
    assert dec_l.update(f) is ShoulderAbductionState.UNKNOWN
    print(f"  左肘丢失  L={_fmt(f['shoulder_abduction_left_deg'])} "
          f"R={_fmt(f['shoulder_abduction_right_deg'])}  "
          f"状态 L={dec_l.state.value} R={dec_r.state.value}")

    # 肩→肘只剩不到 1 像素 → 角度没有意义，如实 N/A + geom（不给假读数）
    f = fe.update(_pose_with({7: (0.40, 0.301)}))
    assert f['shoulder_abduction_left_deg'] is None
    assert f['shoulder_abduction_left_invalid_reason'] == 'geom'
    assert f['shoulder_abduction_right_valid'], "另一侧不该被连带作废"
    print("  肩→肘过短  L=N/A [geom]，R 照常")

    # 关键点在画面外（RTMPose 外推点）→ 同样 N/A + out_of_frame
    f = fe.update(_pose_with({7: (0.40, 1.20)}))
    assert f['shoulder_abduction_left_deg'] is None
    assert f['shoulder_abduction_left_invalid_reason'] == 'out_of_frame'
    print("  左肘在画外  L=N/A [out_of_frame]")

    # trunk 参考轴（实验）：躯干点齐全时照常出读数，缺髋点整条 N/A 且不退回 image
    ft = ShoulderAbductionFeatures(reference='trunk')
    f = ft.update(_pose_with({7: (0.60, 0.30)}))
    assert f['shoulder_abduction_ref'] == 'trunk'
    assert f['shoulder_abduction_left_valid']
    f = ft.update(_pose_with(drop=(12,)))
    assert f['shoulder_abduction_left_deg'] is None
    assert f['shoulder_abduction_right_deg'] is None
    assert f['shoulder_abduction_left_invalid_reason'] == 'no_trunk'
    assert f['shoulder_abduction_right_invalid_reason'] == 'no_trunk'
    print("  trunk 参考  正常出数；缺髋点 → 两侧 N/A [no_trunk]")

    print("  -> 肩关节外展 OK")


# ---------------------------------------------------------------------------
# 2. 骨盆倾斜
# ---------------------------------------------------------------------------

def test_pelvic_tilt() -> None:
    print("\n[2] 骨盆倾斜 PelvicTiltFeatures")
    fe = PelvicTiltFeatures()
    dec = PelvicTiltDecision()

    # 双髋等高 → 0° / NEUTRAL；等高时没有"哪侧更高"
    f = fe.update(_pose_with())
    _close(f['pelvic_tilt_deg'], 0.0)
    assert f['pelvic_tilt_higher'] is None
    assert f['pelvic_tilt_invalid_reason'] is None
    st = dec.update(f)
    assert st is PelvicTiltState.NEUTRAL
    print(f"  中立      pelvis={_fmt(f['pelvic_tilt_deg'])}  "
          f"状态={st.value}")

    # 画面右侧髋抬高 → 正角度 / TILTED，并标出"R higher"
    f = fe.update(_pose_with({12: (0.58, 0.50)}))
    expected = math.degrees(math.atan2(0.10 * SIZE[1], 0.16 * SIZE[0]))
    _close(f['pelvic_tilt_deg'], expected, eps=1e-4)
    assert f['pelvic_tilt_higher'] == 'R'
    st = dec.update(f)
    assert st is PelvicTiltState.TILTED
    print(f"  右髋抬高  pelvis={_fmt(f['pelvic_tilt_deg'])}  "
          f"状态={st.value}  高侧=R")

    # 左髋抬高 → 负角度，但触发看绝对值，仍是 TILTED；高侧=L
    dec = PelvicTiltDecision()
    f = fe.update(_pose_with({11: (0.42, 0.50)}))
    _close(f['pelvic_tilt_deg'], -expected, eps=1e-4)
    assert f['pelvic_tilt_higher'] == 'L'
    st = dec.update(f)
    assert st is PelvicTiltState.TILTED
    print(f"  左髋抬高  pelvis={_fmt(f['pelvic_tilt_deg'])}  "
          f"状态={st.value}  高侧=L")

    # 迟滞带：进入 TILTED 后回到 8°（enter=10 与 exit=5 之间）应保持
    dec = PelvicTiltDecision(enter_deg=10.0, exit_deg=5.0)
    dec.update(f)
    assert dec.update({'pelvic_tilt_deg': 8.0, 'pelvic_tilt_valid': True}) \
        is PelvicTiltState.TILTED, "迟滞带内应保持 TILTED"
    assert dec.update({'pelvic_tilt_deg': 4.0, 'pelvic_tilt_valid': True}) \
        is PelvicTiltState.NEUTRAL, "越过 exit 应回 NEUTRAL"
    print("  迟滞带    8° 保持 TILTED，4° 回 NEUTRAL")

    # 缺一个髋 → 整条 N/A，不做单侧降级；原因如实
    f = fe.update(_pose_with(drop=(12,)))
    assert not f['pelvic_tilt_valid'] and f['pelvic_tilt_deg'] is None
    assert f['pelvic_tilt_invalid_reason'] == 'no_point'
    assert dec.update(f) is PelvicTiltState.UNKNOWN
    print(f"  右髋丢失  pelvis={_fmt(f['pelvic_tilt_deg'])}  "
          f"状态={dec.state.value}  原因=no_point")

    # 双髋几乎重合 → 肩线/髋线太短，角度无意义，N/A + geom
    f = fe.update(_pose_with({12: (0.42, 0.601)}))
    assert f['pelvic_tilt_deg'] is None
    assert f['pelvic_tilt_invalid_reason'] == 'geom'
    print("  双髋过近  pelvis=N/A [geom]")

    print("  -> 骨盆倾斜 OK")


# ---------------------------------------------------------------------------
# 3. 肩胛对称性
# ---------------------------------------------------------------------------

def test_scapular_symmetry() -> None:
    print("\n[3] 肩胛对称性 ScapularSymmetryFeatures")
    fe = ScapularSymmetryFeatures()
    dec = ScapularSymmetryDecision()

    # 双肩等高 → 0° / SYMMETRIC；等高时没有"哪侧更高"
    f = fe.update(_pose_with())
    _close(f['scapular_symmetry_deg'], 0.0)
    assert f['scapular_symmetry_higher'] is None
    assert f['scapular_symmetry_invalid_reason'] is None
    st = dec.update(f)
    assert st is ScapularSymmetryState.SYMMETRIC
    print(f"  中立      scap={_fmt(f['scapular_symmetry_deg'])}  "
          f"状态={st.value}")

    # 画面右侧肩抬高 → 正角度 / ASYMMETRIC，高侧=R
    f = fe.update(_pose_with({6: (0.60, 0.20)}))
    expected = math.degrees(math.atan2(0.10 * SIZE[1], 0.20 * SIZE[0]))
    _close(f['scapular_symmetry_deg'], expected, eps=1e-4)
    assert f['scapular_symmetry_higher'] == 'R'
    st = dec.update(f)
    assert st is ScapularSymmetryState.ASYMMETRIC
    print(f"  右肩抬高  scap={_fmt(f['scapular_symmetry_deg'])}  "
          f"状态={st.value}  高侧=R")

    # 左侧肩抬高 → 负角度，按绝对值仍触发；高侧=L
    dec = ScapularSymmetryDecision()
    f = fe.update(_pose_with({5: (0.40, 0.20)}))
    _close(f['scapular_symmetry_deg'], -expected, eps=1e-4)
    assert f['scapular_symmetry_higher'] == 'L'
    st = dec.update(f)
    assert st is ScapularSymmetryState.ASYMMETRIC
    print(f"  左肩抬高  scap={_fmt(f['scapular_symmetry_deg'])}  "
          f"状态={st.value}  高侧=L")

    # 缺一个肩 → 整条 N/A，不做单侧降级；原因如实
    f = fe.update(_pose_with(drop=(6,)))
    assert not f['scapular_symmetry_valid'] and \
        f['scapular_symmetry_deg'] is None
    assert f['scapular_symmetry_invalid_reason'] == 'no_point'
    assert dec.update(f) is ScapularSymmetryState.UNKNOWN
    print(f"  右肩丢失  scap={_fmt(f['scapular_symmetry_deg'])}  "
          f"状态={dec.state.value}  原因=no_point")

    # 双肩几乎重合 → 肩线太短，N/A + geom
    f = fe.update(_pose_with({6: (0.40, 0.301)}))
    assert f['scapular_symmetry_deg'] is None
    assert f['scapular_symmetry_invalid_reason'] == 'geom'
    print("  双肩过近  scap=N/A [geom]")

    print("  -> 肩胛对称性 OK")


# ---------------------------------------------------------------------------
# 4. 平滑 + 叠加输出
# ---------------------------------------------------------------------------

def test_smoothing_and_overlay() -> None:
    print("\n[4] smoothing + output 叠加")
    fe = ShoulderAbductionFeatures()
    smoother = AngleSmoother(alpha=0.3)
    dec = ShoulderAbductionDecision(side='L')

    # 同一姿势人为加抖动：平滑后标准差应明显变小，状态始终 ABDUCTED
    raw = [80.0, 90.0, 70.0, 85.0, 75.0, 88.0]
    smoothed = []
    for deg in raw:
        f = smoother.update({'shoulder_abduction_left_deg': deg,
                             'shoulder_abduction_left_valid': True})
        smoothed.append(f['shoulder_abduction_left_deg'])
        assert dec.update(f) is ShoulderAbductionState.ABDUCTED
    raw_std = statistics.pstdev(raw)
    smooth_std = statistics.pstdev(smoothed)
    assert smooth_std < raw_std, \
        f"平滑后抖动应变小：raw={raw_std:.2f} smooth={smooth_std:.2f}"
    print(f"  EMA       原始 std={raw_std:.2f}° → 平滑后 std={smooth_std:.2f}°")

    # 丢帧仍 N/A，不维持旧值
    f = smoother.update({'shoulder_abduction_left_deg': None,
                         'shoulder_abduction_left_valid': False})
    assert f['shoulder_abduction_left_deg'] is None
    print("  丢帧      平滑后仍为 N/A（不维持上一有效值）")

    # 合成一帧同时含三个动作的读数
    abd = fe.update(_pose_with({7: (0.60, 0.30)}))
    pelvic = PelvicTiltFeatures().update(_pose_with({12: (0.58, 0.50)}))
    scap = ScapularSymmetryFeatures().update(_pose_with({6: (0.60, 0.20)}))

    items = [
        ('abduction', {'features': abd,
                       'states': {'L': ShoulderAbductionState.ABDUCTED,
                                  'R': ShoulderAbductionState.ADDUCTED}}),
        ('pelvic', {'features': pelvic,
                    'state': PelvicTiltState.TILTED, 'enter_deg': 10.0}),
        ('scapular', {'features': scap,
                      'state': ScapularSymmetryState.ASYMMETRIC,
                      'enter_deg': 10.0}),
    ]
    ren = FrameRenderer(draw_skeleton=False)
    frame = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    out = ren.draw_all_action_overlay(frame, items, show_geometry=True)
    assert out is frame and frame.shape == (SIZE[1], SIZE[0], 3)
    assert frame.any(), "叠加后画面应有内容"
    print("  叠加      三个动作 + 几何线绘制成功")

    # 规则关闭 / 读数无效也必须能画：report only + N/A
    bad = PelvicTiltFeatures().update(_pose_with(drop=(12,)))
    items = [
        ('abduction', {'features': abd, 'states': {}}),
        ('pelvic', {'features': bad, 'state': PelvicTiltState.UNKNOWN}),
        ('scapular', {'features': scap, 'state': None}),
    ]
    frame = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    ren.draw_all_action_overlay(frame, items, show_geometry=False)
    assert frame.any()
    print("  降级      report-only / N/A 叠加成功")

    print("  -> 平滑与输出 OK")


# ---------------------------------------------------------------------------
# 5. 优化项：自适应 EMA / 多帧确认 / N/A 原因显示
# ---------------------------------------------------------------------------

def test_optimizations() -> None:
    print("\n[5] 优化项：自适应 EMA / 多帧确认 / 原因显示")

    # adaptive：小抖动仍按 alpha 压，大跳变当帧跟上（真动作不滞后）
    s = AngleSmoother(alpha=0.3, adaptive=True, fast_delta_deg=10.0)
    s.update({'a_deg': 100.0})
    small = s.update({'a_deg': 101.0})['a_deg']
    assert 100.0 < small < 101.0, f"小抖动应被平滑，实际 {small}"
    assert s.update({'a_deg': 120.0})['a_deg'] == 120.0, \
        "大跳变（> fast_delta）应当帧等于原始值"
    print(f"  自适应 EMA  小抖动→{small:.2f}°（未到原始值），大跳变→当帧跟上")

    # deadband：死区内保持；慢速漂移最终仍能跟上（不永久冻住）
    s = AngleSmoother(alpha=0.5, deadband_deg=2.0)
    s.update({'a_deg': 100.0})
    assert s.update({'a_deg': 101.0})['a_deg'] == 100.0
    outs = [s.update({'a_deg': 101.0 + i})['a_deg'] for i in range(10)]
    assert outs[-1] > 100.5, f"慢速漂移被死区冻住了：{outs}"
    print(f"  死区        带内保持 100°，慢速漂移 10 帧后到 {outs[-1]:.2f}°")

    # confirm_frames=2：一帧不翻，两帧才翻；丢帧立即 UNKNOWN 并清计数
    dec = ShoulderAbductionDecision(side='L', confirm_frames=2)
    active = {'shoulder_abduction_left_deg': 90.0,
              'shoulder_abduction_left_valid': True}
    inactive = {'shoulder_abduction_left_deg': 0.0,
                'shoulder_abduction_left_valid': True}
    assert dec.update(active) is ShoulderAbductionState.UNKNOWN, \
        "confirm_frames=2 时首帧不应翻状态"
    assert dec.update(active) is ShoulderAbductionState.ABDUCTED, \
        "连续两帧才应提交 ABDUCTED"
    # 回程同样要两帧；中间夹一帧迟滞带 → 计数清零
    assert dec.update(inactive) is ShoulderAbductionState.ABDUCTED
    dec.update({'shoulder_abduction_left_deg': 50.0,
                'shoulder_abduction_left_valid': True})     # 迟滞带：打断计数
    assert dec.update(inactive) is ShoulderAbductionState.ABDUCTED, \
        "被打断后要重新累计"
    assert dec.update(inactive) is ShoulderAbductionState.ADDUCTED
    # 丢帧不是"一帧反证"：立即失去证据并清掉未完成计数
    assert dec.update(active) is ShoulderAbductionState.ADDUCTED
    assert dec.update({'shoulder_abduction_left_deg': None,
                       'shoulder_abduction_left_valid': False}) \
        is ShoulderAbductionState.UNKNOWN
    assert dec.update(active) is ShoulderAbductionState.UNKNOWN, \
        "丢帧后计数应清零，不能接着上一段凑够两帧"
    print("  多帧确认    一帧未知、两帧才翻；迟滞带/丢帧会清零计数")

    # N/A 原因与高侧标签进画面：仍能画出，不因新键崩溃
    abd_bad = ShoulderAbductionFeatures().update(
        _pose_with({7: (0.40, 0.301)}))
    pelvic_bad = PelvicTiltFeatures().update(_pose_with(drop=(12,)))
    abd_ok = ShoulderAbductionFeatures().update(_pose_with({7: (0.60, 0.30)}))
    pelvic_ok = PelvicTiltFeatures().update(_pose_with({12: (0.58, 0.50)}))
    items = [
        ('abduction', {'features': abd_bad, 'states': {}}),
        ('pelvic', {'features': pelvic_bad, 'state': None}),
    ]
    ren = FrameRenderer(draw_skeleton=False)
    frame = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    ren.draw_all_action_overlay(frame, items, show_geometry=True)
    assert frame.any(), "N/A 原因行也必须画出来"
    items = [
        ('abduction', {'features': abd_ok, 'states': {}}),
        ('pelvic', {'features': pelvic_ok,
                    'state': PelvicTiltState.TILTED}),
    ]
    frame = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    ren.draw_all_action_overlay(frame, items, show_geometry=True)
    assert frame.any(), "带高侧标签的读数行也应画出来"
    print("  原因显示    N/A 原因与 (R higher) 标签均可叠加")

    # trunk 参考轴的几何线：画肩中点→髋中点那根轴，不画画面竖直轴
    abd_trunk = ShoulderAbductionFeatures(reference='trunk').update(
        _pose_with({7: (0.60, 0.30)}))
    items = [('abduction', {'features': abd_trunk, 'states': {}})]
    frame_trunk = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    ren.draw_all_action_overlay(frame_trunk, items, show_geometry=True)
    assert frame_trunk.any()
    print("  trunk 参考  几何线绘制成功")

    print("  -> 优化项 OK")


def main() -> None:
    print("===== 新动作识别冒烟测试 =====")
    test_shoulder_abduction()
    test_pelvic_tilt()
    test_scapular_symmetry()
    test_smoothing_and_overlay()
    test_optimizations()
    print("\ntest_new_actions: ALL PASSED")


if __name__ == "__main__":
    main()
