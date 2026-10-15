"""
全身动作识别 demo（举手 / 仰头）—— **独立动作线**，2026-10-12 新增。

和现有的坐姿 / CVA / 头前伸那条线**没有任何关系**：本入口不装配、不 import 那条
线的任何对象（没有 FeatureExtractor / PostureFeatures / CvaProxyFeatures /
FhpDecision，也没有任何提醒），画面上只有举手和仰头两组读数 + 状态。反过来，
另外两个入口（main_demo.py = 45° 前侧坐姿线 / main_front_demo.py = 正面实验行）
也完全不受本文件影响 —— 这条线是旁路，进出都不碰它们。

机位要求（**重要**）：
    本入口假定摄像头在正前方偏一侧（约 45° 前侧），与 main_demo.py 同机位
    （机位由入口文件声明，与 CLAUDE.md「机位路由」同思路）。两个动作对这个
    机位的依赖完全不同：
      * **举手**：任意机位都能测 —— 上臂朝上在哪个视角都看得见；
      * **仰头**：**只在侧对 / 45° 前侧有意义**。正对摄像头时仰头主要是"朝镜头
        方向的透视缩短"，耳-鼻在像面上几乎不动，读数不反映仰头大小（见
        features.HeadTiltFeatures 的"近似在哪"）。所以本入口不是"正面入口"。
    换机位（尤其换到正对）要重做/重标，别把这里的读数当跨机位通用。

两个角怎么构造（用哪些关键点，全在 features 层，像素空间）：
    举手（features.ArmRaiseFeatures）：
        arm_raise_deg = 肩(5/6)→肘(7/8) 与 **向上竖直轴** 的夹角，度
                        （0 = 完全上举 / 90 = 平举 / 180 = 自然下垂）
        elbow_deg     = 肩-肘-腕 在肘处的内角（180 = 手臂伸直）
        —— 触发**用抬升角**：肘角在下垂、平举、上举伸直三种姿势下都是 180°，
           单独拿它判"举手"必然把下垂/平举一起误判进来（肘角仍照常显示作对照）。
           抬升角只要肩+肘，**腕丢了不影响触发**（腕只喂显示用的肘角）；左右
           两侧独立算，取"举得更高的那一侧"。
    仰头（features.HeadTiltFeatures）：
        head_pitch_deg = 耳中点(3/4)→鼻尖(0) 与 **水平线** 的夹角，度
                         （正 = 鼻在耳上方 = 仰头，负 = 低头）
        —— COCO 17 **没有下巴/头顶/耳屏**，用"耳 + 鼻"代替头部前后轴，是经典的
           2D head-pose 代理，**不是真实俯仰角**（阈值只能本项目自己定，别套文献）；
           2D 投影下 45° 机位水平方向被压缩，读数只保证**单调**；**不补偿转头
           （yaw）** —— 画面上那条耳-鼻几何线转没转一眼能看出来。
           刻意**不用现成的 PostureFeatures.head_neck_angle**：耳朵几乎就在头后仰
           的旋转轴心上，仰头时耳几乎不动，那个角量的是头相对躯干的**平移**前伸，
           不是**转动**俯仰，两回事。
           零点不在 0：45° 机位下中立站姿实测约 -20~-30°（鼻尖天生略低于耳中点），
           所以仰头阈值要按画面上的中立读数往上加，别照数字本身调。
           横向退化门控（--tilt-min-dx-ratio）：正对镜头时这个角会退化成
           sign(θ)·90°（只剩符号、没有量值，还长得像正常读数）→ 判几何退化、
           报 N/A，画面上标 [geometry degenerate]（与"关键点丢了"分开）。

v1 的两个刻意取舍（**别悄悄改**）：
    1) **不做任何平滑**：不 EMA、不中值、不"维持上一有效值"。关键点丢帧就标
       invalid、读数写 N/A，抖动就原样跳出来 —— 本轮的命题是"单目关键点撑不撑
       得住全身动作"，加平滑/维持恰好会把这两个最该被看到的信号掩盖掉
       （与 CvaProxyFeatures 的"降级维持上一有效值"故意相反）。
    2) **举手这条线默认只报数、不判状态**（--arm-rule 默认 off）：先看几眼数据，
       再决定触发用抬升角还是肘角。判据类（decision.ArmRaiseDecision）已经写好并
       自测通过，`--arm-rule raise` 就能开。仰头那条线按计划带状态机（角度唯一、
       没有歧义）。

阈值都是**占位值**、不是标定结果（--arm-raise-enter/-exit、--head-tilt-enter/
-head-tilt-exit 可现场改）：读数的中立基线随摄像头高度/机位/人体比例变化，
先跑起来看画面上的数再定。

用法（在 rtm_demo/ 目录下运行。--camera 默认 = capture.DEFAULT_CAMERA_INDEX，
本机 = USB 外接那台、固定在 45° 前侧；想用别的摄像头显式传 --camera N）：
    python main_action_demo.py                              # 举手只报数，仰头判状态
    python main_action_demo.py --arm-rule raise              # 举手也开状态机
    python main_action_demo.py --arm-raise-enter 30 --arm-raise-exit 50
    python main_action_demo.py --head-tilt-enter -5 --head-tilt-exit -15
    python main_action_demo.py --no-geometry                 # 不画角度几何辅助线
    python main_action_demo.py --snapshot data/action_snap.png  # 按 s 存当前帧
    python main_action_demo.py --camera 0                    # 换回笔记本内置（正对机位，仰头会判退化）
    python main_action_demo.py --selftest                    # 合成数据自测（不碰摄像头）

按键：s 存快照（需 --snapshot 指定路径，否则忽略）、ESC 退出。
首帧较慢（惰性初始化 pose 检测器），正常。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 本目录（rtm_demo）内的模块直接 import（与另外两个入口同约定）
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from capture import CameraCapture, DEFAULT_CAMERA_INDEX   # noqa: E402
from pose_estimation import detect_pose                  # noqa: E402
from features import (ArmRaiseFeatures,                  # noqa: E402
                      HeadTiltFeatures,
                      selftest_arm_raise, selftest_head_tilt)
from decision import (ArmRaiseDecision, ArmRaiseState,    # noqa: E402
                      HeadTiltDecision, HeadTiltState,
                      selftest_arm_raise_decision,
                      selftest_head_tilt_decision)
from output import FrameRenderer, selftest_action_overlay  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构造参数表（本入口专属，与另外两个入口的参数表互不影响）。"""
    p = argparse.ArgumentParser(
        description="全身动作识别 demo（举手 / 仰头）—— 45° 前侧机位，独立动作线")

    p.add_argument("--camera", type=int, metavar="INDEX",
                   default=DEFAULT_CAMERA_INDEX,
                   help=f"摄像头编号（默认 {DEFAULT_CAMERA_INDEX} = USB 外接/45° 那台；"
                        f"传 0 用笔记本内置，内置是正对机位，仰头特征会判几何退化）")
    p.add_argument("--arm-rule", choices=("off", "raise"), default="off",
                   help="举手这条线怎么判：off（默认）= 只报角度不判状态；"
                        "raise = 按上臂抬升角（<= enter 判 RAISED）跑状态机。"
                        "默认 off 是因为要先看几眼数据再定触发用抬升角还是肘角")
    p.add_argument("--arm-raise-enter", type=float, default=45.0,
                   help="举手触发：抬升角 <= 此值判 RAISED（度；越小越举）。"
                        "默认 45 = 抬到 45° 斜上以里；**占位值**，不是标定结果")
    p.add_argument("--arm-raise-exit", type=float, default=60.0,
                   help="举手恢复：抬升角 >= 此值判 DOWN（度，必须大于 enter）。"
                        "两者之间是迟滞带（防阈值附近抖动报警）")
    p.add_argument("--head-tilt-enter", type=float, default=20.0,
                   help="仰头触发：耳→鼻 与水平线夹角 >= 此值判 TILTED_BACK（度，"
                        "正=仰头）。**占位值**，不是标定结果")
    p.add_argument("--head-tilt-exit", type=float, default=10.0,
                   help="仰头恢复：角度 <= 此值判 NEUTRAL（度，必须小于 enter）")
    p.add_argument("--tilt-min-dx-ratio", type=float, default=0.15,
                   help="仰头的横向退化门控：|Δx|（鼻相对耳中点的水平偏移）÷ 双耳"
                        "间距 低于此值视为几何退化（正对镜头），读数报 N/A 而不是"
                        "给出 ±90 的假数。实测 45° 前侧 0.43~0.52、正对 0.013，"
                        "默认 0.15 取在空档中间")
    p.add_argument("--no-geometry", dest="geometry", action="store_false",
                   help="不画角度几何辅助线（举手的肩-肘-腕 + 向上竖直参考线、"
                        "仰头的耳中点→鼻 + 水平参考线）")
    p.set_defaults(geometry=True)
    p.add_argument("--no-skeleton", dest="draw_skeleton", action="store_false",
                   help="不把姿态骨架（关键点+连线+pid/置信度标签）叠到画面上")
    p.set_defaults(draw_skeleton=True)
    p.add_argument("--snapshot", metavar="PATH", default=None,
                   help="按 s 把当前帧（含叠加）存成图片到该路径；不指定则 s 无效。"
                        "注意 data/ 下的实拍画面是个人数据，.gitignore 只挡了部分"
                        "文件名模式 —— 存进去前想清楚要不要提交")
    p.add_argument("--selftest", action="store_true",
                   help="用合成数据自测本动作线各层，不打开摄像头")
    return p


def run_pipeline(args: argparse.Namespace) -> None:
    """动作线主循环（单进程）：摄像头 → pose → 两个动作特征 → 两个判定 → 画面。

    只装配本动作线的对象：坐姿/CVA/头前伸那条线的对象**一个都不建**
    （不更新、不画、不提醒）。举手那条线的判定按 --arm-rule 决定建不建。
    """
    import cv2

    src = CameraCapture(args.camera if args.camera is not None else DEFAULT_CAMERA_INDEX,
                        width=1280, height=720)
    arm_feats = ArmRaiseFeatures()
    head_feats = HeadTiltFeatures(min_dx_span_ratio=args.tilt_min_dx_ratio)
    # 举手：默认（off）**不建判定对象** —— 画面上明说"只报数不判"，不假装判过
    arm_dec = (None if args.arm_rule == "off" else
               ArmRaiseDecision(enter_deg=args.arm_raise_enter,
                                exit_deg=args.arm_raise_exit))
    head_dec = HeadTiltDecision(enter_deg=args.head_tilt_enter,
                                exit_deg=args.head_tilt_exit)
    ren = FrameRenderer(draw_skeleton=args.draw_skeleton)

    if args.snapshot:
        Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)

    print(f"[action] 摄像头：#{src.index}"
          + ("（= 默认，本机为 USB 外接/45° 那台）" if src.index == DEFAULT_CAMERA_INDEX
             else f"（默认是 #{DEFAULT_CAMERA_INDEX}，这里被 --camera 覆盖了）"))
    print("[action] 机位：45° 前侧（与 main_demo.py 同）。举手任意机位可测；"
          "**仰头只在侧对/45° 前侧有意义**，正对摄像头测不到")
    print("[action] 举手判据："
          + ("off —— 只报抬升角/肘角，不判状态（先看数据）"
             if arm_dec is None else
             f"raise —— 抬升角 <= {arm_dec.enter_deg:.1f}° 判 RAISED，"
             f">= {arm_dec.exit_deg:.1f}° 回 DOWN（迟滞带 "
             f"{arm_dec.enter_deg:.1f}~{arm_dec.exit_deg:.1f}°）"))
    print(f"[action] 仰头判据：耳→鼻 与水平线夹角 >= {head_dec.enter_deg:.1f}° "
          f"判 TILTED_BACK，<= {head_dec.exit_deg:.1f}° 回 NEUTRAL"
          f"（**注意**：这个角的零点不是「坐直」—— 45° 机位下中立站姿实测约 "
          f"-20°，阈值要按画面上的中立读数往上加，不是照数字本身）")
    print(f"[action] 仰头退化门控：|Δx|/双耳间距 < {args.tilt_min_dx_ratio:.2f} "
          f"判几何退化（正对镜头）→ 读数 N/A、状态 UNKNOWN（不给假数）")
    print("[action] 阈值都是**占位值**（不是标定结果），读数基线随机位/人变；"
          "本线无提醒、不平滑（丢帧即 N/A），与坐姿/CVA 那条线无关")

    # 控制台只在**状态变化**时打一行（用于确认状态确实在变，不刷屏）
    prev = {ArmRaiseState.UNKNOWN, HeadTiltState.UNKNOWN}
    try:
        with src:
            while True:
                frame = src.read()
                if frame is None:
                    break
                ts = time.monotonic()

                pose = detect_pose(frame)
                arm = arm_feats.update(pose)
                head = head_feats.update(pose)
                arm_state = arm_dec.update(arm) if arm_dec is not None else None
                head_state = head_dec.update(head)

                ren.draw_skeleton_frame(frame, pose)
                ren.draw_action_overlay(
                    frame, arm, arm_state, head, head_state,
                    arm_enter_deg=(None if arm_dec is None else arm_dec.enter_deg),
                    head_enter_deg=head_dec.enter_deg,
                    show_geometry=args.geometry)

                cur = (arm_state, head_state)
                if cur != prev:
                    # 举手默认不判状态：如实打 report-only，别打 N/A ——
                    # "N/A" 会被读成"没读到数"，而这里读数是有、只是没做判定
                    alabel = "report-only" if arm_state is None else arm_state.value
                    deg = arm.get('arm_raise_deg')
                    hdeg = head.get('head_pitch_deg')
                    # 仰头行带上诊断量：这个角在"正对镜头"或"只认出一只耳"时会
                    # 退化成 dx≈0（角度由噪声定符号），光看角度分不清是哪一种 ——
                    # 来源/置信度/两个像素点一摆出来就能分辨
                    conf = head.get('head_pitch_conf')
                    geo = head.get('head_tilt_geom') or {}
                    em, ns = geo.get('ear_mid'), geo.get('nose')
                    ratio = head.get('head_pitch_dx_ratio')
                    diag = (f"src={head.get('head_pitch_pts') or '--'}"
                            f" conf={'--' if conf is None else f'{conf:.2f}'}"
                            f" dx/span={'--' if ratio is None else f'{ratio:.2f}'}")
                    if head.get('head_pitch_degenerate'):
                        diag += " DEGENERATE(正对镜头)"
                    if em and ns:
                        diag += (f" ear_mid=({em[0]:.0f},{em[1]:.0f})"
                                 f" nose=({ns[0]:.0f},{ns[1]:.0f})")
                    print(f"[action] arm={alabel}"
                          f" ({'--' if deg is None else f'{deg:.1f}deg'})"
                          f"  head={head_state.value}"
                          f" ({'--' if hdeg is None else f'{hdeg:.1f}deg'})"
                          f"  [{diag}]",
                          flush=True)
                    prev = cur

                cv2.imshow("Action Demo (RTMPose) - 45deg side", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:      # ESC
                    break
                if key == ord('s') and args.snapshot:
                    cv2.imwrite(args.snapshot, frame)
                    print(f"[snapshot] 已存 {args.snapshot} (t={ts:.1f})")
    finally:
        pass

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# 自测：合成数据把整条链路走一遍（不碰摄像头）
# ---------------------------------------------------------------------------

def _pose(pts: dict, size: tuple, vis: float = 1.0) -> dict:
    """构造一帧合成 pose（17 点，pts 里的点可见，其余 visibility=0）。

    与 features/decision 各自自测里用的构造方式同构，这里独立写一份，避免
    跨模块 import 私有下划线函数。
    """
    landmarks = []
    for pid in range(17):
        if pid in pts:
            x, y = pts[pid]
            landmarks.append((x, y, 0.0, vis))
        else:
            landmarks.append((0.0, 0.0, 0.0, 0.0))
    return {'landmarks': landmarks, 'image_size': size}


def _selftest_end_to_end() -> None:
    """合成帧走完 pose → features → decision → output，确认角度与状态真的会变。

    这是"流程通不通"的检查（不碰摄像头）：同一个 45° 前侧构图下，摆出
    下垂 / 上举 / 仰头 / 丢点 四种情况，看读数与状态是否跟着动、丢点是否如实变 N/A。
    """
    import numpy as np

    size = (640, 480)
    # 45° 前侧构图：双肩 (224,216)/(384,216)px，双耳 (288,120)/(352,120)px，鼻 (397,120)px
    base = {5: (0.35, 0.45), 6: (0.60, 0.45),
            3: (0.45, 0.25), 4: (0.55, 0.25), 0: (0.62, 0.25)}

    arm_feats = ArmRaiseFeatures()
    head_feats = HeadTiltFeatures()
    arm_dec = ArmRaiseDecision(enter_deg=45.0, exit_deg=60.0)
    head_dec = HeadTiltDecision(enter_deg=20.0, exit_deg=10.0)
    ren = FrameRenderer(draw_skeleton=False)

    def step(pts):
        """跑一帧，返回 (arm_feat, arm_state, head_feat, head_state, frame)。"""
        p = _pose(pts, size)
        a = arm_feats.update(p)
        h = head_feats.update(p)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ren.draw_action_overlay(frame, a, arm_dec.update(a), h,
                                head_dec.update(h))
        return a, arm_dec.state, h, head_dec.state, frame

    # 1) 左臂自然下垂（肘在肩正下方）→ 抬升角 ≈180°，判 DOWN
    down = {**base, 7: (0.35, 0.65), 9: (0.35, 0.80)}
    a, a_st, h, h_st, frame = step(down)
    assert a_st is ArmRaiseState.DOWN, f"下垂应判 DOWN，实际 {a_st}"
    assert a['arm_raise_left_deg'] > 170.0, \
        f"下垂抬升角应接近 180°，实际 {a['arm_raise_left_deg']}"
    assert h_st is HeadTiltState.NEUTRAL, "同一帧仰头应 NEUTRAL"
    assert int(np.count_nonzero(frame)) > 0, "画面必须有叠加像素"

    # 2) 左臂上举 → 抬升角很小 → RAISED，且两侧读数都在（右臂仍下垂）
    up = {**base, 7: (0.33, 0.30), 9: (0.31, 0.18)}
    a, a_st, *_ = step(up)
    assert a_st is ArmRaiseState.RAISED, f"上举应判 RAISED，实际 {a_st}"
    assert a['arm_raise_side'] == 'L', "应取举得更高的左侧"
    assert a['arm_raise_deg'] < 30.0, f"上举抬升角应很小，实际 {a['arm_raise_deg']}"
    assert a['arm_raise_right_deg'] is None, "右臂点缺失时右侧应为 N/A（不补假数）"

    # 3) 仰头：鼻抬高（y 变小）→ 俯仰角转正 → TILTED_BACK；手臂放下 → DOWN
    look_up = {**base, 0: (0.62, 0.10), 7: (0.35, 0.65), 9: (0.35, 0.80)}
    a, a_st, h, h_st, *_ = step(look_up)
    assert h_st is HeadTiltState.TILTED_BACK, f"仰头应判 TILTED_BACK，实际 {h_st}"
    assert h['head_pitch_deg'] > 20.0, f"仰头角度应超过 enter，实际 {h['head_pitch_deg']}"
    assert a_st is ArmRaiseState.DOWN, "同一帧举手应仍为 DOWN"

    # 4) 低头（鼻低于耳）→ 角度为负 → 绝不误报仰头
    a, a_st, h, h_st, *_ = step({**base, 0: (0.62, 0.35)})
    assert h['head_pitch_deg'] < 0.0 and h_st is HeadTiltState.NEUTRAL, \
        "低头不该判仰头"

    # 5) 丢点帧（只剩鼻）：两条线都如实变 N/A，**不沿用上一帧的状态/读数**
    a, a_st, h, h_st, *_ = step({0: (0.62, 0.25)})
    assert a_st is ArmRaiseState.UNKNOWN and h_st is HeadTiltState.UNKNOWN, \
        "关键点全丢时应判 UNKNOWN（不沿用旧状态）"
    assert a['arm_raise_deg'] is None and h['head_pitch_deg'] is None

    # 6) 正对镜头（耳都在，鼻几乎在耳中点正前方）→ 横向退化门控作废，
    #    这与上面的"丢点"在画面上是两回事（机位问题 vs 检测问题），
    #    两种情况都必须是 UNKNOWN、都不给假数
    a, a_st, h, h_st, *_ = step({**base, 0: (0.505, 0.25)})
    assert h_st is HeadTiltState.UNKNOWN and h['head_pitch_deg'] is None, \
        "正对镜头应被门控作废（否则会给出 ±90 的假读数）"
    assert h['head_pitch_degenerate'] is True, "作废原因要能被画面/控制台读出来"

    # 6) 画面对丢点帧也要画得出来（N/A 不能被吞掉）
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ren.draw_action_overlay(frame, a, a_st, h, h_st)
    assert int(np.count_nonzero(frame)) > 0, "丢点帧也要画出 N/A 行"

    print("  selftest_end_to_end: OK")


def selftest() -> None:
    """本动作线各层自测 + 端到端合成帧检查（不碰摄像头）。

    只跑本线的用例：坐姿/CVA/头前伸那条线的自测在各自入口里
    （main_demo.py --selftest），这条线不碰它们。
    """
    print("===== 1. features: arm raise angles =====")
    selftest_arm_raise()
    print("===== 2. features: head tilt (pitch proxy) =====")
    selftest_head_tilt()
    print("===== 3. decision: arm raise state =====")
    selftest_arm_raise_decision()
    print("===== 4. decision: head tilt state =====")
    selftest_head_tilt_decision()
    print("===== 5. output: action overlay =====")
    selftest_action_overlay()
    print("===== 6. end-to-end: pose -> features -> decision -> overlay =====")
    _selftest_end_to_end()
    print("\nACTION-LINE SELFTESTS PASSED.")


def main() -> None:
    """动作线入口：只装举手 / 仰头这两个动作，不碰坐姿/CVA 那条线。"""
    args = build_parser().parse_args()
    if args.selftest:
        selftest()
        return
    run_pipeline(args)


if __name__ == "__main__":
    main()
