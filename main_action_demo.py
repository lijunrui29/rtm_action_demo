"""
RTMPose 全身动作识别 demo（十一个动作）—— **单机入口**
（另有双摄入口 main_dual_demo.py，两台并排、只报数不判状态）。

数据源**有且仅有一台**：那台 USB 外接摄像头（见 capture.USB_CAMERA_INDEX），
编号写死在 capture 层，本入口连"选哪台"的参数都没有。机位固定为 45° 前侧，
由**入口文件**声明（不是命令行开关）—— 换机位要重做/重标，别把这里的读数当
跨机位通用。

原有两个动作对这个机位的依赖完全不同（**别混为一谈**）：
  * **举手**：任意机位都能测 —— 上臂朝上在哪个视角都看得见；
  * **仰头**：**只在侧对 / 45° 前侧有意义**。正对摄像头时仰头主要是"朝镜头
    方向的透视缩短"，耳-鼻在像面上几乎不动，读数不反映仰头大小（见
    features.HeadTiltFeatures 的"近似在哪"）。所以本入口**不是"正面入口"**。

原有两个角怎么构造（用哪些关键点，全在 features 层，像素空间）：
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
           刻意**不用"耳→肩"那一类角度**：耳朵几乎就在头后仰的旋转轴心上，仰头
           时耳几乎不动，那个角量的是头相对躯干的**平移**前伸，不是**转动**俯仰，
           两回事（features 层因此没有这类特征）。
           零点不在 0：45° 机位下中立站姿实测约 -20~-30°（鼻尖天生略低于耳中点），
           所以仰头阈值要按画面上的中立读数往上加，别照数字本身调。
           横向退化门控（--tilt-min-dx-ratio）：正对镜头时这个角会退化成
           sign(θ)·90°（只剩符号、没有量值，还长得像正常读数）→ 判几何退化、
           报 N/A，画面上标 [geometry degenerate]（与"关键点丢了"分开）。

2026-10-18 新增九个动作（features 里各一个特征类、decision 里各一个状态机）：
    躯干前屈（肩中点→髋中点 vs 竖直）、颈部侧屈（双耳连线 vs 水平）、
    肘屈伸 / 膝屈伸 / 髋前屈（三点内角，**左右独立、不合并**）、
    躯干旋转（肩线 vs 髋线方向差，看肩髋是否同步转动）、
    肩关节外展（肩→肘 vs 向下竖直轴，**左右独立**）、
    骨盆倾斜（双髋连线 vs 水平）、肩胛对称性（双肩连线 vs 水平）。
    九个都**默认只报数**（`--*-rule` 默认 off），要判状态加对应开关；
    `--actions arm,head,...` 控制本次显示哪些（默认全开，画面共 22 行）。

两个刻意取舍（**别悄悄改**）：
    1) **平滑只做 EMA，且只平滑有效帧**（2026-10-18 用户要求加入，推翻了原先
       "完全不平滑"）：`--smooth-alpha` 默认 0.3，传 1.0 就是原始读数。
       **仍然不"维持上一有效值"**：关键点丢帧照旧标 invalid、读数 N/A；
       几何辅助线仍画原始坐标，不平滑像素点。要看原始抖动就传 1.0。
    2) **举手 + 九个新动作默认只报数、不判状态**（--arm-rule / --*-rule 默认
       off）：先看几眼数据，再决定触发量/阈值。判据类已经写好并自测通过，
       加对应开关就能开。仰头那条线按计划带状态机（角度唯一、没有歧义）。

阈值都是**占位值**、不是标定结果（--arm-raise-enter/-exit、--head-tilt-enter/
-head-tilt-exit 可现场改）：读数的中立基线随摄像头高度/机位/人体比例变化，
先跑起来看画面上的数再定。

用法（在 rtm_action_demo/ 目录下运行；摄像头只能是 USB_CAMERA_INDEX 那台，
没有"选摄像头"的参数 —— 换插口顺序变了就改 capture.USB_CAMERA_INDEX）：
    python main_action_demo.py                              # 十一动作全显示；新动作只报数，仰头判状态
    python main_action_demo.py --actions arm,head            # 只显示原有两条线
    python main_action_demo.py --arm-rule raise              # 举手也开状态机
    python main_action_demo.py --trunk-flex-rule on          # 躯干前屈也开状态机
    python main_action_demo.py --arm-raise-enter 30 --arm-raise-exit 50
    python main_action_demo.py --head-tilt-enter -5 --head-tilt-exit -15
    python main_action_demo.py --smooth-alpha 0.2            # 更强的平滑（默认 0.3）
    python main_action_demo.py --smooth-alpha 1.0            # 关掉平滑，看原始读数
    python main_action_demo.py --no-geometry                 # 不画角度几何辅助线
    python main_action_demo.py --snapshot data/action_snap.png  # 按 s 存当前帧
    python main_action_demo.py --selftest                    # 合成数据自测（不碰摄像头）

按键：s 存快照（需 --snapshot 指定路径，否则忽略）、ESC 退出。
首帧较慢（惰性初始化 pose 检测器），正常。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 本目录（rtm_action_demo）内的模块直接 import
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from capture import CameraCapture                        # noqa: E402
from pose_estimation import detect_pose                  # noqa: E402
from features import (ArmRaiseFeatures,                  # noqa: E402
                      HeadTiltFeatures,
                      TrunkFlexFeatures, NeckLateralTiltFeatures,
                      ElbowFlexFeatures, TrunkRotationFeatures,
                      KneeFlexFeatures, HipFlexFeatures,
                      ShoulderAbductionFeatures, PelvicTiltFeatures,
                      ScapularSymmetryFeatures,
                      selftest_arm_raise, selftest_head_tilt,
                      selftest_trunk_flex, selftest_neck_lateral,
                      selftest_elbow_flex, selftest_trunk_rotation,
                      selftest_knee_flex, selftest_hip_flex,
                      selftest_shoulder_abduction, selftest_pelvic_tilt,
                      selftest_scapular_symmetry, selftest_geometry_gates)
from decision import (ArmRaiseDecision, ArmRaiseState,    # noqa: E402
                      HeadTiltDecision, HeadTiltState,
                      TrunkFlexDecision, NeckLateralTiltDecision,
                      ElbowFlexDecision, TrunkRotationDecision,
                      KneeFlexDecision, HipFlexDecision,
                      ShoulderAbductionDecision, PelvicTiltDecision,
                      ScapularSymmetryDecision,
                      selftest_arm_raise_decision,
                      selftest_head_tilt_decision,
                      selftest_trunk_flex_decision,
                      selftest_neck_lateral_decision,
                      selftest_elbow_flex_decision,
                      selftest_trunk_rotation_decision,
                      selftest_knee_flex_decision,
                      selftest_hip_flex_decision,
                      selftest_shoulder_abduction_decision,
                      selftest_pelvic_tilt_decision,
                      selftest_scapular_symmetry_decision)
from output import (ACTION_ORDER, FrameRenderer,          # noqa: E402
                    selftest_action_overlay,
                    selftest_extended_action_overlay)
from smoothing import AngleSmoother, selftest_angle_smoother  # noqa: E402


def _parse_actions(text: str) -> tuple:
    """解析 --actions：逗号分隔的动作名 → 按 ACTION_ORDER 规范顺序去重的元组。

    顺序固定为画面上从上到下的行序，命令行里写反了也不改变；重复项忽略。
    未知名字直接报错（拼错不该静默少画一块）。
    """
    names = [t.strip() for t in text.split(",") if t.strip()]
    if not names:
        raise argparse.ArgumentTypeError("--actions 不能为空")
    bad = [n for n in names if n not in ACTION_ORDER]
    if bad:
        raise argparse.ArgumentTypeError(
            f"未知动作 {bad}；可选：{','.join(ACTION_ORDER)}")
    chosen = set(names)
    return tuple(a for a in ACTION_ORDER if a in chosen)


def _parse_alpha(text: str) -> float:
    """--smooth-alpha：EMA 系数必须在 (0, 1]；1.0 = 关掉平滑。"""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"alpha 必须是数字（收到 {text!r}）")
    if not (0.0 < value <= 1.0):
        raise argparse.ArgumentTypeError(
            f"--smooth-alpha 必须在 (0, 1] 内（收到 {value}）；1.0 = 不平滑")
    return value


def _parse_positive_float(text: str) -> float:
    """--smooth-fast-delta：必须 > 0（adaptive EMA 的"大跳变"门限）。"""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"必须是数字（收到 {text!r}）")
    if value <= 0.0:
        raise argparse.ArgumentTypeError(f"必须 > 0（收到 {value}）")
    return value


def _parse_nonneg_float(text: str) -> float:
    """--smooth-deadband：必须 >= 0（0 = 关闭死区）。"""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"必须是数字（收到 {text!r}）")
    if value < 0.0:
        raise argparse.ArgumentTypeError(f"必须 >= 0（收到 {value}）")
    return value


def _parse_confirm_frames(text: str) -> int:
    """--confirm-frames：目标状态需连续保持的帧数，必须 >= 1。"""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"必须是整数（收到 {text!r}）")
    if value < 1:
        raise argparse.ArgumentTypeError(f"必须 >= 1（收到 {value}）")
    return value


def build_parser() -> argparse.ArgumentParser:
    """构造参数表（全是动作线自己的旋钮；**没有摄像头选择** —— 只有那一台）。"""
    p = argparse.ArgumentParser(
        description="RTMPose 全身动作识别 demo（举手 / 仰头 + 九个新动作）—— "
                    "45° 前侧机位，数据源固定为 USB 那一台摄像头")

    p.add_argument("--actions", type=_parse_actions,
                   default=ACTION_ORDER, metavar="LIST",
                   help="本次运行显示哪些动作（逗号分隔，默认全开）："
                        + ",".join(ACTION_ORDER) +
                        "。只影响显示，不改变各动作各自的判断开关")
    p.add_argument("--smooth-alpha", type=_parse_alpha, default=0.3, metavar="A",
                   help="角度读数 EMA 平滑系数，取值 (0,1]（默认 0.3）。"
                        "越小越平滑、跟动作越滞后；1.0 = 关掉平滑。"
                        "只平滑有效读数：丢帧仍是 N/A（不维持旧值），"
                        "几何辅助线仍画原始关键点")
    p.add_argument("--smooth-adaptive", choices=("on", "off"), default="on",
                   help="自适应 EMA（默认 on）：读数单帧跳变 >= fast-delta 时把 "
                        "alpha 临时推向 1，真动作当帧跟上；小抖动照常压。"
                        "模块默认 off，入口默认开")
    p.add_argument("--smooth-fast-delta", type=_parse_positive_float,
                   default=10.0, metavar="DEG",
                   help="自适应 EMA 判定'大跳变'的单帧差值（度，默认 10.0）")
    p.add_argument("--smooth-deadband", type=_parse_nonneg_float,
                   default=0.0, metavar="DEG",
                   help="死区（度，默认 0 = 关闭）：原始读数与上一输出值之差"
                        "小于它时沿用上一输出。与相邻两帧原始值比较不同，"
                        "慢速漂移会累积到越带后更新，不会被永久冻住")
    p.add_argument("--confirm-frames", type=_parse_confirm_frames, default=2,
                   metavar="N",
                   help="状态确认帧数（默认 2）：目标状态需连续保持 N 帧才提交，"
                        "压住单帧噪声造成的假状态。1 = 旧行为（无额外延迟）；"
                        "丢帧仍立即回 UNKNOWN")
    p.add_argument("--abduction-ref", choices=("image", "trunk"),
                   default="image",
                   help="肩关节外展的参考轴（默认 image = 画面向下竖直轴）："
                        "trunk = 肩中点→髋中点的躯干轴（更接近'躯干竖直轴'的"
                        "定义，但肩髋中点本身有抖动，属实验选项）；trunk 模式缺"
                        "躯干点直接 N/A，不静默退回 image")
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

    # ---- 九个新动作：判断默认关（只报数），阈值全是占位值、未标定 ----
    p.add_argument("--trunk-flex-rule", choices=("off", "on"), default="off",
                   help="躯干前屈/后伸：on = 开启迟滞状态机（默认 off 只报数）")
    p.add_argument("--trunk-flex-enter", type=float, default=25.0,
                   help="躯干前屈触发：|肩髋连线与竖直轴夹角| >= 此值判 FLEXED"
                        "（度）。**占位值，未标定**")
    p.add_argument("--trunk-flex-exit", type=float, default=15.0,
                   help="躯干前屈恢复：|角度| <= 此值判 NEUTRAL（度，须小于 enter）")
    p.add_argument("--neck-tilt-rule", choices=("off", "on"), default="off",
                   help="颈部侧屈：on = 开启迟滞状态机（默认 off 只报数）")
    p.add_argument("--neck-tilt-enter", type=float, default=15.0,
                   help="颈部侧屈触发：|双耳连线与水平线夹角| >= 此值判 TILTED"
                        "（度）。**占位值，未标定**")
    p.add_argument("--neck-tilt-exit", type=float, default=8.0,
                   help="颈部侧屈恢复：|角度| <= 此值判 NEUTRAL（度，须小于 enter）")
    p.add_argument("--elbow-flex-rule", choices=("off", "on"), default="off",
                   help="肘关节屈伸：on = 左右各开一套迟滞状态机"
                        "（默认 off 只报数）")
    p.add_argument("--elbow-flex-enter", type=float, default=100.0,
                   help="肘屈触发：肘角 <= 此值判 FLEXED（度，180 = 伸直）。"
                        "**占位值，未标定**")
    p.add_argument("--elbow-flex-exit", type=float, default=130.0,
                   help="肘屈恢复：肘角 >= 此值判 EXTENDED（度，须大于 enter）")
    p.add_argument("--trunk-rotation-rule", choices=("off", "on"), default="off",
                   help="躯干旋转：on = 开启迟滞状态机（默认 off 只报数）")
    p.add_argument("--trunk-rotation-enter", type=float, default=15.0,
                   help="躯干旋转触发：|双肩连线与双髋连线方向差| >= 此值判 "
                        "ROTATED（度）。**占位值，未标定**")
    p.add_argument("--trunk-rotation-exit", type=float, default=8.0,
                   help="躯干旋转恢复：|方向差| <= 此值判 NEUTRAL（度，须小于 enter）")
    p.add_argument("--knee-flex-rule", choices=("off", "on"), default="off",
                   help="膝关节屈伸：on = 左右各开一套迟滞状态机"
                        "（默认 off 只报数）")
    p.add_argument("--knee-flex-enter", type=float, default=120.0,
                   help="屈膝触发：膝角 <= 此值判 FLEXED（度，180 = 伸直）。"
                        "**占位值，未标定**")
    p.add_argument("--knee-flex-exit", type=float, default=150.0,
                   help="屈膝恢复：膝角 >= 此值判 EXTENDED（度，须大于 enter）")
    p.add_argument("--hip-flex-rule", choices=("off", "on"), default="off",
                   help="髋关节前屈：on = 左右各开一套迟滞状态机"
                        "（默认 off 只报数）")
    p.add_argument("--hip-flex-enter", type=float, default=120.0,
                   help="屈髋触发：肩-髋-膝在髋处的内角 <= 此值判 FLEXED（度，"
                        "180 = 直立）。**占位值，未标定**")
    p.add_argument("--hip-flex-exit", type=float, default=150.0,
                   help="屈髋恢复：髋角 >= 此值判 EXTENDED（度，须大于 enter）")
    p.add_argument("--shoulder-abduction-rule", choices=("off", "on"),
                   default="off",
                   help="肩关节外展：on = 左右各开一套迟滞状态机"
                        "（默认 off 只报数）")
    p.add_argument("--shoulder-abduction-enter", type=float, default=60.0,
                   help="肩外展触发：外展角 >= 此值判 ABDUCTED（度，0 = 手臂"
                        "垂在体侧、90 = 平举）。**占位值，未标定**")
    p.add_argument("--shoulder-abduction-exit", type=float, default=40.0,
                   help="肩外展恢复：外展角 <= 此值判 ADDUCTED（度，须小于 enter）")
    p.add_argument("--pelvic-tilt-rule", choices=("off", "on"), default="off",
                   help="骨盆倾斜：on = 开启迟滞状态机（默认 off 只报数）")
    p.add_argument("--pelvic-tilt-enter", type=float, default=10.0,
                   help="骨盆倾斜触发：|双髋连线与水平线夹角| >= 此值判 TILTED"
                        "（度）。**占位值，未标定**")
    p.add_argument("--pelvic-tilt-exit", type=float, default=5.0,
                   help="骨盆倾斜恢复：|角度| <= 此值判 NEUTRAL（度，须小于 enter）")
    p.add_argument("--scapular-symmetry-rule", choices=("off", "on"),
                   default="off",
                   help="肩胛对称性：on = 开启迟滞状态机（默认 off 只报数）")
    p.add_argument("--scapular-symmetry-enter", type=float, default=10.0,
                   help="肩胛不对称触发：|双肩连线与水平线夹角| >= 此值判 "
                        "ASYMMETRIC（度）。**占位值，未标定**")
    p.add_argument("--scapular-symmetry-exit", type=float, default=5.0,
                   help="肩胛恢复：|角度| <= 此值判 SYMMETRIC（度，须小于 enter）")
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
    """动作线主循环（单进程）：USB 摄像头 → pose → 各动作特征 → 各动作判定 → 画面。

    arm/head 始终计算；九个新动作只算 `--actions` 里显示的。举手与九个新动作的
    判定按各自的 `--*-rule` 决定建不建（默认 off = 只报数）。
    """
    import cv2

    src = CameraCapture(width=1280, height=720)
    # 平滑器有跨帧状态：单机只有一路，建一个即可（双摄入口是每路一个）
    smoother = AngleSmoother(
        alpha=args.smooth_alpha,
        adaptive=(args.smooth_adaptive == 'on'),
        fast_delta_deg=args.smooth_fast_delta,
        deadband_deg=args.smooth_deadband)
    # 特征对象都是无跨帧状态的（只算不平滑），按 selected 每帧取用
    feats_obj = {
        'arm': ArmRaiseFeatures(),
        'head': HeadTiltFeatures(min_dx_span_ratio=args.tilt_min_dx_ratio),
        'trunk': TrunkFlexFeatures(),
        'neck': NeckLateralTiltFeatures(),
        'elbow': ElbowFlexFeatures(),
        'rotation': TrunkRotationFeatures(),
        'knee': KneeFlexFeatures(),
        'hip': HipFlexFeatures(),
        'abduction': ShoulderAbductionFeatures(reference=args.abduction_ref),
        'pelvic': PelvicTiltFeatures(),
        'scapular': ScapularSymmetryFeatures(),
    }
    # 举手：默认（off）**不建判定对象** —— 画面上明说"只报数不判"，不假装判过
    arm_dec = (None if args.arm_rule == "off" else
               ArmRaiseDecision(enter_deg=args.arm_raise_enter,
                                exit_deg=args.arm_raise_exit,
                                confirm_frames=args.confirm_frames))
    head_dec = HeadTiltDecision(enter_deg=args.head_tilt_enter,
                                exit_deg=args.head_tilt_exit,
                                confirm_frames=args.confirm_frames)
    # 九个新动作默认也只报数：off 时不建判定对象（与 --arm-rule off 同一做法）
    def _dec(rule: str, factory):
        return None if rule == 'off' else factory()

    trunk_dec = _dec(args.trunk_flex_rule, lambda: TrunkFlexDecision(
        enter_deg=args.trunk_flex_enter, exit_deg=args.trunk_flex_exit,
        confirm_frames=args.confirm_frames))
    neck_dec = _dec(args.neck_tilt_rule, lambda: NeckLateralTiltDecision(
        enter_deg=args.neck_tilt_enter, exit_deg=args.neck_tilt_exit,
        confirm_frames=args.confirm_frames))
    rotation_dec = _dec(args.trunk_rotation_rule, lambda: TrunkRotationDecision(
        enter_deg=args.trunk_rotation_enter,
        exit_deg=args.trunk_rotation_exit,
        confirm_frames=args.confirm_frames))
    elbow_decs = (None if args.elbow_flex_rule == 'off' else {
        side: ElbowFlexDecision(side=side, enter_deg=args.elbow_flex_enter,
                                exit_deg=args.elbow_flex_exit,
                                confirm_frames=args.confirm_frames)
        for side in ('L', 'R')})
    knee_decs = (None if args.knee_flex_rule == 'off' else {
        side: KneeFlexDecision(side=side, enter_deg=args.knee_flex_enter,
                               exit_deg=args.knee_flex_exit,
                               confirm_frames=args.confirm_frames)
        for side in ('L', 'R')})
    hip_decs = (None if args.hip_flex_rule == 'off' else {
        side: HipFlexDecision(side=side, enter_deg=args.hip_flex_enter,
                              exit_deg=args.hip_flex_exit,
                              confirm_frames=args.confirm_frames)
        for side in ('L', 'R')})
    abduction_decs = (None if args.shoulder_abduction_rule == 'off' else {
        side: ShoulderAbductionDecision(
            side=side, enter_deg=args.shoulder_abduction_enter,
            exit_deg=args.shoulder_abduction_exit,
            confirm_frames=args.confirm_frames)
        for side in ('L', 'R')})
    pelvic_dec = _dec(args.pelvic_tilt_rule, lambda: PelvicTiltDecision(
        enter_deg=args.pelvic_tilt_enter, exit_deg=args.pelvic_tilt_exit,
        confirm_frames=args.confirm_frames))
    scapular_dec = _dec(args.scapular_symmetry_rule,
                        lambda: ScapularSymmetryDecision(
                            enter_deg=args.scapular_symmetry_enter,
                            exit_deg=args.scapular_symmetry_exit,
                            confirm_frames=args.confirm_frames))
    single_decs = {'trunk': trunk_dec, 'neck': neck_dec,
                   'rotation': rotation_dec,
                   'pelvic': pelvic_dec, 'scapular': scapular_dec}
    bilateral_decs = {'elbow': elbow_decs, 'knee': knee_decs,
                      'hip': hip_decs, 'abduction': abduction_decs}
    ren = FrameRenderer(draw_skeleton=args.draw_skeleton)

    if args.snapshot:
        Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)

    print(f"[action] 数据源：USB 摄像头 #{src.index}（capture.USB_CAMERA_INDEX，"
          f"编号写死、没有 --camera）")
    print("[action] 机位：45° 前侧。举手任意机位可测；"
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
          "本线无提醒")
    smooth_bits = [f"EMA alpha={args.smooth_alpha:g}"]
    if args.smooth_alpha < 1.0:
        smooth_bits.append(
            f"adaptive={'on' if args.smooth_adaptive == 'on' else 'off'}")
        if args.smooth_adaptive == 'on':
            smooth_bits.append(f"fast-delta={args.smooth_fast_delta:g}°")
        smooth_bits.append(f"deadband={args.smooth_deadband:g}°")
    print("[action] 平滑：" + "，".join(smooth_bits)
          + ("（关闭，原始读数）" if args.smooth_alpha >= 1.0 else "")
          + "；只平滑有效帧（丢帧仍 N/A、不维持旧值），几何线仍用原始坐标")
    print(f"[action] 状态确认：confirm_frames={args.confirm_frames}"
          "（目标状态连续保持 N 帧才提交；丢帧立即回 UNKNOWN）")
    print(f"[action] 肩外展参考轴：{args.abduction_ref}"
          + ("（实验：肩中点→髋中点，缺躯干点直接 N/A）"
             if args.abduction_ref == 'trunk' else
             "（画面向下竖直轴）"))
    print(f"[action] 本次显示动作（--actions）：{','.join(args.actions)}")
    rule_desc = [f"trunk={'off' if trunk_dec is None else 'on'}",
                 f"neck={'off' if neck_dec is None else 'on'}",
                 f"elbow={'off' if elbow_decs is None else 'on'}",
                 f"rotation={'off' if rotation_dec is None else 'on'}",
                 f"knee={'off' if knee_decs is None else 'on'}",
                 f"hip={'off' if hip_decs is None else 'on'}",
                 f"abduction={'off' if abduction_decs is None else 'on'}",
                 f"pelvic={'off' if pelvic_dec is None else 'on'}",
                 f"scapular={'off' if scapular_dec is None else 'on'}"]
    print("[action] 新增动作判据：" + ", ".join(rule_desc)
          + "（off = 只报数；on 时左右独立动作左右各一套状态机）")
    print("[action] 控制台仍只在 arm/head 状态变化时打诊断行；新动作读数看画面")

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
                # arm/head 始终计算：控制台的“状态变化”日志沿用原有行为。
                # 平滑在 features 之后、decision / output 之前：判定与画面用的
                # 都是平滑后的读数（丢帧仍 None，不沿用旧值）
                arm = smoother.update(feats_obj['arm'].update(pose))
                head = smoother.update(feats_obj['head'].update(pose))
                feats = {'arm': arm, 'head': head}
                arm_state = arm_dec.update(arm) if arm_dec is not None else None
                head_state = head_dec.update(head)
                states = {'arm': arm_state, 'head': head_state}

                # 新动作只在本次显示时才计算（--actions 控制，省掉没必要的工作）
                for kind in args.actions:
                    if kind not in ('arm', 'head'):
                        feats[kind] = smoother.update(feats_obj[kind].update(pose))
                for kind, dec in single_decs.items():
                    if kind in feats:
                        states[kind] = (None if dec is None else dec.update(feats[kind]))
                bstates = {}
                for kind, decs in bilateral_decs.items():
                    if kind not in feats:
                        continue
                    bstates[kind] = ({} if decs is None else
                                     {side: decs[side].update(feats[kind])
                                      for side in ('L', 'R')})

                items = []
                for kind in args.actions:      # 已按 ACTION_ORDER 规范化
                    if kind in bilateral_decs:
                        items.append((kind, {'features': feats[kind],
                                             'states': bstates.get(kind, {})}))
                    else:
                        if kind == 'arm':
                            dec = arm_dec
                        elif kind == 'head':
                            dec = head_dec
                        else:
                            dec = single_decs[kind]
                        items.append((kind, {
                            'features': feats[kind],
                            'state': states.get(kind),
                            'enter_deg': None if dec is None else dec.enter_deg}))

                ren.draw_skeleton_frame(frame, pose)
                ren.draw_all_action_overlay(frame, items,
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


def _selftest_smoothing_pipeline() -> None:
    """确认平滑真的接在 features 与 decision/output 之间（不是只存在于模块里）。"""
    size = (640, 480)
    base = {5: (0.35, 0.45), 6: (0.60, 0.45)}
    down = {**base, 7: (0.35, 0.65), 9: (0.35, 0.80)}
    up = {**base, 7: (0.33, 0.30), 9: (0.31, 0.18)}
    feats = ArmRaiseFeatures()
    sm = AngleSmoother(alpha=0.5)

    f1 = feats.update(_pose(down, size))
    s1 = sm.update(f1)
    assert s1['arm_raise_deg'] == f1['arm_raise_deg'], "首帧应从原始值起步"

    f2 = feats.update(_pose(up, size))
    s2 = sm.update(f2)
    lo, hi = sorted((f1['arm_raise_deg'], f2['arm_raise_deg']))
    assert lo <= s2['arm_raise_deg'] <= hi, "第二帧应落在两帧原始值之间"
    assert s2['arm_raise_deg'] > f2['arm_raise_deg'], "平滑应被上一帧的旧值拉高"

    # 判定吃的是平滑后的值：原始帧已上举，平滑后仍在下垂侧 → 判 DOWN
    assert ArmRaiseDecision(enter_deg=45, exit_deg=60).update(f2) \
        is ArmRaiseState.RAISED
    assert ArmRaiseDecision(enter_deg=45, exit_deg=60).update(s2) \
        is ArmRaiseState.DOWN, "decision 必须消费平滑后的读数"

    # 丢点帧照旧 N/A：平滑层不维持上一有效值
    s3 = sm.update(feats.update(_pose({0: (0.62, 0.25)}, size)))
    assert s3['arm_raise_deg'] is None

    print("  selftest_smoothing_pipeline: OK")


def _selftest_extended_end_to_end() -> None:
    """九个新动作走完 features → decision → output（合成全身姿态，不碰摄像头）。

    一帧里刻意让**左右不同**（右肘/右膝/右髋屈、左侧伸直），确认左右独立动作
    真的是两套读数/两套状态机；再逐类删点，确认无效侧只影响自己、读数为 N/A。
    """
    import numpy as np

    size = (640, 480)
    # 全身合成姿态（归一化坐标）。右肩略高 → 肩线倾斜，肩髋线方向差 ≈11°
    full = {
        5: (0.40, 0.37), 6: (0.60, 0.33),
        3: (0.45, 0.18), 4: (0.55, 0.18), 0: (0.50, 0.18),
        7: (0.35, 0.50), 9: (0.32, 0.63),      # 左臂伸直
        8: (0.68, 0.45), 10: (0.64, 0.28),     # 右肘上折
        11: (0.42, 0.60), 12: (0.58, 0.60),
        13: (0.43, 0.75), 15: (0.43, 0.90),    # 左腿伸直
        14: (0.72, 0.62), 16: (0.68, 0.75),    # 右膝抬起
    }
    trunk_feats, neck_feats = TrunkFlexFeatures(), NeckLateralTiltFeatures()
    elbow_feats, rotation_feats = ElbowFlexFeatures(), TrunkRotationFeatures()
    knee_feats, hip_feats = KneeFlexFeatures(), HipFlexFeatures()
    abduction_feats, pelvic_feats = ShoulderAbductionFeatures(), PelvicTiltFeatures()
    scapular_feats = ScapularSymmetryFeatures()
    elbow_decs = {s: ElbowFlexDecision(side=s) for s in ('L', 'R')}
    knee_decs = {s: KneeFlexDecision(side=s) for s in ('L', 'R')}
    hip_decs = {s: HipFlexDecision(side=s) for s in ('L', 'R')}
    abduction_decs = {s: ShoulderAbductionDecision(side=s) for s in ('L', 'R')}
    trunk_dec = TrunkFlexDecision()
    neck_dec = NeckLateralTiltDecision()
    rotation_dec = TrunkRotationDecision()
    pelvic_dec = PelvicTiltDecision()
    scapular_dec = ScapularSymmetryDecision()

    def run(pts):
        p = _pose(pts, size)
        return {'trunk': trunk_feats.update(p), 'neck': neck_feats.update(p),
                'elbow': elbow_feats.update(p),
                'rotation': rotation_feats.update(p),
                'knee': knee_feats.update(p), 'hip': hip_feats.update(p),
                'abduction': abduction_feats.update(p),
                'pelvic': pelvic_feats.update(p),
                'scapular': scapular_feats.update(p)}

    # 1) 正常读数 + 左右分离：左侧伸直、右侧屈
    f = run(full)
    assert f['trunk']['trunk_flex_valid'] and \
        abs(f['trunk']['trunk_flex_deg']) < 5.0, "直立躯干应接近 0°"
    assert f['neck']['neck_lateral_valid'] and \
        abs(f['neck']['neck_lateral_deg']) < 3.0, "双耳水平应接近 0°"
    assert f['rotation']['trunk_rotation_valid'] and \
        abs(f['rotation']['trunk_rotation_deg']) > 5.0, "肩线倾斜应产生非零方向差"
    assert f['elbow']['elbow_flex_left_deg'] > 150.0
    assert f['elbow']['elbow_flex_right_deg'] < 60.0
    assert f['knee']['knee_flex_left_deg'] > 150.0
    assert f['knee']['knee_flex_right_deg'] < 100.0
    assert f['hip']['hip_flex_left_deg'] > 150.0
    assert f['hip']['hip_flex_right_deg'] < 110.0
    # 新增三个：肩外展左右独立；骨盆/肩胛各一条连线角
    assert f['abduction']['shoulder_abduction_left_valid'] and \
        f['abduction']['shoulder_abduction_right_valid']
    assert 20.0 < f['abduction']['shoulder_abduction_left_deg'] < 35.0
    assert 35.0 < f['abduction']['shoulder_abduction_right_deg'] < 50.0
    assert f['pelvic']['pelvic_tilt_valid']
    assert abs(f['pelvic']['pelvic_tilt_deg']) < 1.0, "双髋水平应接近 0°"
    assert f['scapular']['scapular_symmetry_valid']
    assert abs(f['scapular']['scapular_symmetry_deg']) > 5.0, \
        "右肩更高应产生非零肩线角"
    # 左右各一套状态机，绝不合并
    e_st = {s: elbow_decs[s].update(f['elbow']) for s in ('L', 'R')}
    k_st = {s: knee_decs[s].update(f['knee']) for s in ('L', 'R')}
    h_st = {s: hip_decs[s].update(f['hip']) for s in ('L', 'R')}
    a_st = {s: abduction_decs[s].update(f['abduction']) for s in ('L', 'R')}
    assert e_st['L'].name == 'EXTENDED' and e_st['R'].name == 'FLEXED'
    assert k_st['L'].name == 'EXTENDED' and k_st['R'].name == 'FLEXED'
    assert h_st['L'].name == 'EXTENDED' and h_st['R'].name == 'FLEXED'
    assert a_st['L'].name == 'ADDUCTED' and a_st['R'].name == 'ADDUCTED'  # <40°
    # 外展左右独立：用更低的阈值验证两侧能分别触发
    a_low = {s: ShoulderAbductionDecision(side=s, enter_deg=20.0,
                                          exit_deg=10.0)
             for s in ('L', 'R')}
    assert a_low['L'].update(f['abduction']).name == 'ABDUCTED'
    assert a_low['R'].update(f['abduction']).name == 'ABDUCTED'
    assert trunk_dec.update(f['trunk']).name == 'NEUTRAL'
    assert neck_dec.update(f['neck']).name == 'NEUTRAL'
    assert rotation_dec.update(f['rotation']).name == 'NEUTRAL'  # 11° < enter 15
    assert pelvic_dec.update(f['pelvic']).name == 'NEUTRAL'      # 0° < enter 10
    assert scapular_dec.update(f['scapular']).name == 'SYMMETRIC'  # 8.5° < enter 10

    # 2) 单侧丢点：只作废该侧，另一侧照常
    no_left_wrist = {k: v for k, v in full.items() if k != 9}
    f = run(no_left_wrist)
    assert f['elbow']['elbow_flex_left_deg'] is None
    assert f['elbow']['elbow_flex_right_deg'] is not None, "右肘不该被左肘拖累"

    no_left_ear = {k: v for k, v in full.items() if k != 3}
    f = run(no_left_ear)
    assert f['neck']['neck_lateral_valid'] is False and \
        f['neck']['neck_lateral_deg'] is None, "缺一只耳应整条 invalid（不降级）"

    # 新增三个的门控：肩外展单侧缺点只废该侧；骨盆/肩胛缺一个点整条 N/A
    no_left_elbow = {k: v for k, v in full.items() if k != 7}
    f = run(no_left_elbow)
    assert f['abduction']['shoulder_abduction_left_deg'] is None
    assert f['abduction']['shoulder_abduction_right_deg'] is not None
    no_right_hip = {k: v for k, v in full.items() if k != 12}
    f = run(no_right_hip)
    assert f['pelvic']['pelvic_tilt_valid'] is False and \
        f['pelvic']['pelvic_tilt_deg'] is None, "缺一髋应整条 invalid（不降级）"
    no_right_shoulder = {k: v for k, v in full.items() if k != 6}
    f = run(no_right_shoulder)
    assert f['scapular']['scapular_symmetry_valid'] is False and \
        f['scapular']['scapular_symmetry_deg'] is None, \
        "缺一肩应整条 invalid（不降级）"

    # 3) 只剩单侧髋可见：另一侧不参与中点，但中点仍有效（spec 明确允许）
    one_hip = {k: v for k, v in full.items() if k != 11}
    f = run(one_hip)
    assert f['trunk']['trunk_flex_valid'] and \
        f['trunk']['trunk_flex_deg'] is not None, "单侧髋可见时中点应仍可算"

    # 4) 双髋全丢：躯干/旋转/膝/髋全部 invalid，读数 N/A
    no_hips = {k: v for k, v in full.items() if k not in (11, 12)}
    f = run(no_hips)
    for key, deg_key in (('trunk', 'trunk_flex'), ('rotation', 'trunk_rotation')):
        assert f[key][f'{deg_key}_valid'] is False
        assert f[key][f'{deg_key}_deg'] is None
    for side, side_key in (('L', 'left'), ('R', 'right')):
        for prefix, kind in (('knee_flex', 'knee'), ('hip_flex', 'hip')):
            assert f[kind][f'{prefix}_{side_key}_deg'] is None

    # 5) 全量叠加帧画得出来（arm/head 未判 + 九个新动作按需判/只报数）
    ren = FrameRenderer(draw_skeleton=False)
    p = _pose(full, size)
    items = [
        ('arm', {'features': ArmRaiseFeatures().update(p), 'state': None}),
        ('head', {'features': HeadTiltFeatures().update(p), 'state': None}),
        ('trunk', {'features': trunk_feats.update(p), 'state': None}),
        ('neck', {'features': neck_feats.update(p), 'state': None}),
        ('elbow', {'features': elbow_feats.update(p), 'states': e_st}),
        ('rotation', {'features': rotation_feats.update(p), 'state': None}),
        ('knee', {'features': knee_feats.update(p), 'states': k_st}),
        ('hip', {'features': hip_feats.update(p), 'states': h_st}),
        ('abduction', {'features': abduction_feats.update(p),
                       'states': a_st}),
        ('pelvic', {'features': pelvic_feats.update(p), 'state': None}),
        ('scapular', {'features': scapular_feats.update(p), 'state': None}),
    ]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ren.draw_all_action_overlay(frame, items)
    assert int(np.count_nonzero(frame)) > 0, "全量叠加必须有像素"

    # 6) --actions 解析：规范顺序、去重、未知名字报错
    assert _parse_actions('hip,arm,trunk,arm') == ('arm', 'trunk', 'hip')
    assert _parse_actions(','.join(ACTION_ORDER)) == ACTION_ORDER
    try:
        _parse_actions('arm,bogus')
    except argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError("未知动作名应报错，不能静默忽略")

    print("  selftest_extended_end_to_end: OK")


def selftest() -> None:
    """各层自测 + 端到端合成帧检查（不碰摄像头、不碰模型）。

    覆盖 features 十一个动作、decision 十一个动作、output 两套叠加、EMA 平滑，
    再加端到端：原举手/仰头链路、九个新动作的全身链路、平滑真的接在判定之前。
    """
    print("===== 1. features: arm raise angles =====")
    selftest_arm_raise()
    print("===== 2. features: head tilt (pitch proxy) =====")
    selftest_head_tilt()
    print("===== 3. features: trunk flex =====")
    selftest_trunk_flex()
    print("===== 4. features: neck lateral tilt =====")
    selftest_neck_lateral()
    print("===== 5. features: elbow flex (L/R) =====")
    selftest_elbow_flex()
    print("===== 6. features: trunk rotation =====")
    selftest_trunk_rotation()
    print("===== 7. features: knee flex (L/R) =====")
    selftest_knee_flex()
    print("===== 8. features: hip flex (L/R) =====")
    selftest_hip_flex()
    print("===== 9. features: shoulder abduction (L/R) =====")
    selftest_shoulder_abduction()
    print("===== 10. features: pelvic tilt =====")
    selftest_pelvic_tilt()
    print("===== 11. features: scapular symmetry =====")
    selftest_scapular_symmetry()
    print("===== 12. features: geometry gates (short segment / out of frame) =====")
    selftest_geometry_gates()
    print("===== 13. decision: arm raise =====")
    selftest_arm_raise_decision()
    print("===== 14. decision: head tilt =====")
    selftest_head_tilt_decision()
    print("===== 15. decision: trunk flex =====")
    selftest_trunk_flex_decision()
    print("===== 16. decision: neck lateral =====")
    selftest_neck_lateral_decision()
    print("===== 17. decision: elbow flex =====")
    selftest_elbow_flex_decision()
    print("===== 18. decision: trunk rotation =====")
    selftest_trunk_rotation_decision()
    print("===== 19. decision: knee flex =====")
    selftest_knee_flex_decision()
    print("===== 20. decision: hip flex =====")
    selftest_hip_flex_decision()
    print("===== 21. decision: shoulder abduction =====")
    selftest_shoulder_abduction_decision()
    print("===== 22. decision: pelvic tilt =====")
    selftest_pelvic_tilt_decision()
    print("===== 23. decision: scapular symmetry =====")
    selftest_scapular_symmetry_decision()
    print("===== 24. output: original action overlay =====")
    selftest_action_overlay()
    print("===== 25. output: extended action overlay =====")
    selftest_extended_action_overlay()
    print("===== 26. smoothing: EMA angle smoother =====")
    selftest_angle_smoother()
    print("===== 27. end-to-end: arm/head chain =====")
    _selftest_end_to_end()
    print("===== 28. end-to-end: nine new actions =====")
    _selftest_extended_end_to_end()
    print("===== 29. end-to-end: smoothing before decision =====")
    _selftest_smoothing_pipeline()
    print("\nACTION-LINE SELFTESTS PASSED.")


def main() -> None:
    """入口：装举手 / 仰头 + 九个新动作（--actions 控制显示，判据各自开关）。"""
    args = build_parser().parse_args()
    if args.selftest:
        selftest()
        return
    run_pipeline(args)


if __name__ == "__main__":
    main()
