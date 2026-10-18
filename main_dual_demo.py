"""
RTMPose 全身动作识别 demo —— **双摄入口**（扩展坞上两台 USB 摄像头并排）。

和单机入口 `main_action_demo.py` 并列：那边是"有且仅有一台"（编号写死在
`capture.USB_CAMERA_INDEX`，连选摄像头的参数都没有）；这边专管两台，编号由
`--cameras A,B` 指定（不传就用 `capture` 里的两个默认常量）。**两个入口互不
影响**，单机那台照旧跑。

画面：**一个窗口左右并排**（左 = cam A、右 = cam B），每半幅各有自己的骨架 +
四行读数/状态，左下角标 `cam A` / `cam B`。`s` 存的是**拼好的整张**。

**两半都必须在屏幕内**：两台 1280x720 并排是 2560 宽，屏幕只有 1280 宽时右半幅
会整个跑到屏幕外（只看得到左边那台，很容易误以为"没拼成两半"）。所以每半幅在
**推理之后**把画面等比缩到「屏幕宽 × 96% ÷ 2」，再绘制 —— 推理仍用摄像头原分辨率
（人像框裁剪能拿到更多像素），pose 是归一化坐标、换掉 image_size 即可；缩放是
等比的，角度类读数不受影响，字号按显示尺寸画，不会随窗口变小。缩放上限可用
`--display-width` 覆盖，传一个很大的值（如 99999）就不缩放。控制台启动时会打一行
实际用的上限。窗口首帧还会被钉到屏幕左上角 —— Windows 默认把新窗口级联放到偏右下，
宽度缩够了也会被位置顶出屏幕。

**本入口两台都只报数、不判状态**（这是刻意的，不是还没做完）：
    两个机位里有几个能测仰头、读数基线各是多少，都要先看数据才知道 ——
    所以两台都不建 decision 对象，画面上两行状态统一写 `not decided (report only)`。
    要判定就用单机入口 `main_action_demo.py`（它带着仰头的状态机）。

`--cameras` 为什么存在：扩展坞/USB Hub 的**枚举顺序常常不稳**，哪台是 A 哪台是 B
不能靠猜。两台各自打开时会打印"编号 + 实到分辨率"，认不准就照着改编号重跑。
带宽也是一样 —— 两台挂在一个 Hub 上可能只能跑低分辨率，`--width/--height` 是
逃生口（一台正常、另一台长期读不到帧，通常就是带宽不够）。

两个动作对机位的依赖完全不同，两台要分开看（详见 CLAUDE.md）：
    * **举手**：任意机位都能测；
    * **仰头**：**只在侧对 / 45° 前侧有意义**。某台若是正对机位，它的仰头读数
      会自己判几何退化 → N/A + 画面上标 [geometry degenerate]（这是特征层已有的
      行为，不需要额外配置）。控制台每台那一行的诊断量（src/conf/dx/span/两个像素点）
      就是用来判断"这台到底能不能测仰头"的。

用法（在 rtm_action_demo/ 目录下运行）：
    python main_dual_demo.py --list-cameras                   # 先查：0..5 哪些编号能用
    python main_dual_demo.py                                 # 两台用 capture 里的默认编号
    python main_dual_demo.py --cameras 2,1                    # 显式指定两台编号
    python main_dual_demo.py --width 640 --height 480         # 带宽不够时降分辨率
    python main_dual_demo.py --display-width 1920              # 窗口整宽上限（默认按屏幕自动）
    python main_dual_demo.py --log-interval 0                 # 控制台不打读数行
    python main_dual_demo.py --no-geometry                    # 不画角度几何辅助线
    python main_dual_demo.py --snapshot data/dual_snap.png    # 按 s 存拼好的当前帧
    python main_dual_demo.py --selftest                       # 合成数据自测（不碰摄像头/模型）

按键：s 存快照（需 --snapshot 指定路径，否则忽略）、ESC 退出（两路一起关）。
各层自测在单机入口（`python main_action_demo.py --selftest`），本入口不重复。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from capture import (CameraCapture, probe_cameras,         # noqa: E402
                     USB_CAMERA_INDEX, USB_CAMERA_INDEX_B)
from pose_estimation import detect_pose                    # noqa: E402
from features import ArmRaiseFeatures, HeadTiltFeatures    # noqa: E402
from output import FrameRenderer                           # noqa: E402

# 拼图窗口标题；两台在拼图/日志/画面标签里的名字（顺序 = 左右 = --cameras 的顺序）
WINDOW_TITLE = "Action Demo (RTMPose) - dual cam"
STREAM_NAMES = ("cam A", "cam B")

# 数据源标签的颜色：正常那路白字，读不到帧那路灰字（"断了"要和"画面暗"分开）
_LABEL_COLOR_OK = (255, 255, 255)
_LABEL_COLOR_DEAD = (170, 170, 170)


class _Stream:
    """一路数据源的全部运行时对象。

    **每台一套，绝不共用**：features 那两个类目前不持有跨帧状态、本可以共用，
    但 decision 那两个类是**有跨帧状态**的（迟滞带靠上一帧的状态），共用就会把
    两台的证据搅在一起。统一按"每路一套"建 —— 便宜，且以后给特征加跨帧状态时
    不会突然出事。（本入口两条线都只报数，干脆不建 decision，见模块说明。）
    """

    def __init__(self, name: str, index: int, width: int, height: int,
                 tilt_min_dx_ratio: float, draw_skeleton: bool) -> None:
        self.name = name
        self.src = CameraCapture(index=index, width=width, height=height)
        self.arm_feats = ArmRaiseFeatures()
        self.head_feats = HeadTiltFeatures(min_dx_span_ratio=tilt_min_dx_ratio)
        self.ren = FrameRenderer(draw_skeleton=draw_skeleton)
        self.last_log_ts = 0.0
        # 最近一次读到的帧尺寸 (h, w)。只记尺寸、不存像素 —— 读不到帧时那半幅
        # 要有个尺寸对得上的黑底占位，但**不拿上一帧顶上**（v1 纪律：丢帧就如实
        # 标出来，不维持上一有效值）。
        self.last_shape: Optional[Tuple[int, int]] = None

    @property
    def index(self) -> int:
        return self.src.index


def _parse_cameras(text: str) -> Tuple[int, int]:
    """`--cameras A,B` → (A, B)。要求正好两个**不同**的整数编号。"""
    parts = [p.strip() for p in text.replace("，", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"要两个编号，例如 --cameras 2,1（收到 {text!r}）")
    try:
        a, b = (int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"编号必须是整数（收到 {text!r}）")
    if a == b:
        raise argparse.ArgumentTypeError(
            f"两个编号不能相同（{a}）—— 那是同一台摄像头，不是两台")
    return a, b


def build_parser() -> argparse.ArgumentParser:
    """构造参数表（本入口专属）。**没有 --arm-rule**：本入口不判状态。"""
    p = argparse.ArgumentParser(
        description="RTMPose 全身动作识别 demo —— 双摄入口：两台 USB 摄像头"
                    "并排在一个窗口，两台都只报数、不判状态")

    p.add_argument("--cameras", type=_parse_cameras, metavar="A,B",
                   default=(USB_CAMERA_INDEX, USB_CAMERA_INDEX_B),
                   help=f"两台的摄像头编号（左 cam A、右 cam B），用逗号分隔，必须不同。"
                        f"默认 {USB_CAMERA_INDEX},{USB_CAMERA_INDEX_B}"
                        f"（capture.USB_CAMERA_INDEX / USB_CAMERA_INDEX_B）。"
                        f"扩展坞枚举顺序不稳 —— 认不准就按两台各自打印的编号试")
    p.add_argument("--width", type=int, default=1280,
                   help="期望分辨率宽（默认 1280）。两台挂一个扩展坞带宽不够时"
                        "降到 640 试试")
    p.add_argument("--height", type=int, default=720,
                   help="期望分辨率高（默认 720，见 --width）")
    p.add_argument("--display-width", type=int, default=0, metavar="PX",
                   help="拼图整窗的最大显示宽度（默认 0 = 自动按屏幕宽度留 4%% 边距）。"
                        "两台 1280x720 并排是 2560 宽，屏幕只有 1280 宽时右半幅会整个"
                        "跑到屏幕外 —— 推理仍用原分辨率，之后每半幅等比缩到这个上限再绘制。"
                        "传一个很大的值（如 99999）可关掉缩放")
    p.add_argument("--tilt-min-dx-ratio", type=float, default=0.15,
                   help="仰头的横向退化门控：|Δx|（鼻相对耳中点的水平偏移）÷ 双耳"
                        "间距 低于此值视为几何退化（正对镜头），读数报 N/A 而不是"
                        "给出 ±90 的假数。实测 45° 前侧 0.43~0.52、正对 0.013，"
                        "默认 0.15 取在空档中间")
    p.add_argument("--log-interval", type=float, default=0.5, metavar="SEC",
                   help="控制台每台每 SEC 秒打一行实时读数（默认 0.5s；0 = 不打）。"
                        "本入口两台都不判状态，没法靠「状态变化」触发打印，所以按时间限流")
    p.add_argument("--no-geometry", dest="geometry", action="store_false",
                   help="不画角度几何辅助线（举手的肩-肘-腕 + 向上竖直参考线、"
                        "仰头的耳中点→鼻 + 水平参考线）")
    p.set_defaults(geometry=True)
    p.add_argument("--no-skeleton", dest="draw_skeleton", action="store_false",
                   help="不把姿态骨架（关键点+连线+pid/置信度标签）叠到画面上")
    p.set_defaults(draw_skeleton=True)
    p.add_argument("--snapshot", metavar="PATH", default=None,
                   help="按 s 把当前帧（拼好的整张、含叠加、显示尺寸）存成图片到该路径；"
                        "不指定则 s 无效。注意 data/ 下的实拍画面是个人数据")
    p.add_argument("--list-cameras", nargs="?", type=int, const=6, default=None,
                   metavar="N",
                   help="只枚举、不跑主循环：把编号 0..N-1 挨个试着打开，打印哪些能开"
                        "和实到分辨率（不带 N 就扫 0..5）。扩展坞枚举顺序不稳、"
                        "--cameras 认不准时先跑这个")
    p.add_argument("--selftest", action="store_true",
                   help="用合成数据自测本入口新增的逻辑（拼图/标签/两路互不干扰），"
                        "不打开摄像头")
    return p


# ---------- 拼图 ----------

def _fit_height(frame: np.ndarray, h: int) -> np.ndarray:
    """把帧等比缩放到高度 h（只缩不放 —— 调用方给的是两者中较小的那个）。"""
    if frame.shape[0] == h:
        return frame
    scale = h / frame.shape[0]
    w = max(1, int(round(frame.shape[1] * scale)))
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def _fit_width(frame: np.ndarray, max_w: int) -> np.ndarray:
    """把帧等比缩到宽度 max_w（只缩不放；max_w <= 0 原样返回）。"""
    if max_w <= 0 or frame.shape[1] <= max_w:
        return frame
    scale = max_w / frame.shape[1]
    h = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (max_w, h), interpolation=cv2.INTER_AREA)


def _retag_image_size(pose: dict, w: int, h: int) -> dict:
    """把 pose 的 image_size 换成显示尺寸（landmarks 是归一化坐标，原样保留）。

    只用于"推理后为显示缩小画面"：特征层与绘制层都用 image_size 把归一化坐标
    换回像素，换掉它就等于"同一组关键点画在缩小的画面上"，角度不受影响。
    """
    return {'landmarks': pose['landmarks'], 'image_size': (w, h)}


def _screen_size() -> Optional[Tuple[int, int]]:
    """主屏幕分辨率（只用来把窗口缩进屏幕）。取不到返回 None。

    Windows 用 user32；其它平台暂不处理 —— 取不到就不缩放，退回原始拼接尺寸。
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        user32 = ctypes.windll.user32
        w, h = int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return None
    return (w, h) if w > 0 and h > 0 else None


def _compose_pair(fa: np.ndarray,
                  fb: np.ndarray) -> Tuple[np.ndarray, int]:
    """左右拼成一张：两半幅等比缩到**同一高度**（取较小的那个，不放大）再 hstack。

    返回 (拼好的图, 左半幅宽度) —— 宽度给调用方定位标签用（缩放后左半幅的宽度
    不等于原帧宽，不能拿原值去算）。
    """
    h = min(fa.shape[0], fb.shape[0])
    a, b = _fit_height(fa, h), _fit_height(fb, h)
    return np.hstack([a, b]), a.shape[1]


def _log_line(name: str, arm: dict, head: dict) -> str:
    """一路的实时读数行（控制台），诊断量与单机入口完全一致。

    仰头那个角在"正对镜头"或"只认出一只耳"时会退化成 dx≈0（角度由噪声定符号），
    光看角度分不清是哪一种 —— 来源/置信度/两个像素点一摆出来就能分辨。双摄时这套
    诊断量正是判断**第二台机位能不能测仰头**的依据，所以照单机入口原样保留。
    """
    deg = arm.get('arm_raise_deg')
    side = arm.get('arm_raise_side') or '--'
    hdeg = head.get('head_pitch_deg')
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
    return (f"[{name}] arm={side}"
            f" {'--' if deg is None else f'{deg:.1f}deg'}"
            f"  head={'--' if hdeg is None else f'{hdeg:.1f}deg'}"
            f"  [{diag}]")


def _open_all(streams: List[_Stream], cameras: Tuple[int, int]) -> bool:
    """打开两台。**少一台就明确失败**（不静默退化成"只跑一台"）。

    用户要的就是两台：真只开起来一台却照样跑，画面上少半幅、控制台少一路，
    很容易被当成"另一台没人站"而不是"另一台没打开"。
    """
    for st in streams:
        if st.src.open():
            continue
        # 失败提示要能指路：把**用户刚敲的那两个编号**回给他等于没提示，
        # 直接指向"先查有哪些编号"那一步。
        print(f"[dual] {st.name}（编号 {st.index}）没打开 —— 两台都得开起来才开跑。"
              f"先看有哪些编号可用：python main_dual_demo.py --list-cameras"
              f"，再按实际编号 --cameras A,B（当前用的是 {cameras[0]},{cameras[1]}）。"
              f"若两台都开得了仍失败，试试降分辨率 --width 640 --height 480")
        for s in streams:       # 已经开起来的那台也放掉，别占着设备
            s.src.release()
        return False
    return True


def run_list_cameras(count: int) -> int:
    """枚举编号 0..count-1（`--list-cameras`）：把"哪一号能用"直接打出来。

    这一档只回答"有哪些编号"，不做别的判断 —— 认设备靠**实到分辨率**（摄像头的
    默认模式，见 `capture.probe_cameras`）。返回退出码：找到至少一台是 0，一台都
    没有是 1。
    """
    count = max(1, count)
    print(f"[dual] 挨个试编号 0..{count - 1}（每个编号最慢几秒，稍等）……")
    results = probe_cameras(count)
    for r in results:
        if r["opened"]:
            h, w = r["shape"]
            print(f"  编号 {r['index']}: 可用  {w}x{h}")
        else:
            print(f"  编号 {r['index']}: 打不开")

    found = [r["index"] for r in results if r["opened"]]
    if not found:
        print("[dual] 一个都没开起来 —— 摄像头没插好、驱动没装上，"
              "或正被别的程序（相机应用 / 会议软件）占着")
        return 1

    print(f"[dual] 一共 {len(found)} 台可用：{', '.join(str(i) for i in found)}")
    # 分辨率只是"认设备"的辅助指纹，两台同型号时它一样 —— 这时候要提醒一句：
    # 要想知道哪台是 45° 前侧那台，得看画面内容（仰头的读数基线也跟着机位变）。
    sizes = [tuple(r["shape"]) for r in results if r["opened"]]
    if len(found) >= 2 and len(set(sizes)) < len(sizes):
        print("[dual] 注：几台的默认分辨率一样，光看这个数分不出哪个编号是哪个机位 ——"
              " 看画面内容最准（哪台正对着人、哪台在 45° 前侧）")
    if len(found) == 2:
        print(f"[dual] 正好两台，接着跑："
              f"python main_dual_demo.py --cameras {found[0]},{found[1]}")
    elif len(found) > 2:
        print("[dual] 不止两台 —— 挑出要用的那两台，用 --cameras A,B 指定"
              "（左 cam A / 右 cam B）")
    else:
        print("[dual] 只找到一台 —— 本入口要两台都开起来才能跑（单机入口一台就够）")
    return 0


def run_pipeline(args: argparse.Namespace) -> int:
    """双摄主循环：两台各读一帧 → 各自 pose/特征 → 各自叠加 → 拼成一个窗口。

    返回进程退出码（0 = 正常退出，1 = 配置/打开失败）。
    """
    idx_a, idx_b = args.cameras
    if idx_a == idx_b:
        # 默认常量被改重了也会走到这里（--cameras 那条路已经在解析时挡掉）
        print(f"[dual] 两台的编号都是 {idx_a} —— 那是同一台摄像头。"
              f"改用 --cameras {idx_a},{idx_b} 指定两台，或改 capture 里的默认常量")
        return 1

    streams = [_Stream(STREAM_NAMES[0], idx_a, args.width, args.height,
                       args.tilt_min_dx_ratio, args.draw_skeleton),
               _Stream(STREAM_NAMES[1], idx_b, args.width, args.height,
                       args.tilt_min_dx_ratio, args.draw_skeleton)]

    if not _open_all(streams, args.cameras):
        return 1

    if args.snapshot:
        Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)

    # 显示宽度上限：两台 1280x720 并排是 2560 宽，屏幕只有 1280 宽时右半幅整个
    # 跑到屏幕外（用户只看得到左边那台）。按屏幕宽度给每半幅定上限，在**读帧之后、
    # 推理之前**缩：读数和骨架都按显示尺寸算，字号不用跟着整窗缩放一起变小
    # （0.5 号字在最长的仰头那行约 580px，640 宽的半幅仍放得下）。
    max_composite_w = args.display_width
    if max_composite_w > 0:
        size_desc = f"--display-width {max_composite_w}"
    else:
        screen = _screen_size()
        max_composite_w = int(screen[0] * 0.96) if screen else 0
        size_desc = (f"屏幕 {screen[0]}x{screen[1]} 留 4% 边距"
                     if screen else "取不到屏幕尺寸，不缩放")
    half_w_limit = max_composite_w // len(streams) if max_composite_w else 0

    print(f"[dual] 两台：{STREAM_NAMES[0]} = #{idx_a}（左），"
          f"{STREAM_NAMES[1]} = #{idx_b}（右）—— 编号认不准就改 --cameras A,B")
    print("[dual] 本入口**两条线都只报数、不判状态**（两台都写 not decided）——"
          "先把两个机位的数看明白；要判状态用单机入口 main_action_demo.py")
    print(f"[dual] 分辨率上限 {args.width}x{args.height}；"
          f"控制台 {'不打' if args.log_interval <= 0 else f'每 {args.log_interval:g}s 每台一行'}"
          f"；某台正对机位时它的仰头会判几何退化、读数 N/A（正常现象）")
    if half_w_limit:
        print(f"[dual] 显示：{size_desc} → 每半幅最宽 {half_w_limit}px"
              f"（整窗约 {half_w_limit * len(streams)}px，两台并排都在屏幕内）")
    else:
        print("[dual] 显示：不缩小，整窗按原始拼接尺寸显示")

    window_ready = False
    try:
        while True:
            halves: List[np.ndarray] = []
            dead: List[bool] = []
            for st in streams:
                frame = st.src.read()
                dead.append(frame is None)
                if dead[-1]:
                    # 这一路断了：那半幅画成黑底占位、循环继续（**不拿旧帧顶上**）
                    h, w = st.last_shape or (args.height, args.width)
                    halves.append(np.zeros((h, w, 3), dtype=np.uint8))
                    continue

                full = frame
                # cache_key=st.name：这个人是"哪一路"，给 pose 层的人像框缓存
                # 分桶用 —— 不分桶的话这路失败帧会拿到另一路的框去裁（见 capture 层）
                pose = detect_pose(full, cache_key=st.name)

                # **显示缩放在推理之后**：推理仍用摄像头原分辨率（人像框裁剪能拿到
                # 更多像素），之后把画面缩到半幅宽度，pose 的 landmarks 是归一化
                # 坐标、只需换掉 image_size —— 特征层按新尺寸换算像素，等比缩放
                # 不改变任何角度读数；字号（0.5）按显示尺寸画，不会随窗口一起变小。
                frame = _fit_width(full, half_w_limit)
                st.last_shape = frame.shape[:2]
                if pose is not None and frame is not full:
                    pose = _retag_image_size(pose, frame.shape[1], frame.shape[0])
                arm = st.arm_feats.update(pose)
                head = st.head_feats.update(pose)

                st.ren.draw_skeleton_frame(frame, pose)
                # 状态一律传 None：本入口两台都不判状态，画面上两行写
                # "not decided (report only)"——不假装判过
                st.ren.draw_action_overlay(frame, arm, None, head, None,
                                           show_geometry=args.geometry)
                halves.append(frame)

                now = time.monotonic()
                if args.log_interval > 0 and now - st.last_log_ts >= args.log_interval:
                    st.last_log_ts = now
                    print(_log_line(st.name, arm, head), flush=True)

            composite, split_x = _compose_pair(halves[0], halves[1])
            # 标签画在**拼好的整张**上（不跟着半幅的缩放走），位置在左下角 ——
            # 上面四行读数从 y=30 起排到 ~114，左下角空着，不打架
            ch = composite.shape[0]
            for i, st in enumerate(streams):
                x = 12 if i == 0 else split_x + 12
                text = f"{st.name}  (no signal)" if dead[i] else st.name
                st.ren.draw_stream_label(
                    composite, text, (x, ch - 12),
                    color=_LABEL_COLOR_DEAD if dead[i] else _LABEL_COLOR_OK)

            if not window_ready:
                # 首帧建窗口并**钉到左上角**：Windows 会把新窗口级联放到 (200,200)
                # 一类的偏移位置，就算窗口宽度已经按屏幕缩过，右边缘仍会出一截屏幕
                cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow(WINDOW_TITLE, 0, 0)
                window_ready = True
            cv2.imshow(WINDOW_TITLE, composite)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:      # ESC
                break
            if key == ord('s') and args.snapshot:
                cv2.imwrite(args.snapshot, composite)
                print(f"[snapshot] 已存 {args.snapshot}（拼好的整张，"
                      f"{composite.shape[1]}x{composite.shape[0]}）")
    finally:
        for st in streams:
            st.src.release()
        cv2.destroyAllWindows()

    return 0


# ---------------------------------------------------------------------------
# 自测：只测本入口新增的逻辑（拼图 / 标签 / 两路互不干扰），不碰摄像头
# ---------------------------------------------------------------------------

def _pose(pts: dict, size: tuple, vis: float = 1.0) -> dict:
    """构造一帧合成 pose（17 点，pts 里的点可见，其余 visibility=0）。"""
    landmarks = []
    for pid in range(17):
        if pid in pts:
            x, y = pts[pid]
            landmarks.append((x, y, 0.0, vis))
        else:
            landmarks.append((0.0, 0.0, 0.0, 0.0))
    return {'landmarks': landmarks, 'image_size': size}


def selftest() -> None:
    """本入口新增逻辑的自测：拼图尺寸/标签/两路互不干扰。

    各层自己的自测在单机入口（main_action_demo.py --selftest），这里不重复，
    也不 import 那个入口模块（入口之间不互相 import）。
    """
    lab = FrameRenderer(draw_skeleton=False)

    def label_pixels(img, x0, x1, y0, y1):
        """某个区域里的非黑像素数（用来确认标签真的画上去了）。"""
        return int(np.count_nonzero(img[y0:y1, x0:x1].any(axis=2)))

    # 1) 同尺寸两帧：等高、宽度相加、两半幅内容都在
    fa = np.zeros((120, 160, 3), dtype=np.uint8)
    fa[40:80, 40:120] = 200                     # 左半幅中间一块亮区
    fb = np.zeros((120, 200, 3), dtype=np.uint8)
    fb[10:30, 10:60] = 120                      # 右半幅左上角一块亮区
    comp, split = _compose_pair(fa, fb)
    assert comp.shape == (120, 360, 3), f"拼图尺寸不对：{comp.shape}"
    assert split == 160, f"左半幅宽度应是 160，实际 {split}"
    assert int(comp[40:80, 40:120].max()) == 200, "左半幅内容丢了"
    assert int(comp[10:30, 170:220].max()) == 120, "右半幅内容丢了"

    # 2) 不同高度：缩到较小的那个（不放大），宽按比例 —— 只有高的那半缩
    tall = np.zeros((120, 160, 3), dtype=np.uint8)
    short = np.zeros((60, 100, 3), dtype=np.uint8)
    comp, split = _compose_pair(tall, short)
    assert comp.shape[0] == 60, f"应缩到较小高度 60，实际 {comp.shape[0]}"
    assert split == 80, f"120x160 缩到高 60 后宽应是 80，实际 {split}"
    assert comp.shape[1] == 80 + 100, \
        f"总宽应是 80+100（矮的那半不动），实际 {comp.shape[1]}"
    # 反过来放：高的在右，矮的仍在左且不动
    comp2, split2 = _compose_pair(short, tall)
    assert split2 == 100 and comp2.shape == (60, 180, 3), \
        f"左右互换后应仍是 100+80 @60，实际 {split2} / {comp2.shape}"

    # 3) 一帧是"读不到帧"的黑底占位：拼图照样成立（这就是那半幅的实时画面）
    black = np.zeros((120, 160, 3), dtype=np.uint8)
    comp, split = _compose_pair(black, fa)
    assert comp.shape == (120, 320, 3), f"含占位黑底的拼图尺寸不对：{comp.shape}"
    assert split == 160

    # 4) 数据源标签：画在拼图左下角，两半幅各一个；两侧都能数到像素
    lab.draw_stream_label(comp, "cam A", (12, comp.shape[0] - 12))
    lab.draw_stream_label(comp, "cam B", (split + 12, comp.shape[0] - 12))
    assert label_pixels(comp, 8, 70, comp.shape[0] - 22, comp.shape[0]) > 0, \
        "cam A 标签没画上"
    assert label_pixels(comp, split + 8, split + 80,
                        comp.shape[0] - 22, comp.shape[0]) > 0, "cam B 标签没画上"
    assert label_pixels(comp, 75, split - 5, comp.shape[0] - 22, comp.shape[0]) == 0, \
        "两个标签之间不该有像素（标签没跑到别人的半幅上）"

    # 5) 两路各一套对象 → 一路丢点不影响另一路的读数与画面
    #    （features 目前无跨帧状态、本可共用，但"每路一套"这个模式要钉住：
    #     decision 那两个类是有跨帧状态的，共用就会把两台的证据搅在一起）
    size = (640, 480)
    base = {5: (0.35, 0.45), 6: (0.60, 0.45), 3: (0.45, 0.25), 4: (0.55, 0.25),
            0: (0.62, 0.25)}
    a_arm_feats, a_head_feats = ArmRaiseFeatures(), HeadTiltFeatures()
    b_arm_feats, b_head_feats = ArmRaiseFeatures(), HeadTiltFeatures()
    a_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    b_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # A 路：只剩一个鼻点 → 两条线都算不出，如实 N/A
    a_pose = _pose({0: (0.62, 0.25)}, size)
    a_arm = a_arm_feats.update(a_pose)
    a_head = a_head_feats.update(a_pose)
    # B 路：上举 + 仰头 → 两条线都有读数
    b_pose = _pose({**base, 0: (0.62, 0.10), 7: (0.33, 0.30), 9: (0.31, 0.18)}, size)
    b_arm = b_arm_feats.update(b_pose)
    b_head = b_head_feats.update(b_pose)

    assert a_arm['arm_raise_deg'] is None and a_head['head_pitch_deg'] is None, \
        "A 路丢点时应如实 N/A"
    assert b_arm['arm_raise_deg'] is not None and b_arm['arm_raise_deg'] < 30.0, \
        "B 路上举应有很小的抬升角"
    assert b_head['head_pitch_deg'] is not None and b_head['head_pitch_deg'] > 20.0, \
        "B 路仰头应有正角"

    # 两路各画各的帧：都画出四行（含 N/A 与 not decided），互不覆盖
    FrameRenderer(draw_skeleton=False).draw_action_overlay(
        a_frame, a_arm, None, a_head, None)
    FrameRenderer(draw_skeleton=False).draw_action_overlay(
        b_frame, b_arm, None, b_head, None)
    assert int(np.count_nonzero(a_frame[30:115, :])) > 0, \
        "A 路的读数/状态行应在行位内画出来（N/A 也要画）"
    assert int(np.count_nonzero(b_frame[30:115, :])) > 0, "B 路的读数/状态行没画出来"
    # 两帧都是黑底，读数行区域之外应当干干净净（没有几何线画进来）
    assert int(np.count_nonzero(a_frame[120:, :])) == 0, \
        "只给读数不给几何时，下半幅不该有笔画"

    print("  selftest_dual: OK")


def main() -> int:
    """双摄入口：两台各跑一条只报数的动作线，拼成一个窗口。"""
    args = build_parser().parse_args()
    if args.selftest:
        selftest()
        print("dual selftest: ALL PASSED")
        return 0
    if args.list_cameras is not None:
        # 纯枚举，不跑主循环 —— 先查编号，认准了再跑
        return run_list_cameras(args.list_cameras)
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
