"""
capture 层：只负责拿到一帧一帧的图像，不关心图像来自哪里。

上层（pose estimation 等）只依赖 FrameSource 的三个方法：
    open()    打开图像源，返回是否成功
    read()    读取一帧，返回 BGR 帧；结束/出错返回 None
    release() 释放图像源

当前只有 CameraCapture（USB 摄像头）。以后要换成机器人摄像头，
只需新增一个 FrameSource 子类（如 RTCameraCapture），上层代码不用改。

约定：read() 返回的帧是 BGR 顺序的 numpy 数组（OpenCV 默认格式）。
需要 RGB 的转换（如 mediapipe）留给上层自己做。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

# 一帧图像的别名：BGR 的 numpy 数组
Frame = np.ndarray


class FrameSource(ABC):
    """图像源抽象基类：统一管理"打开/读取/释放"的生命周期。

    子类只需实现 _open/_read/_release 三个方法，
    生命周期状态和 with 语句支持由基类统一处理。
    """

    def __init__(self) -> None:
        self._cap = None          # 子类用它保存底层句柄（如 cv2.VideoCapture）
        self._is_opened = False

    @property
    def is_opened(self) -> bool:
        return self._is_opened

    def open(self) -> bool:
        """打开图像源，返回是否成功。"""
        self.release()  # 重复调用 open 时先清理旧资源
        self._cap = self._open()
        self._is_opened = self._cap is not None
        return self._is_opened

    def read(self) -> Optional[Frame]:
        """读取一帧（BGR）。图像源结束或出错时返回 None。"""
        if not self._is_opened:
            return None
        return self._read()

    def release(self) -> None:
        """关闭图像源并释放资源。"""
        if self._cap is not None:
            self._release()
            self._cap = None
        self._is_opened = False

    def __enter__(self) -> "FrameSource":
        if not self.open():
            raise RuntimeError("图像源打开失败")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # ---------- 以下由子类实现 ----------

    @abstractmethod
    def _open(self):
        """打开图像源，成功返回句柄对象，失败返回 None。"""

    @abstractmethod
    def _read(self) -> Optional[Frame]:
        """用 self._cap 读取一帧，返回 BGR 帧；结束/失败返回 None。"""

    @abstractmethod
    def _release(self) -> None:
        """释放 self._cap 指向的底层资源。"""


# Windows 摄像头后端依次尝试顺序：DSHOW → MSMF → 平台默认（None）。
# MSMF 常见"能打开但一直抓不到帧"（cap_msmf.cpp 报 -1072875772）的毛病，
# DSHOW 一般更稳；非 Windows 上 DSHOW/MSMF 打开会失败，自然落到平台默认。
_CAMERA_BACKENDS = (cv2.CAP_DSHOW, cv2.CAP_MSMF, None)
_OPEN_PROBE_RETRIES = 3    # 打开后实抓一帧验证，最多试几次（有的后端 opened() 为真但抓不到）
_READ_RETRIES = 5          # 读帧瞬时失败重试次数（避免一帧抽风把整条链路当"视频结束"）
_READ_RETRY_SLEEP = 0.05   # 每次重试前的等待（秒）


# 摄像头编号。**机器相关的猜测**，换机器 / 换插口顺序会变 —— 那时改这里。
# 本机实测的设备顺序与"机位指纹"（用仰头的 |Δx|/双耳间距 区分，见 features.py）：
#     index 0 = 笔记本内置（正对着人 → 仰头特征判几何退化）
#     index 1 = USB 外接，正对机位
#     index 2 = USB 外接，扩展坞上那台，固定在 45° 前侧
# 认不准就先把 0/1/2… 各试一遍，存一帧看画面里人是从哪个角度被拍的。
#
# 单机入口（main_action_demo.py）**只用 USB_CAMERA_INDEX 这一台**、且不接受任何
# 编号参数 —— "有且仅有一台"在它那里是结构上的保证。
# 2026-10-18 实拍确认（看存下来的帧比对，不是只凭 dx/span 猜）：
#     index 0 = 笔记本内置（Windows 相机隐私关着，目前只返回灰色占位图）
#     index 1 = **正对机位**那台（直连 USB 口）
#     index 2 = **扩展坞上的 45° 前侧**那台
# 早先一度把 1/2 记反了（45° 那台记成了 1），导致单机入口其实对着正面机位跑。
USB_CAMERA_INDEX = 2

# 双摄入口（main_dual_demo.py）的第二台：正对机位那台。
# ⚠ 扩展坞/USB Hub 的枚举顺序常常不稳（还会留下已拔设备的幽灵条目），
#    插上后大概率要用 `--cameras A,B` 覆盖它。认不准时看两台各自打印的
#    "编号 + 实到分辨率"和控制台 dx/span。
USB_CAMERA_INDEX_B = 1


class CameraCapture(FrameSource):
    """USB 摄像头图像源（用 OpenCV 的 cv2.VideoCapture）。

    编号**默认取模块常量 `USB_CAMERA_INDEX`**（那台 45° 前侧的 USB 摄像头）：
    单机入口不传编号，用的就是它。双摄入口会**显式传**两台各自的编号 ——
    编号的默认值与覆盖入口都收敛在这个模块里，不让上层记住"第几号是哪个机位"。
    """

    def __init__(self, index: Optional[int] = None,
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 quiet: bool = False):
        super().__init__()
        self.index = USB_CAMERA_INDEX if index is None else index
        self.width = width      # 可选的期望分辨率，摄像头不一定支持
        self.height = height
        # 探编号时（probe_cameras 一口气试好几个编号）每号都打一行"失败/释放"
        # 会淹掉结果表，这时置 True 静音。跑主循环时保持默认 False。
        self.quiet = quiet

    def _open(self):
        for backend in _CAMERA_BACKENDS:
            try:
                cap = (cv2.VideoCapture(self.index, backend)
                       if backend is not None else cv2.VideoCapture(self.index))
            except Exception:
                continue
            if not cap.isOpened():
                cap.release()
                continue
            if self.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # 实抓一帧验证：有的后端能打开但一直读不到帧，那种直接弃用换下一个
            for _ in range(_OPEN_PROBE_RETRIES):
                ok, frame = cap.read()
                if ok:
                    # 打编号 + 实到分辨率：编号认不准时（曾把"本地摄像头 #1"
                    # 读成"内置那台"）这两个数就是判断依据，别只打"USB 摄像头"
                    if not self.quiet:
                        size = ("" if frame is None
                                else f", {frame.shape[1]}x{frame.shape[0]}")
                        print(f"[capture] 已打开 USB 摄像头 #{self.index} "
                              f"(backend={backend if backend is not None else 'default'}"
                              f"{size})")
                    return cap
            cap.release()
        if not self.quiet:
            print(f"[capture] 打开 USB 摄像头失败：编号 {self.index} 不存在、被占用，"
                  f"或不是想要的那台 —— 若插口/扩展坞顺序变了：单机入口改 "
                  f"capture.USB_CAMERA_INDEX；双摄入口先跑 --list-cameras 看有哪些编号，"
                  f"再用 --cameras A,B 指定")
        return None

    def _read(self) -> Optional[Frame]:
        for _ in range(_READ_RETRIES):
            ret, frame = self._cap.read()
            if ret:
                return frame
            time.sleep(_READ_RETRY_SLEEP)
        return None

    def _release(self) -> None:
        self._cap.release()
        if not self.quiet:
            print("[capture] 摄像头已释放")


def probe_cameras(count: int = 6) -> List[Dict[str, object]]:
    """把编号 `0..count-1` 挨个试着打开、实抓一帧 —— 回答"哪一号是我那台摄像头"。

    扩展坞/USB Hub 的**枚举顺序不稳**（哪台是 #1 哪台是 #3 不能靠猜），
    双摄入口的 `--list-cameras` 就是用这个函数把可用的编号直接列出来。

    **不设分辨率**：报出来的是摄像头的默认模式 —— 认设备靠这条指纹最省事
    （比如"1080P USB Camera"报 1920x1080，内置那台报 1280x720）。
    要试某个特定分辨率能不能跑，那是 `--width/--height` 的事，不在这里。

    返回 `[{"index": int, "opened": bool, "shape": (高, 宽) 或 None}, ...]`，
    按编号升序。**本函数不抛异常**：某个编号打不开就是 opened=False。
    """
    results: List[Dict[str, object]] = []
    for i in range(max(0, count)):
        src = CameraCapture(index=i, quiet=True)
        opened = src.open()
        frame = src.read() if opened else None
        src.release()
        results.append({
            "index": i,
            # opened 已是"实抓到一帧"的意思（_open 里探测过），再确认一次读到帧
            "opened": bool(opened and frame is not None),
            "shape": None if frame is None else frame.shape[:2],
        })
    return results



