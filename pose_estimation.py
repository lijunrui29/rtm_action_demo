"""
Pose Estimation 核心模块：输入一帧图像，输出 17 个 COCO 关键点。

用 RTMPose（OpenMMLab）的 ONNX 模型做推理，替换原来的 MediaPipe。
只做检测，不做任何判断。返回归一化坐标，不包含绘制/规则逻辑。

模型文件（end2end.onnx）在本文件同目录的 models/ 下，用绝对路径，
避免依赖工作目录。运行前请先装 onnxruntime：
    pip install onnxruntime

RTMPose 是 top-down 模型，本来需要先框出人的包围盒再推理。本项目**没有检测器**，
按"单一人"处理：**整帧推理**（letterbox）就是常态路径；整帧分数掉到阈值下时，
再用 本帧锚点 / 最近成功帧的人像框 裁剪重推一次（见 detect_pose_with_score）。

"每帧都先裁剪再推理"（PERSON_CROP_ALWAYS=True）**实测过，默认关闭**：同一段
录像回放下，裁剪路径的关键点抖动反而更大（4.1px → 13.8px），因为裁剪框每帧
由上一帧结果重算，框自己会抖，把误差反馈进了下一帧。别再默认打开。
要接多人 / 人不在画面中心，见 CLAUDE.md 的后续路线（接 RTMDet）。

**每一路画面各自独立**：推理会话 `_sess` 是无状态的、可共用；但"最近成功帧的
人像框"是跨帧缓存，**必须按数据源分桶** —— 见 detect_pose 的 cache_key 参数。
多台摄像头同时跑时每路传一个不同的 key，否则一路的失败帧会拿另一路的框去裁。

接口约定（保持稳定，别改签名，与 MediaPipe 版一致）：
    detect_pose(frame_bgr) -> dict | None
        返回 {'landmarks': [(x, y, z, visibility) x17], 'image_size': (w, h)}
        关键点编号是 COCO 17 点标准（不是 MediaPipe 33 点）：
            0 鼻, 1 左眼, 2 右眼, 3 左耳, 4 右耳,
            5 左肩, 6 右肩, 7 左肘, 8 右肘,
            9 左腕, 10 右腕, 11 左髋, 12 右髋,
            13 左膝, 14 右膝, 15 左踝, 16 右踝
        两个动作用到的点：肩 5/6、肘 7/8、腕 9/10（举手），耳 3/4、鼻 0（仰头）。
    detect_pose_with_score(frame_bgr) -> (dict | None, float)
        调试/标定接口：结果与 detect_pose 一致，额外返回本帧整体
        置信度分数（未检测到也返回），供对照 DETECT_SCORE_THRESHOLD 用。
    landmarks_to_pixels(landmarks, image_size) -> [(x, y) x17]
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / 'models'
# 用 glob 找 onnx，模型目录（含 20230831/rtmpose_onnx/<name>/）一旦换版本也能找到
_ONNX_FILES = sorted(MODEL_DIR.rglob('end2end.onnx'))
if _ONNX_FILES:
    MODEL_PATH = str(_ONNX_FILES[0])
else:
    MODEL_PATH = str(MODEL_DIR / 'end2end.onnx')  # 占位，get_detector() 里报错

# RTMPose 输入尺寸（与 pipeline.json 的 image_size 一致，顺序 [w, h]）
INPUT_SIZE = (192, 256)

# 归一化参数（ImageNet 统计，RTMPose 训练用；顺序 BGR/RGB 见下）
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# "检测到人"的整体置信度阈值：17 个关键点 SIMCC 峰值（softmax 最大值）的平均。
# 2026-08-16 从 0.3 降到 0.15：下肢/踝部常被截断出画或离得远，
# 这些低分点把平均值拖到 0.3 以下 → 频繁判"未检测到人"。0.15 更易检出。
# 调太高会漏检，调太低会对空场景误检出随机关键点；真机标定可在此改。
DETECT_SCORE_THRESHOLD = 0.15

# 整帧失败时的"人像区域重试"（2026-08-18，实测标定，详见 detect_pose_with_score）：
# 人占画面偏小/身体转过一定角度时，整帧推理 17 点一起掉到阈值下，但其实人还在
# 画面里、模型只是没锁住。此时用 置信度 >= ANCHOR_CONF 的点（或最近一次成功帧的
# 人像框，时间连续）裁剪放大重推，通常能把 mean 拉回阈值上。
# ⚠ 这是无检测器下的**临时替代**（见 CLAUDE.md 开放问题）：将来接 RTMDet 后可以去掉。
ANCHOR_CONF = 0.2          # 视为"锚点"的最低置信度（实测失败帧最高点 ~0.22）
PERSON_BBOX_VALID_SEC = 1.0  # 上次成功帧人像框的有效期（人不动时沿用；走远/换人自动失效）

# 每帧都先按人像框裁剪推理（top-down）——**实验开关，默认关闭**（2026-10-17 实测）：
# 理论上裁剪后人在模型输入里占满，精度应更好；但在本项目的数据上**实测更差**：
# 同一段 120 帧录像回放，整帧推理相邻帧关键点抖动 4.1px，裁剪路径 13.8px。
# 原因是裁剪框每帧都从上一帧的结果重算，框本身会抖，误差被反馈进下一帧；
# 换成固定框/加大外扩也都没赢过整帧推理。所以常态路径保持整帧推理，
# 裁剪只用于"整帧失败时重试"。要把这里改成 True 做对照，请先在实拍录像上量化。
PERSON_CROP_ALWAYS = False
PERSON_CROP_PAD = 1.25     # 关键点包围盒外扩倍数（MMPose 的 padding 惯例）
PERSON_CROP_MIN_PIX = 48   # 裁剪框任一边小于此值就不裁（人太远，放大也是糊的）

# 「最近一次成功帧的人像框」缓存**按数据源分桶**（cache_key）。多台摄像头同时跑时，
# 这个键必须是"哪一路"，否则 B 台失败帧会捡到 A 台成功帧的像素坐标（不同画面、
# 可能不同分辨率）当裁剪框，重推出一个看着正常、其实是别的画面的结果。
# 单路调用不传 cache_key → 全部落在 None 这一格，行为与分桶前完全一致。
_last_person_bbox = {}        # {cache_key: (x1, y1, x2, y2)，原图像素}
_last_person_bbox_t = {}      # {cache_key: time.monotonic()}

_sess = None


def get_detector():
    """单例：只初始化一次 onnxruntime 会话。"""
    global _sess
    if _sess is None:
        import onnxruntime as ort
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"找不到 RTMPose 模型：{MODEL_PATH}。请先下载 end2end.onnx"
                "（openmmlab 的 rtmpose-s_simcc-body7_pt-body7_420e-256x192 包），"
                "解压放到 models/ 目录下。")
        _sess = ort.InferenceSession(MODEL_PATH,
                                     providers=['CPUExecutionProvider'])
    return _sess


def _letterbox_dims(w: int, h: int) -> tuple:
    """等比缩放 + 灰边到 INPUT_SIZE 的变换参数。

    返回 (scale, pad_x, pad_y)：原图按 scale 等比缩放后放到 INPUT_SIZE 画布上，
    左上角起始于 (pad_x, pad_y)（输入图像素坐标）。整帧拉伸（16:9 → 3:4）会把
    人形压扁变形，letterbox 保留比例，实测对一般场景整体置信度明显更高
    （2026-08-18：正面帧 mean 0.258 → 0.456）。
    """
    scale = min(INPUT_SIZE[0] / w, INPUT_SIZE[1] / h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    pad_x = (INPUT_SIZE[0] - new_w) // 2
    pad_y = (INPUT_SIZE[1] - new_h) // 2
    return scale, pad_x, pad_y


def _normalize_to_tensor(img: np.ndarray) -> np.ndarray:
    """把已 resize 到 INPUT_SIZE 的 RGB 图归一化成 (1, 3, 256, 192) 张量。

    归一化用 ImageNet 统计（RTMPose 训练配置），通道顺序 RGB。
    """
    img = img.astype(np.float32)
    img = (img - _MEAN) / _STD  # RGB 通道顺序
    # HWC -> CHW，加 batch 维
    img = img.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(img)


def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR 帧 → RTMPose 输入张量 (1, 3, 256, 192)：letterbox 等比缩放 + 灰边。

    RTMPose 训练用的是 ImageNet 统计的 RGB 归一化（pipeline.json 里 to_rgb: true）。
    归一化是对每个通道独立做 (x - mean) / std，通道混不混 RGB 只影响通道顺序。
    为保险按训练配置走 RGB：BGR 先翻转成 RGB，再按 RGB 通道的 mean/std 归一化。
    坐标映射回原图时要用 _letterbox_dims 的 (scale, pad)，见 detect_pose_with_score。
    """
    h, w = frame_bgr.shape[:2]
    scale, pad_x, pad_y = _letterbox_dims(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    canvas = np.zeros((INPUT_SIZE[1], INPUT_SIZE[0], 3), dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cv2.resize(
        rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return _normalize_to_tensor(canvas)


def _decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray) -> np.ndarray:
    """把 SIMCC 输出解码成关键点坐标（相对输入图 192x256 的像素坐标）。

    RTMPose 的 SIMCC head 把每个关键点的 x/y 各自编码成一条 softmax 分布
    （长度 = 输入边长 * simcc_split_ratio，这里是 384 / 512）。
    argmax 取峰值位置，再除以 simcc_split_ratio 还原成输入图的像素坐标。

    simcc_x/simcc_y: (batch, 17, 384) / (batch, 17, 512)
    返回: (17, 2) 的 float 数组，顺序 [x, y]（相对输入图，未还原到原图）。
    """
    split_ratio = 2.0  # 与 pipeline.json 的 simcc_split_ratio 一致
    x_bins = simcc_x[0].argmax(axis=1).astype(np.float32) / split_ratio
    y_bins = simcc_y[0].argmax(axis=1).astype(np.float32) / split_ratio
    return np.stack([x_bins, y_bins], axis=1)


def detect_pose(frame_bgr, cache_key=None):
    """检测一帧图像中的人体姿态。

    参数:
        frame_bgr: OpenCV 读取的 BGR 图像 (H, W, 3)。
        cache_key: **哪一路数据源**的标识（多台摄像头同时跑时必须各不相同），
                   只是给"最近一次成功帧的人像框"缓存分桶用，见 detect_pose_with_score。
                   单路调用不传（None）= 与从前完全一致。

    返回:
        None —— 该帧未检测到人体（score 低于阈值 / 无人）。
        否则返回 dict:
            {
                'landmarks': 长度为 17 的列表，每个元素是 (x, y, z, visibility)
                             - x, y: 归一化坐标 0~1（相对图像宽高）
                             - z:     深度，恒为 0（RTMPose 不输出深度）
                             - visibility: 该点置信度 0~1（SIMCC 峰值即置信度）
                'image_size': (width, height)
            }
        关键点编号对应关系（COCO 17 点标准，不是 MediaPipe 33 点）：
            0 鼻, 1 左眼, 2 右眼, 3 左耳, 4 右耳,
            5 左肩, 6 右肩, 7 左肘, 8 右肘,
            9 左腕, 10 右腕, 11 左髋, 12 右髋,
            13 左膝, 14 右膝, 15 左踝, 16 右踝
    """
    result, _score = detect_pose_with_score(frame_bgr, cache_key=cache_key)
    return result


def detect_pose_with_score(frame_bgr, cache_key=None):
    """同 detect_pose，额外返回本帧"整体置信度分数"（17 点平均，未检测到也返回）。

    这是新加的调试/标定接口（加能力走"加新方法"，detect_pose 签名保持不变）：
    想看分数贴着 DETECT_SCORE_THRESHOLD 有多近、据此判断阈值该往上还是往下调时用。
    单机入口 main_action_demo.py 用的是 detect_pose（只读数不看分数）。

    参数:
        cache_key: 同 detect_pose —— "哪一路数据源"。**多路必须传不同的值**：
                   "最近成功帧人像框"是跨帧缓存，不分桶的话 B 台失败帧会拿
                   A 台的像素框去裁自己的画面。默认 None = 单路，行为不变。

    推理顺序（PERSON_CROP_ALWAYS=False，默认）：
    1) 整帧推理。过阈 → 直接用整帧结果（只记下锚点框供失败帧重试用）。
    2) 整帧失败 → "人像区域重试"：本帧锚点框，或**本路**最近成功帧的框
       （PERSON_BBOX_VALID_SEC 内），裁剪放大重推一次，过阈才接受。

    PERSON_CROP_ALWAYS=True 时会在第 1 步前后各加一次裁剪推理（top-down）。
    **默认关闭**：同一段录像回放实测裁剪路径抖动更大（4.1px → 13.8px），
    原因见文件头与常量处的注释。

    返回:
        (result, overall_score)：result 与 detect_pose 的返回完全一致（dict | None）；
        overall_score 恒为 float（0~1，越小越像空场景）。未检测到时 result 为 None，
        但分数照样返回，方便看"差多少才够到阈值"。
    """
    h, w = frame_bgr.shape[:2]

    # 1) 有本路新鲜的人像框 → 直接裁剪推理（top-down 的常态路径）
    cached = _fresh_cached_bbox(cache_key)
    if PERSON_CROP_ALWAYS and cached is not None:
        crop = _infer_crop(frame_bgr, cached)
        if crop is not None and crop[1] >= DETECT_SCORE_THRESHOLD:
            _remember_person_bbox(crop[0], cache_key)
            return crop
        # 框里的信号不够（人走出去/框偏了）：丢掉这个框，回到整帧重新找人
        _forget_person_bbox(cache_key)

    # 2) 整帧推理找人（raw 结果留着给第 3 步算锚点框，避免重复推理）
    scores, coords = _infer_whole_raw(frame_bgr)
    overall_score = float(np.mean(scores))
    if overall_score >= DETECT_SCORE_THRESHOLD:
        whole = _whole_from_raw(frame_bgr, scores, coords)
        if PERSON_CROP_ALWAYS:
            bbox = _crop_bbox_from_result(whole)
            if bbox is not None:
                crop = _infer_crop(frame_bgr, bbox)
                if crop is not None and crop[1] >= DETECT_SCORE_THRESHOLD:
                    _remember_person_bbox(crop[0], cache_key)
                    return crop
        _remember_person_bbox(whole, cache_key)
        return whole, overall_score

    # 3) 整帧失败：人像区域重试（锚点优先，其次最近成功帧的人像框）
    retry = _retry_person_crop(frame_bgr, scores, coords, cache_key)
    if retry is not None:
        crop_result, crop_score = retry
        _remember_person_bbox(crop_result, cache_key)
        return crop_result, crop_score
    return None, overall_score


def _infer_whole_raw(frame_bgr):
    """整帧推理，返回 (scores, coords)（coords 相对 192x256 输入图）。

    只给"人像区域重试"用 —— 它要拿原始 scores/coords 算锚点框；正常返回走
    _whole_from_raw（映射回原图）。
    """
    simcc_x, simcc_y = get_detector().run(None, {'input': _preprocess(frame_bgr)})
    scores = simcc_x[0].max(axis=1).astype(np.float64)
    coords = _decode_simcc(simcc_x, simcc_y)
    return scores, coords


def _whole_from_raw(frame_bgr, scores, coords) -> dict:
    """把整帧推理的 raw 结果映射回原图（letterbox 逆变换）。"""
    h, w = frame_bgr.shape[:2]
    scale, pad_x, pad_y = _letterbox_dims(w, h)
    return _make_result(h, w, scores, coords,
                        lambda x, y: ((x - pad_x) / scale, (y - pad_y) / scale))


def _infer_crop(frame_bgr: np.ndarray, bbox):
    """按人像框裁剪推理（top-down），坐标映射回原图。

    bbox: (x1, y1, x2, y2) 原图像素，调用方负责按模型宽高比外扩。
    返回 (result, overall_score)；框退化/裁剪失败返回 None。
    裁剪图**直接拉伸**到 INPUT_SIZE：框已经按 3:4 的模型宽高比外扩过，
    拉伸不会把人形压扁（和 MMPose 的 top-down 预处理一致）。
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_tensor = _normalize_to_tensor(
        cv2.resize(crop_rgb, INPUT_SIZE, interpolation=cv2.INTER_LINEAR))
    simcc_x, simcc_y = get_detector().run(None, {'input': crop_tensor})
    scores = simcc_x[0].max(axis=1).astype(np.float64)
    coords = _decode_simcc(simcc_x, simcc_y)

    in_w, in_h = INPUT_SIZE
    bw, bh = x2 - x1, y2 - y1
    result = _make_result(
        h, w, scores, coords,
        lambda px, py: (x1 + px * (bw / in_w), y1 + py * (bh / in_h)))
    return result, float(np.mean(scores))


def _model_bbox(x1: float, y1: float, x2: float, y2: float,
                w: int, h: int, pad: float = PERSON_CROP_PAD):
    """把关键点包围盒扩成模型输入的宽高比（192:256 = 3:4）并外扩 pad。

    先按中心缩放、再夹回帧内（越界部分裁掉；numpy 负索引会绕回帧尾，必须先夹）。
    框任一边小于 PERSON_CROP_MIN_PIX 时返回 None（人太远，裁出来也是糊的）。
    """
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(1.0, (x2 - x1) * pad), max(1.0, (y2 - y1) * pad)
    ratio = INPUT_SIZE[0] / INPUT_SIZE[1]      # 0.75
    if bw / bh < ratio:
        bw = bh * ratio
    else:
        bh = bw / ratio
    x1, x2 = max(0.0, cx - bw / 2), min(float(w), cx + bw / 2)
    y1, y2 = max(0.0, cy - bh / 2), min(float(h), cy + bh / 2)
    if x2 - x1 < PERSON_CROP_MIN_PIX or y2 - y1 < PERSON_CROP_MIN_PIX:
        return None
    return (int(x1), int(y1), int(x2), int(y2))


def _crop_bbox_from_result(result: dict):
    """从一次成功结果里 置信度 >= ANCHOR_CONF 的点 算 top-down 裁剪框。"""
    w, h = result['image_size']
    xs, ys = [], []
    for (x, y, _z, vis) in result['landmarks']:
        if vis >= ANCHOR_CONF:
            xs.append(x * w)
            ys.append(y * h)
    if len(xs) < 2:
        return None
    return _model_bbox(min(xs), min(ys), max(xs), max(ys), w, h)


def _fresh_cached_bbox(cache_key):
    """本路缓存的人像框（PERSON_BBOX_VALID_SEC 内有效），否则 None。"""
    bbox = _last_person_bbox.get(cache_key)
    if bbox is None:
        return None
    if time.monotonic() - _last_person_bbox_t.get(cache_key, 0.0) > PERSON_BBOX_VALID_SEC:
        return None
    return bbox


def _forget_person_bbox(cache_key) -> None:
    """丢掉本路缓存的人像框（框失效时调用，别让它拖着错误位置）。"""
    _last_person_bbox.pop(cache_key, None)
    _last_person_bbox_t.pop(cache_key, None)


def _make_result(h: int, w: int, scores: np.ndarray, coords: np.ndarray,
                 remap) -> dict:
    """把输入图坐标 (N,2) 用 remap(px, py) -> (原图像素x, y) 映射回原图，
    构建与 detect_pose 一致的结果 dict（归一化坐标 + 该点置信度）。
    """
    landmarks = []
    for i in range(17):
        px, py = remap(float(coords[i, 0]), float(coords[i, 1]))
        landmarks.append((px / w, py / h, 0.0, float(scores[i])))
    return {'landmarks': landmarks, 'image_size': (w, h)}


def _bbox_from_points(frame_bgr: np.ndarray, scores: np.ndarray,
                      coords: np.ndarray) -> tuple | None:
    """从 置信度 >= ANCHOR_CONF 的点 算人像包围盒（原图像素）。

    需要 >=2 个锚点、包围盒足够大，否则返回 None（整帧推理全塌，无从锚定）。
    边距：头顶多留 0.3 倍、左右/底部 0.15 倍（给头留空间）。
    """
    h, w = frame_bgr.shape[:2]
    scale, pad_x, pad_y = _letterbox_dims(w, h)
    xs, ys = [], []
    for i in range(17):
        if scores[i] >= ANCHOR_CONF:
            xs.append((coords[i, 0] - pad_x) / scale)
            ys.append((coords[i, 1] - pad_y) / scale)
    if len(xs) < 2:
        return None
    x1, x2 = max(0.0, min(xs)), min(float(w), max(xs))
    y1, y2 = max(0.0, min(ys)), min(float(h), max(ys))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0.0, x1 - bw * 0.15)
    x2 = min(float(w), x2 + bw * 0.15)
    y1 = max(0.0, y1 - bh * 0.30)
    y2 = min(float(h), y2 + bh * 0.15)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return (int(x1), int(y1), int(x2), int(y2))


def _retry_person_crop(frame_bgr: np.ndarray, scores: np.ndarray,
                       coords: np.ndarray, cache_key=None):
    """整帧失败时的人像区域重试：返回 (crop_result, crop_overall) 或 None。

    1) 优先用本帧 置信度 >= ANCHOR_CONF 的锚点 框人像；锚点不足时
    2) 用**本路**（cache_key）最近一次成功帧缓存的人像框（PERSON_BBOX_VALID_SEC
       内有效，人静止时位置基本不变，裁剪放大正好补上整帧看不出的信号）；
    3) 裁剪 → 拉伸到输入尺寸 → 重推；重推 mean 过阈才接受。
    """
    bbox = _bbox_from_points(frame_bgr, scores, coords)
    if bbox is None:
        bbox = _fresh_cached_bbox(cache_key)
    if bbox is None:
        return None

    crop = _infer_crop(frame_bgr, bbox)
    if crop is None or crop[1] < DETECT_SCORE_THRESHOLD:
        return None  # 裁剪也没救活（人走远/彻底塌陷）
    return crop


def _remember_person_bbox(result: dict, cache_key=None) -> None:
    """把一次成功结果的 top-down 裁剪框缓存下来，供**同一路**（cache_key）的
    失败帧做"人像区域重试"用（时间连续，人没动；PERSON_CROP_ALWAYS=True 时
    也用于下一帧直接裁剪推理）。框已按模型宽高比外扩，见 _crop_bbox_from_result。
    """
    bbox = _crop_bbox_from_result(result)
    if bbox is None:
        return
    _last_person_bbox[cache_key] = bbox
    _last_person_bbox_t[cache_key] = time.monotonic()


def landmarks_to_pixels(landmarks, image_size):
    """把归一化坐标转成像素坐标（纯几何换算，不做判断）。

    参数:
        landmarks: detect_pose 返回的 'landmarks'（归一化坐标列表）
        image_size: (width, height)

    返回:
        (x_px, y_px) 元组列表，长度与 landmarks 相同。
    """
    w, h = image_size
    return [(int(x * w), int(y * h)) for x, y, _z, _vis in landmarks]
