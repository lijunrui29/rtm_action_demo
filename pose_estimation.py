"""
Pose Estimation 核心模块：输入一帧图像，输出 17 个 COCO 关键点。

用 RTMPose（OpenMMLab）的 ONNX 模型做推理，替换原来的 MediaPipe。
只做检测，不做任何判断。返回归一化坐标，不包含绘制/规则逻辑。

模型文件（end2end.onnx）在本文件同目录的 models/ 下，用绝对路径，
避免依赖工作目录。运行前请先装 onnxruntime：
    pip install onnxruntime

RTMPose 是 top-down 模型，本来需要先框出人的包围盒再推理。本项目**没有检测器**，
按"单一人"处理：**整帧推理**（letterbox）就是常态路径；整帧判不到人时，再用
候选裁剪框（本帧锚点 / 最近成功帧的框 / 放大一档的框 / 画面中心）依次重推
（见 detect_pose_with_score 与 _retry_person_crop）。

**"有没有人"的判据（2026-11-10 改）**：不再用 17 点平均分。45° 机位下下半身
经常出画/被桌子挡住，低分点会把均值拖到阈值以下，人还在也判 None；反过来空场景
里几个随机高点又可能凑过均值。现在改成"核心点计数 + 躯干锚点"（见 CORE_* 常量
与 _has_person）：核心点（鼻/耳/肩/肘/髋）里至少 CORE_MIN_COUNT 个达到
CORE_CONF_THRESHOLD，且肩或髋至少一个可见。平均分仍作为诊断量返回。

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
        置信度分数（未检测到也返回），供观察分数高低用。
    detect_pose_debug(frame_bgr) -> (dict | None, dict)
        **加能力走加新方法**：结果同 detect_pose，额外返回诊断信息
        （整体分、核心点计数、各核心点置信度、本帧走的是整帧还是哪个候选
        裁剪框）。入口的 --pose-debug / --dump-pose-fail 用这个。
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

# "检测到人"的判据（2026-11-10 改）：核心点计数 + 躯干锚点，不再用 17 点平均分。
# 核心点只取动作真正依赖的部位（鼻/耳/肩/肘/髋），下半身出画不再拖累判定。
#   1) 核心点中 conf >= CORE_CONF_THRESHOLD 的个数 >= CORE_MIN_COUNT；
#   2) 且肩(5/6)或髋(11/12)至少有一个达到该置信度（必须有躯干锚点，
#      防止只有面部/手部噪点也凑成"有人"）。
CORE_KEYPOINT_IDS = (0, 3, 4, 5, 6, 7, 8, 11, 12)
CORE_CONF_THRESHOLD = 0.3
CORE_MIN_COUNT = 4
TORSO_ANCHOR_IDS = (5, 6, 11, 12)

# 17 点平均分：**只作诊断参考，不再是判定条件**（2026-11-10 起）。
# 保留这个常量是因为 detect_pose_with_score 还返回分数，对照数值时用得到。
DETECT_SCORE_THRESHOLD = 0.15

# 整帧失败时的"人像区域重试"（2026-08-18，实测标定，详见 detect_pose_with_score）：
# 人占画面偏小/身体转过一定角度时，整帧推理 17 点一起掉到阈值下，但其实人还在
# 画面里、模型只是没锁住。此时用 置信度 >= ANCHOR_CONF 的点（或最近一次成功帧的
# 人像框，时间连续）裁剪放大重推，通常能把 mean 拉回阈值上。
# ⚠ 这是无检测器下的**临时替代**（见 CLAUDE.md 开放问题）：将来接 RTMDet 后可以去掉。
ANCHOR_CONF = 0.2          # 视为"锚点"的最低置信度（实测失败帧最高点 ~0.22）
PERSON_BBOX_VALID_SEC = 1.0  # 上次成功帧人像框的有效期（人不动时沿用；走远/换人自动失效）

# 人像框缓存做 EMA（2026-11-10）：框由关键点包围盒算出，本身会抖；
# 直接把抖动的框记下来、下一帧又拿它裁，会把误差反馈进下一帧
# （PERSON_CROP_ALWAYS 实测变差的根因）。这里让缓存框缓慢跟随（0.6 = 每帧
# 向新框靠 60%），既跟得上人移动，又不把单帧抖动原样带下去。
PERSON_BBOX_SMOOTH = 0.6

# 失败重试的候选框（2026-11-10）。按"最可能命中"的顺序依次试，
# **第一个通过人物判据的就接受**（不在所有候选里比分数，失败帧才付多次推理）：
#   1) 本帧锚点框：模型在本帧已经锁到 2 个以上点，只是整体没到判据；
#   2) 最近成功帧的框：人基本没动，放大后能把信号拉起来；
#   3) 最近成功帧的框再放大一档：人走近/走远 / 缓存框略偏；
#   4) 画面中心 3:4 区域：人不在缓存框附近（刚入画/换位置）时的最后兜底。
PERSON_BBOX_WIDE_PAD = 1.6   # 候选 3 相对缓存框的放大倍数

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


def _resize_rgb(rgb: np.ndarray, size: tuple) -> np.ndarray:
    """把 RGB 图缩放到 size=(w, h)。

    下采样用 INTER_AREA（按区域平均，抗锯齿；细肢体/远处的人比 INTER_LINEAR
    更不容易糊成一团），上采样用 INTER_LINEAR（INTER_AREA 在放大时退化成
    最近邻，反而更差）。2026-11-10 起整帧预处理与裁剪推理都走这里。
    """
    dst_w, dst_h = size
    src_h, src_w = rgb.shape[:2]
    shrinking = dst_w <= src_w and dst_h <= src_h
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(rgb, (dst_w, dst_h), interpolation=interp)


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
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = _resize_rgb(
        rgb, (new_w, new_h))
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
        None —— 该帧未检测到人体（核心点不足 / 无人）。
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

    分数**不再参与"有没有人"的判定**（2026-11-10 起改成核心点计数，见
    _has_person）；这里保留它只作诊断/对照用。单机入口 main_action_demo.py
    平时用 detect_pose；要看诊断信息用 detect_pose_debug。

    参数:
        cache_key: 同 detect_pose —— "哪一路数据源"。**多路必须传不同的值**：
                   "最近成功帧人像框"是跨帧缓存，不分桶的话 B 台失败帧会拿
                   A 台的像素框去裁自己的画面。默认 None = 单路，行为不变。

    推理顺序（PERSON_CROP_ALWAYS=False，默认）：
    1) 整帧推理。核心点够 → 直接用整帧结果（只记下锚点框供失败帧重试用）。
    2) 整帧判不到人 → 候选裁剪框依次重推（本帧锚点 / 最近成功帧的框 /
       放大一档的框 / 画面中心），第一个通过人物判据的就接受。
       见 _retry_person_crop 与 PERSON_BBOX_WIDE_PAD 处的说明。

    PERSON_CROP_ALWAYS=True 时会在第 1 步前后各加一次裁剪推理（top-down）。
    **默认关闭**：同一段录像回放实测裁剪路径抖动更大（4.1px → 13.8px），
    原因见文件头与常量处的注释。

    返回:
        (result, overall_score)：result 与 detect_pose 的返回完全一致（dict | None）；
        overall_score 恒为 float（0~1）。整帧成功时是整帧 17 点平均；走了裁剪
        重试时是**接受那次裁剪**的 17 点平均；完全失败时是整帧平均。
    """
    result, info = _detect_with_info(frame_bgr, cache_key)
    return result, info['overall_score']


def detect_pose_debug(frame_bgr, cache_key=None):
    """同 detect_pose，额外返回诊断信息（2026-11-10 加，供入口 --pose-debug 用）。

    返回 (result, info)：
        result —— 与 detect_pose 完全一致（dict | None）。
        info —— dict:
            overall_score: float，17 点平均（见 detect_pose_with_score）
            core_count:    int，核心点中 conf >= CORE_CONF_THRESHOLD 的个数
            core_scores:   {关键点编号: 置信度}，便于看是哪些部位在/不在
            accepted:      bool，本帧最终有没有判到人
            stage:         str，本帧结果来自哪一步：
                           'whole' / 'crop_always' / 'anchor' / 'last' /
                           'last_wide' / 'center' / 'none'
            bbox:          (x1, y1, x2, y2) 或 None，裁剪路径用的框（原图像素）
    这个接口只加观测，不改判定；缓存语义与 detect_pose 相同（cache_key 分桶）。
    """
    return _detect_with_info(frame_bgr, cache_key)


def _has_person(scores) -> bool:
    """本路 17 点置信度是否构成"画面里有人"。

    判据见 CORE_* 常量：核心点计数够 + 有躯干锚点。返回 bool，供推理路径用；
    要看计数/逐点分数用 _presence。
    """
    return _presence(scores)[0]


def _presence(scores) -> tuple:
    """返回 (是否有人, 核心点可见数)。

    核心点只看动作真正依赖的部位（鼻/耳/肩/肘/髋），下半身出画不再把
    "人还在"判成没人；同时要求肩或髋至少一个可见，空场景里几个面部/手部
    随机高点凑不成"有人"。
    """
    core = [i for i in CORE_KEYPOINT_IDS if scores[i] >= CORE_CONF_THRESHOLD]
    if len(core) < CORE_MIN_COUNT:
        return False, len(core)
    if not any(scores[i] >= CORE_CONF_THRESHOLD for i in TORSO_ANCHOR_IDS):
        return False, len(core)
    return True, len(core)


def _make_info(scores, accepted: bool, stage: str, bbox) -> dict:
    """构造 detect_pose_debug 的 info（只读观测，不参与任何判定）。"""
    _ok, core_count = _presence(scores)
    return {
        'overall_score': float(np.mean(scores)),
        'core_count': int(core_count),
        'core_scores': {int(i): float(scores[i]) for i in CORE_KEYPOINT_IDS},
        'accepted': bool(accepted),
        'stage': stage,
        'bbox': None if bbox is None else tuple(int(v) for v in bbox),
    }


def _detect_with_info(frame_bgr, cache_key=None):
    """检测主流程：整帧 → （可选常态裁剪）→ 候选裁剪重试。

    返回 (result | None, info)。detect_pose / detect_pose_with_score /
    detect_pose_debug 都走这里，保证三条入口的行为一致。
    """
    # 1) 有本路新鲜的人像框 → 直接裁剪推理（top-down 的常态路径）
    cached = _fresh_cached_bbox(cache_key)
    if PERSON_CROP_ALWAYS and cached is not None:
        crop = _infer_crop(frame_bgr, cached)
        if crop is not None and _has_person(crop[1]):
            _remember_person_bbox(crop[0], cache_key)
            return crop[0], _make_info(crop[1], True, 'crop_always', cached)
        # 框里的信号不够（人走出去/框偏了）：丢掉这个框，回到整帧重新找人
        _forget_person_bbox(cache_key)

    # 2) 整帧推理找人（raw 结果留着给第 3 步算锚点框，避免重复推理）
    scores, coords = _infer_whole_raw(frame_bgr)
    if _has_person(scores):
        whole = _whole_from_raw(frame_bgr, scores, coords)
        if PERSON_CROP_ALWAYS:
            bbox = _crop_bbox_from_result(whole)
            if bbox is not None:
                crop = _infer_crop(frame_bgr, bbox)
                if crop is not None and _has_person(crop[1]):
                    _remember_person_bbox(crop[0], cache_key)
                    return crop[0], _make_info(crop[1], True,
                                               'crop_always', bbox)
        _remember_person_bbox(whole, cache_key)
        return whole, _make_info(scores, True, 'whole', None)

    # 3) 整帧判不到人：候选裁剪框依次重试
    retry = _retry_person_crop(frame_bgr, scores, coords, cache_key)
    if retry is not None:
        crop_result, crop_scores, stage, bbox = retry
        _remember_person_bbox(crop_result, cache_key)
        return crop_result, _make_info(crop_scores, True, stage, bbox)
    return None, _make_info(scores, False, 'none', None)


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
    返回 (result, scores)（scores 是 17 点原始置信度，供人物判据用）；
    框退化/裁剪失败返回 None。
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
    crop_tensor = _normalize_to_tensor(_resize_rgb(crop_rgb, INPUT_SIZE))
    simcc_x, simcc_y = get_detector().run(None, {'input': crop_tensor})
    scores = simcc_x[0].max(axis=1).astype(np.float64)
    coords = _decode_simcc(simcc_x, simcc_y)

    in_w, in_h = INPUT_SIZE
    bw, bh = x2 - x1, y2 - y1
    result = _make_result(
        h, w, scores, coords,
        lambda px, py: (x1 + px * (bw / in_w), y1 + py * (bh / in_h)))
    return result, scores


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


def _expand_bbox(bbox, factor: float, w: int, h: int):
    """把框绕中心放大 factor 倍（保持宽高比），夹回帧内；退化返回 None。"""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * factor, (y2 - y1) * factor
    x1, x2 = max(0.0, cx - bw / 2), min(float(w), cx + bw / 2)
    y1, y2 = max(0.0, cy - bh / 2), min(float(h), cy + bh / 2)
    if x2 - x1 < PERSON_CROP_MIN_PIX or y2 - y1 < PERSON_CROP_MIN_PIX:
        return None
    return (int(x1), int(y1), int(x2), int(y2))


def _center_crop_bbox(w: int, h: int):
    """画面中心的 3:4 区域（尽量占满高度），作为重试的兜底候选。

    机位固定、人通常就在画面中间；这个框**不依赖上一帧结果**，所以没有
    "框抖动反馈"的问题。人明显偏在一侧时它可能裁不到，只当最后一档。
    """
    ratio = INPUT_SIZE[0] / INPUT_SIZE[1]      # 0.75
    bh = float(h)
    bw = bh * ratio
    if bw > w:
        bw = float(w)
        bh = bw / ratio
    cx, cy = w / 2.0, h / 2.0
    return (int(cx - bw / 2), int(cy - bh / 2),
            int(cx + bw / 2), int(cy + bh / 2))


def _bbox_iou(a, b) -> float:
    """两个像素框的交并比，用于跳过重复候选（避免对同一块区域反复推理）。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
    area_b = float(max(1, bx2 - bx1) * max(1, by2 - by1))
    return inter / (area_a + area_b - inter)


def _retry_person_crop(frame_bgr: np.ndarray, scores: np.ndarray,
                       coords: np.ndarray, cache_key=None):
    """整帧判不到人时的候选裁剪重试。

    按顺序试（第一个通过人物判据的就接受，不在候选间比分数）：
      1) 'anchor'    本帧置信度 >= ANCHOR_CONF 的锚点框；
      2) 'last'      **本路**（cache_key）最近成功帧缓存的人像框；
      3) 'last_wide' 上一框再放大 PERSON_BBOX_WIDE_PAD 倍（人走近/走远/框略偏）；
      4) 'center'    画面中心 3:4 区域（人刚入画/换位置，缓存框够不着）。
    返回 (crop_result, crop_scores, stage, bbox) 或 None。
    重复候选（IoU > 0.9）跳过，失败帧最多付 3~4 次推理。
    """
    h, w = frame_bgr.shape[:2]
    candidates = []
    anchor = _bbox_from_points(frame_bgr, scores, coords)
    if anchor is not None:
        candidates.append(('anchor', anchor))
    cached = _fresh_cached_bbox(cache_key)
    if cached is not None:
        candidates.append(('last', cached))
        wide = _expand_bbox(cached, PERSON_BBOX_WIDE_PAD, w, h)
        if wide is not None:
            candidates.append(('last_wide', wide))
    candidates.append(('center', _center_crop_bbox(w, h)))

    tried = []
    for stage, bbox in candidates:
        if any(_bbox_iou(bbox, t) > 0.9 for t in tried):
            continue
        tried.append(bbox)
        crop = _infer_crop(frame_bgr, bbox)
        if crop is not None and _has_person(crop[1]):
            return crop[0], crop[1], stage, bbox
    return None  # 所有候选都没救活（人走远/彻底塌陷）


def _remember_person_bbox(result: dict, cache_key=None) -> None:
    """把一次成功结果的 top-down 裁剪框缓存下来，供**同一路**（cache_key）的
    失败帧做"人像区域重试"用（时间连续，人没动；PERSON_CROP_ALWAYS=True 时
    也用于下一帧直接裁剪推理）。框已按模型宽高比外扩，见 _crop_bbox_from_result。

    新框与旧框做 EMA（PERSON_BBOX_SMOOTH）：框来自关键点包围盒，本身会抖；
    直接原样缓存会把单帧抖动反馈进下一帧的裁剪（PERSON_CROP_ALWAYS 实测
    变差的根因）。EMA 让框缓慢跟随，既跟得上人移动，又不放大抖动。
    """
    bbox = _crop_bbox_from_result(result)
    if bbox is None:
        return
    old = _last_person_bbox.get(cache_key)
    if old is not None:
        a = PERSON_BBOX_SMOOTH
        bbox = tuple(int(round(a * n + (1.0 - a) * o))
                     for n, o in zip(bbox, old))
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


# ---------------------------------------------------------------------------
# 自测：人物判据与候选框（纯合成分数/几何，不碰模型与摄像头）
# ---------------------------------------------------------------------------

def selftest_pose_presence() -> None:
    """人物判据 / 候选框几何的自测。

    覆盖：
    1) 上半身够、下半身全丢 → 判有人（旧 17 点平均分会漏）；
    2) 空场景零星高点 → 判没人；
    3) 只有脸/手、没有肩髋 → 判没人（躯干锚点）；
    4) 中心框/放大框不越界，IoU 能认出重复框。
    """
    def make(**vals) -> np.ndarray:
        s = np.zeros(17, dtype=np.float64)
        for pid, v in vals.items():
            s[int(pid[1:])] = v
        return s

    # 1) 上半身 5 个核心点 0.4，其余（含下肢）0.01：
    #    旧均值 ≈ 0.125 < DETECT_SCORE_THRESHOLD，旧判据会判"没人"
    up = make(p0=0.4, p3=0.4, p5=0.4, p6=0.4, p7=0.4)
    up[8:] = 0.01
    assert float(np.mean(up)) < DETECT_SCORE_THRESHOLD
    ok, count = _presence(up)
    assert ok and count == 5, "上半身可见就该判有人（下半身出画不拖累）"

    # 2) 空场景：只有 3 个核心点 0.5，计数不够
    assert not _has_person(make(p0=0.5, p3=0.5, p4=0.5))

    # 3) 4 个核心点但全是脸/肘，没有肩髋锚点 → 不算有人
    assert not _has_person(make(p0=0.5, p3=0.5, p4=0.5, p7=0.5))

    # 4) 4 个核心点含肩/髋 → 有人
    assert _has_person(make(p0=0.5, p5=0.5, p7=0.5, p11=0.5))

    # 5) 候选框几何：中心框保持模型宽高比、放大框不越界、IoU 能区分
    w, h = 1280, 720
    center = _center_crop_bbox(w, h)
    assert 0 <= center[0] < center[2] <= w and 0 <= center[1] < center[3] <= h
    ratio = (center[2] - center[0]) / float(center[3] - center[1])
    assert abs(ratio - INPUT_SIZE[0] / INPUT_SIZE[1]) < 0.02, \
        "中心候选框应保持模型 3:4 宽高比"
    wide = _expand_bbox(center, PERSON_BBOX_WIDE_PAD, w, h)
    assert wide is not None
    assert 0 <= wide[0] < wide[2] <= w and 0 <= wide[1] < wide[3] <= h
    assert _bbox_iou(center, center) > 0.99
    assert _bbox_iou(center, wide) < 0.99, "放大一档后应被当成不同候选"

    print("  selftest_pose_presence: OK")
