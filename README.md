# RTMPose 全身动作识别 Demo

基于 RTMPose（OpenMMLab）的人体姿态识别 Demo：检测人体关键点，实时识别**全身动作**——目前是**举手**和**仰头**两个动作，桌面上直接给出实时读数与状态。

> 本项目是**「后续用机器人检测人体动作、判断后做出反应」这一目标的简易 Demo 前置版**：先把「检测关键点 → 算动作特征 → 判动作」的整条链路在本机跑通，为机器人集成打基础。

- 桌面版：OpenCV 实时窗口，本机（USB）摄像头
- **这是动作识别仓库，不是坐姿仓库**。它与坐姿项目 `rtm_demo` 是**各自独立的两个仓库**（本仓库由后者分叉而来），本仓库**只做全身动作**。

## 功能

| 动作 | 特征量 | 判定 |
|---|---|---|
| 举手 | **上臂抬升角**：肩→肘 与 向上竖直轴 的夹角（0° = 完全上举 / 90° = 平举 / 180° = 下垂）。另有肘角（肩-肘-腕内角，180° = 伸直）供对照 | **默认只报数、不判状态**（`--arm-rule` 默认 `off`）；加 `--arm-rule raise` 可开状态机 → `DOWN / RAISED` |
| 仰头 | **头俯仰代理角**：耳中点(3/4)→鼻尖(0) 与 水平线 的夹角（正 = 鼻在耳上方 = 仰头，负 = 低头） | `NEUTRAL / TILTED_BACK`（迟滞双阈值状态机） |

画面上显示：两条动作的**实时读数** + **状态标签**，以及角度几何辅助线（抬升角的竖直参考轴、耳→鼻连线）。骨架照常画（COCO 全身）。

**这条线是旁路**：两个动作各有各的特征与判定，互不干扰，也不影响本仓库里仍混着的坐姿代码（见「已知范围与限制」）。

## 环境要求

- Windows，Python 3.x
- 依赖：`opencv-python`、`numpy`、`onnxruntime`（模型推理用）
- 模型文件 `end2end.onnx`（rtmpose-s）：见下方「模型文件」

安装依赖（用你自己环境里的 Python）：

```bash
python -m pip install -r requirements.txt
```

下面命令里的 `python` 都指这个装好依赖的解释器。想用虚拟环境隔离，可先 `python -m venv .venv` 激活后再装。

## 模型文件

`models/`（含 `end2end.onnx`，约 21MB）**未提交到仓库**，clone 后需自行准备。

模型是 **RTMPose-s**（body7, 256×192）的 ONNX 导出，原始 PyTorch 权重（官方 mmpose 下载站）：

```
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth
```

`end2end.onnx` 由该权重经 **mmdeploy** 导出（导出得到 `deploy.json / pipeline.json / end2end.onnx` 三个文件，即 mmdeploy SDK 包格式），对应 mmpose 配置为 `projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py`。

准备好后把整个包解压到 `models/` 下任意位置即可——代码用 `rglob('end2end.onnx')` 自动查找，不依赖固定路径。

## 快速开始

在 `rtm_action_demo/` 目录下运行。命令里的 `python` 指装了依赖的解释器（见「环境要求」）。

### 桌面版

```bash
python main_action_demo.py                                   # 默认：USB 那台（45° 前侧）；举手只报数，仰头判状态
python main_action_demo.py --arm-rule raise                   # 举手也开状态机
python main_action_demo.py --camera 0                         # 换回笔记本内置摄像头（正对机位；仰头会判几何退化）
python main_action_demo.py --no-geometry                      # 不画角度几何辅助线
python main_action_demo.py --no-skeleton                      # 不画骨架
python main_action_demo.py --snapshot data/action_snap.png    # 按 s 存当前帧
python main_action_demo.py --selftest                         # 合成数据自测（不碰摄像头/模型）
```

按键：**`s` 存快照**（需 `--snapshot` 指定路径）、**ESC 退出**。首帧较慢（惰性初始化模型），正常。

### 摄像头怎么选

- **`--camera 0`** = 笔记本**内置**摄像头（**正对着人**）
- **`--camera 1`** = **USB 外接**摄像头（**默认**，摆在前侧约 45°）

**这两个动作对机位的要求不一样**：

- **举手**：竖直方向的动作，**哪个机位都能量**；
- **仰头**：**必须侧对 / 45° 前侧**。正对着摄像头时，仰头主要是朝镜头方向的透视缩短，耳-鼻在画面上几乎不动——**程序会判「几何退化」，读数直接显示 N/A**，而不是给你一个看着像真的错数（详见「已知范围与限制」）。

所以默认跑的就是 USB / 45° 那台。摄像头序号由插拔顺序决定，认不准就用 `--camera N` 试。

## 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--camera N` | `1` | 摄像头序号（`0` = 内置/正对，`1` = USB/45°） |
| `--arm-rule {off,raise}` | `off` | **`off` = 举手只报数不判状态**；`raise` = 开启举手状态机 |
| `--arm-raise-enter D` | `45.0` | 抬升角 **≤ D** 判「已举起」（度；**占位值，待标定**） |
| `--arm-raise-exit D` | `60.0` | 抬升角 **≥ D** 判「已放下」（度；迟滞带上沿，**占位值**） |
| `--head-tilt-enter D` | `20.0` | 头俯仰角 **≥ D** 判「仰头」（度；**占位值，且零点不在 0**，见「阈值标定」） |
| `--head-tilt-exit D` | `10.0` | 头俯仰角 **≤ D** 判「中立」（度；迟滞带下沿，**占位值**） |
| `--tilt-min-dx-ratio R` | `0.15` | 仰头**横向退化门控**：耳→鼻的水平跨度 ÷ 双耳间距 < R → 判几何退化、读数作废 |
| `--no-geometry` | 关 | 不画角度几何辅助线 |
| `--no-skeleton` | 关 | 不画骨架 |
| `--snapshot PATH` | 无 | 按 `s` 把当前帧存到 PATH |
| `--selftest` | — | 合成数据自测各层（不碰摄像头 / 模型） |

## 阈值标定

**两个动作的阈值目前都是占位值，不是标定结果**，先按画面读数标定再定。

画面与控制台上直接显示实时读数：

- **举手**：`arm_raise` 是抬升角（**越小越举得高**：0° 完全上举 / 90° 平举 / 180° 下垂），`elbow` 是肘角（180° = 伸直）。两侧独立显示。
- **仰头**：`head_pitch` 是耳→鼻夹角（**正 = 仰头，负 = 低头**）。

标定方法：

1. 摆好**中立**姿势（举手：手臂自然垂下；仰头：正常平视），记下读数；
2. 摆出**动作**姿势（举手：举过肩 / 举过头顶；仰头：明显后仰），记下读数；
3. 把 enter 阈值定在两档之间靠近中立一侧，exit 定得更靠动作一侧，**中间留出迟滞带**（避免边界抖动来回跳）。

**两个容易踩的坑**：

- **仰头的零点不在 0**：45° 机位下「中立站姿」实测约 **-20~-30°**（鼻尖天生比耳中点低一点）。所以阈值要**按画面上中立时的读数往上加**，别照数字本身调。中立基线还会随机位 / 摄像头高度 / 人体比例变化，**换个条件就要重看**。
- **只有那一个门控是既定值**：`--tilt-min-dx-ratio`（默认 0.15）不是阈值而是「这个机位量不量得了」的门控，有实测支撑（45° 前侧 0.43~0.52，正对 0.013），可以直接用。

## 项目结构（分层架构）

```
rtm_action_demo/
├── main_action_demo.py   # 【动作线入口】装配各层、跑主循环；只装举手/仰头
├── requirements.txt      # Python 依赖（pip install -r requirements.txt）
├── capture.py            # 图像输入：FrameSource / CameraCapture（默认 index 1 = USB/45°）
├── pose_estimation.py    # RTMPose 推理：17 个 COCO 关键点
├── features.py           # 特征：ArmRaiseFeatures（上臂抬升角+肘角）、HeadTiltFeatures（耳→鼻俯仰角）
├── decision.py           # 判断：ArmRaiseDecision（DOWN/RAISED）、HeadTiltDecision（NEUTRAL/TILTED_BACK）
├── output.py             # 输出：骨架绘制 + 动作读数/状态/几何辅助线
├── pose_schema.py        # 姿态对齐：COCO17 / 机器人 27 舵机 → 统一 27 DOF 规范 schema（为对接机器人预留）
├── models/               # RTMPose ONNX 模型（未提交）
└── data/                 # 输出目录（快照等；被 .gitignore 挡住，仓库里只提交 .gitkeep）

  ※ 以下为本仓库里仍混着的、从坐姿仓库分叉带来的文件，动作线不使用：
    main_demo.py / main_front_demo.py / web_demo.py / perf_logger.py / tools/
```

数据流：`capture → pose_estimation → features → decision → output`，各层职责单一、接口稳定。

关键点编号为 **COCO 17 点标准**：0 鼻、3/4 耳、5/6 肩、7/8 肘、9/10 腕、11/12 髋等。

## 后续计划

- **先跑几轮实拍**，把举手 / 仰头的 enter/exit 从占位值换成标定值；顺便决定举手是开状态机还是继续只报数。
- **加更多动作**（挥手 / 转身 / 蹲下…）：照 `ArmRaiseFeatures` + `ArmRaiseDecision` 的样子各加一对类，接口不变。
- **机器人摄像头**：当前用的是本机摄像头；后续接入机器人摄像头时，只需在 `capture.py` 新增一个图像源子类，上层数据流不用改。
- **人体 → 机器人关节映射**：在 `pose_schema.py` 的规范 27 DOF 空间里做（人体侧角度/位置 → 机器人目标角）。
- **多目标 / 人不在画面中心**：`pose_estimation.py` 接 RTMDet 检测器，给 `detect_pose` 加可选 bbox，上层不用改。

## 已知范围与限制

- **动作线目前没有任何提醒**：只显示读数与状态，不触发提醒、不做时长判断。要不要接提醒，待定。
- **两个动作的阈值都是占位值，未标定**；读数的中立基线随机位 / 摄像头高度 / 人体比例变化，换条件要重看。
- **仰头只在侧对 / 45° 前侧有效**：正对摄像头时仰头主要是朝镜头方向的透视缩短、耳-鼻在画面上几乎不动，程序判为**几何退化**、读数显示 N/A。更隐蔽的是此时 `|Δx| ≈ 0` 会让角度退化成「只剩符号」的 ±90°，长得像正常读数——所以特征层主动把它作废，不给假数。
- **仰头角是 2D 代理，不是真实俯仰角**：COCO 17 没有下巴 / 耳屏，用「耳点 + 鼻尖」代替头部轴；45° 机位下水平方向被压缩，读数比真实角大，**只保证单调**（仰头越大读数越大），量值与任何文献的头屈伸角都不可比。**不补偿转头**（yaw），头往侧面转也会串进读数。
- **举手角是像素空间的上臂朝向**：单目 2D，不含深度；只看「肩→肘」朝向，手腕丢失不影响触发。
- **v1 刻意不做平滑**：不 EMA、不中值、不维持上一有效值。关键点丢帧就标无效、读数写 N/A。这是为了先看清楚「单目关键点撑不撑得住全身动作」这个问题本身。
- **单 / 多人**：目前按「整帧单一人」处理（top-down 无检测器）。
- **同仓库还混着坐姿版代码**：本仓库由坐姿项目 `rtm_demo` 分叉而来，那些坐姿 / CVA / 头前伸的代码与 `tools/` 下的标定脚本**都还在**，但**动作线完全不用它们**。相关说明见旧仓库 `rtm_demo` 的文档。
- **模型**：使用 rtmpose-s（`models/` 内）
