# RTMPose 全身动作识别 Demo

基于 RTMPose（OpenMMLab）的人体姿态识别 Demo：检测人体关键点，实时识别**全身动作**——目前有**举手 / 仰头**两个原有动作，以及**躯干前屈 / 颈部侧屈 / 肘屈伸 / 躯干旋转 / 膝屈伸 / 髋前屈 / 肩关节外展 / 骨盆倾斜 / 肩胛对称性**九个新增动作，桌面上直接给出实时读数与状态。

> 本项目是**「后续用机器人检测人体动作、判断后做出反应」这一目标的简易 Demo 前置版**：先把「检测关键点 → 算动作特征 → 判动作」的整条链路在本机跑通，为机器人集成打基础。

- 桌面版：OpenCV 实时窗口。单机入口用**一台 USB 外接摄像头**（固定摆在 45° 前侧机位）；另有**双摄入口**同时跑两台（扩展坞），左右并排一个窗口
- **这是动作识别仓库，不是坐姿仓库**。它与坐姿项目 `rtm_demo` 是**各自独立的两个仓库**（本仓库由后者分叉而来）：本仓库只做全身动作，分叉时带过来的坐姿代码已于 2026-10-15 全部删除，坐姿那套代码的家在旧仓库 `rtm_demo`。

## 功能

| 动作 | 特征量 | 判定 |
|---|---|---|
| 举手 | **上臂抬升角**：肩→肘 与 向上竖直轴 的夹角（0° = 完全上举 / 90° = 平举 / 180° = 下垂）。另有肘角（肩-肘-腕内角，180° = 伸直）供对照 | **默认只报数、不判状态**（`--arm-rule` 默认 `off`）；加 `--arm-rule raise` 可开状态机 → `DOWN / RAISED` |
| 仰头 | **头俯仰代理角**：耳中点(3/4)→鼻尖(0) 与 水平线 的夹角（正 = 鼻在耳上方 = 仰头，负 = 低头） | `NEUTRAL / TILTED_BACK`（迟滞双阈值状态机） |
| 躯干前屈/后伸 | 肩中点(5/6)→髋中点(11/12) 连线与竖直轴的**有符号**夹角（正 = 肩中点偏向画面右）。单侧肩/髋不可见时该侧不参与中点；双肩或双髋全不可见 → N/A | 默认只报数；`--trunk-flex-rule on` → `NEUTRAL / FLEXED` |
| 颈部侧屈 | 双耳(3/4)连线与水平线的夹角（正 = 画面右侧的耳更高）。缺一只耳 → N/A，**不做单耳降级** | 默认只报数；`--neck-tilt-rule on` → `NEUTRAL / TILTED` |
| 肘屈伸（左右独立） | 肩-肘-腕 在肘处的内角（180° = 伸直）。三点任一不可见 → 该侧 N/A，左右不合并 | 默认只报数；`--elbow-flex-rule on` → 左右**各一套** `EXTENDED / FLEXED` |
| 躯干旋转 | 双肩连线与双髋连线的方向差（归一化到 ±90°）。四点任一不可见 → N/A。用途：看肩髋是否同步转动、判断动作代偿 | 默认只报数；`--trunk-rotation-rule on` → `NEUTRAL / ROTATED` |
| 膝屈伸（左右独立） | 髋-膝-踝 在膝处的内角（180° = 腿伸直）。三点任一不可见 → 该侧 N/A | 默认只报数；`--knee-flex-rule on` → 左右各一套 `EXTENDED / FLEXED` |
| 髋前屈（左右独立） | 肩-髋-膝 在髋处的内角（180° = 直立）。三点任一不可见 → 该侧 N/A | 默认只报数；`--hip-flex-rule on` → 左右各一套 `EXTENDED / FLEXED` |
| 肩关节外展（左右独立） | 肩→肘 与**向下**竖直轴（手臂自然下垂方向）的夹角（0° = 手臂贴体侧 / 90° = 平举 / 180° = 举过头顶）。肩或肘任一不可见 → 该侧 N/A | 默认只报数；`--shoulder-abduction-rule on` → 左右各一套 `ADDUCTED / ABDUCTED` |
| 骨盆倾斜 | 双髋(11/12)连线与水平线的夹角（正 = 画面右侧髋更高）。缺一个髋 → N/A，**不做单侧降级** | 默认只报数；`--pelvic-tilt-rule on` → `NEUTRAL / TILTED` |
| 肩胛对称性 | 双肩(5/6)连线与水平线的夹角（正 = 画面右侧肩更高）。缺一个肩 → N/A，**不做单侧降级** | 默认只报数；`--scapular-symmetry-rule on` → `SYMMETRIC / ASYMMETRIC` |

**九个新动作默认全部只报数**（与举手的 `--arm-rule off` 同一做法），每类各有一个开关控制是否启用状态机；左右独立动作（肘 / 膝 / 髋 / 肩关节外展）在判定层是**左右各一套状态机**，不合并。

角度读数默认经过一层 **EMA 平滑**（`--smooth-alpha`，默认 0.3；传 1.0 = 关），用来压住「人没动、数字却一直跳」的关键点抖动。入口默认还开着**自适应 EMA**（`--smooth-adaptive on`）：小抖动按 alpha 压，大跳变当帧跟上，减少「压噪」和「跟动作」之间的取舍；另可开**死区**（`--smooth-deadband`，默认 0 = 关）——原始读数与上一输出值之差小于死区时保持上一输出，慢速漂移仍会累积到越界后更新，不会永久冻住。平滑只作用于**有效帧之间**：关键点丢帧照旧显示 N/A，不维持上一有效值；几何辅助线仍画原始关键点坐标。

**几何门控**（features 层，2026-10-18 优化轮）：所有角度都依赖一条线段，线段短于**画面对角线的 1%**（`MIN_SEGMENT_RATIO`，另带 2px 下限）时角度对像素噪声极敏感，读数判为 N/A；RTMPose 对画面外的关节会给**外推坐标**（不是 None），归一化坐标越出 `[-2%, 102%]` 的点也按不可见处理。N/A 时画面上会带原因短码，例如 `N/A  [geom too small]`、`[no point]`、`[out of frame]`、`[no trunk ref]`；控制台同样在 `--` 后附 `(geom)` 一类短码。**宁可不判，不给假数。**

**多帧确认**：单机入口的判定默认要目标状态**连续保持 2 帧**才提交（`--confirm-frames`，默认 2；`1` = 关），避免单帧噪声打出一个假状态。代价是约 1 帧延迟（30fps 下 ~33ms）。双摄入口不判状态，所以没有这个参数。

画面上显示：本次运行选中的动作各两行（读数 + 状态；左右独立动作左、右各一行），以及对应几何辅助线（肩髋连线、双耳连线、三点关节链、肩线/髋线、肩→肘 + 向下/水平参考轴、双髋连线 + 水平参考线、双肩连线 + 水平参考线）。骨架照常画（COCO 全身）。全部动作默认开启时共 22 行；`--actions` 可只显示其中一部分（如 `--actions arm,head`）。

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

### 桌面版（单机，一台摄像头）

```bash
python main_action_demo.py                                   # 默认：USB 那台（45° 前侧）；十一动作全显示，新动作只报数
python main_action_demo.py --actions arm,head                 # 只显示原有两条线（画面更清爽）
python main_action_demo.py --arm-rule raise                   # 举手也开状态机
python main_action_demo.py --trunk-flex-rule on               # 躯干前屈也开状态机（其余新动作同理）
python main_action_demo.py --shoulder-abduction-rule on       # 肩关节外展开状态机（左右各一套）
python main_action_demo.py --pelvic-tilt-rule on              # 骨盆倾斜开状态机
python main_action_demo.py --scapular-symmetry-rule on        # 肩胛对称性开状态机
python main_action_demo.py --smooth-alpha 0.2                 # 更强的平滑（默认 0.3）
python main_action_demo.py --smooth-adaptive off              # 关掉自适应 EMA（退回固定 alpha）
python main_action_demo.py --smooth-deadband 1.5              # 死区：±1.5° 内保持上一输出（默认 0 = 关）
python main_action_demo.py --smooth-alpha 1.0                 # 关掉平滑，看原始读数
python main_action_demo.py --confirm-frames 1                 # 关掉多帧确认（默认 2 帧）
python main_action_demo.py --abduction-ref trunk              # 肩外展改用躯干轴（实验，缺躯干点 → N/A）
python main_action_demo.py --no-geometry                      # 不画角度几何辅助线
python main_action_demo.py --no-skeleton                      # 不画骨架
python main_action_demo.py --snapshot data/action_snap.png    # 按 s 存当前帧
python main_action_demo.py --selftest                         # 合成数据自测（不碰摄像头/模型）
python test_new_actions.py                                    # 新增动作 + 优化项独立冒烟测试（合成姿态）
```

按键：**`s` 存快照**（需 `--snapshot` 指定路径）、**ESC 退出**。首帧较慢（惰性初始化模型），正常。

### 双摄版（扩展坞上两台，一个窗口并排）

```bash
python main_dual_demo.py --list-cameras                       # 先查：0..5 哪些编号能用（认编号用）
python main_dual_demo.py                                     # 两台用 capture 里的默认编号
python main_dual_demo.py --cameras 2,1                        # 显式指定两台编号（左 A / 右 B）
python main_dual_demo.py --actions arm,head                   # 每半幅只显示原有两条线
python main_dual_demo.py --smooth-alpha 0.2                   # 更强的平滑（两台各用各的平滑状态）
python main_dual_demo.py --smooth-adaptive on --smooth-deadband 1.5   # 两台同一组平滑参数
python main_dual_demo.py --abduction-ref trunk                # 肩外展改躯干轴（实验；两台都改）
python main_dual_demo.py --width 640 --height 480             # 带宽不够时降分辨率
python main_dual_demo.py --display-width 1920                 # 窗口整宽上限（默认按屏幕自动）
python main_dual_demo.py --snapshot data/dual_snap.png        # 按 s 存拼好的整张
python main_dual_demo.py --selftest                           # 合成数据自测（不碰摄像头/模型）
```

左半幅 = `cam A`、右半幅 = `cam B`（顺序就是 `--cameras` 的顺序），各有自己的骨架 + 每个动作两行读数，左下角标着名字。**这个入口所有动作都只报数、不判状态**（状态行都写 `not decided (report only)`）——先把两个机位的数看明白，要判状态用单机入口。控制台每台限流打印一行，N/A 读数后面会附原因短码（如 `elbow=L--(no_point)`），正好用来查第二机位为什么测不到。按键同上，ESC 一次关掉两路。

**窗口会按屏幕宽度自动缩**：两台 1280x720 并排是 2560 宽，屏幕只有 1280 宽时右半幅会整个跑到屏幕外（只看得到左边那台）。所以每半幅读帧后先等比缩到「屏幕宽 × 96% ÷ 2」再推理/绘制（角度读数不受等比缩放影响），窗口首帧钉到左上角。缩放上限可用 `--display-width PX` 改，传很大的值（如 `99999`）就不缩。启动时控制台会打一行实际用的上限。半幅变矮后，`output.overlay_layout` 会**自动收紧行高和字号**（十一个动作 22 行仍留在半幅内）；只想看几条线可用 `--actions`。

### 摄像头与机位

**单机入口**只认**那一台** USB 外接摄像头，编号写死在 `capture.py` 的常量里：

```python
USB_CAMERA_INDEX = 2     # 2026-10-18 实拍确认：2 = 扩展坞上那台（45° 前侧），1 = 正对机位那台
USB_CAMERA_INDEX_B = 1   # 双摄入口的第二台——枚举顺序不稳，插上后按实际编号改
```

单机入口**没有"选摄像头"这个参数**（连 `--camera` 都不存在），接上就能跑；换机器、插口顺序变了导致编号不同，就改 `USB_CAMERA_INDEX`。**双摄入口**则接受 `--cameras A,B` 显式指定两台（不传就用上面两个常量）——USB 扩展坞的枚举顺序常常不稳，所以那边留了开关；两台打开时会各自打印「编号 + 实到分辨率」，认不准就照着改。

**各动作对机位的要求不一样**：

- **举手**：竖直方向的动作，**哪个机位都能量**；
- **仰头**：**必须侧对 / 45° 前侧**。正对着摄像头时，仰头主要是朝镜头方向的透视缩短，耳-鼻在画面上几乎不动——**程序会判「几何退化」，读数直接显示 N/A**，而不是给你一个看着像真的错数（详见「已知范围与限制」）；
- **九个新动作**：都是 2D 像面量，角度会随机位/朝向变化（尤其躯干旋转、肩关节外展）。同一机位下标定出来的阈值别直接搬到别的机位。

所以本项目的机位就是 USB / 45° 那台（机位由入口文件声明，不是命令行开关）。

## 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--actions LIST` | 全部十一个 | 本次显示哪些动作（逗号分隔）：`arm,head,trunk,neck,elbow,rotation,knee,hip,abduction,pelvic,scapular`。只影响显示/控制台，不改变各动作自己的判据开关 |
| `--smooth-alpha A` | `0.3` | 角度读数 EMA 平滑系数，取值 `(0,1]`。越小越平滑、跟动作越滞后；`1.0` = 关掉平滑。只平滑有效帧，丢帧仍 N/A；几何线保持原始坐标 |
| `--smooth-adaptive {on,off}` | `on` | **自适应 EMA**（入口默认开；`smoothing.py` 模块默认关）：小抖动按 alpha 压，大跳变（相邻原始值差 ≥ `--smooth-fast-delta`）当帧跟上。`--smooth-alpha 1.0` 时本项不生效 |
| `--smooth-fast-delta D` | `10.0` | 自适应 EMA 判「大跳变」的差值（度），必须 > 0 |
| `--smooth-deadband D` | `0.0` | 死区（度，≥ 0，0 = 关）：原始读数与**上一输出值**之差小于 D 时沿用上一输出。慢速漂移会累积越界后更新，不会被冻住 |
| `--confirm-frames N` | `2` | 判定层**多帧确认**：目标状态连续保持 N 帧才提交（N ≥ 1；`1` = 旧行为）。丢帧立即 UNKNOWN 并清计数 |
| `--abduction-ref {image,trunk}` | `image` | 肩外展参考轴：`image` = 画面向下竖直轴（原行为）；`trunk` = 肩中点→髋中点轴（**实验**；缺任一肩/髋 → 两侧 N/A `[no trunk ref]`，**不静默退回 image**） |
| `--arm-rule {off,raise}` | `off` | **`off` = 举手只报数不判状态**；`raise` = 开启举手状态机 |
| `--arm-raise-enter D` | `45.0` | 抬升角 **≤ D** 判「已举起」（度；**占位值，待标定**） |
| `--arm-raise-exit D` | `60.0` | 抬升角 **≥ D** 判「已放下」（度；迟滞带上沿，**占位值**） |
| `--head-tilt-enter D` | `20.0` | 头俯仰角 **≥ D** 判「仰头」（度；**占位值，且零点不在 0**，见「阈值标定」） |
| `--head-tilt-exit D` | `10.0` | 头俯仰角 **≤ D** 判「中立」（度；迟滞带下沿，**占位值**） |
| `--tilt-min-dx-ratio R` | `0.15` | 仰头**横向退化门控**：耳→鼻的水平跨度 ÷ 双耳间距 < R → 判几何退化、读数作废 |
| `--trunk-flex-rule {off,on}` | `off` | 躯干前屈状态机（`--trunk-flex-enter/exit`，默认 25 / 15 度；**占位值**） |
| `--neck-tilt-rule {off,on}` | `off` | 颈部侧屈状态机（`--neck-tilt-enter/exit`，默认 15 / 8 度；**占位值**） |
| `--elbow-flex-rule {off,on}` | `off` | 肘屈伸状态机，左右各一套（`--elbow-flex-enter/exit`，默认 100 / 130 度；**占位值**） |
| `--trunk-rotation-rule {off,on}` | `off` | 躯干旋转状态机（`--trunk-rotation-enter/exit`，默认 15 / 8 度；**占位值**） |
| `--knee-flex-rule {off,on}` | `off` | 膝屈伸状态机，左右各一套（`--knee-flex-enter/exit`，默认 120 / 150 度；**占位值**） |
| `--hip-flex-rule {off,on}` | `off` | 髋前屈状态机，左右各一套（`--hip-flex-enter/exit`，默认 120 / 150 度；**占位值**） |
| `--shoulder-abduction-rule {off,on}` | `off` | 肩关节外展状态机，左右各一套（`--shoulder-abduction-enter/exit`，默认 60 / 40 度；**占位值**） |
| `--pelvic-tilt-rule {off,on}` | `off` | 骨盆倾斜状态机（`--pelvic-tilt-enter/exit`，默认 10 / 5 度；**占位值**） |
| `--scapular-symmetry-rule {off,on}` | `off` | 肩胛对称性状态机（`--scapular-symmetry-enter/exit`，默认 10 / 5 度；**占位值**） |
| `--no-geometry` | 关 | 不画角度几何辅助线（控制全部十一个动作的几何线） |
| `--no-skeleton` | 关 | 不画骨架 |
| `--snapshot PATH` | 无 | 按 `s` 把当前帧存到 PATH |
| `--selftest` | — | 合成数据自测各层（不碰摄像头 / 模型） |

`main_dual_demo.py` 的参数（**没有 `--arm-rule`：它不判状态**）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--actions LIST` | 全部十一个 | 每半幅显示哪些动作；语义同单机入口。本入口对所有动作**都只报数**，没有 `--*-rule` |
| `--smooth-alpha A` | `0.3` | 每半幅的 EMA 平滑系数，语义同单机入口；**两台各用各的平滑状态** |
| `--smooth-adaptive` / `--smooth-fast-delta` / `--smooth-deadband` | `on` / `10.0` / `0.0` | 语义同单机入口；两台同一组参数、各自的平滑状态 |
| `--abduction-ref {image,trunk}` | `image` | 语义同单机入口（trunk 为实验项）；两台都按这个参考轴 |
| `--list-cameras [N]` | — | **只枚举、不跑主循环**：把编号 `0..N-1` 挨个试着打开，打印哪些能用和实到分辨率（不带 N 就扫 `0..5`）。认不准 `--cameras` 时先跑这个 |
| `--cameras A,B` | `2,1` | 两台的摄像头编号（左 A / 右 B），必须不同。默认取 `capture.USB_CAMERA_INDEX` / `USB_CAMERA_INDEX_B` |
| `--width` / `--height` | `1280` / `720` | 期望分辨率。两台挂一个扩展坞带宽不够时降到 `640x480` |
| `--display-width PX` | `0`（自动） | 拼图整窗最大显示宽度；默认按屏幕宽度留 4% 边距。每半幅先等比缩到这个上限的 1/2 再推理/绘制。传很大的值可关掉缩放 |
| `--tilt-min-dx-ratio R` | `0.15` | 同单机入口（仰头横向退化门控） |
| `--log-interval SEC` | `0.5` | 控制台每台每 SEC 秒打一行实时读数；`0` = 不打（本入口不判状态，没法靠状态变化触发打印，所以按时间限流） |
| `--no-geometry` / `--no-skeleton` / `--snapshot PATH` / `--selftest` | — | 同单机入口 |

## 阈值标定

**十一个动作的阈值目前都是占位值，不是标定结果**，先按画面读数标定再定。

画面与控制台上直接显示实时读数：

- **举手**：`arm_raise` 是抬升角（**越小越举得高**：0° 完全上举 / 90° 平举 / 180° 下垂），`elbow` 是肘角（180° = 伸直）。两侧独立显示。
- **仰头**：`head_pitch` 是耳→鼻夹角（**正 = 仰头，负 = 低头**）。
- **躯干前屈**：夹角绝对值越大越偏离竖直；符号只表示偏向画面左/右（**触发看绝对值**）。
- **颈部侧屈**：符号 = 哪一侧的耳更高（正 = 画面右侧），**触发看绝对值**。
- **肘 / 膝 / 髋屈伸**：180° = 伸直，越小越屈；**触发是「≤ enter」**，恢复是「≥ exit」。
- **躯干旋转**：肩线与髋线方向差的绝对值；0 = 肩髋平行，**触发看绝对值**。
- **肩关节外展**：肩→肘与**向下**竖直轴的夹角（0° 手臂贴体侧 / 90° 平举 / 180° 举过头顶），两侧独立；**触发是「≥ enter」**，恢复是「≤ exit」。
- **骨盆倾斜 / 肩胛对称性**：连线与水平线的夹角，符号只表示画面哪一侧更高（正 = 画面右侧），**触发看绝对值**。

标定方法：

1. 摆好**中立**姿势（举手：手臂自然垂下；仰头：正常平视；新动作：自然站立、肘膝伸直、躯干不歪不转、双肩双髋水平），记下读数；
2. 摆出**动作**姿势（举手：举过肩 / 举过头顶；仰头：明显后仰；外展：手臂向侧方平举 / 上举；躯干/颈/肘/膝/髋/骨盆/肩胛：做到位），记下读数；
3. 把 enter 阈值定在两档之间靠近中立一侧，exit 定得更靠动作一侧，**中间留出迟滞带**（避免边界抖动来回跳）。**注意方向**：躯干/颈/旋转/外展/骨盆/肩胛是「绝对值 ≥ enter」，肘/膝/髋是「角度 ≤ enter」。

**两个容易踩的坑**：

- **仰头的零点不在 0**：45° 机位下「中立站姿」实测约 **-20~-30°**（鼻尖天生比耳中点低一点）。所以阈值要**按画面上中立时的读数往上加**，别照数字本身调。中立基线还会随机位 / 摄像头高度 / 人体比例变化，**换个条件就要重看**。
- **只有那一个门控是既定值**：`--tilt-min-dx-ratio`（默认 0.15）不是阈值而是「这个机位量不量得了」的门控，有实测支撑（45° 前侧 0.43~0.52，正对 0.013），可以直接用。

## 项目结构（分层架构）

```
rtm_action_demo/
├── main_action_demo.py   # 【单机入口】装配各层、跑主循环；一台 USB 摄像头
├── main_dual_demo.py     # 【双摄入口】两台并排一个窗口；两台都只报数、不判状态
├── requirements.txt      # Python 依赖（pip install -r requirements.txt）
├── capture.py            # 图像输入：FrameSource / CameraCapture（编号默认写死，双摄入口可传）
├── pose_estimation.py    # RTMPose 推理：17 个 COCO 关键点
├── features.py           # 特征：十一个动作各一个类（举手/仰头/躯干前屈/颈部侧屈/肘屈伸/躯干旋转/膝屈伸/髋前屈/肩外展/骨盆倾斜/肩胛对称性）
├── smoothing.py          # 平滑：AngleSmoother，按 key 独立做 EMA（有跨帧状态，每路一个）
├── decision.py           # 判断：十一个动作各一个状态机；肘/膝/髋/肩外展按 side 各建一套
├── output.py             # 输出：骨架绘制 + 动作读数/状态/几何辅助线
├── pose_schema.py        # 姿态对齐：COCO17 / 机器人 27 舵机 → 统一 27 DOF 规范 schema（为对接机器人预留）
├── test_new_actions.py   # 新动作独立冒烟测试：features → smoothing → decision → output
├── models/               # RTMPose ONNX 模型（未提交）
└── data/                 # 输出目录（快照等；被 .gitignore 挡住，仓库里只提交 .gitkeep）
```

数据流：`capture → pose_estimation → features → smoothing → decision → output`，各层职责单一、接口稳定。

`smoothing.py` 只碰以 `_deg` 结尾的数值键：`*_valid` / `*_conf` / `*_geom` 原样透传，几何辅助线仍用原始像素坐标；`None` 照旧输出 `None` 并清掉该 key 的历史，不跨丢帧沿用旧值。

`pose_schema.py` 是**纯标准库**模块（不 import cv2/numpy），为将来"人体→机器人关节映射"预留，当前数据流不用它；可单独跑 `python pose_schema.py` 自测。

关键点编号为 **COCO 17 点标准**：0 鼻、3/4 耳、5/6 肩、7/8 肘、9/10 腕、11/12 髋等。

## 后续计划

- **先跑几轮实拍**，把举手 / 仰头的 enter/exit 从占位值换成标定值；顺便决定举手是开状态机还是继续只报数。
- **加更多动作**（挥手 / 转身 / 蹲下…）：照 `ArmRaiseFeatures` + `ArmRaiseDecision` 的样子各加一对类，接口不变。
- **机器人摄像头**：当前用的是本机 USB 摄像头；后续接入机器人摄像头时，只需在 `capture.py` 新增一个图像源子类（`FrameSource` 的子类），上层数据流不用改。
- **人体 → 机器人关节映射**：在 `pose_schema.py` 的规范 27 DOF 空间里做（人体侧角度/位置 → 机器人目标角）。
- **多目标 / 人不在画面中心**：`pose_estimation.py` 接 RTMDet 检测器，给 `detect_pose` 加可选 bbox，上层不用改。

## 已知范围与限制

- **动作线目前没有任何提醒**：只显示读数与状态，不触发提醒、不做时长判断。要不要接提醒，待定。
- **十一个动作的阈值都是占位值，未标定**；读数的中立基线随机位 / 摄像头高度 / 人体比例变化，换条件要重看。
- **九个新动作默认只报数**：判据类已写好并自测通过，但要加对应的 `--*-rule on` 才会跑状态机（与举手 v1 的取舍一致）。
- **新动作也是 2D 代理量，不是临床关节角**：
  - 躯干前屈量的是肩髋连线在像面上的倾角，单目投影 + 机位会影响量值，**只保证同机位下单调**；
  - 躯干旋转量的是**肩线/髋线在像面上的方向差**，不是真实轴向旋转（yaw）；人正对镜头时前后方向的旋转在像面上会明显变小；
  - 膝/髋动作要求**下半身真的在画面里**；RTMPose 会把画面外的点外推出坐标（不是 None）。现在超出画面 2% 边距的点会被判 `out of frame`、读数 N/A；但刚好卡在画面边缘的点仍可能是外推值，看骨架和几何线确认。
  - **肩关节外展在 2D 像面上分不清额状面/矢状面**：手臂向侧方抬和向前抬在单目像面上都会让肩→肘角变大，读数无法区分是哪种；180° 附近也分不清是「举过头顶」还是「另一侧的外展/内收」。左右各测各的，不做合并。
  - **骨盆倾斜 / 肩胛对称性量的是连线倾角**：摄像头本身没摆平（roll）会把整段读数整体带偏，**机位水平是这两个量的前提**；它们也只是 2D 像面代理，不等于真实骨盆/肩胛倾斜角。
- **不做单点降级**：骨盆倾斜、肩胛对称性、颈部侧屈都要求两个端点都可见，缺一个直接 N/A；肩关节外展按侧独立，缺该侧肩或肘只作废该侧。
- **左右独立动作不做合并**：肘/膝/髋/肩外展左右各一套读数与状态机；一侧丢点只作废该侧，另一侧照常输出。
- **仰头只在侧对 / 45° 前侧有效**：正对摄像头时仰头主要是朝镜头方向的透视缩短、耳-鼻在画面上几乎不动，程序判为**几何退化**、读数显示 N/A。更隐蔽的是此时 `|Δx| ≈ 0` 会让角度退化成「只剩符号」的 ±90°，长得像正常读数——所以特征层主动把它作废，不给假数。
- **仰头角是 2D 代理，不是真实俯仰角**：COCO 17 没有下巴 / 耳屏，用「耳点 + 鼻尖」代替头部轴；45° 机位下水平方向被压缩，读数比真实角大，**只保证单调**（仰头越大读数越大），量值与任何文献的头屈伸角都不可比。**不补偿转头**（yaw），头往侧面转也会串进读数。
- **举手角是像素空间的上臂朝向**：单目 2D，不含深度；只看「肩→肘」朝向，手腕丢失不影响触发。
- **平滑只有 EMA（含自适应/死区），不维持上一有效值**：默认 `--smooth-alpha 0.3`、自适应开（`1.0` = 全关）。它只压「有效帧之间」的抖动；关键点丢帧照旧标无效、读数写 N/A，下一有效帧从原始值重新起步。自适应 EMA 让大跳变当帧跟上，**但不是运动预测**——真动作仍会跟手，只是不再靠牺牲压噪换响应。死区默认关；打开后固定噪声更稳，但极慢的动作可能被压到死区边界。想看原始读数（判断抖动有多大、关键点撑不撑得住）传 `--smooth-alpha 1.0`。几何辅助线不做平滑，抖动时线和数值可能有轻微出入。
- **多帧确认引入约 1 帧延迟**：`--confirm-frames 2`（默认）在 30fps 下约 33ms、60fps 下约 17ms；对阈值附近的快速来回最有用。嫌滞后就传 `1`（等于旧行为）。它只影响判定层，不影响画面读数。
- **几何门控会让更多帧显示 N/A**：线段短于画面对角线 1%、关键点越出画面 2% 边距时，读数如实作废并标原因（`geom too small` / `out of frame`）。这是有意的「宁可不判、不给假数」；比例可在 `features.MIN_SEGMENT_RATIO` 调，但改完要重跑各层自测。
- **N/A 原因短码**：`no pose`（没人/模型没返回）、`no frame size`、`no point`（关键点不可见）、`out of frame`（RTMPose 外推点）、`geom too small`（线段过短）、`geometry degenerate`（仅仰头：正对镜头）、`no trunk ref`（仅 `--abduction-ref trunk`：肩/髋点不全）。画面上跟在 N/A 后，控制台跟在 `--` 后。
- **`--abduction-ref trunk` 是实验项**：原始需求写的是「躯干竖直轴」，当前默认仍是画面向下竖直轴。trunk 模式用肩中点→髋中点；躯干侧倾时会改变读数，但肩髋中点本身抖动也会串进来，未实拍验证前不建议当正式判据。缺躯干点直接两侧 N/A，**不会偷偷退回 image 参考**。
- **单 / 多人**：目前按「整帧单一人」处理（top-down 无检测器；整帧失败时按人像区域裁剪重推，是接检测器之前的临时替代）。
- **双摄入口不判状态**：`main_dual_demo.py` 两台的所有动作都只报数（状态行写 `not decided (report only)`）。两个机位里有几个能测仰头、读数基线各是多少，要先看数据才知道，所以这一档刻意不判。要判定用单机入口。
- **双摄的帧率**：两台是**顺序**读帧、顺序推理（没有多线程），帧率约为单机的一半。看数够用；真要两台都跑满再单独做一轮。
- **双摄的编号**：USB 扩展坞的枚举顺序常常不稳，`--cameras A,B` 就是为此留的。**认不准就先跑 `--list-cameras`**，它把 `0..N-1` 挨个开一遍、列出哪些能用和实到分辨率；两台同型号时分辨率也一样，那就得看画面内容才知道哪个编号对应哪个机位。两台打开时还会各自打印「编号 + 实到分辨率」。
- **模型**：使用 rtmpose-s（`models/` 内）
