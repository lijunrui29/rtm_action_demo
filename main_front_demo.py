"""
正面机位入口：只跑 Sternum–Ear 角度的**实验显示行**。

和侧面入口 main_demo.py 的区别（机位由**入口文件**声明，不是命令行参数）：

| | main_demo.py（侧面 / 45° 前侧） | 本入口（明显正面） |
|---|---|---|
| 头前伸线 | CVA-like proxy 三档 + 提醒 + CSV | Sternum–Ear 两档**只显示** |
| 画的东西 | 45° 读数块 + 几何 + Head Fwd 摘要行 | 正面实验行（两行）+ V 形几何 |
| 提醒 | 有（FhpAlert） | **无**（没有配套 Alert，不参与任何判定） |

指标口径（2026-09-28 起，取代原来的 Ear–Shoulder 水平位移）：顶点 = 双肩中点
（近似胸骨中点，构造出处 Molaeifar et al. 2021），两条边 = 顶点 → 左耳 / 右耳
（近似耳屏），**像素空间**夹角，**越大越前伸**。它是工程代理，不是临床 FHD。
换它的原因：正面实测那条 Ear–Shoulder 线**不单调**（第一段 Obvious 中位数低于
第一段 Slight），且抖动 0.252 对本角度的 0.076。**但别当成多出一个能力** ——
它与颈压缩高度冗余（段级 Pearson −0.989），是同一个几何的另一种写法。

**45° 那条线在本入口一个对象都不装** —— 它的正面预设（自标定的 84.82 / 79.32）
实测不可用（坐直段仍误报 ~12%，恢复点 86.55° 高于坐直中位数 86.28°，见
CLAUDE.md 开放问题②），装上就是实机那次误报的来源。那个预设 2026-09-29 已被
换成文献切点（见 decision.CVA_PROXY_VIEW_PRESETS 的注释与
data/archive/measured_basis_2026-09-29.md），但本入口照旧一个对象都不装。

用法（在 rtm_demo/ 目录下运行）：
    python main_front_demo.py --camera 0
    python main_front_demo.py --camera 0 --front-fhp-normal-max 58   # 重标后覆盖上界（度）
    python main_front_demo.py --no-fhp-geometry                       # 不画 V 形几何
    python main_front_demo.py --selftest                              # 合成数据自测各层

⚠ 上界 54.23° 是 1 人 × 3 段的 **in-sample 种子值**（规则与旧的 0.065 相同：Normal
均值与 FHP 合并均值的中点）。**不可跨机位**（正面阈值套 45° → FHP 漏报 100%，实测），
也不可跨人/跨次沿用（同人同机位两轮"坐直"基线是从 0.067 漂到 0.027 的那种量级）。
换条件请用 tools/test_molaeifar_sternum_angle.py 重采后用本参数覆盖。数据 CSV
与标定产物一律落在 data/（实拍读数 = 个人身体数据，被 .gitignore 挡住）。

ESC 退出。主循环与侧面入口共用 main_demo.run_pipeline，没有第二份。
"""

from main_demo import build_parser, run_pipeline, selftest


def main() -> None:
    """正面机位入口：装正面实验显示行，45° 那条线不装配。"""
    args = build_parser(front=True).parse_args()
    if args.selftest:
        selftest()   # 各层自测与机位无关，两个入口共用
        return
    run_pipeline(args, front=True)


if __name__ == "__main__":
    main()
