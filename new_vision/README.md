# 这是Robocup人型的工作区

赛道形态：主体是一个封闭的环形赛道，整体呈椭圆形或"体育场"形状（由两条平行的长直道和两个半圆形弯道组成）。
双环结构：赛道由两条黑色的同心闭合曲线构成，分别是外圈边界和内圈边界。两条线之间的区域即为机器人的行驶跑道。

凹陷设计：在赛道底部（大约数字50到60之间的区域），内圈和外圈的线条都向内（向圆心方向）有一个半圆形的凹陷。这使得赛道在该段变窄或改变了曲率。
垂直分隔线：在底部偏右的位置，有一个显著的矩形区域结构：
- 红色虚线：有一条垂直的红色虚线横跨赛道宽度（连接内圈和外圈），表示有一定高度的红色的横棍，比赛中需要跨过。
- 黑色实线：在红色虚线的右侧，有一条垂直的黑色实线，同样连接内外圈表示起点和终点。
这两条线与赛道边界共同围成了一个狭长的矩形区域，机器人需要在这个区域内完成特定的动作（如跨越红色横棍）才能继续前进。

## 2026赛季任务

- 沿环形赛道逆时针双足步行一圈（614cm，限宽门最窄240mm）
- 识别3处几何图卡（圆形/五角星/正方形/菱形/十字形/三角形，10cm×10cm白底黑线）并执行对应动作维持3秒以上
- 跨越红色木条障碍（10×20mm截面，竖放/平放自选）
- 评分：竞速0-200 + 图卡3×20 + 越障60 - 干预扣分(5/次) + 附加分20

## 文档与配置放在哪

- **规划 / 研究笔记 / 接手说明**：[`docs/project/`](docs/project/)（`AI_log.md`）
- **场地与对照用 JSON（非主仿真配置）**：[`config/`](config/)，说明见 [`config/README.md`](config/README.md)
- **仿真与脚本默认巡线参数**：仓库根目录 [`line_follow_params.json`](line_follow_params.json)（勿随意改路径，控制器与多脚本硬编码指向此处）
- **2026新规则**：[`docs/新赛事规则2026.md`](docs/新赛事规则2026.md)；2025旧规则历史参考：根目录 `rule.pdf`
- **实车视觉代码**：[`jetson/`](jetson/)（V1巡线 + 红条 + 窄门 + 图卡 + 数字识别）
- **其余技术文档**：[`docs/`](docs/)
- **归档/废弃代码**：[`archive/`](archive/)


## Walking policy UDP integration

For closed-loop ONNX walking, run `jetson/run_policy_vision.py` and the policy's
`--command-source vision` mode. Vision sends directly to policy port 5005 by
default; the root `connector.py` is an optional legacy relay. See the
[detailed setup/test guide](../docs/closed_loop_vision.md). This entry point uses
the new CPU line detector and sends the historical UDP JSON velocity commands;
it does not open the STM32 serial port or execute shape/bar actions.
