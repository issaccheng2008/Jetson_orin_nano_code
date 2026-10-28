# 配置与基线数据分层

本目录放**不随烧机/仿真主循环直接读取**、但用于场地、底图、对照实验的 JSON。  
**例外**：仓库根目录的 `line_follow_params.json` 仍是 Webots 控制器与多数脚本的**默认路径**，请勿在未批量改代码的情况下挪走。

| 路径 | 用途 |
|------|------|
| `field/场地参数基线.json` | 场地外框、赛道尺寸、障碍/起跑线等在 DXF/规则侧的基线；`tools/generate_track_png.ps1` 读取 |
| `presets/*.json` | 巡线参数的只读快照 / 黄金对照（例如仿真 OpenCV 管线对标用），需手动复制到根 `line_follow_params.json` 或 `OpenMV_flash/` 才生效 |

真机刷机三件套仍以 **`OpenMV_flash/`** 为准；根目录 **`line_follow_params.json`** 为仿真与参数迭代主副本。
