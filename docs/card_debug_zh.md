# 图卡识别可视化测试

在 Jetson 连接的屏幕上打开终端：

```bash
cd /home/isaac/jetson_orin_code
bash scripts/run_card_debug.sh
```

程序只使用摄像头和现有 ShapeDetector，不启动策略、串口、网络发送或运动控制。它会读取 `config/button_start.env` 中的相机曝光、白平衡、预处理和亮度映射配置，使测试和当前实车配置一致。没有此配置时使用当前相机参数和 selective/legacy。退出不会恢复手动应用的摄像头参数。

画面顶部始终显示**当前帧**的状态和识别类别；图卡离开后不会继续显示上一次类别。

| 状态 | 含义 |
| --- | --- |
| NO CARD | 当前帧无合格黑框，也无已确认的存在信号 |
| PRESENCE CUE ONLY | 只有较弱的图卡存在信号，尚未找到可分类的合格黑框 |
| BOX FOUND / UNKNOWN | 找到黑框，但内部形状未识别 |
| RECOGNIZED | 已识别出图卡形状，不受动作触发距离或冷却影响 |

绿色四边形是通过验证的黑框，橙色矩形是存在信号候选。左侧显示原始摄像头，右侧显示检测器灰度图、矫正灰度图、局部二值图和原分类器 warp；下方显示选择性二值图和 ring 墨迹图。未找到图卡时，中间图窗口会清空。底部显示候选数量、几何筛选结果、拒绝原因及两条分类路径的结果。CV 规则分类没有概率置信度，closure 是框的评分，不能当作概率。

| 按键 | 操作 |
| --- | --- |
| 0 | 之后的帧标为无卡，用于测误检 |
| 1 / 2 / 3 | 圆形 circle / 五角星 pentagon / 正方形 square |
| 4 / 5 / 6 | 菱形 diamond / 十字 cross / 三角形 triangle |
| U | 取消真实标签，继续观察但不计正确率 |
| P | 暂停/继续；暂停时摄像头继续采集，恢复时使用最新帧 |
| S | 保存当前原图、面板、全部中间图和诊断 JSON |
| R | 清零统计，帧日志保留 reset 事件 |
| Q / Esc | 退出并保存统计 |

切换真实标签从**下一张处理的帧**开始生效。建议先按 P 暂停、更换图卡并选标签，再按 P 继续；每个测试条件使用 R 清零或另开一次运行。每帧正确率包含漏检和无法分类的帧；默认未标注时显示 N/A。它是处理帧的正确率，连续帧相互相关，不等同于独立图卡试验的准确率，也没有替代真实标签的自动判断。

每次运行输出到 `records/card_debug/时间戳/`：

- `session.json`：运行参数、相机配置和检测阈值。
- `camera_controls.json`：相机设置的实际读回值和错误。
- `frames.jsonl`：逐帧真实标签、识别状态、候选数、拒绝原因、框坐标、分类路径、耗时等全部标量诊断。
- `summary.json`：统计和混淆表，正确率覆盖最近一次 R 之后的帧。
- `snapshot_*/`：按 S 或达到 `--max-frames` 时保存的图片和诊断。退出时一定保存 summary，但不会自动保存每帧图片。

输出处于 Git 忽略的 records 目录，不会混入源码提交。

其他用法：

```bash
# 直接指定摄像头；保留当前硬件曝光/白平衡，使用默认识别参数
.venv/bin/python new_vision/jetson/card_debug.py --camera /dev/video0

# 同样的实车参数，20 帧后自动退出（也会保存最后一帧面板）
bash scripts/run_card_debug.sh --max-frames 20

# 无显示器/SSH 环境测试摄像头
bash scripts/run_card_debug.sh --headless --max-frames 50 --expected circle

# 离线视频/图片（直接调用，不额外应用摄像头配置）
.venv/bin/python new_vision/jetson/card_debug.py --video example.mp4 --expected triangle
.venv/bin/python new_vision/jetson/card_debug.py --image example.jpg --headless --expected square

# 使用其他预处理进行对比，命令末尾参数覆盖脚本默认值
bash scripts/run_card_debug.sh --photometric-mode raw --shape-preprocess canny
```

请在 Jetson 桌面终端启动 GUI。没有 DISPLAY 时程序会提示使用桌面或 headless，避免无窗口运行。若无法打开摄像头，检查 USB 和设备名，并退出占用摄像头的视觉程序。这个测试仍使用原来的地面图卡几何验证（标定相机、约 10cm 卡片），请把卡放在地面视野里测试；手举正对镜头的卡可能被现有地面几何约束拒绝，具体原因可看底部和日志。中文显示取决于 PIL 和中文字体，有缺失时使用英文类别，五角星在旧分类器中叫 pentagon。
