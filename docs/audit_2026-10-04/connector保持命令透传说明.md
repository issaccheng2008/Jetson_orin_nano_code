# Connector保持命令透传

本次仅修改`connector.py`及其测试。客户端和最终policy接收层由其他任务配合修改。

UDP导航包可增加`"command_mode":"held"`或`"command_mode":"continuous"`。其他值、数字、布尔、数组、对象以及显式null均拒绝；非法包不更新视觉有效时间。

| 输入 | Connector行为 |
|---|---|
| held普通运行包 | 在已有输入范围校验后，完整透传vx/wz，且同步平滑器内部状态 |
| continuous或没有标签的普通运行包 | 保留现有vx/wz斜率限制及wz限速为0时不限斜率的行为 |
| vx与wz均为0 | 同tick立即双零、清平滑器状态 |
| hold_upright或card_tilt为真 | 同tick立即双零、清平滑器状态；姿态标志、QR和事件字段保留 |
| 视觉超时 | 同tick立即双零、清平滑器状态；旧QR、事件和标签不延续 |
| Connector退出 | 立即发三份零命令，不经过减速斜坡 |

例如客户端发送`{"vx":0.5,"wz":0.5,"qr":-1,"command_mode":"held"}`，Connector第一份输出就是完整0.5/0.5，不会先发0.02/0.04。Connector不强行把所有输入改成0.5；保持客户端选择的停车、直行及转向含义。

`held`描述命令语义，发布仍默认50Hz；既有视觉超时仍默认0.25秒。它不允许客户端只每0.5秒发一份心跳。转向保持时，客户端仍需按原节奏刷新命令。

测试：Linux Python3和Windows Python3.13均通过22项`test_connector.py`测试，包括两种转向符号、完整数值透传、每20ms重复输出、标签校验、标签切换时状态同步、全零/姿态/超时停止及重启不续旧转向。原无标签普通运行斜率回归保持通过。没有连接实车验证。

两源文件原工作树已经为CRLF；局部替换保留原换行。审阅功能差异可用`git diff --ignore-space-at-eol -- connector.py tests/test_connector.py`。
