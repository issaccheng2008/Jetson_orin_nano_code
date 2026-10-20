# 手机 / 电脑双摇杆遥控

手机、电脑和 Jetson 连接同一个 Wi-Fi 或手机热点，浏览器打开 Jetson 提供的页面即可。
不需要域名、云服务器或安装手机 App。网页静态资源全部在本地，断开互联网也能使用。

左摇杆向上推动控制前进速度 `vx`，右摇杆左右推动控制角速度 `wz`。
不能倒退或横移；两个摇杆可以同时操作，松开某个摇杆只让它对应的目标归零。

## 遥控专用自启动（推荐）

本分支的新自启动流程为：开机只提供网页，手机点击“启用控制”后才启动行走上位机。
上位机与 STM32 就绪前，摇杆保持零指令；页面显示“控制中”后再重新推动摇杆。
停止、失焦或失联会结束本次上位机运行，重新启用才会再次启动。

在 Jetson 上执行一次 `bash scripts/install_remote_autostart.sh` 可安装新服务，并停用、备份和屏蔽旧视觉自启动。
配置检查成功后才切换旧服务。详见 [遥控自启动安装说明](remote_autostart_zh.md)。
这里没有远程操作 Jetson 的已安装服务，需在设备上运行安装命令才会生效。

## 手动两终端启动

使用原来能够运行上位机的 Python 环境。遥控网页服务只用 Python 标准库，**不需要额外 pip 安装**。
以下命令在仓库根目录运行，例如 `~/Jetson_orin_nano_code`。

终端 1：

```bash
cd ~/Jetson_orin_nano_code
python3 -m remote_control
```

终端会打印随机 `Access token`（访问口令）。每次重新启动默认生成新口令。
执行 `hostname -I` 查看 Jetson 的局域网 IP，例如 `192.168.1.20`。
在手机 / 电脑上打开 `http://192.168.1.20:8080`，输入终端口令并连接。
手机上使用 Jetson 的 IP，不要输入 `127.0.0.1` 或 `0.0.0.0`。

终端 2：

```bash
cd ~/Jetson_orin_nano_code/humanoid_jetson_deploy
python3 main.py --model models/current_walking.onnx \
  --command-source remote --no-plot
```

**将 `models/current_walking.onnx` 换成你已验证能行走的兼容模型路径**。
该路径是示例，仓库不保证存在这个文件；模型需符合现有上位机的 49 维 walking 接口。
此命令保留现有 dry-run 行为，仍需要 STM32 状态反馈，但不发送电机使能标志。
确认模型、机器人支撑和原有启动流程后，要实际行走时在该命令最后加 `--enable-motors`。
在这个手动两终端模式中，网页的“启用控制”只取得速度指令控制权，电机是否使能由你启动 `main.py` 的参数决定。
上面的自启动模式则会在网页启用时启动带使能标志的上位机。

连接成功后点击“启用控制”，再使用摇杆。另一台手机或电脑可以连接查看指令，但无法抢占控制权。
先在原设备停止，再在另一设备启用，即可切换操作者。

## 操作

| 操作 | 效果 |
| --- | --- |
| 左摇杆向上 | 前进，位移越大指令越快 |
| 右摇杆向左 / 向右 | 左转 / 右转；可以叠加前进 |
| 松开摇杆 | 对应目标归零，正常输出按变化率限制下降 |
| W / ↑ | 按住前进，松开归零 |
| A / ←、D / → | 按住转向，松开归零 |
| 空格 / 页面停止按钮 | 立即发送零速度指令，并撤销本次控制权 |
| 切换标签页、离开页面、窗口失焦 | 停止控制；返回后需要重新启用 |

页面显示的是**目标指令**和**服务端发布指令**；延迟是浏览器到服务的请求往返时间。
UDP 没有执行回执，因此“已连接”不代表电机已启动或机器人已执行。
页面没有编造实际速度、电量或机器人在线状态。

## 速度和转向参数

默认上限：前进 `0.4 m/s`，转向 `0.5 rad/s`。示例：

```bash
python3 -m remote_control --max-vx 0.2 --max-wz 0.3
```

网页服务端硬上限为 `vx ≤ 1.0 m/s`、`|wz| ≤ 0.5 rad/s`。
接口上限不代表任意模型在这个范围都能稳定行走，使用已经验证过的速度。
页面速度档位是服务端上限的比例，无法越过服务端配置。

默认左转对应正角速度。若实机左右相反，只需反转网页输出的转向符号：

```bash
python3 -m remote_control --yaw-sign -1
```

正常运动的变化率默认 `--vx-accel 1.0`（m/s²）、`--wz-accel 2.0`（rad/s²）。
松手会按这两个参数减速。“停止”和失联则直接归零，不等待缓慢减速。

## 断连与停止的实际含义

- 浏览器约每 50 ms 更新输入；遥控服务以 50 Hz 向上位机发布命令。
- 服务 300 ms 没有收到有效新指令，会清除控制会话，并在下一发布周期发送全零速度。
- 超时、停止后的旧请求不能恢复移动；再次移动需要点击“启用控制”。
- 服务本身退出或崩溃后，上位机还有默认 250 ms 的 UDP 超时归零保护。
- 网页停止先将 `vx=0, wz=0`；自启动模式还会结束上位机并沿用其 disable 流程。手动两终端模式保持原来的只归零行为。
- 这些软件停止操作**不是独立硬件急停**，也不是机器人瞬间静止的保证。
- 完整退出上位机时，在运行 `main.py` 的终端按 Ctrl+C，沿用现有发送电机 disable 包的流程。

## 不接机器人，先检查网页与链路

在电脑或 Jetson 上，只启动 `python3 -m remote_control` 就能检查网页操作。
网页不需要 ONNX、串口或电机即可加载，但这不会验证真实机器人运动。

如需检查实际 UDP 数据，可在另一个终端运行下面的接收器（此时不要同时启动 `main.py`，它们会占用同一个端口）：

```bash
python3 -u - <<'PY'
import json
import socket
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
    receiver.bind(("127.0.0.1", 5007))
    count = 0
    while True:
        message = json.loads(receiver.recv(2048))
        count += 1
        if count % 10 == 0:
            print(message)
PY
```

推动摇杆后应看到 `vx` / `wz` 变化，`vy` 永远为 0、`qr` 永远为 -1。
停止或断开浏览器后输出应归零。结束接收器后再启动正式上位机。

## 网络和启动配置

- 默认页面端口 `8080`，可用 `--port 8081` 修改。
- 默认监听 `0.0.0.0` 供同一网络的设备访问；仅本机预览用 `--bind 127.0.0.1`。
- 固定口令可通过环境变量设置，避免把口令放进 URL：

  ```bash
  export ROBOT_REMOTE_TOKEN='replace-with-a-long-random-token'
  python3 -m remote_control
  ```

- 网页访问口令只放在页面内存，不写入 localStorage，刷新后重新输入。
- 页面打不开：检查双方 IP、8080 入站防火墙、路由器访客网络 / AP 隔离；手机热点可能不允许连接它的两个设备互访。
- 使用可信局域网。首版采用 HTTP，不要把端口直接映射到公网或挂到公网域名。
- 遥控 UDP 始终发往 Jetson 本机 `127.0.0.1:5007`。修改端口必须两端一致：网页服务 `--policy-port 5507`、上位机 `--udp-command-port 5507`。
- 视觉默认使用 5005，遥控默认使用 5007。遥控模式不需要启动 `connector.py` 或视觉进程；不要将两路发送器改成同一端口。
- `remote` 只支持 walking 模型，不能与 `--fixed-policy`、`--policy one-foot` 或 `--one-foot-model` 组合。

## 开发验证

```bash
python3 -m unittest discover -s tests -v
cd humanoid_jetson_deploy
python3 -m unittest discover -s tests -v
```

新增控制状态和 HTTP / UDP 测试不连接真实电机。浏览器和软件验证不能代替实机步态、转向符号与无线网络测试。

前端控制状态测试可使用 Node.js 单独执行，不需要安装 npm 包：

```bash
node --test tests/test_remote_ui.cjs
```

可选真实浏览器测试入口为 `tests/browser_remote_smoke.cjs`，需要开发环境安装 Playwright 和 Chromium。
先在一个终端运行本机模拟服务（输出到测试端口 15007）：

```bash
ROBOT_REMOTE_TOKEN=local-test-only python3 -m remote_control \
  --bind 127.0.0.1 --port 8088 --policy-port 15007
```

另一个终端执行：

```bash
ROBOT_REMOTE_TEST_TOKEN=local-test-only node tests/browser_remote_smoke.cjs
```

测试只允许连接 `127.0.0.1`，请使用上述模拟服务；它会主动发送运动指令来验证链路。
本次软件验证于 2026-10-20 完成：两套 Python 测试共 131 项、前端状态测试 11 项、真实浏览器检查 21 项，全部通过。

界面设计参考：[Teenage Engineering TP–7](https://teenage.engineering/products/tp-7) 的实体操控感与 [Porsche 仪表式设计](https://shop.porsche.com/us/en-US/porsche-design-timepieces-custom-built-timepieces) 的清晰刻度；页面布局和代码为本项目实现。
