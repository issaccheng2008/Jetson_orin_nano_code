# Jetson 参数修改与 vi 保存速查

适用于 `policy49-button-test` 的按钮自启动模式。下面的命令直接在 Jetson 终端输入。服务启动顺序为 connector → 视觉 → 等待 PC2 按钮 → 步态。

## 1. 首次安装或更新服务

首次使用，或更新了服务安装脚本时执行：

```bash
cd ~/jetson_orin_code
git fetch origin
git switch policy49-button-test
git pull --ff-only && bash scripts/install_button_autostart.sh
```

安装脚本不要加 `sudo`；它会在需要时请求密码。首次安装生成 `config/button_start.env`，已有配置保留，并启用下次开机自启动。安装本身不立即启动程序。

## 2. 启动程序

```bash
sudo systemctl start humanoid-button-vision.service
```

会先启动 connector，再启动视觉。识别并锁存首张图卡后 PA3 灯亮，但仍须按 PC2 才启动步态；未亮灯也能按按钮启动。STM32 原有上电站姿仍会执行。

命令返回终端、关闭终端或断开 SSH 后，服务仍继续运行。

## 3. 关闭／杀掉当前程序

```bash
sudo systemctl stop humanoid-button-vision.service humanoid-button-connector.service
```

这条命令关闭视觉、它启动的步态子进程和 connector。服务先发送退出信号；20 秒内仍未退出的进程会被 systemd 强制结束。仅停止当前运行，下次开机仍会自启动。

## 4. 重新启动程序

运行中想重新开始，或修改配置后让新参数生效：

```bash
sudo systemctl restart humanoid-button-vision.service
```

会关闭本次视觉和步态，再重新启动视觉、等待本次运行的新按键。按钮一直按住时，需要松开再按。

想把全部进程关闭后再启动，依次执行：

```bash
sudo systemctl stop humanoid-button-vision.service humanoid-button-connector.service
sudo systemctl start humanoid-button-vision.service
```

## 5. 打开配置并修改参数

```bash
cd ~/jetson_orin_code
vi config/button_start.env
```

切换到英文输入法，按小写 `i` 进入编辑模式，再修改需要的参数。例如文件中的这些值：

```bash
CAMERA=0                     # 摄像头，对应 --camera
VX=0.2                       # 前进速度，对应 --vx
MAX_WZ=0.5                   # 最大角速度，对应 --max-wz
WZ_STEP=0.3                  # 角速度档位，对应 --wz-step
HEADING_LOOKAHEAD_CM=50       # 前视距离，对应 --heading-lookahead-cm
HEADING_CORRIDOR_CM=5         # 走廊宽度，对应 --heading-corridor-cm
HEADING_RIGHT_TOLERANCE_DEG=9 # 右侧容差，对应 --heading-right-tolerance-deg
HEADING_LEFT_TOLERANCE_DEG=4  # 左侧容差，对应 --heading-left-tolerance-deg
HEADING_FULL_SCALE_DEG=15    # 满幅角度，对应 --heading-full-scale-deg
HEADING_LEFT_WZ="0.3 0.4 0.5" # 左转档位，多个值保留双引号
CARD_TRIGGER_DIST_CM=30      # 图卡触发距离，对应 --card-trigger-dist-cm
SHAPE_EVERY=2                # 图卡检测间隔，对应 --shape-every
POLICY_MAX_SECONDS=1200      # 步态最长运行秒数，对应 --start-policy-max-seconds
```

以上是当前默认值示例，按实际需要调整。行走模型修改 `WALKING_MODEL`，单脚站模型修改 `ONE_FOOT_MODEL`。

## 6. vi 保存并退出

依次操作：

1. 按 `Esc`，退出编辑模式。
2. 保持英文输入法，按 **Shift + 英文分号键 `;`**，输入冒号 `:`。
3. 确认左下角出现 `:`，再输入小写 `wq`。
4. 左下角应完整显示 `:wq`，按回车保存退出。

如果输入 `w` 后光标在移动，说明还没进入冒号命令行。重新按 `Esc`，先输入 `:`，看到冒号后再输入 `wq`。

放弃本次修改并退出：按 `Esc`，输入 `:q!`，再按回车。

## 7. 保存参数后重启

```bash
sudo systemctl restart humanoid-button-vision.service
```

只修改 `config/button_start.env` 不需要重新安装服务。重启后重新等待 PC2 按钮；下次开机也使用保存后的配置。

只查看配置生成的完整命令，不启动程序：

```bash
bash scripts/run_button_vision.sh --dry-run
```

当前 `--wz-mode heading` 固定在 `scripts/run_button_vision.sh` 中；配置文件没有提供的参数，需修改该脚本后重启服务。

## 8. 查看状态和日志

查看两个服务是否在运行：

```bash
systemctl status humanoid-button-connector.service humanoid-button-vision.service --no-pager
```

持续查看本次开机日志：

```bash
sudo journalctl -u humanoid-button-connector.service -u humanoid-button-vision.service -b -f
```

按 `Ctrl+C` 只退出日志查看，程序继续运行。关闭程序请使用第 3 节的 `stop` 命令。

## 9. 关闭或恢复开机自启动

关闭开机自启动，同时停止当前程序：

```bash
sudo systemctl disable --now humanoid-button-vision.service humanoid-button-connector.service
```

恢复开机自启动：

```bash
sudo systemctl enable humanoid-button-connector.service humanoid-button-vision.service
```

`enable` 只设置下次开机启动；想立即运行，再执行第 2 节的 `start` 命令。

## 10. 提示服务不存在或配置错误

遇到 `Unit ... not found` 或 `bad unit file setting`，先更新本分支并重新安装服务，再启动：

```bash
cd ~/jetson_orin_code
git pull --ff-only &&
bash scripts/install_button_autostart.sh &&
sudo systemctl start humanoid-button-vision.service
```

安装脚本会校验服务文件并执行 `daemon-reload`。若仍报错，查看第 8 节的日志。

以上 `systemctl` 命令管理服务启动的进程。手动运行的旧长命令需在原终端按 `Ctrl+C` 退出；手动运行前先停止两个服务，避免重复占用相机、串口和 UDP 端口。

更多参数和服务操作见 [开机自启说明](button_autostart.md)。
