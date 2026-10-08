# Jetson 开机自启与 PC2 按钮起步

启动顺序是 `connector → vision → 等待 STM32 PC2 按钮 → gait`。两个 systemd 服务只启动 connector 和带 `--start-gate button --start-policy-on-gate` 的 vision；不直接启动 `main.py`。Vision 收到本次运行的新按钮事件后，才通过策略 Python 启动 gait 并等待 STM32 就绪。PC2 接线与固件按当前版本说明完成；安装服务本身不会启动电机。

## 在 Jetson 上安装

以平时运行机器人程序的普通用户登录，仓库放在 `~/jetson_orin_code`。不要给整个安装命令加 `sudo`。

```bash
cd ~/jetson_orin_code
bash scripts/install_button_autostart.sh --dry-run
bash scripts/install_button_autostart.sh
```

首次安装创建 `config/button_start.env`，默认视觉 Python 为仓库的 `.venv/bin/python`，策略 Python 为 `~/venvs/humanoid_policy/bin/python`。两个环境可以不同。安装先检查脚本、两份模型及视觉环境的 `numpy/cv2/serial`、策略环境的 `onnxruntime/serial`，再用 `systemd-analyze verify` 校验生成的服务文件，成功后才写系统服务并设为开机启用。配置生成与程序运行使用当前用户；只有系统文件安装和 `systemctl` 使用 sudo。缺依赖或路径错误时，编辑配置后重跑安装，已有配置保留。更新安装脚本后也需重跑安装；已有配置不会被覆盖。

```bash
vi config/button_start.env
bash scripts/install_button_autostart.sh --check
```

vi 的编辑、保存退出和参数生效步骤见 [参数修改与 vi 保存速查](Jetson参数修改与vi保存速查.md)。

配置使用 Bash 语法，带空格的路径要加引号。默认串口 `/dev/ttyACM0`、摄像头 0；用户须已有串口和摄像头权限（通常为 `dialout`、`video` 组；权限修改后重新登录或重启）。不使用 root 跑视觉或步态。不要把此用户可编辑配置交给不可信账户修改。

默认模型是 `humanoid_jetson_deploy/policy_49_max.onnx` 和 `policy-one-foot-standing_old.onnx`，可以分别修改 `WALKING_MODEL`、`ONE_FOOT_MODEL`。配置还保留当前视觉命令：heading 模式、vx 0.2、max-wz 0.5、wz-step 0.3、lookahead 50 cm、corridor 5 cm、右容差 9°、左容差 4°、full scale 15°、左转档位 0.3/0.4/0.5、卡片触发距离 30 cm、shape-every 2。策略单次最长运行默认 1200 秒，可修改 `POLICY_MAX_SECONDS`。UDP 端口固定 connector 输入 5006、策略输入 5005；connector 使用 `--max-vx-accel 1 --max-wz-accel 0`。

## 启动、停止与检查

首次安装只启用下次开机启动。台架准备好后，可以手动启动 vision，systemd 会先启动 connector：

```bash
sudo systemctl start humanoid-button-vision.service
systemctl status humanoid-button-connector.service humanoid-button-vision.service
journalctl -u humanoid-button-connector.service -u humanoid-button-vision.service -b -f
```

确认日志显示等待按钮，且按按钮前没有 C 步态进程或上位机运动 COMMAND；再按 PC2 验证 gait 就绪后起步。STM32 原有上电电机初始化及 `Action_Goto` 站姿仍会执行，按钮只控制上位机步态程序的启动。避免同时运行原来的手动 connector、vision 或策略命令，否则 UDP、摄像头或串口会冲突。

```bash
# 停止视觉及其 gait 子进程，再停止 connector
sudo systemctl stop humanoid-button-vision.service humanoid-button-connector.service
# 修改配置后重新启动；每次重新启动都要重新按按钮
sudo systemctl restart humanoid-button-vision.service
# 关闭开机启动并停止当前运行
sudo systemctl disable --now humanoid-button-vision.service humanoid-button-connector.service
# 恢复开机启动；当前运行仍需手动 start 或等待重启
sudo systemctl enable humanoid-button-connector.service humanoid-button-vision.service
```

Vision 异常退出后，systemd 在 5 秒后重启并重新等待本次运行的新按钮事件；不会自动恢复步态。正常结束（包括 gait 到达运行时限）不自动重启。服务停止使用 `KillMode=control-group`、先发 SIGINT，20 秒后仍未退出则由 systemd 强制清理，包括 gait 子进程。Connector 停止会连带停止依赖它的 vision；恢复后用 `sudo systemctl start humanoid-button-vision.service` 重新进入等待按钮状态。

视觉记录写入 `records/line_telemetry`，丢线和卡片图片分别写入带时间戳的 `records/loss_*`、`records/shape_*`；策略自身日志仍使用原有 records 规则。查看本次启动全部日志：

```bash
journalctl -u humanoid-button-connector.service -u humanoid-button-vision.service -b --no-pager
bash scripts/run_button_connector.sh --dry-run
bash scripts/run_button_vision.sh --dry-run
```

`--dry-run` 只显示配置、服务或命令，不创建配置、不安装服务、不运行 Python、不打开硬件；`--check` 会运行依赖导入检查，但不启动服务或策略。Windows 工作区仅验证脚本语法和 dry-run 输出，尚未在 Jetson 上部署，也未验证摄像头、串口、按钮或电机。
