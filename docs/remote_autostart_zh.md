# 网页遥控开机启动

本分支使用 `humanoid-remote.service`。开机只启动网页服务；在网页中通过令牌认证并点击“启用”后，服务才启动行走策略进程。等待策略完成准备时，遥控输出保持为零；准备成功后才能发出运动指令。无需实体启动按钮。此流程不启动相机、视觉巡线程序或 `connector.py`。

服务异常退出后的自动重启也只恢复网页服务。需要再次通过网页点击“启用”，策略不会因重启自动运行。停止服务会向整个进程组发送 SIGINT，等待最多 20 秒后由 systemd 清理残留进程。

网页停止、失焦或指令失联会撤销本次控制，并结束本次行走上位机；下次启用会重新启动。
启动过程中也可停止。准备成功后必须重新推动摇杆或重新按方向键，等待时按住的输入不会自动变成运动。

## 安装前检查

在 Jetson 的普通账号中执行，不要使用 `sudo bash`：

```bash
cd /你的路径/Jetson_orin_nano_code
bash scripts/install_remote_autostart.sh --dry-run
```

`--dry-run` 只显示计划、服务内容和配置模板，不创建目录、配置或令牌，不调用 systemctl，也不启动程序。服务脚本自己切换到配置中的仓库目录，支持包含空格的路径。

默认使用：

| 设置 | 默认值 |
| --- | --- |
| Python | `$HOME/venvs/humanoid_policy/bin/python` |
| 行走模型 | `humanoid_jetson_deploy/policy(13).onnx` |
| STM32 串口 | `/dev/ttyACM0` |
| 网页监听 | `0.0.0.0:8080` |
| 策略命令端口 | `5007` |
| 策略准备超时 | 20 秒 |
| 前进速度上限 | 0.4 |
| 转向速度上限 | 0.5 |
| kp / kd 倍率 | 1 / 1 |

Python 环境需具有 `numpy`、`onnxruntime` 和 `pyserial`，普通账号应已具有串口访问权限。安装检查只读取文件、导入模块并验证数值参数，不打开串口、不执行模型推理，也不能证明真实电机、串口连线或机器人状态正常。

## 安装和配置

```bash
bash scripts/install_remote_autostart.sh
```

首次安装生成独立的 `config/remote_start.env`，包含一次随机生成的认证令牌。文件权限设为 0600，已加入 Git 忽略规则。不要把令牌放进命令行、地址栏、截图或提交记录。配置是可信的 Bash 文件，只应由当前账号编辑。

安装器先执行配置、文件、Python 依赖检查，再调用 `systemd-analyze verify` 验证新服务。全部通过后，只停用并停止以下两个旧服务，再持久屏蔽它们：

- `humanoid-button-vision.service`
- `humanoid-button-connector.service`

若旧服务在 `/etc/systemd/system/` 中为普通文件，安装器先将其移动到 `/etc/systemd/system/humanoid-remote-legacy-backup/<唯一时间戳目录>/`，再执行 mask，避免普通文件占用路径导致 mask 失败。重复安装保留此前所有备份，也保留已有遥控配置及令牌，仅收紧配置权限。本分支的旧 `install_button_autostart.sh` 已拒绝实际安装；其 `--dry-run` / `--check` 仅供历史检查。

如果首次安装因 Python 路径、模型或串口配置不合适而退出，生成的私人配置会保留。编辑后先检查再重新安装：

```bash
nano config/remote_start.env
bash scripts/install_remote_autostart.sh --check
bash scripts/run_remote_control.sh --dry-run
bash scripts/install_remote_autostart.sh
```

`--check` 不修改配置、不操作服务。包装脚本的 `--dry-run` 显示准确的 Python 参数数组，令牌仅通过环境传递，不打印到输出。可配置 `POLICY_STEP_CM` 调整步长；空值沿用策略默认步长。配置改变后需要重启网页服务才能生效。

安装只设置下次开机启动，不会立即启动新服务。需要现在启动时明确执行：

```bash
sudo systemctl start humanoid-remote.service
systemctl status humanoid-remote.service
journalctl -u humanoid-remote.service -n 100 --no-pager
```

在同一可信局域网的手机浏览器打开 `http://Jetson的IP:8080`，在页面中输入私人配置里的令牌并点击“启用”，等待状态变为可用后操作。每次启动网页服务后都需要重新启用行走。

## 停止及排查

```bash
sudo systemctl stop humanoid-remote.service
# 取消下次开机启动，同时停止当前服务：
sudo systemctl disable --now humanoid-remote.service
```

若启用失败或准备超时，先查看日志、模型路径、Python 依赖、串口权限及 STM32 状态；排查后在页面中重新启用。若配置权限错误，执行 `chmod 600 config/remote_start.env`。不要通过恢复旧视觉服务绕过故障检查。

检查旧服务屏蔽情况：

```bash
systemctl is-enabled humanoid-button-vision.service humanoid-button-connector.service
```

预期两项均显示 `masked`。备份是保留安装历史的文件，安装器不会自动恢复或重新启用旧服务。
