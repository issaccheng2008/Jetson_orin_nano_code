# 机器人动作示教台

这是独立于行走模型的 50 Hz 示教程序。它直接使用 STM32 的 12 关节 COMMAND / STATE
协议，不需要摄像头、ONNX 或图形桌面。浏览器只负责显示和操作，记录与控制循环运行在 Jetson。

## 已实现的功能

- **松弛**：每 20 ms 把最近收到的角度作为下一帧目标发回，12 关节 Kp=Kd=0，
  COMMAND_ENABLE 保持设置，使电机持续反馈。手动移动时仍有电机机械摩擦。
- **连续记录**：开始后将实际收到的每个 STATE 保存到 `.jsonl` 日志；停止后 flush/fsync 关闭文件。
  不把同一反馈重复写成多帧。STM32 正常 50 Hz 回传时，记录即为 50 Hz。
- **Checkpoint**：每次按钮/命令立即采集最新角度，原子更新同一个 `.json` 文件。
  新建可开启另一组；历史文件可重新加载、改名、排序、删除姿态、编辑每段时间。
- **连续动作回放**：先从当前位置过渡到首帧，再按记录的 STM32 时间轴以 50 Hz 发送目标。
  倍率 1 为原速，0.5 为半速。小于等于 100 ms 的记录缺口按时间线性插值。
- **Checkpoint 回放**：每个姿态的 `duration_s` 表示到下一姿态的时间：
  `q(t)=qA+(qB-qA)*t/T`。每个关节在该段内角速度恒定，12 关节同时到达。
  末帧 `duration_s` 不参与回放；段交界处速度可不连续，这是线性插值本身的行为。
- **锁定**：抓取最近的接收角度作为固定目标，恢复 `config.py` 原有站立 PD（基准 JOINT_KP/KD，
  standing 倍率目前 1/1）。不会回到默认站姿。停止回放同样锁定当时实测位置。
- 回放正常结束后持续保持末帧。停止记录只停止写日志，保持当前控制模式。
- 页面显示实测角度和目标（可切换 ° / rad）、连接状态、记录帧数、缺帧估计；可下载文件。

控制范围是**两腿的 12 个电机**。现有协议不含头部和手臂舵机的角度及 PD，因此这些舵机不参与示教。

## 下位机修改与烧录

本地参考仓库：同级 `STM32_reference/`，来自用户提供的 `抬脚` 分支，基线：
`246f169c2bff3aed2fb1aa8d5d2f8a31e3f8407a`。

修改了 `Core/Inc/jetson_protocol.h`、`Core/Inc/jetson_robot_bridge.h`、
`Core/Src/jetson_robot_bridge.c` 和 `Core/Src/main.c`：

1. `COMMAND_TEACHING = 1<<3` 标记整个示教会话（松弛、锁定、回放、停止都携带）。
2. `STATE_TEACHING_READY = 1<<7` 宣告固件支持。新上位机遇到旧固件会拒绝使能。
   包长、CRC、协议版本保持原有 v2，不增加带宽。
3. 会话内返回真实模型坐标编码器角度，跳过“相对目标 ±0.5 rad”反馈限幅和识别用冻结反馈。
   保留既有方向变换和装配偏置逆变换；不把电机 raw 值直接误作协议角度。
4. 进入时取消本地腿部动作、前倾与冻结；示教期间拒绝识别动作/实体动作按钮抢占。
5. 示教期间 disable / ESTOP / 100 ms COMMAND 看门狗停止电机输出，**不软复位、不自动回开机站姿**。
   会话保持锁存，重新连接不自动恢复 PD；上位机需要明确 reset/release。
   下一个带 ENABLE 的普通行走 COMMAND 可以退出示教会话。

`firmware/teaching/teaching.patch` 是对应基线的补丁，在另一份 STM32 仓库中应用：

```bash
git apply --check /path/to/Jetson_orin_nano_code/firmware/teaching/teaching.patch
git apply /path/to/Jetson_orin_nano_code/firmware/teaching/teaching.patch
cmake --preset Debug
cmake --build --preset Debug
```

已编译的 ELF 和 BIN 在 `artifacts/teaching/`：`stm32-teaching.elf`、`stm32-teaching.bin`。
使用 STM32CubeProgrammer / 原工程烧录方式；BIN 起始地址为 `0x08000000`。
文件针对原工程 STM32H723ZG，不能用于其他 MCU。烧录/复位仍会执行原有开机站姿流程，
应先托住/悬吊机器人。**尚未代为烧录，也未执行实机电机运动。**

## Windows 电脑直接打开

Jetson 独立安装目录约定为 `/home/isaac/robot_teaching/`，使用已存在的
`/home/isaac/venvs/humanoid_policy/bin/python`（已确认 numpy 1.26.4 / pyserial 3.5 可用）。
程序副本与现有行走代码分离；重新调整原项目的 PD/限位后，需要同步这份目录中的 `config.py`。

1. 首次使用先烧录上述 STM32 固件。
2. 托住机器人，停止占用串口的行走/视觉自启动进程。服务若已安装，可在 Jetson 上执行：

   ```bash
   sudo systemctl stop humanoid-button-vision.service humanoid-button-connector.service
   fuser -v /dev/ttyACM0
   ```

   若是手动运行的进程，在其原终端 Ctrl+C。不要同时启动两个串口控制程序。
   新程序会检查 `/proc` 串口占用，并使用 Linux flock + TIOCEXCL 排他打开。
3. 双击仓库根目录 **`open_teaching.cmd`**，输入 SSH 密码；无需把密码写入脚本。
   它通过 `isaac@192.168.43.7` 启动服务，建立 localhost SSH 隧道，然后自动打开浏览器。
   首次 SSH 连接按实际主机指纹确认。保留启动窗口。
4. 页面初始为“仅监控”，即使服务带 `--enable-motors` 也不主动发送电机 COMMAND。
   确认“支持示教”和反馈在线后，托住机器人，点击“松弛”。
5. 连续动作：开始记录 → 手动摆动 → 停止记录 → 文件列表“回放”。
6. Checkpoint：每摆好一次点击“记录当前姿态” → 编辑各段秒数 → 保存 → 对应文件“回放”。
7. 随时点击“锁定当前位置”检查平衡。检查结束先托住再松弛/停止输出。

默认打开 `http://127.0.0.1:8765/`。若本机端口冲突：

```powershell
.\scripts\open_teaching.ps1 -LocalPort 8766
```

服务只监听 Jetson 的 `127.0.0.1:8765`，不开放局域网控制端口。控制请求带随机会话 token，
校验 Host/Origin。电脑浏览器通过 SSH 隧道访问；不需要 npm、额外 Web 框架或云端服务。

## 手动 SSH 启动与命令

若不使用双击入口，在电脑建立带端口转发的连接：

```bash
ssh -L 127.0.0.1:8765:127.0.0.1:8765 isaac@192.168.43.7
cd /home/isaac/robot_teaching
/home/isaac/venvs/humanoid_policy/bin/python -B teaching_console.py serve \
  --port /dev/ttyACM0 --enable-motors \
  --data-dir /home/isaac/jetson_orin_code/records/teaching
```

电脑浏览器访问 localhost 地址。另开一个 SSH 终端可以发送以下命令（与 UI 控制同一个服务）：

```bash
cd /home/isaac/robot_teaching
PY=/home/isaac/venvs/humanoid_policy/bin/python
$PY -B teaching_console.py command release
$PY -B teaching_console.py command 开始记录
$PY -B teaching_console.py command 停止记录
$PY -B teaching_console.py command 记录checkpoint --label "左脚抬起"
$PY -B teaching_console.py command 锁定
$PY -B teaching_console.py command play-log --name recording_实际文件名.jsonl --speed 1 --approach-s 2
$PY -B teaching_console.py command play-checkpoints --name checkpoints_实际文件名.json --approach-s 2
$PY -B teaching_console.py command stop-playback
$PY -B teaching_console.py command estop
$PY -B teaching_console.py command reset
$PY -B teaching_console.py command status
```

英文对应：`start-recording`、`stop-recording`、`checkpoint`、`hold`、`release`。
`reset` 只清除上位机故障锁存，保持停止输出；不会清除硬件本身的电机故障。

## 文件格式与回放约束

记录保存在指定的 `records/teaching/`，每次连续记录和每组 checkpoints 使用独立文件名。
JSONL 首行为元数据，之后每行为一个实际收到的 STATE：

```json
{"t_s":0.02,"host_monotonic_s":123.45,"sequence":100,"timestamp_us":2000000,"position_rad":[0,0,0,0,0,0,0,0,0,0,0,0],"status_flags":153}
```

`t_s` 从首个采样为 0，使用 STM32 微秒时间戳差累加，处理 uint32 回绕。
`host_monotonic_s` 用于本次运行诊断，不适合作为不同运行间的绝对时间。
系统按名义 20 ms 检查缺帧；缺帧计数是估计值，不会填造未收到的反馈。
正常每秒 flush，停止/故障/正常退出 fsync；异常断电可能丢失最后未落盘的数据。

文件显式保存关节名称、顺序、单位与坐标系，回放严格校验。线序为：
**左腿六项，再右腿六项**；每腿依次髋 pitch / roll / yaw、膝 pitch、踝 pitch / roll。
不要用历史 `config.JOINT_NAMES` 的左右标签重新排序；直接沿用 STM32 STATE/COMMAND 顺序。

回放前验证全轨迹的现有限位和速度（默认最大 3 rad/s），超限直接拒绝，避免无提示裁剪动作。
首帧过渡至少为界面设置的时间，必要时按最大速度自动延长。
运行时跟踪误差超过现有 10° 窗口、编码器无效、STATE 故障、反馈/时间戳不更新超过 100 ms、
控制循环延迟超过 80 ms 或串口错误均停止输出并锁存故障。记录的反馈缺口 >100 ms 会停止记录，
这样的连续文件不允许回放。检查原因后 reset，再主动松弛/锁定/回放。

50 Hz 控制采用单独线程和单调时钟，是 Linux 普通调度下的目标频率，不是硬实时保证。
记录线程批量取走全部收到的 STATE，页面 10 Hz 刷新不决定采样频率。多个浏览器共享同一控制器。
关闭浏览器/断开隧道本身不会切换服务器模式；锁定/回放可能继续。
启动窗口退出/SSH PTY 挂断会结束服务并尝试停止输出，STM32 的 100 ms COMMAND 看门狗提供失联保护。
停止输出、急停、断联均会失去电机支撑，需有机械支撑或人员托住。

## 打包、部署与离线验证

重新打包：

```bash
python scripts/package_teaching.py
scp artifacts/teaching/jetson-teaching.tar.gz isaac@192.168.43.7:/home/isaac/
```

Jetson 上首次安装（目录已存在时先备份旧文件；不要覆盖自己的 checkpoint/log）：

```bash
mkdir -p /home/isaac/robot_teaching
tar -xzf /home/isaac/jetson-teaching.tar.gz -C /home/isaac/robot_teaching
```

只依赖 numpy 和 pyserial；可使用原机器人的 Python 虚拟环境。
无硬件的电脑上运行：

```bash
cd humanoid_jetson_deploy
python -B teaching_console.py serve --simulate --data-dir ../records/teaching_demo
python -B -m unittest discover -s tests
```

`--simulate` 不打开串口，自动生成缓慢变化的示教角度；模拟日志与真实日志应分目录。
Linux 上 `test_teaching_serial_pty.py` 使用伪终端模拟 STM32 验证真实串口线程、排他打开、
50 Hz 日志、零增益、锁定和急停；不访问实际 STM32 设备。
