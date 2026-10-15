# 双摄像头设备绑定与生命周期

## 当前端口规则（2026-10-15）

主巡线摄像头可接 USB `2.3` 或 `2.4`，使用 `/dev/robot_main_camera`。这两个端口仅允许出现一只摄像头；两个端口同时出现摄像头时，程序明确报错，不按枚举顺序选择。用户保证不会将辅摄像头接入这两个端口。辅读卡摄像头接其他端口，使用 `/dev/robot_card_camera`，启用双摄时必须恰好一只。

规则只匹配 `video4linux` 子系统、`index=0` 且具有 `:capture:` 能力的设备，为其添加符号链接。没有禁用、独占或限制 USB 端口；STM32 等串口设备可照常接入这两个端口，不参与摄像头角色分配。

换主摄像头端口后，等待设备重新枚举并重新启动视觉程序。此修改支持换口后重新启动识别，不实现运行中的自动重连。若端口损坏导致设备无法枚举，别名规则本身不能修复硬件连接。

下方 2026-10-12/13 的 `2.1` 绑定记录为历史方案，已被上述双端口规则替代。

本分支默认 CARD_CAMERA=1、主 CAMERA=0，-1 显式关闭副摄像头。旧的本机配置不会被模板覆盖；若实际启动命令仍有 --card-camera -1，仍是单摄像头。

新增 CAMERA_DEVICE / CARD_CAMERA_DEVICE（命令参数 --camera-device / --card-camera-device）优先于数字编号，支持 /dev/v4l/by-id 或 /dev/v4l/by-path。启动输出实际选择路径；缺失路径或主副解析为同一个设备会报错，不会交换角色或静默回退。数字编号只是兼容选择，不保证插拔后身份稳定。

2026-10-12 只读查询板子：两台均为 XHH-260602-KTY 2M，ID_SERIAL_SHORT 均为 01.00.00；by-id 名称冲突，无法用此序列号区分。应按物理 USB 路径绑定，角色须由安装位置确认。

2026-10-13 用户最新指定：USB 2.1 为主巡线相机，另一台为副读卡相机。当前另一台在 2.2；不再固定为 2.3。
程序配置仍使用固定设备名称，不用 video0/video2 来决定角色：

```bash
CAMERA_DEVICE=/dev/robot_main_camera
CARD_CAMERA_DEVICE=/dev/robot_card_camera
CARD_CAMERA=1
```

同一主机同时只允许一台其他采集相机；不存在或多于一台时明确报错。程序核对主名称指向 2.3/2.4 中唯一的摄像头、副名称指向两个端口之外的另一采集设备，防止旧别名或插拔后的节点编号变化交换角色。

主摄像头一直用于巡线、接近停车判断与二维码。起点门控等待第一张图卡时打开副摄像头做图卡分类，图卡确认后释放；进入后续停车精准识别准备窗口时再次打开，在 DONE+稳定等待之前的帧不投票。投票结束/动作开始或窗口结束时释放副摄像头，停车阶段打开失败明确报错，不偷偷用主摄像头代投。冷打开相机有初始化时间，尚未双摄像头硬件计时验证。

主副摄像头应分别配置安装几何。副摄像头默认 raw，不共享主相机 LUT。没有修改模型或电机逻辑。

现场 video0 查询时被视觉进程11208占用，该进程后来能运行说明节点可用；不能仅凭事后进程状态确认17:49那次失败的唯一原因，不要同时启动第二个视觉进程抢占它。

验证：摄像头默认值、稳定路径/重复节点、投票帧时间与真实入口路由，加原单摄像头入口回归共22项通过。旧单摄像头回归用例显式添加 --card-camera -1。仅查询了设备与进程，没有开启摄像头、重启服务或使能电机。

## 固定名称（推荐）

规则文件 config/udev/99-robot-cameras.rules 将接口 2.3/2.4 上的摄像头命名为 /dev/robot_main_camera，将两个端口之外的唯一其他采集相机命名为 /dev/robot_card_camera。这是 udev 符号链接，不修改硬件序列号。模板使用这些名字，需要先安装规则；缺少别名会明确报错。

在 Nano 仓库中执行一次：

```bash
sudo install -m 0644 config/udev/99-robot-cameras.rules /etc/udev/rules.d/99-robot-cameras.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=video4linux --action=add
sudo udevadm settle
ls -l /dev/robot_main_camera /dev/robot_card_camera
```

然后修改现有按钮配置（不是仅改模板）：

```bash
CAMERA_DEVICE=/dev/robot_main_camera
CARD_CAMERA_DEVICE=/dev/robot_card_camera
CARD_CAMERA=1
```

手动入口对应 --camera-device /dev/robot_main_camera --card-camera-device /dev/robot_card_camera。程序不会根据名字交换主副用途。拔掉设备后别名随设备消失，重新插入后由规则创建。主相机使用 2.3 或 2.4；副相机使用这两个端口之外的接口，但必须唯一。更新时需同时更新 Python 角色检查与 udev 规则。规则安装后应通过名称与 by-path 核对，再运行脚本 --dry-run。

2026-10-12 后续更新：起点首卡也由副摄像头识别。起点没有低头 DONE 事件，因此使用副摄像头打开后取得的新帧；后续停车仍遵守 DONE+稳定时间。起点二维码、行走模糊停车判断继续使用主摄像头。新增起点副摄像头路由及释放的实际入口测试，相关 23 项通过。

## 2026-10-13 三分支修正

lane_loss_recovery_20261012、boundary_guard_20261012、policy49flitter-vi-gap-straight_20261012 统一采用上述规则。
按钮脚本给没有相机路径字段的旧配置补上两个固定名称；CARD_CAMERA=-1 时清除副路径，不会意外打开副相机。
这次没有改写硬件名称/序列号。此前连接实例命名方案未提交或部署，保留为本地草稿。
