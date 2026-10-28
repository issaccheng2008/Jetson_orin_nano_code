# 双摄像头：显式名称与身份验证（2026-10-13）

当前两台相机的设备名均为 `2M: 2M`，序列号均为 `01.00.00`，厂商/型号/版本相同。不能按相同名称或 by-id 自动区分主副。
旧版根据 USB 2.1/2.3 的 ID_PATH 命名，没有满足“不绑定端口”的要求；旧端口规则已停用。

## 当前方案

程序显式使用 `robot_main_camera`（主巡线）与 `robot_card_camera`（副读卡）。
先按本次连接的唯一设备实例号登记角色，不使用 USB 接口位置作为角色条件，也不把 video0/video2 当作稳定身份。
登记保存到 `/etc/robot/camera_roles.json`，生成 `/dev/robot_main_camera`、`/dev/robot_card_camera` 两个设备名称。
实例号是 USB bus/device number，例如 `usb:001:005`，不是 `2.1/2.2` 端口位置，也不是硬件序列号。

**限制：两台硬件没有不同的持久标识。因此当前命名仅对已登记的连接有效；重启、重插、节点交换或连接重新枚举后拒绝旧登记，需要重新确认。**
校验 boot ID、设备实例号、节点、型号/序列号及设备初始化标记。程序不会根据同名设备猜角色或静默改用数字编号。
若以后更换为唯一序列号相机，才可改成跨重启/插拔的硬件身份绑定；本次没有写相机固件。

## 登记与配置

保持视觉和步态停止；列出设备仅读取元数据，不打开视频流：

```bash
python3 scripts/name_robot_cameras.py --list
```

确认哪一个连接实例对应主/副，再登记。下面的实例号仅是 2026-10-13 本次查询结果，不能在重插后照抄：

```bash
sudo python3 scripts/name_robot_cameras.py --main-id usb:001:005 --card-id usb:001:010
```

实际按钮配置 `config/button_start.env`：

```bash
CAMERA_DEVICE=/dev/robot_main_camera
CARD_CAMERA_DEVICE=/dev/robot_card_camera
CARD_CAMERA=1
```

手动入口显式参数：
`--camera-device /dev/robot_main_camera --card-camera-device /dev/robot_card_camera`。
按钮脚本对没有这些字段的旧配置也默认用两个登记名称。
`CARD_CAMERA=-1` 关闭副相机，并清除副相机路径参数。
有名字的选择优先于兼容数字编号，底层打开设备所用的节点只由已校验名字解析。

旧规则已安装的板子需先备份停用 `/etc/udev/rules.d/99-robot-cameras.rules`，刷新 udev 后再登记名称；不要继续安装旧 ID_PATH 规则。
本次修改三个分支：lane_loss_recovery_20261012、boundary_guard_20261012、policy49flitter-vi-gap-straight_20261012。

## 使用阶段

主相机持续用于巡线、运动中模糊停车判断、二维码和原录像。
起点等待图卡和正常停车精准识别由主副相机独立分类并融合，每个周期最多一票；冲突弃票。
副相机仅在上述静止读卡阶段打开，阶段结束释放；正常停车仍要求 DONE+稳定等待之后的新帧。
副相机安装几何独立可配，默认 raw，不沿用主相机 LUT。

登记与配置验证不启动视觉、电机或策略；实际运行双相机效果仍需实车验证。
