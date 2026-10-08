# Jetson 参数修改与 vi 保存速查

适用于 `policy49-button-test` 的按钮自启动模式。修改的是实际配置 `config/button_start.env`，保存后重启视觉服务生效，无需重新安装服务。

## 1. 打开配置并修改

```bash
cd ~/jetson_orin_code
vi config/button_start.env
```

切换到英文输入法，按小写 `i` 进入编辑模式，再修改需要的参数。例如文件中的这些值：

```bash
VX=0.2                       # 前进速度，对应 --vx
MAX_WZ=0.5                   # 最大角速度，对应 --max-wz
WZ_STEP=0.3                  # 角速度档位，对应 --wz-step
HEADING_LEFT_WZ="0.3 0.4 0.5" # 左转档位，多个值保留双引号
CARD_TRIGGER_DIST_CM=30      # 图卡触发距离，对应 --card-trigger-dist-cm
```

以上是当前默认值示例，按实际需要调整。行走模型修改 `WALKING_MODEL`，单脚站模型修改 `ONE_FOOT_MODEL`。

## 2. 保存并退出

依次操作：

1. 按 `Esc`，退出编辑模式。
2. 保持英文输入法，按 **Shift + 英文分号键 `;`**，输入冒号 `:`。
3. 确认左下角出现 `:`，再输入小写 `wq`。
4. 左下角应完整显示 `:wq`，按回车保存退出。

如果输入 `w` 后光标在移动，说明还没进入冒号命令行。重新按 `Esc`，先输入 `:`，看到冒号后再输入 `wq`。

放弃本次修改并退出：按 `Esc`，输入 `:q!`，再按回车。

## 3. 让新参数生效

```bash
sudo systemctl restart humanoid-button-vision.service
```

重启后重新等待 PC2 按钮，按按钮才启动步态；下次开机也使用保存后的配置。

只查看配置生成的完整命令，不启动程序：

```bash
bash scripts/run_button_vision.sh --dry-run
```

更多参数和服务操作见 [开机自启说明](button_autostart.md)。
