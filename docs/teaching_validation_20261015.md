# 示教功能验证（2026-10-15）

## 验证结果

- Windows 控制目录测试：`python -B -m unittest discover -s tests`，198 项，197 通过，
  1 项 Linux 伪串口专用测试跳过。
- Jetson 独立目录：`python -B -m unittest discover -s tests -p 'test_teaching*.py'`，25 项全部通过。
  Python 环境 `/home/isaac/venvs/humanoid_policy/bin/python`，numpy 1.26.4、pyserial 3.5。
- Linux PTY 集成测试真实运行 SerialLink：模拟 STM32 以 50 Hz 发送 STATE，验证记录时间步长、
  排他串口访问、零增益、恢复 PD、checkpoint 文件和急停 COMMAND。未使用 `/dev/ttyACM0`。
- 浏览器离线检查：松弛、开始/停止连续记录、两个 checkpoint 自动保存、区间时间改为 3.5 秒并保存、
  checkpoint 回放完成后保持末帧、连续日志回放、停止回放锁定、停止输出、故障 reset。
- JavaScript 语法检查、PowerShell 启动脚本语法检查通过。
- STM32 `cmake --preset Debug` / `cmake --build --preset Debug -j 4` 成功。
  GNU ARM GCC 14.3.1；FLASH 92492 B。构建仍有原工程的未使用函数、LCD 格式串、USB 初始化器、
  链接 RWX 段等警告，没有编译/链接错误。
- STM32 补丁 reverse-check 和两个仓库的 `git diff --check` 通过。

## 交付位置

- Jetson 源码已安装到 `/home/isaac/robot_teaching/`，未覆盖现有行走项目。
- 未启动实机控制服务、未烧录 STM32、未发送实体电机动作。
- Windows 双击入口：`open_teaching.cmd`。首次实机使用前需烧录新固件并释放串口。
- 编译固件和 Jetson 部署包：`artifacts/teaching/`。
- 页面预览截图：`artifacts/teaching/teaching-console-preview.jpg`（离线模拟）。

## 已核对的部署 SHA-256

| 文件 | SHA-256 |
| --- | --- |
| teaching.py | `3d7c965d0b89becd168d98238920ebb43337d9fe2fdc862bd187be61d8a7c3b9` |
| teaching_console.py | `5ea77c972a10e5637c6018f9ea5005a43f64d0bb27072851f3d5eb84e95d92e7` |
| jetson-teaching.tar.gz | `30f93b3e706662460692154b902b3b2b9c1b59e62d90ce457cdb451ded8dfb5d` |
| stm32-teaching.bin | `9ca113e6c5f03f0a913ecc40920b50f93b29965998bb7e8ebde54d680281c8ad` |

前三项已与 Jetson 上的实际文件核对一致。实机负载下的调度抖动、电机实际跟踪效果、
零力矩时手动活动范围和各姿态平衡仍需使用新固件在机器人上验证。
