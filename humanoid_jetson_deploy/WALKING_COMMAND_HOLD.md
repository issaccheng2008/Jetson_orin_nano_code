# Walking 模型输入的最小保持合同

正常 walking49 的 `(vx, vy, wz)` 在模型入口至少保持 **0.5秒**。数值硬编码在 `walking_command_hold.py`，没有 CLI 参数。

适用视觉 UDP、固定速度、scripted/curve 速度命令；不作用于固定关节帧、独立单腿、图卡单腿模型或 phase-clock。

- 从 main 首次真正将该值交给 walking 策略的主机单调时间开始计时，不使用视觉首发时间、UDP收包时间或上个循环的 dt。
- 0.5秒内保留当前成对速度/转向，仅更新最新请求。到期后的下一次模型调用才使用最新请求，不排队补发此前请求。
- 重复相同数值不重置期限。恒定直行的 `wz=0` 同样参与正常命令保持。
- **全零停止立即抢占并清锁**。现有UDP超时产生全零，因此失联不会等待0.5秒。图卡 busy 的既有零命令、upright/单腿接管、故障和退出失能同样清锁，接管结束后的启动是新合同。
- 干跑仍执行同一模型输入合同，便于记录检查；不把模型被调用0.5秒当作电机真实执行了0.5秒。
- 本机制不平滑速度，也不修改观测的49维结构、步长语义、关节裁剪或动作映射。上游 A 对 held 模式的 slew 绕过需要配套生效，避免锁住爬坡初值。

独立详细日志新增5列：`requested_cmd_vx/vy/wz` 是源请求（本地安全覆盖之前）；原有 `cmd_vx/vy/wz` 是实际交给 walking 的值；`command_hold_remaining_s` 是剩余最小保持时间；`command_hold_reason` 说明 normal_walking、stop、upright_takeover、onefoot_takeover 或 not_walking。schema 升为 `control_target_trace_v2`，301列，仅新run日志受影响，旧观测 CSV 和模型结构不变。manifest 记录固定0.5秒合同。

验证命令：

```bash
PYTHONPATH=humanoid_jetson_deploy python3 -m unittest discover \
  -s humanoid_jetson_deploy/tests -p test_walking_command_hold.py -v
```

测试使用可控主机时钟、替身串口及真实观测构造方法，核查 main 实际传给模型的 obs[9:11] 在0/.1/.49秒保持，到.5秒采用最新请求；另验全零/真实UDP watchdog抢占、upright/单腿交接清锁、重复值不续期、错误数据和数组所有权。没有实车执行验证。
