# records/ 索引（自动生成，py -3 records/build_index.py）

⚠️ 命令是从日志横幅**反推**的：命令行本身没落盘。反推不到的只有 `--max-seconds`、
`--camera` 这类不在横幅里的开关；`--no-red-detect` 看 `red=off`。

| run | 起 | 时长 | fire | stop | turn | gap | step | vx | C 会话 | C 晚起 | 重叠 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0512 | 05:12:00 | 32s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 0513 | 05:13:00 | 30s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 0516 | 05:16:00 | 22s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 0517 | 05:17:00 | 54s | ? | ? | ? | ? | ? | 0.3 | 05:17:02 | +2s | 15s |
| 0520 | 05:20:00 | 22s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 0533 | 05:33:00 | 2s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 0535 | 05:35:00 | 3s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 0540 | 05:40:00 | 1s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 0553 | 05:53:00 | 18s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 0554 | 05:54:00 | 16s | ? | ? | ? | ? | ? | 0.3 | 05:54:05 | +5s | 8s |
| 0601 | 06:01:00 | 43s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 0604 | 06:04:00 | 26s | ? | ? | ? | ? | ? | 0.3 | 06:04:26 | +26s | 0s |
| 0605 | 06:05:00 | 25s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 0606 | 06:06:00 | 22s | ? | ? | ? | ? | ? | 0.3 | 06:06:04 | +4s | 18s |
| 0709 | 07:09:00 | 20s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 0710 | 07:10:00 | 36s | ? | ? | ? | ? | ? | 0.3 | 07:09:53 | -7s | 4s |
| 091624 | 09:16:24 | 46s | 5.0 | ? | ? | ? | ? | 0.2 | 无 C | - | 0s |
| 092751 | 09:27:51 | 26s | 5.0 | ? | ? | ? | ? | 0.2 | 09:28:12 | +21s | 3s |
| 1407 | 14:07:00 | 22s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 1409 | 14:09:00 | 17s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 1416 | 14:16:00 | 18s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 1422 | 14:22:00 | 26s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 1425 | 14:25:00 | 20s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 1428 | 14:28:00 | 21s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 1447 | 14:47:00 | 36s | ? | ? | ? | ? | ? | 0.3 | 无 C | - | 0s |
| 165235 | 16:52:35 | 2s | 5.0 | ? | ? | ? | ? | 0.2 | 无 C | - | 0s |
| 165248 | 16:52:48 | 20s | 5.0 | ? | ? | ? | ? | 0.2 | 无 C | - | 0s |
| 165343 | 16:53:43 | 11s | 5.0 | ? | ? | ? | ? | 0.2 | 16:53:50 | +7s | 4s |
| 210308 | 21:03:08 | 24s | 7.0 | ? | ? | 2.00 | 0.5 | 0.2 | 21:03:25 | +17s | 7s |
| 210400 | 21:04:00 | 17s | 7.0 | ? | ? | 2.00 | 0.5 | 0.2 | 21:04:07 | +7s | 10s |
| 210445 | 21:04:45 | 26s | 5.0 | ? | ? | 1.00 | 0.5 | 0.2 | 21:04:51 | +6s | 19s |
| 210554 | 21:05:54 | 0s | ? | ? | ? | ? | ? | ? | 无 C | - | 0s |
| 212837 | 21:28:37 | 0s | ? | ? | ? | ? | ? | ? | 无 C | - | 0s |
| 212852 | 21:28:52 | 18s | 4.5 | 1.5 | 2.50 | 1.00 | 0.5 | 0.2 | 21:29:02 | +10s | 8s |
| 213155 | 21:31:55 | 23s | 4.5 | 1.0 | 2.50 | 1.00 | 0.5 | 0.2 | 21:32:07 | +12s | 9s |
| 213254 | 21:32:54 | 13s | 3.5 | 0.1 | 2.50 | 0.50 | 0.5 | 0.2 | 21:33:00 | +6s | 7s |
| 213316 | 21:33:16 | 7s | 3.5 | 0.1 | 2.50 | 0.50 | 0.5 | 0.2 | 无 C | - | 0s |
| 213330 | 21:33:30 | 24s | 3.5 | 0.1 | 2.50 | 0.50 | 0.5 | 0.2 | 无 C | - | 0s |
| 214710 | 21:47:10 | 0s | ? | ? | ? | ? | ? | ? | 无 C | - | 0s |
| 214722 | 21:47:22 | 0s | ? | ? | ? | ? | ? | ? | 无 C | - | 0s |
| 214747 | 21:47:47 | 15s | 4.5 | 4.5 | 2.50 | 1.00 | 0.5 | 0.2 | 21:47:55 | +8s | 7s |
| 215419 | 21:54:19 | 2s | 4.5 | 30.0 | 2.50 | 1.00 | 0.5 | 0.2 | 无 C | - | 0s |
| 215434 | 21:54:34 | 29s | 4.5 | 30.0 | 2.50 | 1.00 | 0.5 | 0.2 | 21:54:40 | +6s | 23s |
| 215533 | 21:55:33 | 23s | 4.5 | 9.0 | 2.50 | 1.00 | 0.5 | 0.2 | 21:55:39 | +6s | 17s |
| 215707 | 21:57:07 | 22s | 4.5 | 7.0 | 2.50 | 0.80 | 0.5 | 0.2 | 21:57:12 | +5s | 17s |
| 215750 | 21:57:50 | 23s | 4.5 | 7.0 | 2.50 | 0.80 | 0.4 | 0.2 | 21:57:57 | +7s | 16s |
| 215926 | 21:59:26 | 16s | 5.0 | 7.0 | 2.50 | 1.00 | 0.5 | 0.2 | 21:59:31 | +5s | 10s |
| 220045 | 22:00:45 | 16s | 5.0 | 3.0 | 2.50 | 1.00 | 0.5 | 0.2 | 22:00:50 | +5s | 12s |
| 220234 | 22:02:34 | 26s | 7.0 | 3.0 | 2.50 | 1.00 | 0.5 | 0.2 | 22:02:45 | +11s | 15s |
| 220328 | 22:03:28 | 24s | 7.0 | 3.0 | 2.50 | 1.00 | 0.5 | 0.2 | 22:03:44 | +16s | 7s |
| 220445 | 22:04:45 | 16s | 8.0 | 3.0 | 2.50 | 1.00 | 0.5 | 0.2 | 22:04:52 | +7s | 9s |
| 220533 | 22:05:33 | 22s | 10.0 | 3.0 | 2.50 | 1.00 | 0.5 | 0.2 | 22:05:44 | +11s | 12s |

## 参数组 → 重建的 B 命令

### fire=10.0 stop=3.0 turn=2.50 gap=1.00 step=0.5（1 趟：220533）
```bash
python -u new_vision/jetson/run_policy_vision.py --camera 0 --headless \
    --no-red-detect --bias-straight-cm 0 --card-trigger-dist-cm 30 \
    --wz-fire-cm 10.0 --wz-stop-cm 3.0 --wz-turn-s 2.50 \
    --wz-gap-s 1.00 --wz-step 0.5 \
    --dump-on-loss records/loss_dump --shape-dump records/shape_dump \
    --max-seconds 90 2>&1 | tee records/run_$(date +%H%M%S).log
```

### fire=3.5 stop=0.1 turn=2.50 gap=0.50 step=0.5（3 趟：213254, 213316, 213330）
```bash
python -u new_vision/jetson/run_policy_vision.py --camera 0 --headless \
    --no-red-detect --bias-straight-cm 0 --card-trigger-dist-cm 30 \
    --wz-fire-cm 3.5 --wz-stop-cm 0.1 --wz-turn-s 2.50 \
    --wz-gap-s 0.50 --wz-step 0.5 \
    --dump-on-loss records/loss_dump --shape-dump records/shape_dump \
    --max-seconds 90 2>&1 | tee records/run_$(date +%H%M%S).log
```

### fire=4.5 stop=1.0 turn=2.50 gap=1.00 step=0.5（1 趟：213155）
```bash
python -u new_vision/jetson/run_policy_vision.py --camera 0 --headless \
    --no-red-detect --bias-straight-cm 0 --card-trigger-dist-cm 30 \
    --wz-fire-cm 4.5 --wz-stop-cm 1.0 --wz-turn-s 2.50 \
    --wz-gap-s 1.00 --wz-step 0.5 \
    --dump-on-loss records/loss_dump --shape-dump records/shape_dump \
    --max-seconds 90 2>&1 | tee records/run_$(date +%H%M%S).log
```

### fire=4.5 stop=1.5 turn=2.50 gap=1.00 step=0.5（1 趟：212852）
```bash
python -u new_vision/jetson/run_policy_vision.py --camera 0 --headless \
    --no-red-detect --bias-straight-cm 0 --card-trigger-dist-cm 30 \
    --wz-fire-cm 4.5 --wz-stop-cm 1.5 --wz-turn-s 2.50 \
    --wz-gap-s 1.00 --wz-step 0.5 \
    --dump-on-loss records/loss_dump --shape-dump records/shape_dump \
    --max-seconds 90 2>&1 | tee records/run_$(date +%H%M%S).log
```

### fire=4.5 stop=30.0 turn=2.50 gap=1.00 step=0.5（2 趟：215419, 215434）
```bash
python -u new_vision/jetson/run_policy_vision.py --camera 0 --headless \
    --no-red-detect --bias-straight-cm 0 --card-trigger-dist-cm 30 \
    --wz-fire-cm 4.5 --wz-stop-cm 30.0 --wz-turn-s 2.50 \
    --wz-gap-s 1.00 --wz-step 0.5 \
    --dump-on-loss records/loss_dump --shape-dump records/shape_dump \
    --max-seconds 90 2>&1 | tee records/run_$(date +%H%M%S).log
```

### fire=4.5 stop=4.5 turn=2.50 gap=1.00 step=0.5（1 趟：214747）
```bash
python -u new_vision/jetson/run_policy_vision.py --camera 0 --headless \
    --no-red-detect --bias-straight-cm 0 --card-trigger-dist-cm 30 \
    --wz-fire-cm 4.5 --wz-stop-cm 4.5 --wz-turn-s 2.50 \
    --wz-gap-s 1.00 --wz-step 0.5 \
    --dump-on-loss records/loss_dump --shape-dump records/shape_dump \
    --max-seconds 90 2>&1 | tee records/run_$(date +%H%M%S).log
```

### fire=4.5 stop=7.0 turn=2.50 gap=0.80 step=0.4（1 趟：215750）
```bash
python -u new_vision/jetson/run_policy_vision.py --camera 0 --headless \
    --no-red-detect --bias-straight-cm 0 --card-trigger-dist-cm 30 \
    --wz-fire-cm 4.5 --wz-stop-cm 7.0 --wz-turn-s 2.50 \
    --wz-gap-s 0.80 --wz-step 0.4 \
    --dump-on-loss records/loss_dump --shape-dump records/shape_dump \
    --max-seconds 90 2>&1 | tee records/run_$(date +%H%M%S).log
```

### fire=4.5 stop=7.0 turn=2.50 gap=0.80 step=0.5（1 趟：215707）
```bash
python -u new_vision/jetson/run_policy_vision.py --camera 0 --headless \
    --no-red-detect --bias-straight-cm 0 --card-trigger-dist-cm 30 \
    --wz-fire-cm 4.5 --wz-stop-cm 7.0 --wz-turn-s 2.50 \
    --wz-gap-s 0.80 --wz-step 0.5 \
    --dump-on-loss records/loss_dump --shape-dump records/shape_dump \
    --max-seconds 90 2>&1 | tee records/run_$(date +%H%M%S).log
```

### fire=4.5 stop=9.0 turn=2.50 gap=1.00 step=0.5（1 趟：215533）
```bash
python -u new_vision/jetson/run_policy_vision.py --camera 0 --headless \
    --no-red-detect --bias-straight-cm 0 --card-trigger-dist-cm 30 \
    --wz-fire-cm 4.5 --wz-stop-cm 9.0 --wz-turn-s 2.50 \
    --wz-gap-s 1.00 --wz-step 0.5 \
    --dump-on-loss records/loss_dump --shape-dump records/shape_dump \
    --max-seconds 90 2>&1 | tee records/run_$(date +%H%M%S).log
```

### fire=5.0 stop=3.0 turn=2.50 gap=1.00 step=0.5（1 趟：220045）
```bash
python -u new_vision/jetson/run_policy_vision.py --camera 0 --headless \
    --no-red-detect --bias-straight-cm 0 --card-trigger-dist-cm 30 \
    --wz-fire-cm 5.0 --wz-stop-cm 3.0 --wz-turn-s 2.50 \
    --wz-gap-s 1.00 --wz-step 0.5 \
    --dump-on-loss records/loss_dump --shape-dump records/shape_dump \
    --max-seconds 90 2>&1 | tee records/run_$(date +%H%M%S).log
```

### fire=5.0 stop=7.0 turn=2.50 gap=1.00 step=0.5（1 趟：215926）
```bash
python -u new_vision/jetson/run_policy_vision.py --camera 0 --headless \
    --no-red-detect --bias-straight-cm 0 --card-trigger-dist-cm 30 \
    --wz-fire-cm 5.0 --wz-stop-cm 7.0 --wz-turn-s 2.50 \
    --wz-gap-s 1.00 --wz-step 0.5 \
    --dump-on-loss records/loss_dump --shape-dump records/shape_dump \
    --max-seconds 90 2>&1 | tee records/run_$(date +%H%M%S).log
```

### fire=5.0 stop=? turn=? gap=1.00 step=0.5（1 趟：210445）

（这几趟的横幅里没有上面打 `?` 的项：是加 `--wz-stop-cm` / 改成「另一侧」语义之前的代码版本，命令只能从对应日期的代码里查。）

### fire=5.0 stop=? turn=? gap=? step=?（5 趟：091624, 092751, 165235, 165248, 165343）

（这几趟的横幅里没有上面打 `?` 的项：是加 `--wz-stop-cm` / 改成「另一侧」语义之前的代码版本，命令只能从对应日期的代码里查。）

### fire=7.0 stop=3.0 turn=2.50 gap=1.00 step=0.5（2 趟：220234, 220328）
```bash
python -u new_vision/jetson/run_policy_vision.py --camera 0 --headless \
    --no-red-detect --bias-straight-cm 0 --card-trigger-dist-cm 30 \
    --wz-fire-cm 7.0 --wz-stop-cm 3.0 --wz-turn-s 2.50 \
    --wz-gap-s 1.00 --wz-step 0.5 \
    --dump-on-loss records/loss_dump --shape-dump records/shape_dump \
    --max-seconds 90 2>&1 | tee records/run_$(date +%H%M%S).log
```

### fire=7.0 stop=? turn=? gap=2.00 step=0.5（2 趟：210308, 210400）

（这几趟的横幅里没有上面打 `?` 的项：是加 `--wz-stop-cm` / 改成「另一侧」语义之前的代码版本，命令只能从对应日期的代码里查。）

### fire=8.0 stop=3.0 turn=2.50 gap=1.00 step=0.5（1 趟：220445）
```bash
python -u new_vision/jetson/run_policy_vision.py --camera 0 --headless \
    --no-red-detect --bias-straight-cm 0 --card-trigger-dist-cm 30 \
    --wz-fire-cm 8.0 --wz-stop-cm 3.0 --wz-turn-s 2.50 \
    --wz-gap-s 1.00 --wz-step 0.5 \
    --dump-on-loss records/loss_dump --shape-dump records/shape_dump \
    --max-seconds 90 2>&1 | tee records/run_$(date +%H%M%S).log
```

### fire=? stop=? turn=? gap=? step=?（27 趟：0512, 0513, 0516, 0517, 0520, 0533, 0535, 0540, 0553, 0554, 0601, 0604, 0605, 0606, 0709, 0710, 1407, 1409, 1416, 1422, 1425, 1428, 1447, 210554, 212837, 214710, 214722）

（这几趟的横幅里没有上面打 `?` 的项：是加 `--wz-stop-cm` / 改成「另一侧」语义之前的代码版本，命令只能从对应日期的代码里查。）

## C 会话（motor_positions CSV ↔ main_*.log ↔ 哪几趟用了它）

- `motor_positions_20261003_051702_124302.csv`  05:17:02 起，15s / 743 行 → main_logs/main_0517.log；run: 0517
- `motor_positions_20261003_055405_770978.csv`  05:54:05 起，8s / 378 行 → main_logs/main_0554.log；run: 0554
- `motor_positions_20261003_060426_203416.csv`  06:04:26 起，12s / 599 行 → main_logs/main_0604.log；run: 0604
- `motor_positions_20261003_060604_099051.csv`  06:06:04 起，24s / 1176 行 → main_logs/main_0606.log；run: 0606
- `motor_positions_20261003_070953_526682.csv`  07:09:53 起，11s / 556 行 → main_logs/main_0709.log；run: 0710
- `motor_positions_20261003_092812_042921.csv`  09:28:12 起，3s / 173 行 → main_logs/main_0928.log；run: 092751
- `motor_positions_20261003_165350_074759.csv`  16:53:50 起，87s / 4326 行 → main_logs/main_1653.log；run: 165343
- `motor_positions_20261003_210325_829239.csv`  21:03:25 起，7s / 353 行 → main_logs/main_2103.log；run: 210308
- `motor_positions_20261003_210407_037210.csv`  21:04:07 起，10s / 505 行 → main_logs/main_2104.log；run: 210400
- `motor_positions_20261003_210451_821099.csv`  21:04:51 起，19s / 956 行 → main_logs/main_2104.log；run: 210445
- `motor_positions_20261003_212902_318805.csv`  21:29:02 起，8s / 421 行 → main_logs/main_2129.log；run: 212852
- `motor_positions_20261003_213207_254841.csv`  21:32:07 起，9s / 439 行 → main_logs/main_2132.log；run: 213155
- `motor_positions_20261003_213300_239003.csv`  21:33:00 起，12s / 581 行 → main_logs/main_2133.log；run: 213254
- `motor_positions_20261003_214755_176879.csv`  21:47:55 起，8s / 409 行 → main_logs/main_2147.log；run: 214747
- `motor_positions_20261003_215440_029495.csv`  21:54:40 起，24s / 1183 行 → main_logs/main_2154.log；run: 215434
- `motor_positions_20261003_215539_195955.csv`  21:55:39 起，25s / 1260 行 → main_logs/main_2155.log；run: 215533
- `motor_positions_20261003_215712_315002.csv`  21:57:12 起，22s / 1092 行 → main_logs/main_2157.log；run: 215707
- `motor_positions_20261003_215757_662027.csv`  21:57:57 起，16s / 814 行 → main_logs/main_2157.log；run: 215750
- `motor_positions_20261003_215931_777170.csv`  21:59:31 起，14s / 698 行 → main_logs/main_2159.log；run: 215926
- `motor_positions_20261003_220050_283095.csv`  22:00:50 起，12s / 591 行 → main_logs/main_2200.log；run: 220045
- `motor_positions_20261003_220245_827085.csv`  22:02:45 起，17s / 829 行 → main_logs/main_2202.log；run: 220234
- `motor_positions_20261003_220344_806888.csv`  22:03:44 起，7s / 355 行 → main_logs/main_2203.log；run: 220328
- `motor_positions_20261003_220452_222952.csv`  22:04:52 起，20s / 1018 行 → main_logs/main_2204.log；run: 220445
- `motor_positions_20261003_220544_720340.csv`  22:05:44 起，24s / 1178 行 → main_logs/main_2205.log；run: 220533

## 有名字的日志（没有时间戳，命令从横幅看）

- `run_card.log`（65 行）：Camera 0: 1280x720; UDP -> 127.0.0.1:5006; vx=0.4 m/s; max_wz=0.5 rad/s; yaw_sign=1; red=on; sl_gain=1.0; bias=1.0->3.0(dead 6.0, full 12.0)
- `run_discrete.log`（31 行）：[wz] 离散模式：wz 只会是 {0, +0.4, +0.5}，err ≥ 5.0cm 打 0.4、≥ 8.0cm 打 0.5，每次 0.15s（约 3 帧）。只在车身偏右（err>0）时才左转，偏左不转；--center-dead-cm / --steer-full-scale-cm / --b
- `run_discrete_0839_FAIL.log`（31 行）：[wz] 离散模式：wz 只会是 {0, +0.4, +0.5}，err ≥ 5.0cm 打 0.4、≥ 8.0cm 打 0.5，每次 0.15s（约 3 帧）。只在车身偏右（err>0）时才左转，偏左不转；--center-dead-cm / --steer-full-scale-cm / --b
- `run_line_card.log`（100 行）：Camera 0: 1280x720; UDP -> 127.0.0.1:5006; vx=0.3 m/s; max_wz=0.5 rad/s (right 0.5); yaw_sign=1; red=off; sl_gain=1.5; center_dead=4.0cm; bias=0.0->2.
- `run_main.log`（68 行）：[0;93m2026-10-02 13:46:22.413039808 [W:onnxruntime:Default, device_discovery.cc:283 GetGpuDevices] Failed to detect devices under "/sys/class/drm/car
