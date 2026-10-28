# 连续图卡人工标注与校准基线

人工标注已从本会话恢复到仓库持久目录：16个连续区间（9正例、7负例）和15张静态照片。原Robocup视频/照片及records只读，运行代码未修改。

## 校正与复核状态

旧临时baseline_c7ad975快照漏了new_vision/config/cameras.json，camera_config静默回退a=1、b=0；该次结果全部撤回。此前修复后的baseline_c7ad975_calibrated包含new_vision/jetson与new_vision/config，断言_CAM与指定源码配置一致，usb_main距离标定a=1.13233、b=-2.4862。

临时目录丢失后，人工标注从会话恢复，并重新运行完整校准基线与当前实现。最终数字均来自下面本次持久输出；旧临时运行数字不作为当前证据。

## 最终集成验收

最新版将模糊候选投影到卡平面测量边框、纸面和内部墨迹，并缓存blackhat。独立使用相同[reviewed GT](reviewed_annotations.json)、c7ad975基线与真实标定重新运行完整247个视频调用及15张静态照片，结果保存于[final_reviewed](final_reviewed/summary.json)和[comparison_final.json](comparison_final.json)。首个集成未改善的视频证据保留于current_reviewed，未覆盖。

| 指标 | 校准基线 | 最终集成版 |
|---|---:|---:|
| 正例presence flag | 40/80 | 48/80 |
| 有当帧位置且匹配真实卡ROI的presence | 35/80 | 44/80 |
| 精确框/正确分类/跨类错误 | 6/6/0 | 6/6/0 |
| 旧正确精确分类保留/丢失 | — | 6/0 |
| 独立负例视频cue gate/presence | 0/0（167调用） | 0/0（167调用） |
| 4张独立静态负例cue gate/presence | 0/0 | 0/0 |
| WIN三角11照片正确/跨类/未分类 | 2/4/5 | 6/0/5 |
| 桌面逐调用p50/p95 | 50.2/78.8ms | 37.4/58.3ms |

| 正例区间 | 基线定位 | 最终定位 | 新增frame | 丢失frame | 首次匹配frame |
|---|---:|---:|---|---|---|
| new_triangle | 3/3 | 3/3 | 无 | 无 | 327→327 |
| new_square_approach | 3/11 | 3/11 | 无 | 无 | 528→528 |
| new_circle_approach | 5/7 | 5/7 | 无 | 无 | 691→691 |
| new_circle_restart | 0/4 | 0/4 | 无 | 无 | null→null |
| new_square_clear | 7/12 | 8/12 | 1143 | 无 | 1137→1137 |
| old_triangle | 7/11 | 7/11 | 无 | 无 | 49→49 |
| old_square | 6/17 | 6/17 | 无 | 无 | 83→83 |
| old_star_long | 1/6 | 5/6 | 88, 91, 94, 97 | 无 | 100→88 |
| old_star_short | 3/9 | 7/9 | 41, 44, 47, 50 | 无 | 53→41 |

新增9次定位全部匹配人工真卡ROI，无旧定位丢失；其中两段真实五角星运动视频各新增4次定位，首次匹配均提前12帧，按保存视频FPS约0.599s。这是离线采样网格内的确认时间变化，不是实车停车/动作延迟。new_square_clear的1143补出当帧位置；原presence flag在该帧已由历史窗口保持，所以总flag只增加8次。

new_circle_approach、旧triangle/square的成功均保持；new_circle_restart仍0/4，798/804/807卡边受到历史HUD遮挡或画面裁断，不能宣称覆盖这些缺边场景。最终helper在集成调用中接受9次，全部成为真实卡的新增cue gate；连续负例全部零cue/presence。

两run的标注SHA、config SHA一致。最终shape_detector SHA为97564859ebe737b7262971db1c1c46ce33ac41a4ce63a20acad02d20081c158d，card_presence SHA为a5009b34591ee63955386d87ae1b1502979a5a0c22f665fbbb2e2ecf994d7cda；完整源码/config哈希和_CAM断言保存于各summary。最终还复核了nested候选修复，所有上述定位/分类/负例结果保持；--ref current在导入前保存实际被测jetson/config到source_snapshot，避免并行编辑混淆版本。桌面耗时为本次回放观测，不作为Jetson性能或控制周期结论；区间间重置检测器，不覆盖卡离开后的历史窗口尾随统计。

## 首个集成审查记录

最终比較使用[reviewed_annotations.json](reviewed_annotations.json)：80个正例检测调用各有独立人工ROI，完全按没有算法输出的原视频坐标网格重标。[manual_review](manual_review/)为逐帧人工复核证据。旧[annotations.json](annotations.json)完整保留供审计，其稀疏关键帧插值在快速摇晃中会偏离卡；原24/80定位统计撤回，不能作为召回对比。

本次完整重跑保存了[基线](baseline_reviewed_c7ad975/summary.json)、[当前集成版](current_reviewed/summary.json)、[逐帧比较](comparison_reviewed.json)，二者标注SHA与全部config SHA一致，_CAM都验证为usb_main a=1.13233、b=-2.4862。源码SHA分别记录在summary中。

| 指标 | c7ad975校准基线 | 当前首个集成版 |
|---|---:|---:|
| 正例视频调用 | 80 | 80 |
| 粗ROI top判定eligible调用 | 57 | 57 |
| presence flag调用 | 40 | 40 |
| 有当前位置且匹配GT的presence调用 | 35 | 35 |
| 匹配精确框/正确分类/错误分类 | 6/6/0 | 6/6/0 |
| 至少一次定位presence的遭遇区间 | 8/9 | 8/9 |
| 负例视频调用 | 167 | 167 |
| 负例cue gate/presence | 0/0 | 0/0 |
| WIN三角11照片正确/跨类/未分类 | 2/4/5 | 6/0/5 |
| 桌面每调用p50/p95 | 50.2/78.8ms | 60.2/87.1ms |

| 正例区间 | calls | 基线定位presence | 当前定位presence | 基线/当前正确分类 | 首次匹配frame(基线/当前) |
|---|---:|---:|---:|---:|---|
| new_triangle | 3 | 3 | 3 | 2/2 | 327/327 |
| new_square_approach | 11 | 3 | 3 | 0/0 | 528/528 |
| new_circle_approach | 7 | 5 | 5 | 0/0 | 691/691 |
| new_circle_restart | 4 | 0 | 0 | 0/0 | null/null |
| new_square_clear | 12 | 7 | 7 | 4/4 | 1137/1137 |
| old_triangle | 11 | 7 | 7 | 0/0 | 49/49 |
| old_square | 17 | 6 | 6 | 0/0 | 83/83 |
| old_star_long | 6 | 1 | 1 | 0/0 | 100/100 |
| old_star_short | 9 | 3 | 3 | 0/0 | 53/53 |

视频比较中presence、cue gate、定位匹配、精确框及分类输出逐帧没有变化；原6次正确精确分类全部保留。首个集成版修复了WIN照片4次跨类错误，但本组9段运动视频尚无新增模糊命中；new_circle_restart仍0/4。不能把照片改善推广成这组真实运动视频已改善。

[修复通道诊断](current_positive_diagnostic/frames.csv)记录80正例中repair被调用37次、19候选、接受0：edge拒绝13、paper拒绝5、marking拒绝1；score/contrast拒绝0。说明本组缺口在候选结构支撑与纸面检验，出口分数没有成为瓶颈。后续实现变化应重新运行完整同标注比较。

80正例含2个去HUD的完全重复scene，69个相邻采样场景变化≥1灰度级；167负例无完全重复scene，140个变化≥1灰度级。数字按连续调用描述，不能作为独立照片准确率的样本数。定位presence要求有当帧位置；窗口在miss帧只保留presence flag的调用并不算有可用定位。

## 人工GT

[reviewed_annotations.json](reviewed_annotations.json)坐标均为原图归一化xyxy外框粗ROI，每个正例实际采样帧都经过人工复核。框匹配检查中心在ROI内且IoU≥0.15，不支持像素精确IoU精度结论。标注依据原始等距接触表人工看图，文件名/HUD/算法命中不作GT。已共享实现agent用于开发验证，不能称为未见过的严格held-out。

| 区间 | 视频 | 帧号（从0开始，首尾包含） | GT |
|---|---|---|---|
| new_triangle | camera_commands3.avi | 327–335 | triangle |
| new_square_approach | camera_commands3.avi | 510–540 | square |
| new_circle_approach | camera_commands3.avi | 688–707 | circle |
| new_circle_restart | camera_commands3.avi | 798–809 | circle |
| new_square_clear | camera_commands3.avi | 1125–1160 | square |
| new_empty_reflection | camera_commands3.avi | 370–460 | negative |
| new_empty_floor | camera_commands3.avi | 850–980 | negative |
| new_cable_powerstrip | camera_commands3.avi | 990–1105 | negative |
| old_triangle | WIN_20260919_18_10_34_Pro_flip.mp4 | 40–70 | triangle |
| old_triangle_empty | WIN_20260919_18_10_34_Pro_flip.mp4 | 95–130 | negative |
| old_square | WIN_20260919_18_11_50_Pro_flip.mp4 | 50–100 | square |
| old_square_empty | WIN_20260919_18_11_50_Pro_flip.mp4 | 115–138 | negative |
| old_star_long | WIN_20260919_18_13_14_Pro_flip.mp4 | 85–100 | pentagon |
| old_star_empty | WIN_20260919_18_13_14_Pro_flip.mp4 | 120–172 | negative |
| old_star_short | WIN_20260919_18_13_53_Pro_flip.mp4 | 35–60 | pentagon |
| old_star_short_empty | WIN_20260919_18_13_53_Pro_flip.mp4 | 80–124 | negative |

WIN20260922的11照片均人工视觉三角形并标粗ROI。三张records/shape_dump_3和6card/背景.png为无卡负例。31张6card六类实拍由主任务的photo evaluator覆盖。

## 解释边界

- 不把同帧、低运动场景、历史HUD当独立准确率证据。逐帧CSV记录去HUD的scene hash和运动变化，按遭遇区间汇总。
- 新AVI中543–558、1178–1212冻结段不纳入；1213后人脚stress不当实走证据；710–797长时间静止圆卡不作正例准确率。
- 有卡与足够近分开。粗ROI顶部低于presence_top_frac的远卡不能简单算presence漏检。
- 旧flip为triangle/square/star，新AVI有circle；十字/菱形由31照片覆盖。
- 每3帧调用约对应保存视频的新AVI0.3s、旧flip0.15s，不等于真实capture/停车延迟。历史窗口以前的不相关框会影响cue确认，需看cue与presence框是否都匹配GT。
- 离线桌面耗时不等于Jetson硬件耗时；未进行实车验证。

## 复现

评估脚本检查快照config存在、配置字节同gitref、_CAM逐项一致，记录所有配置哈希；--ref current使用当前真实配置。--save-gt-sheets按固定采样保存原图与人工ROI接触表，不用算法输出挑帧。

```sh
python scripts/evaluate_card_sequences.py \
  --annotations docs/audit_2026-10-09/card_recognition/reviewed_annotations.json \
  --ref c7ad975 \
  --save-gt-sheets \
  --output docs/audit_2026-10-09/card_recognition/baseline_reviewed_c7ad975

python scripts/evaluate_card_sequences.py \
  --annotations docs/audit_2026-10-09/card_recognition/reviewed_annotations.json \
  --ref current \
  --save-gt-sheets \
  --output docs/audit_2026-10-09/card_recognition/final_reviewed
```
