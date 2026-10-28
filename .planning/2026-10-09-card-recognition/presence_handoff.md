# 模糊 presence 模块交接

## API

`find_broken_border_presence(gray, photometry, cfg)` 返回 `(box, score, scalar_diagnostics)`。`box=(x,y,w,h)` 与传入工作图同坐标；`score` 使用现有 `photo.scale` 换算的参考对比度单位。模块没有分类、动作或时间状态。

建议在原 `_presence_cue` 无框或低于原出口分数时调用，保留原成功通道的框、分数与位置。新候选仍经现有 9 窗 2 命中 / 最近 3 帧规则；只在当前命中且已确认时提供位置。原本未确认的远候选重武装行为不变。

## 结构与边界

BH31/51、12/16/8参考单位阈值与15/19/25像素关闭产生候选。修复跨度不超过候选短边35%；候选四边回到未修复墨迹验证（单边至少40%，均值至少70%，最长断口不超过35%）。有限断边允许凸包提出候选，并不依赖闭合孔；整边消失仍拒绝。同核BH和同墨迹采样带膨胀有缓存。

结构测量先把观测墨迹与原灰度投到96×96图卡平面，内部区域为15:81，要求纸面至少30%、内部图形墨迹2.5%到75%、内部明暗差至少20参考单位。分数为图卡平面纸面比例乘纸面/边框对比，再除`photo.scale`；不宜把其数值与旧环路径分数不加路径标记地混为一列。此弱结构兜底有额外分数底线3，保留原闭合环通道的1.2门槛。候选底边也必须完整；旧闭合环成功路径的下边缘行为不变。

轴对齐外接矩形面积门槛从55%改为35%，因为旋转/透视卡的真实投影常仅占其外接矩形一半；真正的纸面与内部图形条件在图卡平面检查，而不由朝向决定。多个合格候选中，较小quad的中心落在大15%以上的合格outer quad内时，丢掉较小候选，避免内部墨迹与半边框连通导致cy/width偏移。

## 先前探针（临时环境重置前）

四尺度未最终收紧版本在人工16区间的167负例均无候选；80正例中模块单独23候选、粗ROI匹配11。该计数不能代表实际召回：候选需要和真实卡ROI匹配，且线性插值标注会跟不上步态抖动。此前存在 `背景.png` 反光误报，不能把视频负例零误报扩展为所有负例零误报。

首轮四尺度桌面耗时中位10.6ms、p95 14.7ms。不是Nano耗时或停车行为验证。

## 持久化验证

- `tests/test_card_presence.py`：有限断边外框坐标、亮暗一致、横纵模糊、35/45/60度旋转卡的outer位置、整边缺失、无内部图形/实心块/粗线/线缆、远近消失无旧位置、3张历史线缆反光失败帧。
- `presence_validate.py`：使用恢复的完整人工标注，输出每帧候选、粗ROI IoU、原始/增益0.4/横纵11px模糊的负例计数与桌面耗时。
- `presence_validation_intervals.json`：临时目录重置后从人工审阅消息忠实恢复的时间区间备份，不冒充完整ROI标注。

运行：

```sh
/tmp/jetson-loss-check-venv/bin/python -m unittest discover -s tests -p test_card_presence.py -v
/tmp/jetson-loss-check-venv/bin/python .planning/2026-10-09-card-recognition/presence_validate.py --annotations docs/audit_2026-10-09/card_recognition/reviewed_annotations.json --output '/mnt/d/用户/lenovo/桌面/github repository/jetson_orin_nano_code/.planning/2026-10-09-card-recognition/presence_validation_cardplane.json'
```

最终图卡平面版本：8项测试通过。完整人工16区间的167视频负例，加`背景.png`及3张旧dump，共171负例；原始、增益0.4、横向11px模糊、纵向11px模糊条件全部0候选。四条件共684个负例探针；同一帧的反事实不能视为独立采集样本。

最终outer优先版本的80逐帧人工标注正例，原始/暗化/横纵模糊分别40/36/27/31候选，全都匹配对应真实卡ROI。原版无精准框、无cue的9个实拍帧被新通道恢复：`new_square_clear:1143`、`old_star_long:85/88/94/97`、`old_star_short:38/41/44/50`。这不是动作/停车成功率。独立代理已以冻结源`a5009b34591ee63955386d87ae1b1502979a5a0c22f665fbbb2e2ecf994d7cda`重跑全部247calls+15stills：集成逐帧人工GT定位35→44，无旧命中丢失，rawpresence40→48；两段星形卡首次定位各提前12源帧、约0.599秒保存视频时间。167连续视频负例及4静态负例全部零cue/presence。被测源快照与记录见`docs/audit_2026-10-09/card_recognition/final_reviewed`及`comparison_final.json`。此helper参数使用共享开发样本选定，没有未见过的保留测试集。

最终outer优先图卡平面版原图桌面耗时中位5.6ms、p95 11.6ms；完整逐帧数据见`presence_validation_cardplane.json`（OpenCV4.14.0）。不是Nano或实车性能结论。`presence_validation.json`保留为旧版本开发记录，不能引用为当前性能/召回结论。

边界：`new_circle_restart`的798帧外框与历史HUD连通为503像素宽域，804/807的底边被HUD或画面裁掉。没有足够独立完整四边证据，当前仍拒绝，不借标注ROI或分类猜一张卡。
