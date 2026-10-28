# Nano GPU 环境与算子部署

## 用户授权与约束
- 当前SSH：isaac@192.168.1.131。2026-10-10明确授权安装匹配PyTorch、CUDA OpenCV及依赖，迁移视觉重算子，尝试模型GPU。
- 保留系统包、所有原虚拟环境与原本机/板子文件。不卸载旧包，不覆盖队友脏工作区。使用独立venv、构建目录与代码快照。
- 不使能电机；板上验证用离线实拍和记录观测。
- 本机工作分支policy49flitter-vi，提交后经WSL SSH推送同名远程。
- 图卡旧任务已验证但尚未提交，暂保留；用户要求优先GPU任务。

## 阶段
1. 新地址版本/磁盘/GPU/依赖/队友工作区核对：complete
2. 隔离PyTorch GPU环境安装与CUDA算子实测：complete
3. 冻结匹配OpenCV+contrib，隔离CUDA构建/安装与正确Python导入：complete
4. 巡线/图卡CUDA重算子可选后端与模型GPU后端实现：complete（保持CPU默认；板测未显示GPU收益）
5. 上传隔离代码快照，板上原图/观测精度与端到端性能对照：complete
6. 更新部署计划/命令/证据，提交推送，交付可使用环境：结果文档已更新；提交/推送状态待处理

## 分工
root：SSH、环境与安装、隔离上传、整体验收、文档/提交。
precise_cards：视觉CUDA后端与接入，不远程安装。
cards_data_validation：模型GPU后端及离线bench，不远程安装。
blurred_cards：仅官方版本兼容性资料，不安装。
