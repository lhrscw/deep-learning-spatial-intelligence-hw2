# 深度学习与空间智能 HW2 实验报告

## 首页信息

小组成员：待填写  
学号：待填写  
分工：待填写  
代码仓库链接：待填写  
模型权重下载地址：待填写

## 任务一：Flowers102 图像分类

本任务使用 102 Category Flower Dataset 进行细粒度花卉分类。Baseline 采用 ResNet-18，将原始 ImageNet 1000 类全连接层替换为 102 类输出层。训练时使用 ImageNet 预训练参数初始化卷积骨干，新分类头随机初始化；优化器使用 AdamW，并对骨干网络使用较小学习率、对分类头使用较大学习率。

实验还包含两类对比：第一，改变学习率组合与训练轮数观察超参数影响；第二，将 ResNet-18 改为随机初始化训练，观察 ImageNet 预训练带来的提升。注意力机制实验在 ResNet-18 最后一层特征后加入 SE-block，对通道特征进行重标定，并与 Baseline Accuracy 对比。

实验设置：输入尺寸 224×224，batch size 64，优化器 AdamW，weight decay 1e-4，学习率调度 CosineAnnealingLR，评价指标为验证集和测试集 Accuracy。数据集使用 torchvision 的 Flowers102 标准 train/val/test 划分。

本次先进行 8 epoch 对比实验，用于验证模型差异和形成报告结果；完整训练可按 README 中 30 epoch 配置继续提升。

| 模型 | 初始化 | 注意力 | 学习率 | Epoch | Val Acc | Test Acc |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet-18 | ImageNet | 无 | backbone 1e-4, head 1e-3 | 8 | 0.8873 | 0.8653 |
| ResNet-18 | ImageNet | 无 | backbone 3e-5, head 3e-4 | 8 | 0.7108 | 0.6720 |
| ResNet-18 | 随机初始化 | 无 | 1e-3 | 8 | 0.2990 | 0.2340 |
| ResNet-18 | ImageNet | SE-block | backbone 1e-4, head 1e-3 | 8 | 0.8980 | 0.8676 |

结果显示，ImageNet 预训练在小数据集细粒度分类中带来明显提升；随机初始化 8 epoch 内验证准确率只有 0.2990。学习率过小会显著降低早期收敛速度。SE-block 对通道特征重标定后，验证集和测试集准确率均略高于 baseline。

训练曲线保存在 `reports/figures/task1_resnet18_pretrained.png`、`reports/figures/task1_resnet18_pretrained_low_lr.png`、`reports/figures/task1_resnet18_random.png` 和 `reports/figures/task1_resnet18_se.png`。

## 任务二：Road Vehicle 目标检测、多目标跟踪与越线计数

本任务使用 Road Vehicle Images Dataset 训练车辆检测器。数据来自 Dataset Ninja 的 Road Vehicle 数据集，共 3004 张图像，包含 train 2704 张和 valid 300 张，类别数为 21。原始标注为 Supervisely rectangle JSON，本实验将其转换为 YOLOv8 格式。

检测模型采用 YOLOv8n 作为单阶段检测器，并在 Road Vehicle 数据集上微调。训练完成后，使用 YOLOv8 内置 tracking 流程对 10-30 秒测试视频逐帧推理，输出 bounding box、类别和 Tracking ID。跟踪器默认使用 BoT-SORT。

越线计数逻辑：在视频帧上定义一条虚拟线，对每个 Tracking ID 维护上一帧检测框中心点相对虚拟线的符号。当同一 ID 的中心点从线的一侧变为另一侧时，且该 ID 尚未计数，则累计一次。该方法依赖 Tracking ID 的连续性，能避免同一目标多次被检测造成重复计数。

实验设置：YOLOv8n，输入尺寸 640，batch size 16，先训练 20 epoch 得到报告结果；完整配置保留为 80 epoch。评价指标为 mAP50 和 mAP50-95。跟踪输出视频、逐帧 tracking log、越线计数结果保存在 `runs/task2_detection/tracking_demo/`。

检测和跟踪结果如下：

| 模型 | Epoch | mAP50 | mAP50-95 | 测试视频越线计数 |
| --- | --- | --- | --- | --- |
| YOLOv8n | 20 | 0.3770 | 0.2228 | 9 |

跟踪测试视频为 12 秒 Road Vehicle 验证集图像合成片段 `data/videos/road_vehicle_demo.mp4`，用于稳定复现多目标跟踪、遮挡交汇和越线计数流程。跟踪日志 `runs/task2_detection/tracking_demo/tracking_log.csv` 中共有 2677 条检测记录、67 个唯一 Tracking ID，虚拟线为画面中线 `640,0,640,720`。

遮挡与 ID 跳变分析：连续帧 `reports/figures/task2_occlusion/occlusion_00118.jpg` 至 `occlusion_00121.jpg` 展示了多目标在虚拟线附近密集交汇的片段。BoT-SORT 使用检测框位置、运动预测和外观/IoU 匹配来关联相邻帧目标；在该片段中，多数目标在连续帧中能维持原有 ID，但交汇区域存在检测框重叠和局部遮挡，低置信目标更容易出现短暂丢失或新 ID。越线计数只在同一 Tracking ID 的中心点跨越虚拟线时累计一次，因此能减少重复检测造成的重复计数；但如果遮挡导致 ID switch，仍可能出现漏计或重复计。

训练曲线和可视化保存在 `reports/figures/task2_yolov8_results.png`、`reports/figures/task2_confusion_matrix.png`、`reports/figures/task2_val_batch0_pred.jpg`，跟踪视频为 `reports/figures/task2_tracked_counted_demo.mp4`。

## 任务三：U-Net 语义分割与损失函数工程

本任务从零手写 U-Net，在 Stanford Background Dataset 上训练语义分割模型。数据集包含 715 张场景图像和像素级语义标签，类别包括 sky、tree、road、grass、water、building、mountain、foreground，共 8 类；标签中未知区域记为 ignore index，不参与损失和 mIoU 计算。

U-Net 结构包含四级下采样编码器、瓶颈层、四级上采样解码器，以及对应尺度的 skip connection。每个编码和解码模块使用两层 Conv-BN-ReLU。模型不使用任何预训练权重，全部参数随机初始化。

损失函数对比包含三种配置：标准 Cross-Entropy Loss、手动实现的 Dice Loss、Cross-Entropy Loss + Dice Loss。Dice Loss 先对 logits 做 softmax，再构造 one-hot mask，忽略未知像素，并按类别计算 Dice 系数，最终以 1-Dice 作为损失。评价指标为 pixel accuracy 和 mean IoU。

实验设置：输入尺寸 240×320，batch size 8，优化器 AdamW，learning rate 1e-3，weight decay 1e-4，先训练 10 epoch 得到报告结果；完整配置保留为 50 epoch。训练/验证划分为 80%/20%，固定随机种子 42。

结果表如下：

| 损失配置 | Val Pixel Acc | Val mIoU |
| --- | --- | --- |
| Cross-Entropy | 0.7851 | 0.5394 |
| Dice Loss | 0.7484 | 0.5033 |
| Cross-Entropy + Dice Loss | 0.7775 | 0.5308 |

在 10 epoch 设置下，Cross-Entropy 的验证 mIoU 最高。Dice Loss 能直接优化区域重叠，但单独使用时早期训练对多类别背景数据不如 CE 稳定；CE+Dice 兼顾像素级分类和区域重叠，表现接近 CE。训练曲线保存在 `reports/figures/task3_unet_ce.png`、`reports/figures/task3_unet_dice.png` 和 `reports/figures/task3_unet_ce_dice.png`。

## 总结

当前代码已覆盖三个任务的核心要求：分类微调与消融、检测与跟踪计数、U-Net 从零搭建和 Dice Loss 对比。最终提交前还需要填写首页小组信息、公开 GitHub 仓库链接和模型权重网盘地址，并可将 Task 1/2/3 按 README 的完整 epoch 配置继续训练以提升最终分数。
