# 深度学习与空间智能 HW2

本仓库包含 HW2 三个实验的代码、配置、结果日志、可视化图表和最终报告。

- Flowers102 图像分类：ResNet-18 微调、随机初始化对比、SE 注意力对比。
- Road Vehicle 检测与跟踪：YOLOv8n 训练、多目标跟踪、Tracking ID 记录和虚拟线计数。
- Stanford Background 语义分割：手写 U-Net、Dice Loss、CE / Dice / CE+Dice 对比。

## 目录

```text
configs/              实验配置
src/                  数据处理、训练、检测、跟踪、分割代码
outputs/figures/      结果图表
outputs/logs/         精简结果日志
reports/              最终报告和报告图表
docs/                 作业说明摘要
requirements.txt      Python 依赖
```

数据集、训练过程目录、视频和模型权重不直接放入 Git 仓库。模型权重和跟踪演示文件通过 Release 提供：

```text
https://github.com/lhrscw/deep-learning-spatial-intelligence-hw2/releases/download/v1.0.0/hw2_weights.zip
https://github.com/lhrscw/deep-learning-spatial-intelligence-hw2/releases/download/v1.0.0/hw2_tracking_demo.zip
```

## 环境

```bash
python -m pip install -r requirements.txt
```

## 数据目录

默认相对路径如下：

```text
data/
  flowers102/
  road_vehicle/
  stanford_background/
  videos/
runs/
  task1_flowers/
  task2_detection/
  task3_segmentation/
```

任务一的 Flowers102 可由 `torchvision` 自动下载。任务二和任务三按作业数据集整理到上面的相对路径后运行脚本。

## 任务一：Flowers102 分类

预训练 ResNet-18：

```bash
python -m src.task1_flowers.train_flowers \
  --run-name resnet18_pretrained_lr1e-4_1e-3 \
  --model resnet18 \
  --epochs 30 \
  --batch-size 64 \
  --backbone-lr 1e-4 \
  --head-lr 1e-3 \
  --amp
```

随机初始化：

```bash
python -m src.task1_flowers.train_flowers \
  --run-name resnet18_random_init \
  --model resnet18 \
  --random-init \
  --epochs 30 \
  --batch-size 64 \
  --backbone-lr 1e-3 \
  --head-lr 1e-3 \
  --amp
```

SE 注意力：

```bash
python -m src.task1_flowers.train_flowers \
  --run-name resnet18_se_pretrained \
  --model resnet18 \
  --attention se \
  --epochs 30 \
  --batch-size 64 \
  --backbone-lr 1e-4 \
  --head-lr 1e-3 \
  --amp
```

## 任务二：Road Vehicle 检测与跟踪

转换 Road Vehicle 标注：

```bash
python -m src.task2_detection.prepare_road_vehicle --download
```

训练 YOLOv8n：

```bash
python -m src.task2_detection.train_yolov8 \
  --data data/road_vehicle/yolo/data.yaml \
  --model yolov8n.pt \
  --epochs 80 \
  --batch 16 \
  --device 0
```

生成 12 秒演示视频并进行跟踪计数：

```bash
python -m src.task2_detection.make_demo_video \
  --list-file data/videos/detected_val_images.txt \
  --output data/videos/road_vehicle_demo.mp4 \
  --seconds 12 \
  --fps 24 \
  --object-width 520

python -m src.task2_detection.track_count \
  --weights runs/task2_detection/yolov8n_road_vehicle_e20/weights/best.pt \
  --video data/videos/road_vehicle_demo.mp4 \
  --output-dir runs/task2_detection/tracking_demo \
  --line 640,0,640,720 \
  --save-frames 118 119 120 121
```

提取连续帧：

```bash
python -m src.task2_detection.extract_occlusion_frames \
  --video runs/task2_detection/tracking_demo/tracked_counted.mp4 \
  --start 118 \
  --count 4
```

## 任务三：Stanford Background 分割

```bash
python -m src.task3_segmentation.train_unet --loss ce --epochs 50 --batch-size 8
python -m src.task3_segmentation.train_unet --loss dice --epochs 50 --batch-size 8
python -m src.task3_segmentation.train_unet --loss ce_dice --epochs 50 --batch-size 8
```

## 结果

- Flowers102：ResNet-18 + ImageNet + SE，Val Acc `0.8980`，Test Acc `0.8676`。
- Road Vehicle：YOLOv8n 20 epoch，mAP50 `0.3770`，mAP50-95 `0.2228`；演示视频越线计数 `9` 次。
- Stanford Background：U-Net + Cross-Entropy，Val Pixel Acc `0.7851`，Val mIoU `0.5394`。

结果日志：

```text
outputs/logs/task1_results.csv
outputs/logs/task2_results.json
outputs/logs/task3_results.csv
```

最终报告：

```text
reports/25210980063_李浩然_hw2.pdf
```
