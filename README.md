# 深度学习与空间智能 HW2

本仓库整理 HW2 三个实验的代码、配置、结果图和报告初稿：

- 任务一：在 Flowers102 上微调 ResNet-18，比较 ImageNet 预训练、随机初始化和 SE 注意力模块。
- 任务二：在 Road Vehicle 数据集上训练 YOLOv8n，并完成多目标跟踪、Tracking ID 记录和虚拟线越线计数。
- 任务三：从零实现 U-Net，在 Stanford Background Dataset 上比较 Cross-Entropy、Dice Loss 和 CE+Dice。

## 文件说明

- `src/`：数据处理、模型、损失函数、训练、检测、跟踪计数和报告生成工具。
- `scripts/collect_outputs.py`：汇总报告图表和小型结果日志到 `outputs/`。
- `configs/`：三项实验的推荐运行配置。
- `reports/`：可编辑 Word 报告初稿、PDF 草稿、Markdown 草稿和完整图表。
- `outputs/`：适合公开仓库保留的报告副本、图表和精简结果日志。
- `docs/assignment_requirements.md`：作业说明摘录。
- `docs/github_publish_steps.md`：按 HW1 风格整理的 GitHub 发布步骤。

数据集、视频、训练过程目录和模型权重不放入 Git 仓库。模型权重与演示视频通过 GitHub Release 单独提供。

## 环境

```bash
python -m pip install -r requirements.txt
```

主要依赖包括 PyTorch、torchvision、Ultralytics、OpenCV、pandas、matplotlib、tqdm、PyYAML 和 wandb。

## 数据与权重

本仓库默认使用下面的相对路径：

```text
data/
  flowers-102/
  road_vehicle/
  stanford_background/
  videos/
runs/
  task1_flowers/
  task2_detection/
  task3_segmentation/
```

公开仓库中不直接包含 `data/`、`runs/`、`*.pt`、`*.mp4`。当前 Release 下载地址为：

```text
模型权重：https://github.com/lhrscw/deep-learning-spatial-intelligence-hw2/releases/download/v1.0.0/hw2_weights.zip
跟踪演示视频与日志：https://github.com/lhrscw/deep-learning-spatial-intelligence-hw2/releases/download/v1.0.0/hw2_tracking_demo.zip
```

## 任务一：Flowers102 分类

预训练 ResNet-18 baseline：

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

随机初始化消融：

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

SE 注意力版本：

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

输出保存在：

```text
runs/task1_flowers/<run_name>/
```

## 任务二：Road Vehicle 检测与跟踪计数

准备 YOLOv8 格式数据：

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

对 10-30 秒测试视频进行跟踪和越线计数：

```bash
python -m src.task2_detection.track_count \
  --weights runs/task2_detection/yolov8n_road_vehicle/weights/best.pt \
  --video data/videos/test_road.mp4 \
  --line 320,0,320,640 \
  --save-frames 60 61 62 63
```

如果暂时没有实拍视频，可以生成可复现的 12 秒演示视频：

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

提取连续帧用于遮挡分析：

```bash
python -m src.task2_detection.extract_occlusion_frames \
  --video runs/task2_detection/tracking_demo/tracked_counted.mp4 \
  --start 118 \
  --count 4
```

## 任务三：Stanford Background U-Net 分割

三种损失函数配置：

```bash
python -m src.task3_segmentation.train_unet --loss ce --epochs 50 --batch-size 8
python -m src.task3_segmentation.train_unet --loss dice --epochs 50 --batch-size 8
python -m src.task3_segmentation.train_unet --loss ce_dice --epochs 50 --batch-size 8
```

输出保存在：

```text
runs/task3_segmentation/unet_<loss>/
```

## 当前结果

- 任务一 8 epoch 最优：ResNet-18 + ImageNet + SE，验证集 Acc `0.8980`，测试集 Acc `0.8676`。
- 任务二 20 epoch YOLOv8n：mAP50 `0.3770`，mAP50-95 `0.2228`；演示视频越线计数 `9` 次。
- 任务三 10 epoch 最优：U-Net + Cross-Entropy，验证集 Pixel Acc `0.7851`，验证集 mIoU `0.5394`。

精简结果日志：

```text
outputs/logs/task1_results.csv
outputs/logs/task2_results.json
outputs/logs/task3_results.csv
```

## 报告

当前优先修改 Word 初稿：

```text
reports/HW2_report_draft.docx
```

同步副本位于：

```text
outputs/report/HW2_report_draft.docx
```

重新生成 Word：

```bash
python -m src.common.make_docx_report \
  --output reports/HW2_report_draft.docx
```

汇总公开仓库保留的小文件：

```bash
python scripts/collect_outputs.py
```

最终提交前需要在 Word 中补充小组成员、学号和分工，再手动导出 PDF。

## GitHub

公开仓库：

```text
https://github.com/lhrscw/deep-learning-spatial-intelligence-hw2
```

发布步骤见：

```text
docs/github_publish_steps.md
```

整体方式与 HW1 一致：Git 仓库保留代码、README、报告、图表和小型结果日志；数据集、视频、模型权重和完整训练目录单独上传并在 README/报告中引用。
