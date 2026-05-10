# GitHub 发布步骤

本说明沿用 HW1 的整理方式：公开仓库只保留代码、README、报告、图表和小型结果日志；数据集、模型权重、视频和完整训练目录单独上传。

## 1. 发布前检查

从 HW2 根目录运行：

```bash
python scripts/collect_outputs.py
git status --short --ignored
```

建议放入 Git 的内容：

```text
src/
scripts/
configs/
docs/assignment_requirements.md
docs/github_publish_steps.md
README.md
requirements.txt
.gitignore
reports/
outputs/
```

建议保持忽略的内容：

```text
data/
runs/
weights/
release_assets/
*.pt
*.pth
*.mp4
HW2_深度学习与空间智能*.pdf
```

## 2. 本地 Git 整理

如果是第一次初始化：

```bash
git init
git branch -M main
git add .gitignore README.md requirements.txt configs docs/assignment_requirements.md docs/github_publish_steps.md scripts src reports outputs
git commit -m "Clean HW2 homework repository"
```

如果已经有本地提交，只需要提交本轮修改：

```bash
git add .gitignore README.md docs/github_publish_steps.md reports outputs scripts src configs requirements.txt
git commit -m "Update HW2 GitHub-ready files"
```

如需设置当前仓库作者信息：

```bash
git config user.name "Your Name"
git config user.email "your_email@example.com"
```

## 3. 创建公开仓库并推送

在 GitHub 网页创建 Public repository，仓库名可使用：

```text
deep-learning-spatial-intelligence-hw2
```

如果沿用 HW1 的 GitHub 用户名，推送命令形如：

```bash
git remote add origin git@github.com:lhrscw/deep-learning-spatial-intelligence-hw2.git
git push -u origin main
```

如果本地已经有 `origin`，先查看：

```bash
git remote -v
```

需要替换远端时使用：

```bash
git remote set-url origin git@github.com:lhrscw/deep-learning-spatial-intelligence-hw2.git
git push -u origin main
```

## 4. 上传大型文件

当前本地已经准备好两个压缩包：

```text
release_assets/hw2_weights.zip
release_assets/hw2_tracking_demo.zip
```

如果需要重新打包：

```bash
mkdir -p release_assets
zip -j release_assets/hw2_weights.zip \
  runs/task1_flowers/resnet18_se_pretrained_e8/best.pt \
  runs/task2_detection/yolov8n_road_vehicle_e20/weights/best.pt \
  runs/task3_segmentation/unet_ce/best.pt

zip -j release_assets/hw2_tracking_demo.zip \
  runs/task2_detection/tracking_demo/tracked_counted.mp4 \
  runs/task2_detection/tracking_demo/tracking_log.csv
```

推荐在 GitHub Release 中创建 `v1.0.0`，上传：

```text
hw2_weights.zip
hw2_tracking_demo.zip
```

上传完成后，把 Release 链接补入：

```text
README.md
reports/HW2_report_draft.docx
outputs/report/HW2_report_draft.docx
```

## 5. 报告最后处理

优先修改：

```text
reports/HW2_report_draft.docx
```

需要补充：

```text
小组成员
学号
分工
GitHub 仓库链接
模型权重下载链接
```

Word 修改完成后，再从 Word/WPS 手动导出最终 PDF。若报告文件有变化，运行：

```bash
python scripts/collect_outputs.py
git add reports outputs README.md
git commit -m "Update final HW2 report"
git push
```
