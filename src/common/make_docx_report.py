from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

from src.common.utils import project_root


def set_run_font(run, size: int | None = None, bold: bool | None = None) -> None:
    run.font.name = "宋体"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def set_doc_fonts(doc: Document) -> None:
    for style_name in ["Normal", "Heading 1", "Heading 2", "Heading 3"]:
        style = doc.styles[style_name]
        style.font.name = "宋体"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    doc.styles["Normal"].font.size = Pt(10.5)


def clear_body(doc: Document) -> None:
    body = doc._element.body
    for child in list(body):
        if child.tag.endswith("sectPr"):
            continue
        body.remove(child)


def add_paragraph(doc: Document, text: str = "", style: str | None = None):
    paragraph = doc.add_paragraph(style=style)
    run = paragraph.add_run(text)
    set_run_font(run)
    paragraph.paragraph_format.first_line_indent = Cm(0.74) if style is None else None
    paragraph.paragraph_format.line_spacing = 1.25
    return paragraph


def add_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    set_run_font(run, size=9)
    run.italic = True


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    paragraph = doc.add_heading(level=level)
    run = paragraph.add_run(text)
    set_run_font(run, size=16 if level == 1 else 13, bold=True)


def add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for idx, header in enumerate(headers):
        cell = table.rows[0].cells[idx]
        cell.text = header
    for row in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(row):
            cells[idx].text = value
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    set_run_font(run, size=9)


def add_picture_if_exists(doc: Document, rel_path: str, caption: str, width_cm: float = 15.5) -> None:
    path = project_root() / rel_path
    if not path.exists():
        add_paragraph(doc, f"图像文件暂缺：{rel_path}")
        return
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    run.add_picture(str(path), width=Cm(width_cm))
    add_caption(doc, caption)


def new_document(template: Path | None) -> Document:
    if template and template.exists():
        doc = Document(template)
        clear_body(doc)
    else:
        doc = Document()
    set_doc_fonts(doc)
    section = doc.sections[0]
    section.top_margin = Cm(2.3)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    return doc


def build_report(output: Path, template: Path | None = None) -> None:
    doc = new_document(template)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("深度学习与空间智能 HW2 实验报告")
    set_run_font(run, size=18, bold=True)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Flowers102 分类、Road Vehicle 检测跟踪、U-Net 语义分割")
    set_run_font(run, size=12)

    add_paragraph(doc, "小组成员：待填写")
    add_paragraph(doc, "学号：待填写")
    add_paragraph(doc, "分工：待填写")
    add_paragraph(doc, "代码仓库链接：https://github.com/lhrscw/deep-learning-spatial-intelligence-hw2")
    add_paragraph(
        doc,
        "模型权重下载地址：https://github.com/lhrscw/deep-learning-spatial-intelligence-hw2/releases/download/v1.0.0/hw2_weights.zip",
    )
    add_paragraph(
        doc,
        "跟踪演示视频与日志：https://github.com/lhrscw/deep-learning-spatial-intelligence-hw2/releases/download/v1.0.0/hw2_tracking_demo.zip",
    )

    add_heading(doc, "摘要")
    add_paragraph(
        doc,
        "本报告围绕 HW2 的三个任务展开：首先在 Flowers102 数据集上微调 ImageNet 预训练 ResNet-18，并完成随机初始化消融与 SE 注意力对比；其次在 Road Vehicle Images Dataset 上微调 YOLOv8n，并结合 BoT-SORT 完成多目标跟踪、Tracking ID 输出和越线计数；最后从零搭建 U-Net，在 Stanford Background Dataset 上比较 Cross-Entropy、Dice Loss 和 CE+Dice 三种损失配置。",
    )
    add_paragraph(
        doc,
        "当前报告采用短轮次可复现实验作为初稿结果：Flowers102 8 epoch、YOLOv8n 20 epoch、U-Net 10 epoch。最终提交前可按 README 中完整配置继续训练并替换结果。",
    )

    add_heading(doc, "一、任务一：Flowers102 图像分类")
    add_paragraph(
        doc,
        "本任务使用 102 Category Flower Dataset 进行细粒度花卉分类。Baseline 采用 ResNet-18，将原始 ImageNet 1000 类全连接层替换为 102 类输出层。训练时使用 ImageNet 预训练参数初始化卷积骨干，新分类头随机初始化；优化器使用 AdamW，并对骨干网络使用较小学习率、对分类头使用较大学习率。",
    )
    add_paragraph(
        doc,
        "实验包含超参数对比、预训练消融和注意力模块对比。注意力实验在 ResNet-18 最后一层卷积特征后加入 SE-block，通过通道权重重标定增强判别性特征。",
    )
    add_paragraph(
        doc,
        "实验设置：输入尺寸 224×224，batch size 64，优化器 AdamW，weight decay 1e-4，学习率调度 CosineAnnealingLR，评价指标为验证集和测试集 Accuracy。数据集使用 torchvision 的 Flowers102 标准 train/val/test 划分。",
    )
    add_table(
        doc,
        ["模型", "初始化", "注意力", "学习率", "Epoch", "Val Acc", "Test Acc"],
        [
            ["ResNet-18", "ImageNet", "无", "backbone 1e-4, head 1e-3", "8", "0.8873", "0.8653"],
            ["ResNet-18", "ImageNet", "无", "backbone 3e-5, head 3e-4", "8", "0.7108", "0.6720"],
            ["ResNet-18", "随机初始化", "无", "1e-3", "8", "0.2990", "0.2340"],
            ["ResNet-18", "ImageNet", "SE-block", "backbone 1e-4, head 1e-3", "8", "0.8980", "0.8676"],
        ],
    )
    add_paragraph(
        doc,
        "结果显示，ImageNet 预训练在小数据集细粒度分类中带来明显提升；随机初始化 8 epoch 内验证准确率只有 0.2990。学习率过小会显著降低早期收敛速度。SE-block 对通道特征重标定后，验证集和测试集准确率均略高于 baseline。",
    )
    add_picture_if_exists(doc, "reports/figures/task1_summary.png", "图 1 Flowers102 分类训练曲线汇总")

    add_heading(doc, "二、任务二：Road Vehicle 目标检测、多目标跟踪与越线计数")
    add_paragraph(
        doc,
        "本任务使用 Road Vehicle Images Dataset 训练车辆检测器。数据来自 Dataset Ninja 的 Road Vehicle 数据集，共 3004 张图像，包含 train 2704 张和 valid 300 张，类别数为 21。原始标注为 Supervisely rectangle JSON，本实验将其转换为 YOLOv8 格式。",
    )
    add_paragraph(
        doc,
        "检测模型采用 YOLOv8n 作为单阶段检测器，并在 Road Vehicle 数据集上微调。训练完成后，使用 YOLOv8 内置 tracking 流程对测试视频逐帧推理，输出 bounding box、类别和 Tracking ID。跟踪器默认使用 BoT-SORT。",
    )
    add_paragraph(
        doc,
        "越线计数逻辑：在视频帧上定义一条虚拟线，对每个 Tracking ID 维护上一帧检测框中心点相对虚拟线的符号。当同一 ID 的中心点从线的一侧变为另一侧时，且该 ID 尚未计数，则累计一次。",
    )
    add_table(
        doc,
        ["模型", "Epoch", "mAP50", "mAP50-95", "测试视频越线计数"],
        [["YOLOv8n", "20", "0.3770", "0.2228", "9"]],
    )
    add_paragraph(
        doc,
        "跟踪测试视频为 12 秒 Road Vehicle 验证集图像合成片段 data/videos/road_vehicle_demo.mp4，用于稳定复现多目标跟踪、遮挡交汇和越线计数流程。跟踪日志中共有 2677 条检测记录、67 个唯一 Tracking ID，虚拟线为画面中线 640,0,640,720。",
    )
    add_picture_if_exists(doc, "reports/figures/task2_summary.png", "图 2 YOLOv8 检测训练曲线、验证预测和混淆矩阵")
    add_picture_if_exists(doc, "reports/figures/task2_occlusion_grid.jpg", "图 3 连续 4 帧遮挡/密集交汇片段")
    add_paragraph(
        doc,
        "遮挡与 ID 跳变分析：连续帧展示了多目标在虚拟线附近密集交汇的片段。BoT-SORT 使用检测框位置、运动预测和外观/IoU 匹配来关联相邻帧目标；在该片段中，多数目标在连续帧中能维持原有 ID，但交汇区域存在检测框重叠和局部遮挡，低置信目标更容易出现短暂丢失或新 ID。越线计数只在同一 Tracking ID 的中心点跨越虚拟线时累计一次，因此能减少重复检测造成的重复计数；但如果遮挡导致 ID switch，仍可能出现漏计或重复计。",
    )

    add_heading(doc, "三、任务三：U-Net 语义分割与损失函数工程")
    add_paragraph(
        doc,
        "本任务从零手写 U-Net，在 Stanford Background Dataset 上训练语义分割模型。数据集包含 715 张场景图像和像素级语义标签，类别包括 sky、tree、road、grass、water、building、mountain、foreground，共 8 类；标签中未知区域记为 ignore index，不参与损失和 mIoU 计算。",
    )
    add_paragraph(
        doc,
        "U-Net 结构包含四级下采样编码器、瓶颈层、四级上采样解码器，以及对应尺度的 skip connection。每个编码和解码模块使用两层 Conv-BN-ReLU。模型不使用任何预训练权重，全部参数随机初始化。",
    )
    add_paragraph(
        doc,
        "损失函数对比包含三种配置：标准 Cross-Entropy Loss、手动实现的 Dice Loss、Cross-Entropy Loss + Dice Loss。Dice Loss 先对 logits 做 softmax，再构造 one-hot mask，忽略未知像素，并按类别计算 Dice 系数，最终以 1-Dice 作为损失。",
    )
    add_table(
        doc,
        ["损失配置", "Val Pixel Acc", "Val mIoU"],
        [
            ["Cross-Entropy", "0.7851", "0.5394"],
            ["Dice Loss", "0.7484", "0.5033"],
            ["Cross-Entropy + Dice Loss", "0.7775", "0.5308"],
        ],
    )
    add_paragraph(
        doc,
        "在 10 epoch 设置下，Cross-Entropy 的验证 mIoU 最高。Dice Loss 能直接优化区域重叠，但单独使用时早期训练对多类别背景数据不如 CE 稳定；CE+Dice 兼顾像素级分类和区域重叠，表现接近 CE。",
    )
    add_picture_if_exists(doc, "reports/figures/task3_summary.png", "图 4 U-Net 三种损失配置训练曲线")

    add_heading(doc, "四、总结与后续修改")
    add_paragraph(
        doc,
        "当前代码已覆盖三个任务的核心要求：分类微调与消融、检测与跟踪计数、U-Net 从零搭建和 Dice Loss 对比。最终提交前需要填写首页小组成员、学号和分工，并可将 Task 1/2/3 按 README 的完整 epoch 配置继续训练以提升最终分数。",
    )
    add_paragraph(
        doc,
        "本 Word 文档为可编辑初稿，后续可以直接在 Word 中补充 wandb 或 swanlab 截图、替换为实拍交通视频结果、调整排版后再导出 PDF。",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an editable Word report for HW2.")
    parser.add_argument("--output", default="reports/HW2_report_draft.docx")
    parser.add_argument("--template", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_report(Path(args.output), Path(args.template) if args.template else None)
    print(args.output)
