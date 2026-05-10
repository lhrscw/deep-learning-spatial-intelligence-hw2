from __future__ import annotations

import argparse
from pathlib import Path

import fitz


FIGURES = [
    ("Task 1 training curves", "reports/figures/task1_summary.png"),
    ("Task 2 detection and tracking overview", "reports/figures/task2_overview_grid.jpg"),
    ("Task 2 occlusion and dense crossing frames", "reports/figures/task2_occlusion_grid.jpg"),
    ("Task 3 loss comparison grid", "reports/figures/task3_loss_grid.png"),
]


def add_wrapped_text(page: fitz.Page, text: str, rect: fitz.Rect, font_size: int = 11, line_gap: float = 1.25) -> float:
    font = fitz.Font(fontname="china-s")
    lines: list[str] = []
    current = ""
    for ch in text:
        candidate = current + ch
        if font.text_length(candidate, fontsize=font_size) <= rect.width or not current:
            current = candidate
        else:
            lines.append(current)
            current = ch
    if current:
        lines.append(current)
    y = rect.y0
    for line in lines:
        if y + font_size > rect.y1:
            break
        page.insert_text((rect.x0, y), line, fontsize=font_size, fontname="china-s", color=(0, 0, 0))
        y += font_size * line_gap
    return y


def add_page(doc: fitz.Document, title: str) -> fitz.Page:
    page = doc.new_page(width=595, height=842)
    page.insert_text((54, 56), title, fontsize=18, fontname="china-s", color=(0.08, 0.08, 0.08))
    page.draw_line((54, 70), (541, 70), color=(0.6, 0.6, 0.6), width=0.6)
    return page


def markdown_to_pdf(markdown_path: Path, output_path: Path) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    doc = fitz.open()
    page = add_page(doc, "深度学习与空间智能 HW2 实验报告")
    y = 96
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            y += 8
            continue
        if y > 780:
            page = add_page(doc, "实验报告续")
            y = 96
        if line.startswith("# "):
            page.insert_text((54, y), line[2:], fontsize=17, fontname="china-s")
            y += 28
            continue
        if line.startswith("## "):
            page.insert_text((54, y), line[3:], fontsize=14, fontname="china-s", color=(0.1, 0.1, 0.1))
            y += 22
            continue
        if line.startswith("### "):
            page.insert_text((54, y), line[4:], fontsize=12.5, fontname="china-s")
            y += 18
            continue
        if line.startswith("- "):
            line = "• " + line[2:]
        y = add_wrapped_text(page, line, fitz.Rect(62, y, 536, 790), font_size=10.5)
        y += 6
    base_dir = markdown_path.parent.parent
    for title, rel_path in FIGURES:
        image_path = base_dir / rel_path
        if not image_path.exists():
            continue
        page = add_page(doc, title)
        rect = fitz.Rect(40, 96, 555, 790)
        page.insert_image(rect, filename=str(image_path), keep_proportion=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the report markdown draft to a PDF.")
    parser.add_argument("--markdown", default="reports/report_draft.md")
    parser.add_argument("--output", default="reports/HW2_report_draft.pdf")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    markdown_to_pdf(Path(args.markdown), Path(args.output))
    print(args.output)
