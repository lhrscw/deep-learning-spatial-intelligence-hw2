from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def task1_summary() -> None:
    runs = [
        "resnet18_pretrained_e8_lr1e-4_1e-3",
        "resnet18_pretrained_e8_lr3e-5_3e-4",
        "resnet18_random_e8",
        "resnet18_se_pretrained_e8",
    ]
    rows = []
    for run in runs:
        run_dir = ROOT / "runs" / "task1_flowers" / run
        metrics = run_dir / "metrics.csv"
        test_file = run_dir / "test_metrics.txt"
        if not metrics.exists() or not test_file.exists():
            continue
        df = pd.read_csv(metrics)
        best = df.loc[df["val_acc"].idxmax()]
        test_acc = None
        for line in test_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("test_acc"):
                test_acc = float(line.split(":")[1].strip())
        rows.append(
            {
                "run": run,
                "best_epoch": int(best["epoch"]),
                "val_acc": round(float(best["val_acc"]), 4),
                "test_acc": round(float(test_acc), 4) if test_acc is not None else "",
                "metrics": rel(metrics),
                "weights": rel(run_dir / "best.pt"),
            }
        )
    pd.DataFrame(rows).to_csv(ROOT / "outputs" / "logs" / "task1_results.csv", index=False)


def task2_summary() -> None:
    result_csv = ROOT / "runs" / "task2_detection" / "yolov8n_road_vehicle_e20" / "results.csv"
    tracking_csv = ROOT / "runs" / "task2_detection" / "tracking_demo" / "tracking_log.csv"
    summary = {}
    if result_csv.exists():
        df = pd.read_csv(result_csv)
        df.columns = [c.strip() for c in df.columns]
        best = df.loc[df["metrics/mAP50(B)"].idxmax()]
        summary.update(
            {
                "model": "yolov8n",
                "epochs": 20,
                "best_epoch": int(best["epoch"]),
                "precision": round(float(best["metrics/precision(B)"]), 4),
                "recall": round(float(best["metrics/recall(B)"]), 4),
                "mAP50": round(float(best["metrics/mAP50(B)"]), 4),
                "mAP50_95": round(float(best["metrics/mAP50-95(B)"]), 4),
                "results_csv": rel(result_csv),
                "weights": "runs/task2_detection/yolov8n_road_vehicle_e20/weights/best.pt",
            }
        )
    if tracking_csv.exists():
        tracking = pd.read_csv(tracking_csv)
        summary.update(
            {
                "tracking_rows": int(len(tracking)),
                "unique_tracking_ids": int(tracking["track_id"].nunique()) if len(tracking) else 0,
                "line_crossing_count": int(tracking["counted"].sum()) if len(tracking) else 0,
                "tracking_log": rel(tracking_csv),
                "tracked_video": "runs/task2_detection/tracking_demo/tracked_counted.mp4",
            }
        )
    out = ROOT / "outputs" / "logs" / "task2_results.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def task3_summary() -> None:
    rows = []
    for loss in ["ce", "dice", "ce_dice"]:
        run_dir = ROOT / "runs" / "task3_segmentation" / f"unet_{loss}"
        metrics = run_dir / "metrics.csv"
        if not metrics.exists():
            continue
        df = pd.read_csv(metrics)
        best = df.loc[df["val_miou"].idxmax()]
        rows.append(
            {
                "loss": loss,
                "best_epoch": int(best["epoch"]),
                "val_pixel_acc": round(float(best["val_pixel_acc"]), 4),
                "val_miou": round(float(best["val_miou"]), 4),
                "metrics": rel(metrics),
                "weights": rel(run_dir / "best.pt"),
            }
        )
    pd.DataFrame(rows).to_csv(ROOT / "outputs" / "logs" / "task3_results.csv", index=False)


def copy_report_assets() -> None:
    report_dir = ROOT / "reports"
    out_report = ROOT / "outputs" / "report"
    for name in ["HW2_report_draft.docx", "HW2_report_draft.pdf", "report_draft.md"]:
        copy_if_exists(report_dir / name, out_report / name)

    figure_names = [
        "task1_summary.png",
        "task2_summary.png",
        "task2_occlusion_grid.jpg",
        "task2_yolov8_results.png",
        "task2_val_batch0_pred.jpg",
        "task2_confusion_matrix.png",
        "task3_summary.png",
    ]
    for name in figure_names:
        copy_if_exists(report_dir / "figures" / name, ROOT / "outputs" / "figures" / name)


def main() -> None:
    (ROOT / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
    task1_summary()
    task2_summary()
    task3_summary()
    copy_report_assets()
    print("Collected report assets and result summaries under outputs/.")


if __name__ == "__main__":
    main()
