from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_csv(csv_path: Path, output: Path, title: str, y_columns: list[str]) -> None:
    df = pd.read_csv(csv_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5), dpi=160)
    for col in y_columns:
        if col in df.columns:
            plt.plot(df["epoch"], df[col], marker="o", linewidth=1.8, label=col)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def run(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    output = Path(args.output)
    plot_csv(csv_path, output, args.title, args.columns)
    print(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training metrics from a CSV file.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Training Metrics")
    parser.add_argument("--columns", nargs="+", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
