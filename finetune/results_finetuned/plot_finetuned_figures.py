import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = [
    "schulzeschulbus_b2df/Qwen2.5-14B-Instruct-recency_QWEN2_5_14B-852a4208",
    "moonshotai/Kimi-K2-Instruct-0905",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "deepseek-ai/DeepSeek-V3",
]

MODEL_LABELS = {
    "schulzeschulbus_b2df/Qwen2.5-14B-Instruct-recency_QWEN2_5_14B-852a4208": "Qwen2.5-14B (FT)",
    "moonshotai/Kimi-K2-Instruct-0905": "Kimi",
    "Qwen/Qwen2.5-72B-Instruct-Turbo": "Qwen2.5-72B",
    "deepseek-ai/DeepSeek-V3": "DeepSeek-V3",
}

LABELS = [
    "An-Hour", "A-Few-Hours", "A-Day", "A-Few-Days",
    "A-Week", "A-Few-Weeks", "A-Month", "A-Few-Months",
    "A-Year", "A-Few-Years", "Many-Years", "Never"
]

PARAM_COUNTS_B = {
    "schulzeschulbus_b2df/Qwen2.5-14B-Instruct-recency_QWEN2_5_14B-852a4208": 14,
    "moonshotai/Kimi-K2-Instruct-0905": 1000,
    "Qwen/Qwen2.5-72B-Instruct-Turbo": 72,
    "deepseek-ai/DeepSeek-V3": 671,
}


def load_summary(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_accuracy(summary: list[dict], out_dir: Path) -> None:
    summary_map = {entry["model"]: entry for entry in summary}
    models = [MODEL_LABELS[m] for m in MODEL_ORDER]
    acc = [summary_map[m]["ALL"]["accuracy"] * 100 for m in MODEL_ORDER]
    tol = [summary_map[m]["ALL"]["tolerant_f1"] * 100 for m in MODEL_ORDER]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 4.8))
    plt.bar(x - width / 2, acc, width, label="Accuracy")
    plt.bar(x + width / 2, tol, width, label="Tolerant F1")

    plt.xticks(x, models, rotation=15)
    plt.ylabel("Score (\%)")
    plt.title("Fine-tuned 14B vs. Baselines")
    plt.legend()

    for i, val in enumerate(acc):
        plt.text(x[i] - width / 2, val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    for i, val in enumerate(tol):
        plt.text(x[i] + width / 2, val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "finetuned_accuracy_barplot.pdf")
    plt.savefig(out_dir / "finetuned_accuracy_barplot.png", dpi=200)
    plt.close()


def plot_error_direction(summary: list[dict], out_dir: Path) -> None:
    summary_map = {entry["model"]: entry for entry in summary}
    models = [MODEL_LABELS[m] for m in MODEL_ORDER]

    too_early = [summary_map[m]["ALL"]["error_direction"]["too_early"] for m in MODEL_ORDER]
    too_late = [summary_map[m]["ALL"]["error_direction"]["too_late"] for m in MODEL_ORDER]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 4.8))
    plt.bar(x - width / 2, too_early, width, label="Too early")
    plt.bar(x + width / 2, too_late, width, label="Too late")

    plt.xticks(x, models, rotation=15)
    plt.ylabel("Count")
    plt.title("Directional Recency Errors")
    plt.legend()

    for i, val in enumerate(too_early):
        plt.text(x[i] - width / 2, val + 4, str(val), ha="center", va="bottom", fontsize=9)
    for i, val in enumerate(too_late):
        plt.text(x[i] + width / 2, val + 4, str(val), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "finetuned_error_direction_barplot.pdf")
    plt.savefig(out_dir / "finetuned_error_direction_barplot.png", dpi=200)
    plt.close()


def plot_aggregated_confusion(summary: list[dict], out_dir: Path) -> None:
    agg = np.zeros((12, 12), dtype=float)
    for entry in summary:
        agg += np.array(entry["ALL"]["confusion_matrix"])

    row_sums = agg.sum(axis=1, keepdims=True)
    norm = np.divide(agg, row_sums, where=row_sums != 0)

    plt.figure(figsize=(9, 7))
    im = plt.imshow(norm, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Fraction of samples")

    plt.xticks(range(12), LABELS, rotation=45, ha="right")
    plt.yticks(range(12), LABELS)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Model-averaged Recency Confusion Matrix")

    for i in range(13):
        plt.axhline(i - 0.5, color="white", linewidth=0.5)
        plt.axvline(i - 0.5, color="white", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "finetuned_confusion_heatmap.pdf")
    plt.savefig(out_dir / "finetuned_confusion_heatmap.png", dpi=200)
    plt.close()


def plot_param_efficiency(summary: list[dict], out_dir: Path) -> None:
    summary_map = {entry["model"]: entry for entry in summary}
    models = MODEL_ORDER
    params = np.array([PARAM_COUNTS_B[m] for m in models], dtype=float)
    x = np.log10(params)

    acc = np.array([summary_map[m]["ALL"]["accuracy"] * 100 for m in models])
    tol = np.array([summary_map[m]["ALL"]["tolerant_f1"] * 100 for m in models])

    ft_model = "schulzeschulbus_b2df/Qwen2.5-14B-Instruct-recency_QWEN2_5_14B-852a4208"
    ft_idx = models.index(ft_model)

    base_mask = np.arange(len(models)) != ft_idx
    ft_x = x[ft_idx]
    ft_acc = acc[ft_idx]
    ft_tol = tol[ft_idx]
    base_x = x[base_mask]
    base_acc = acc[base_mask]
    base_tol = tol[base_mask]

    plt.figure(figsize=(7.5, 5))
    plt.scatter(base_x, base_acc, s=80, alpha=0.75, label="Accuracy (baselines)")
    plt.scatter(base_x, base_tol, s=80, alpha=0.75, marker="s", label="Tolerant F1 (baselines)")
    plt.scatter([ft_x], [ft_acc], s=160, color="#d62728", edgecolor="black", linewidth=0.6, label="Accuracy (14B FT)")
    plt.scatter([ft_x], [ft_tol], s=160, color="#ff7f0e", edgecolor="black", linewidth=0.6, marker="s", label="Tolerant F1 (14B FT)")

    # Fit simple linear trends vs log10(params)
    acc_coeff = np.polyfit(x, acc, 1)
    tol_coeff = np.polyfit(x, tol, 1)
    x_fit = np.linspace(x.min() - 0.05, x.max() + 0.05, 200)
    acc_fit = np.polyval(acc_coeff, x_fit)
    tol_fit = np.polyval(tol_coeff, x_fit)
    plt.plot(x_fit, acc_fit, color="#1f77b4", linestyle="--", linewidth=1.3, label="Acc trend")
    plt.plot(x_fit, tol_fit, color="#ff7f0e", linestyle=":", linewidth=1.3, label="Tol F1 trend")

    for i, m in enumerate(models):
        if i == ft_idx:
            continue
        plt.text(x[i] + 0.01, acc[i] + 0.6, MODEL_LABELS[m], fontsize=9)

    plt.annotate(
        "14B fine-tuned",
        xy=(ft_x, ft_acc),
        xytext=(ft_x + 0.08, ft_acc + 6),
        arrowprops=dict(arrowstyle="->", lw=1.0),
        fontsize=10,
        fontweight="bold",
    )

    plt.xlabel("log10(Parameters in billions)")
    plt.ylabel("Score (\%)")
    plt.title("Parameter Efficiency: Fine-tuned vs. Baselines")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "finetuned_param_efficiency.pdf")
    plt.savefig(out_dir / "finetuned_param_efficiency.png", dpi=200)
    plt.close()


def plot_param_normalized(summary: list[dict], out_dir: Path) -> None:
    summary_map = {entry["model"]: entry for entry in summary}
    models = MODEL_ORDER
    labels = [MODEL_LABELS[m] for m in models]

    params = np.array([PARAM_COUNTS_B[m] for m in models], dtype=float)
    log_params = np.log10(params)

    acc = np.array([summary_map[m]["ALL"]["accuracy"] * 100 for m in models])
    tol = np.array([summary_map[m]["ALL"]["tolerant_f1"] * 100 for m in models])

    acc_norm = acc / log_params
    tol_norm = tol / log_params

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 4.8))
    colors_acc = ["#d62728" if "14B" in labels[i] else "#1f77b4" for i in range(len(labels))]
    colors_tol = ["#ff7f0e" if "14B" in labels[i] else "#2ca02c" for i in range(len(labels))]

    plt.bar(x - width / 2, acc_norm, width, color=colors_acc, label="Acc / log10(params)")
    plt.bar(x + width / 2, tol_norm, width, color=colors_tol, label="Tol F1 / log10(params)")

    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Score per log10(params)")
    plt.title("Parameter-Normalized Performance")
    plt.legend()

    for i, val in enumerate(acc_norm):
        plt.text(x[i] - width / 2, val + 0.2, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    for i, val in enumerate(tol_norm):
        plt.text(x[i] + width / 2, val + 0.2, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "finetuned_param_normalized.pdf")
    plt.savefig(out_dir / "finetuned_param_normalized.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures for the fine-tuned comparison.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(__file__).resolve().parent / "summary.json",
        help="Path to summary.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "Paper" / "res",
        help="Directory to write figures",
    )
    args = parser.parse_args()

    summary = load_summary(args.summary)
    ensure_dir(args.out_dir)

    plot_accuracy(summary, args.out_dir)
    plot_error_direction(summary, args.out_dir)
    plot_aggregated_confusion(summary, args.out_dir)
    plot_param_efficiency(summary, args.out_dir)
    plot_param_normalized(summary, args.out_dir)


if __name__ == "__main__":
    main()
