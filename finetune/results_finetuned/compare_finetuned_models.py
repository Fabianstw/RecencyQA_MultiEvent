import argparse
import json
from pathlib import Path

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


def pct(value: float) -> float:
    return round(value * 100.0, 1)


def load_summary(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {entry["model"]: entry for entry in data}


def build_rows(summary_by_model: dict) -> list[dict]:
    rows = []
    for model in MODEL_ORDER:
        entry = summary_by_model[model]
        allm = entry["ALL"]
        stat_yes = entry["Stationary_YES"]
        stat_no = entry["Stationary_NO"]
        single = entry.get("Event_Single-Event")
        multi = entry.get("Event_Multi-Event")

        rows.append({
            "Model": MODEL_LABELS.get(model, model),
            "Overall Acc": pct(allm["accuracy"]),
            "Overall Tol": pct(allm["tolerant_f1"]),
            "St Acc": pct(stat_yes["accuracy"]),
            "St Tol": pct(stat_yes["tolerant_f1"]),
            "Non-St Acc": pct(stat_no["accuracy"]),
            "Non-St Tol": pct(stat_no["tolerant_f1"]),
            "Single Acc": pct(single["accuracy"]) if single else None,
            "Single Tol": pct(single["tolerant_f1"]) if single else None,
            "Multi Acc": pct(multi["accuracy"]) if multi else None,
            "Multi Tol": pct(multi["tolerant_f1"]) if multi else None,
            "Too Early": allm["error_direction"]["too_early"],
            "Too Late": allm["error_direction"]["too_late"],
            "Correct": allm["error_direction"]["correct"],
            "Invalid": allm["invalid"],
        })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")


def write_latex_table(rows: list[dict], path: Path) -> None:
    header = (
        "\\begin{tabular}{l|cc|cc|cc|cc|cc}\n"
        "\\toprule\n"
        " & \\multicolumn{2}{c|}{Overall} & \\multicolumn{2}{c|}{St.} & \\multicolumn{2}{c|}{Non-St.} "
        "& \\multicolumn{2}{c|}{Single-event} & \\multicolumn{2}{c}{Multi-event} \\\\\n"
        "Model & Acc & Tol. & Acc & Tol. & Acc & Tol. & Acc & Tol. & Acc & Tol. \\\\\n"
        "\\midrule\n"
    )
    lines = [header]
    for row in rows:
        line = (
            f"{row['Model']} & {row['Overall Acc']:.1f} & {row['Overall Tol']:.1f} "
            f"& {row['St Acc']:.1f} & {row['St Tol']:.1f} "
            f"& {row['Non-St Acc']:.1f} & {row['Non-St Tol']:.1f} "
            f"& {row['Single Acc']:.1f} & {row['Single Tol']:.1f} "
            f"& {row['Multi Acc']:.1f} & {row['Multi Tol']:.1f} \\\\\n"
        )
        lines.append(line)
    lines.append("\\bottomrule\n\\end{tabular}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fine-tuned 14B model against baselines.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(__file__).resolve().parent / "summary.json",
        help="Path to summary.json",
    )
    parser.add_argument("--csv-out", type=Path, default=None, help="Write comparison table as CSV")
    parser.add_argument("--latex-out", type=Path, default=None, help="Write comparison table as LaTeX tabular")
    args = parser.parse_args()

    summary_by_model = load_summary(args.summary)
    rows = build_rows(summary_by_model)

    if args.csv_out:
        write_csv(rows, args.csv_out)
    if args.latex_out:
        write_latex_table(rows, args.latex_out)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
