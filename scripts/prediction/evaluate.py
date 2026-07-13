from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.csv_utils import save_to_csv
from utils.data import load_json_data
from utils.path import get_data_path, get_results_path

SEVERETY_DICT = {
    "no_action": 0,
    "warning": 1,
    "reprimand": 1,
    "fine": 2,
    "penalty_points": 3,
    "time_penalty": 4,
    "grid_penalty": 5,
    "dsq": 6,
    "race_suspension": 6,
}


def add_severity(data):
    for item in data:
        item["actual_severity"] = SEVERETY_DICT[item["decision_category"]]
        item["predicted_severity"] = SEVERETY_DICT[item["predicted_decision_category"]]


def compute_binary_confusion_matrix(data, save_path=None):
    """Binary confusion matrix: penalty vs no_action, saved as a heatmap image."""
    tp = sum(
        1
        for r in data
        if r["decision_category"] != "no_action"
        and r["predicted_decision_category"] != "no_action"
    )
    tn = sum(
        1
        for r in data
        if r["decision_category"] == "no_action"
        and r["predicted_decision_category"] == "no_action"
    )
    fp = sum(
        1
        for r in data
        if r["decision_category"] == "no_action"
        and r["predicted_decision_category"] != "no_action"
    )
    fn = sum(
        1
        for r in data
        if r["decision_category"] != "no_action"
        and r["predicted_decision_category"] == "no_action"
    )

    # Rows = Actual, Cols = Predicted: [[TN, FP], [FN, TP]]
    matrix = np.array([[tn, fp], [fn, tp]])
    labels = ["No Action", "Penalty"]
    norm_matrix = matrix / len(data)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10, labelpad=8)
    ax.set_ylabel("Actual", fontsize=10, labelpad=8)
    # ax.set_title("Binary Confusion Matrix — Penalty Detection", fontsize=12, pad=12)

    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            count = matrix[i, j]
            pct = norm_matrix[i, j]
            text_color = "white" if pct > 0.55 else "black"
            ax.text(
                j,
                i,
                f"{cell_labels[i][j]}\n{count} ({pct:.0%})",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=text_color,
            )

    plt.tight_layout()

    if save_path is None:
        save_path = Path("binary_confusion_matrix.png")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-- Binary Confusion Matrix saved to: {save_path} --")

    return {"matrix": matrix, "save_path": save_path}


def compute_penalty_detection_metrics(data):
    """Binary metrics: was a penalty needed at all?"""
    total = len(data)
    tp = sum(
        1
        for r in data
        if r["decision_category"] != "no_action"
        and r["predicted_decision_category"] != "no_action"
    )
    tn = sum(
        1
        for r in data
        if r["decision_category"] == "no_action"
        and r["predicted_decision_category"] == "no_action"
    )
    fp = sum(
        1
        for r in data
        if r["decision_category"] == "no_action"
        and r["predicted_decision_category"] != "no_action"
    )
    fn = sum(
        1
        for r in data
        if r["decision_category"] != "no_action"
        and r["predicted_decision_category"] == "no_action"
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / total

    print("\n-- Penalty Detection (was a penalty needed?) --")
    print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
    print(f"Accuracy       : {accuracy * 100:.2f}")
    print(f"Precision      : {precision * 100:.2f}")
    print(f"Recall         : {recall * 100:.2f}")
    print(f"F1-score       : {f1 * 100:.2f}")

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": f"{accuracy * 100:.2f}",
        "precision": f"{precision * 100:.2f}",
        "recall": f"{recall * 100:.2f}",
        "f1": f"{f1 * 100:.2f}",
    }


def compute_category_match_metrics(data):
    """Category match rates overall and among correctly detected penalties."""
    total = len(data)
    penalized_correctly = [
        r
        for r in data
        if r["decision_category"] != "no_action"
        and r["predicted_decision_category"] != "no_action"
    ]

    category_match_rate_all = (
        sum(
            1
            for r in data
            if r["decision_category"] == r["predicted_decision_category"]
        )
        / total
        if total
        else 0.0
    )

    category_match_rate = (
        sum(
            1
            for r in penalized_correctly
            if r["decision_category"] == r["predicted_decision_category"]
        )
        / len(penalized_correctly)
        if penalized_correctly
        else 0.0
    )

    # Macro F1: unweighted average across all categories
    all_categories = set(r["decision_category"] for r in data) | set(
        r["predicted_decision_category"] for r in data
    )
    per_class_f1 = {}
    for cat in all_categories:
        tp_c = sum(
            1
            for r in data
            if r["decision_category"] == cat and r["predicted_decision_category"] == cat
        )
        fp_c = sum(
            1
            for r in data
            if r["decision_category"] != cat and r["predicted_decision_category"] == cat
        )
        fn_c = sum(
            1
            for r in data
            if r["decision_category"] == cat and r["predicted_decision_category"] != cat
        )
        p = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        r = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1_c = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class_f1[cat] = f1_c
    macro_f1_category = np.mean(list(per_class_f1.values())) if per_class_f1 else 0.0

    print("\n-- Category Match --")
    print(
        f"Correct category match rate (all)              : {category_match_rate_all * 100:.2f}"
    )
    print(
        f"Correct category match rate (penalties only)   : {category_match_rate * 100:.2f}"
    )
    print(
        f"Macro F1 (all categories)                      : {macro_f1_category * 100:.2f}"
    )

    return {
        "category_match_rate_all": f"{category_match_rate_all * 100:.2f}",
        "category_match_rate_penalties": f"{category_match_rate * 100:.2f}",
        "macro_f1_category": f"{macro_f1_category * 100:.2f}",
    }


def compute_severity_metrics(data):
    """Ordinal severity MAE and bias."""
    add_severity(data)
    severity_errors = [r["predicted_severity"] - r["actual_severity"] for r in data]
    mae = np.mean([abs(e) for e in severity_errors])
    rmse = np.sqrt(np.mean([e**2 for e in severity_errors]))
    bias = np.mean(severity_errors)

    print("\n-- Severity (ordinal) --")
    print(f"MAE      : {mae:.2f}  (0=perfect, 1=off by one tier)")
    print(f"RMSE     : {rmse:.2f}  (penalises large errors more than MAE)")
    print(f"Bias     : {bias:.2f}  (negative=too lenient, positive=too harsh)")

    return {"mae": f"{mae:.2f}", "rmse": f"{rmse:.2f}", "bias": f"{bias:.2f}"}


def compute_equality_metrics(data):
    """Exact penalty equality metrics."""
    total = len(data)
    penalized_correctly = [
        r
        for r in data
        if r["decision_category"] != "no_action"
        and r["predicted_decision_category"] != "no_action"
    ]

    exact_penalty_accuracy = sum(1 for r in data if r["equal"]) / total

    exact_penalty_when_penalty_accuracy = (
        sum(1 for r in penalized_correctly if r["equal"]) / len(penalized_correctly)
        if penalized_correctly
        else 0.0
    )

    categorized_correctly = [
        r
        for r in data
        if r["decision_category"] == r["predicted_decision_category"]
        and r["decision_category"] != "no_action"
    ]
    exact_penalty_when_category_accuracy = (
        sum(1 for r in categorized_correctly if r["equal"]) / len(categorized_correctly)
        if categorized_correctly
        else 0.0
    )

    print("\n-- Equality --")
    print(
        f"Equality Accuracy (all)                    : {exact_penalty_accuracy * 100:.2f}"
    )
    print(
        f"Equality Accuracy (penalty given)          : {exact_penalty_when_penalty_accuracy * 100:.2f}"
    )
    print(
        f"Equality Accuracy (correct category)       : {exact_penalty_when_category_accuracy * 100:.2f}"
    )

    return {
        "exact_penalty_accuracy": f"{exact_penalty_accuracy * 100:.2f}",
        "exact_penalty_when_penalty_accuracy": f"{exact_penalty_when_penalty_accuracy * 100:.2f}",
        "exact_penalty_when_category_accuracy": f"{exact_penalty_when_category_accuracy * 100:.2f}",
    }


def compute_confusion_matrix(data, save_path=None):
    """Confusion matrix across all decision categories, saved as a heatmap image."""
    present = set(
        [r["decision_category"] for r in data]
        + [r["predicted_decision_category"] for r in data]
    )
    # Sort by severity tier, then alphabetically within the same tier
    all_categories = sorted(present, key=lambda c: (SEVERETY_DICT.get(c, 99), c))
    n = len(all_categories)
    idx = {cat: i for i, cat in enumerate(all_categories)}

    # Build numeric matrix: rows = actual, cols = predicted
    matrix = np.zeros((n, n), dtype=int)
    for r in data:
        matrix[idx[r["decision_category"]], idx[r["predicted_decision_category"]]] += 1

    # Normalised row percentages for colour mapping (avoid div-by-zero)
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.where(row_sums > 0, matrix / row_sums, 0.0)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(max(7, n * 1.1), max(6, n)))
    ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    # Axis ticks & labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_categories, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(all_categories, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10, labelpad=8)
    ax.set_ylabel("Actual", fontsize=10, labelpad=8)
    # ax.set_title("Confusion Matrix — Decision Categories", fontsize=13, pad=12)

    # Cell annotations: raw count + percentage
    for i in range(n):
        for j in range(n):
            count = matrix[i, j]
            pct = norm_matrix[i, j]
            text_color = "white" if pct > 0.55 else "black"
            ax.text(
                j,
                i,
                f"{count}{f'\n({pct:.0%})' if count > 0 else ''}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    plt.tight_layout()

    # Save
    if save_path is None:
        save_path = Path("confusion_matrix.png")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-- Confusion Matrix saved to: {save_path} --")

    return {"categories": all_categories, "matrix": matrix, "save_path": save_path}


def print_table(results, name):
    metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1_category",
        "exact_penalty_accuracy",
    ]

    text = f"\\texttt{{{name.replace('_', '\\_')}}}"
    for metric_key in metric_keys:
        if metric_key not in results:
            print(f"Warning: {metric_key} not in metrics")
            continue
        text += f" & {results[metric_key]}"
    text += " \\\\"
    print()
    print(text)
    print()


def evaluate(path):
    path = Path(path)
    print("-" * 50, "\n", path, "\n", "-" * 50)
    data = load_json_data(path)
    print(f"Total: {len(data)}")

    base_path = get_results_path() / "prediciton" / path.parent.stem

    name = path.stem
    results = {
        **compute_penalty_detection_metrics(data),
        **compute_category_match_metrics(data),
        **compute_severity_metrics(data),
        **compute_equality_metrics(data),
    }

    print_table(results, name)

    save_to_csv({"name": name, **results}, base_path / "results.csv")

    compute_binary_confusion_matrix(
        data, save_path=base_path / f"{path.stem}_binary_confusion.png"
    )
    compute_confusion_matrix(
        data, save_path=base_path / f"{path.stem}_category_confusion.png"
    )


if __name__ == "__main__":
    # path = get_data_path() / "predicted/2025/nomic-v1.5_Q8_qwen2.5-coder-7b-2025.json"
    # path = get_data_path() / "predicted/2025/nomic-v1.5_Q8_gemma4-e2b-2025.json"
    # path = get_data_path() / "predicted/2025/qwen2.5-coder-7b_K1.json"
    # path = get_data_path() / "predicted/2025/qwen2.5-coder-7b_K5.json"
    # path = get_data_path() / "predicted/2025/qwen2.5-coder-7b_K10.json"
    # path = get_data_path() / "predicted/2025/qwen2.5-coder-7b_K20.json"
    # path = get_data_path() / "predicted/2025/qwen2.5-coder-7b_K30.json"
    # path = get_data_path() / "predicted/2025/qwen2.5-coder-7b_K40.json"
    # path = get_data_path() / "predicted/2025/qwen2.5-coder-7b-no-cases-no-rules_K0.json"
    # path = get_data_path() / "predicted/2025/qwen2.5-coder-7b-no-cases_K5.json"
    # path = get_data_path() / "predicted/2025/qwen2.5-coder-7b-no-rules_K5.json"
    # path = get_data_path() / "predicted/2025/gemma4-e4b_K10.json"
    # path = get_data_path() / "predicted/2025/qwen3.5-9b_K10.json"
    # path = get_data_path() / "predicted/2025/qwen3.5-9b_K20.json"
    # path = get_data_path() / "predicted/2025/gpt-oss-20b_low_K10.json"
    # evaluate(path)

    for path in (get_data_path() / "predicted/2025/").glob("qwen2.5*.json"):
        evaluate(path)
