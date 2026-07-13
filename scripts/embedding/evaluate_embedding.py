from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils.logger import Logger
from utils.path import get_logs_path, get_results_path
from utils.server import DB, COLLECTIONS

# Define specific collections for Plot and Table
COLLECTIONS_Plot: list[DB] = [
    COLLECTIONS["minilm"],
    COLLECTIONS["nomic-v1.5_Q8"],
]

COLLECTIONS_Table: list[DB] = [
    COLLECTIONS["minilm"],
    COLLECTIONS["gemma"],
    COLLECTIONS["qwen3-4B"],
    COLLECTIONS["bge-large"],
    COLLECTIONS["mxbai"],
    COLLECTIONS["nomic-v1.5"],
    COLLECTIONS["nomic-v1.5_Q8"],
    COLLECTIONS["nomic-v2"],
]

K_Plot = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
K_Table = [1, 5, 10, 20, 30, 40]


###########################################

def filter_docs(all_docs):
    """
    filter documents that have section that exist more than > 1
    """
    metadatas = all_docs["metadatas"]
    section_counts = Counter(section for m in metadatas for section in set(m.get("sections", [])))
    valid_sections = {sec for sec, count in section_counts.items() if count > 1}
    valid_indices = [i for i, m in enumerate(metadatas) if not valid_sections.isdisjoint(m.get("sections", []))]

    filtered_docs = {}
    for key in ["ids", "documents", "metadatas"]:
        if key in all_docs and all_docs[key] is not None:
            filtered_docs[key] = [all_docs[key][i] for i in valid_indices]
    return filtered_docs


def evaluate_all_ks(all_docs, ks, collection):
    # Query once at max(k) per document, then slice neighbors for each k.
    ids = all_docs["ids"]
    metadatas = all_docs["metadatas"]
    max_k = max(ks)

    infringements = [set(m.get("sections", [])) for m in metadatas]  # inf[i] = set of sections that that doc contains

    # Pre-fetch neighbors once at max_k for every document
    all_neighbors = []
    for i in range(len(metadatas)):
        possible_matches = sum(1 for j, secs in enumerate(infringements) if j != i and (secs & infringements[i]))

        description = metadatas[i].get("description", "")
        results = collection.query(
            query_texts=[description],
            n_results=max_k + 1,
            include=["metadatas", "distances"]
        )

        neighbors = [
            m for m, id_ in zip(results["metadatas"][0], results["ids"][0]) if id_ != ids[i]
        ][:max_k]

        all_neighbors.append((neighbors, infringements[i], possible_matches))

    # Now compute metrics for each k by slicing the cached neighbors
    metrics_by_k = {}
    for k in ks:
        precision_arr, recall_arr, hitrates_any, hitrates_all, f1_arr = [], [], [], [], []

        for neighbors_full, inf_i, possible_matches in all_neighbors:
            neighbors = neighbors_full[:k]

            neighbor_sections = [set(n.get("sections", [])) for n in neighbors]

            any_overlap = sum(1 for n_sec in neighbor_sections if n_sec & inf_i)
            all_overlap = sum(1 for n_sec in neighbor_sections if n_sec & inf_i == n_sec)

            precision = any_overlap / len(neighbors) if neighbors else 0.0
            recall = any_overlap / possible_matches if possible_matches else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
            hitrate_any = 1 if any_overlap > 0 else 0
            hitrate_all = 1 if all_overlap > 0 else 0

            precision_arr.append(precision)
            recall_arr.append(recall)
            f1_arr.append(f1)
            hitrates_any.append(hitrate_any)
            hitrates_all.append(hitrate_all)

        metrics = {
            "precision": np.mean(precision_arr) * 100,
            "recall": np.mean(recall_arr) * 100,
            "f1": np.mean(f1_arr) * 100,
            "hitrate_any": np.mean(hitrates_any) * 100,
            "hitrate_all": np.mean(hitrates_all) * 100,
        }

        print(f"k = {k}:")
        print(f"    Precision:     {metrics['precision']:.2f}")
        print(f"    Recall:        {metrics['recall']:.2f}")
        print(f"    F1:            {metrics['f1']:.2f}")
        print(f"    Hit Rate Any:  {metrics['hitrate_any']:.2f}")
        print(f"    Hit Rate All:  {metrics['hitrate_all']:.2f}")

        metrics_by_k[k] = metrics

    return metrics_by_k


def plot_results(results_by_collection: dict, ks: list):
    metric_keys = ["precision", "recall", "f1", "hitrate_any", "hitrate_all"]
    metric_labels = ["Precision", "Recall", "F1", "Hit Rate Any", "Hit Rate All"]
    file_names = ["precision.png", "recall.png", "f1.png", "hitrate_any.png", "hitrate_all.png"]

    save_basename = get_results_path() / "embedding"
    save_basename.mkdir(parents=True, exist_ok=True)

    colors = plt.cm.tab10.colors

    for metric_key, metric_label, file_name in zip(metric_keys, metric_labels, file_names):
        fig, ax = plt.subplots(figsize=(9, 5))

        for idx, (collection_name, metrics_list) in enumerate(results_by_collection.items()):
            values = [m[metric_key] for m in metrics_list]
            color = colors[idx % len(colors)]

            ax.plot(ks, values, marker="o", linewidth=2, markersize=6,
                    label=collection_name, color=color)

            for k_val, v in zip(ks, values):
                ax.annotate(f"{v:.1f}", xy=(k_val, v), xytext=(0, 8), textcoords="offset points", ha="center",
                            fontsize=8, color=color, )

        ax.set_xlabel("K", fontsize=11)
        ax.set_ylabel(f"{metric_label} (%)", fontsize=11)
        ax.set_xticks(ks)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="lower right")

        plt.tight_layout()
        save_path = save_basename / f"{file_name}"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
        plt.show()
        plt.close(fig)


def print_table(results_by_collection):
    metric_keys = ["precision", "recall", "f1", "hitrate_any", "hitrate_all"]

    for metric_key in metric_keys:
        print(f"\n{metric_key}:")
        for collection_name, metrics_list in results_by_collection.items():
            values = [m[metric_key] for m in metrics_list]

            text = f"\\texttt{{{collection_name.replace("_", "\\_")}}}"
            for val in values:
                text += f" & {val:.2f}"
            text += " \\\\"
            print(text)


if __name__ == "__main__":
    # 1. Union of K values
    K_All = sorted(set(K_Plot) | set(K_Table))

    # 2. Union of Collections (using 'id' as the unique identifier key)
    unique_collections = {c.id: c for c in COLLECTIONS_Plot + COLLECTIONS_Table}
    COLLECTIONS_All = list(unique_collections.values())

    with Logger(get_logs_path() / "embedding/evaluate.txt"):
        # Temporary nested structure to hold ALL evaluations: { collection_name: { k_value: metrics } }
        raw_evaluations = {}

        for db in COLLECTIONS_All:
            collection = db.get_collection()
            docs = filter_docs(collection.get(include=["documents", "metadatas"]))

            print(f"\n{db.name}:")
            # Single pass: fetch at max(K_All), slice for each k
            raw_evaluations[db.name] = evaluate_all_ks(docs, K_All, collection)

        # 3. Filter down for Plot (only selected collections, only K_Plot values)
        plot_results_filtered = {
            c.name: [raw_evaluations[c.name][k] for k in K_Plot]
            for c in COLLECTIONS_Plot
        }

        # 4. Filter down for Table (only selected collections, only K_Table values)
        table_results_filtered = {
            c.name: [raw_evaluations[c.name][k] for k in K_Table]
            for c in COLLECTIONS_Table
        }

        # 5. Output executions
        plot_results(plot_results_filtered, K_Plot)
        print_table(table_results_filtered)
