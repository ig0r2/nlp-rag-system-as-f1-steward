from typing import Literal

import numpy as np
import umap
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from utils.path import get_results_path
from utils.server import COLLECTIONS

# COLLECTION = COLLECTIONS["minilm"]
# COLLECTION = COLLECTIONS["gemma"]
# COLLECTION = COLLECTIONS["qwen3-4B"]
# COLLECTION = COLLECTIONS["bge-large"]
# COLLECTION = COLLECTIONS["mxbai"]
# COLLECTION = COLLECTIONS["nomic-v1.5"]
COLLECTION = COLLECTIONS["nomic-v1.5_Q8"]
# COLLECTION = COLLECTIONS["nomic-v2"]

# EXCLUDE_SECTIONS = {"SR Article 34.7", "TR Article 4.1", "SR Article 40.9", "SR Article 28.2", "SR Article 29.2"}
EXCLUDE_SECTIONS = set()

YEARS = ['2022', '2023', '2024', '2025']

COLORS: Literal["article", "decision_category"] = "article"

####################################################################

save_path = get_results_path() / "plots"
save_path.mkdir(parents=True, exist_ok=True)
save_basename = f"{COLLECTION.name}{"_f" if EXCLUDE_SECTIONS else ""}"

data = COLLECTION.get_collection().get(include=["embeddings", "metadatas"])

embeddings = np.array(data["embeddings"])
metadatas = data["metadatas"]
total_raw = len(metadatas)

# Filter out excluded sections
keep_mask = [not EXCLUDE_SECTIONS.intersection(m.get("sections", [])) for m in metadatas]
embeddings = embeddings[keep_mask]
metadatas = [m for m, keep in zip(metadatas, keep_mask) if keep]


def first_section(meta):
    sections = meta.get("sections", [])
    return sections[0] if sections else "unknown"


def category(meta):
    return meta["decision_category"]


def run_plots(embeddings_subset: np.ndarray, metadatas_subset: list, out_dir, basename: str):
    """Fit UMAP and save all three plot variants to out_dir / basename_*.png."""
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(metadatas_subset)

    # Colors and labels
    color_labels = [
        first_section(m) if COLORS == "article" else category(m)
        for m in metadatas_subset
    ]
    unique_labels = sorted(set(color_labels))

    cmap = plt.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")
    section_to_color = {s: cmap(i % cmap.N) for i, s in enumerate(unique_labels)}
    point_colors = [section_to_color[s] for s in color_labels]

    text_labels = [m["fact"][:40] for m in metadatas_subset]

    # Reduce to 2D
    reducer_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=100)
    embeddings_2d = reducer_2d.fit_transform(embeddings_subset)

    # 2D Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        s=100, c=point_colors, edgecolors="white", linewidths=0.4, zorder=2,
    )

    excluded_str = "\nExcluded sections: " + ",".join(sorted(EXCLUDE_SECTIONS)) if EXCLUDE_SECTIONS else ""
    ax.set_title(f"n_docs={total}{excluded_str}", fontsize=11)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(out_dir / f"umap2d_{basename}_{total}_1.png", dpi=150)

    for i, label in enumerate(text_labels):
        ax.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=7, xytext=(5, 5), textcoords="offset points")
    plt.savefig(out_dir / f"umap2d_{basename}_{total}_2.png", dpi=150)

    # Legend
    legend_handles = [Patch(facecolor=section_to_color[s], label=s) for s in unique_labels]
    ax.legend(handles=legend_handles, title="Section", loc="best",
              fontsize=8, title_fontsize=9, framealpha=0.7)
    plt.savefig(out_dir / f"umap2d_{basename}_{total}_3.png", dpi=150)

    plt.show()
    plt.close(fig)


####################################################################
# 1. All documents
run_plots(embeddings, metadatas, save_path, save_basename)

# 2. Per-year subsets
for year in YEARS:
    year_mask = [m.get("year") == year for m in metadatas]
    year_embeddings = embeddings[np.array(year_mask)]
    year_metadatas = [m for m, keep in zip(metadatas, year_mask) if keep]

    if not year_metadatas:
        print(f"No documents found for year {year}, skipping.")
        continue

    year_out_dir = save_path / year
    year_basename = f"{save_basename}_{year}"
    run_plots(year_embeddings, year_metadatas, year_out_dir, year_basename)
