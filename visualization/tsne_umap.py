import os
from typing import Callable, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import to_rgba

from joblib import Parallel, delayed
from sklearn.manifold import TSNE
import umap

plt.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.sans-serif": "Libertine",
})

plt.rc('text.latex', preamble=r'\usepackage{amsmath, bm, mathrsfs, mathtools, color}')

# ----------------------------------------------------------------------
# Data utilities
# ----------------------------------------------------------------------

def normalize_replace_zeros(matrix: np.ndarray) -> np.ndarray:
    """
    Row-normalize a 2D matrix. If a row sums to zero, replace it
    by a uniform distribution over columns.
    """
    matrix = np.asarray(matrix)
    row_sums = matrix.sum(axis=1, keepdims=True)
    n_cols = matrix.shape[1]

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(matrix, row_sums, where=row_sums != 0)

    zero_sum_rows = (row_sums == 0).flatten()
    normalized[zero_sum_rows] = 1.0 / n_cols
    return normalized


def load_matched_df(
    pickle_path: str,
    perturb_percent: float = 0.0
) -> pd.DataFrame:
    """
    Load the sampled_matched_perturbed_df and:
      - filter by given perturb_percent
      - deduplicate russian users
    Returns a cleaned dataframe.
    """
    df = pd.read_pickle(pickle_path)
    df = df[df["perturb_percent"] == perturb_percent]

    mask_russian = df["russian"] == 1
    df_russian = df[mask_russian].drop_duplicates(subset="user", keep="first")
    df_other = df[~mask_russian]

    df_clean = pd.concat([df_russian, df_other], ignore_index=True)
    return df_clean


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Attach normalized traj_counts_perturbed, and return:
        X_norm   : array of normalized matrices
        X_raw    : array of raw matrices
        labels   : array of 0/1 labels (non-russian / russian)
    """
    df = df.copy()
    df["traj_counts_perturbed_normalised"] = df["traj_counts_perturbed"].apply(
        normalize_replace_zeros
    )

    X_norm = df["traj_counts_perturbed_normalised"].values
    X_raw = df["traj_counts_perturbed"].values
    labels = df["russian"].values.astype(int)

    return X_norm, X_raw, labels


# ----------------------------------------------------------------------
# Distance utilities
# ----------------------------------------------------------------------

def euclidean_dist(A: np.ndarray, B: np.ndarray) -> float:
    """Flatten then compute L2 distance between two matrices."""
    return np.linalg.norm(A - B)


def weighted_symmetric_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    p_counts: np.ndarray,
    q_counts: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """
    Weighted symmetric KL divergence between two probability matrices p, q
    with weights derived from corresponding count matrices p_counts, q_counts.
    """
    p_sum = np.sum(p_counts, axis=1)
    q_sum = np.sum(q_counts, axis=1)

    pw = p_sum / np.sum(p_sum)
    qw = q_sum / np.sum(q_sum)

    p = np.asarray(p)
    q = np.asarray(q)

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    p_kl = p * np.log(p / q)
    q_kl = q * np.log(q / p)

    kl_div_pq = pw * np.sum(p_kl, axis=1)
    kl_div_qp = qw * np.sum(q_kl, axis=1)

    symmetric_kl = (np.sum(kl_div_pq) + np.sum(kl_div_qp)) / 2.0
    return symmetric_kl


def build_distance_matrix(
    n: int,
    pairwise_fn: Callable[[int, int], float],
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Generic parallel distance-matrix builder.

    pairwise_fn(i, j) should return the distance between objects i and j.
    """
    def compute_row(i: int) -> Tuple[int, np.ndarray]:
        row = np.zeros(n)
        for j in range(i + 1, n):
            row[j] = pairwise_fn(i, j)
        return i, row

    dist_matrix = np.zeros((n, n))

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(compute_row)(i) for i in range(n)
    )

    for i, row in results:
        dist_matrix[i, i + 1:] = row[i + 1:]
        dist_matrix[i + 1:, i] = row[i + 1:]

    return dist_matrix


def build_euclidean_distance_matrix(X: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    """Distance matrix using Euclidean distance on matrices in X."""
    n = len(X)

    def pairwise_fn(i: int, j: int) -> float:
        return euclidean_dist(X[i], X[j])

    return build_distance_matrix(n, pairwise_fn, n_jobs=n_jobs)


def build_swkl_distance_matrix(
    X_prob: np.ndarray,
    X_counts: np.ndarray,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Distance matrix using the weighted symmetric KL divergence
    between probability matrices X_prob and count matrices X_counts.
    """
    n = len(X_prob)

    def pairwise_fn(i: int, j: int) -> float:
        return weighted_symmetric_kl_divergence(
            X_prob[i], X_prob[j], X_counts[i], X_counts[j]
        )

    return build_distance_matrix(n, pairwise_fn, n_jobs=n_jobs)


# ----------------------------------------------------------------------
# Embedding & plotting
# ----------------------------------------------------------------------

def run_tsne(dist_matrix: np.ndarray, random_state: int = 42) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        random_state=random_state,
    )
    return tsne.fit_transform(dist_matrix)


def run_umap(dist_matrix: np.ndarray, random_state: int = 42) -> np.ndarray:
    umap_model = umap.UMAP(
        n_components=2,
        metric="precomputed",
        random_state=random_state,
    )
    return umap_model.fit_transform(dist_matrix)

def plot_embedding(
    X_embedded: np.ndarray,
    labels: np.ndarray,
    save_name: str,
    save_folder="figures/",
    label_names=("Organics", "Trolls"),
    label_colors=('#006DFF', '#FF6B6B'),
    alpha: float = 0.3,
    s: int = 32,
) -> None:
    """
    Scatter plot with opaque edges (alpha=1) and transparent fill.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define alphas for each group (fill transparency)
    fill_alphas = (0.22, 0.22)  # Organics: 0.3, Trolls: 0.3
    
    for label_idx, (label, color) in enumerate(zip(label_names, label_colors)):
        mask = labels == label_idx
        if mask.any():
            # Create RGBA colors with different alpha values
            # Fill: color with transparency
            fill_color_rgba = to_rgba(color, alpha=fill_alphas[label_idx])
            
            # Edge: same color but FULLY OPAQUE (alpha=1.0)
            edge_color_rgba = to_rgba(color, alpha=1.0)
            
            # Size
            group_size = s if label_idx == 0 else int(s * 1.2)
            
            # Create a list of colors for all points in this group
            n_points = mask.sum()
            fill_colors = [fill_color_rgba] * n_points
            
            ax.scatter(
                X_embedded[mask, 0], 
                X_embedded[mask, 1], 
                c=fill_colors,           # Fill with transparency
                s=group_size, 
                edgecolors=edge_color_rgba,  # Edges fully opaque
                linewidth=0.4,           # Slightly thicker for visibility
                label=label,
                zorder=2
            )
    
    legend = ax.legend(
        frameon=True,
        framealpha=1.0,
        edgecolor='black',
        facecolor='white',
        loc='upper right',
        fontsize=11,
        handletextpad=0.5,
        borderpad=0.5,
        labelspacing=0.5
    )
    legend.get_frame().set_linewidth(0.5)
    
    ax.set_xlabel(r"\textbf{Dimension 1}", fontsize=12)
    ax.set_ylabel(r"\textbf{Dimension 2}", fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.25, zorder=1)

    ax.set_facecolor('#f8f9fa')
    
    # Set nice limits with 5% padding
    x_min, x_max = X_embedded[:, 0].min(), X_embedded[:, 0].max()
    y_min, y_max = X_embedded[:, 1].min(), X_embedded[:, 1].max()
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Equal aspect for proper representation
    ax.set_aspect('equal', adjustable='box')
    
    os.makedirs(save_folder, exist_ok=True)
    save_name = os.path.join(save_folder, f"{save_name}.pdf")
    fig.savefig(save_name, bbox_inches="tight")
    plt.close(fig)

def run_embedding_pipeline(
    dist_matrix: np.ndarray,
    labels: np.ndarray,
    base_title: str,
) -> None:
    """
    Given a distance matrix and labels, run t-SNE and UMAP and plot both.
    """
    # t-SNE
    tsne_embedding = run_tsne(dist_matrix)
    plot_embedding(tsne_embedding, labels, save_name=f"t-sne-{base_title}")

    # UMAP
    umap_embedding = run_umap(dist_matrix)
    plot_embedding(umap_embedding, labels, save_name=f"umap-{base_title}")

# ----------------------------------------------------------------------
# Main experiment logic
# ----------------------------------------------------------------------

def main():
    pickle_path = "../../../io-detection-reddit/data-analysis/sampled_matched_perturbed_df_final.pkl"

    # 1) Load and prepare data
    df = load_matched_df(pickle_path, perturb_percent=0.0)
    X_norm, X_raw, labels = prepare_features(df)
    X_counts = df["traj_counts_perturbed"].values  # for SWKL

    # 2) Euclidean on normalized trajectories
    dist_euclid_norm = build_euclidean_distance_matrix(X_norm, n_jobs=-1)
    run_embedding_pipeline(
        dist_euclid_norm,
        labels,
        base_title="euclid-dist-norm-traj-counts",
    )

    # 3) SWKL on normalized probabilities with counts
    dist_swkl = build_swkl_distance_matrix(X_norm, X_counts, n_jobs=-1)
    run_embedding_pipeline(
        dist_swkl,
        labels,
        base_title="swkl-dist-norm-traj-counts",
    )


if __name__ == "__main__":
    main()
