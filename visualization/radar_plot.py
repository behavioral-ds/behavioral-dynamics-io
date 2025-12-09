import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.sans-serif": "Libertine",
})

plt.rc('text.latex', preamble=r'\usepackage{amsmath, bm, mathrsfs, mathtools, color}')

save_folder = 'figures/'

# Original names coming from your data/vectors:
actions = [
    "Wait reply",
    "New thread",
    "Root comment",
    "Reply comment (agree)",
    "Reply comment (neut)",
    "Reply comment (disagree)"
]

# Canonical plotting order (what you want everywhere):
new_order = [
    "New thread",
    "Root comment",
    "Reply comment (agree)",
    "Reply comment (neut)",
    "Reply comment (disagree)",
    "Wait reply"
]

# Map "new_order" columns to positions in "actions"
order_idx = [actions.index(a) for a in new_order]

def reorder_cols(M, idx):
    """
    Reorder columns of a (n_samples, 6) matrix M to match new_order.
    """
    M = np.asarray(M, float)
    return M[:, idx]

sampled_matched_gail_df = pd.read_pickle(
    "../../../io-detection-reddit/data-analysis/sampled_matched_perturbed_df_w_gail_opt.pkl"
)

# Baselines
trolls_df = sampled_matched_gail_df[
    (sampled_matched_gail_df.russian == 1) &
    (sampled_matched_gail_df.run == 0) &
    (sampled_matched_gail_df.perturb_percent == 0.0)
].copy().reset_index(drop=True)

non_trolls_df = sampled_matched_gail_df[
    (sampled_matched_gail_df.russian == 0) &
    (sampled_matched_gail_df.run == 0) &
    (sampled_matched_gail_df.perturb_percent == 0.0)
].copy().reset_index(drop=True)

# Flatten each policy to mean over time (axis=0) -> length-6 vector
trolls_df["flattened_policies"] = [np.mean(p, axis=0) for p in trolls_df.policy.values]
non_trolls_df["flattened_policies"] = [np.mean(p, axis=0) for p in non_trolls_df.policy.values]

# Build matrices
troll_policies_raw = np.vstack(trolls_df.flattened_policies.values)      # shape: (Nt, 6)
non_troll_policies_raw = np.vstack(non_trolls_df.flattened_policies.values)  # (Nn, 6)

# Reorder columns to the canonical plotting order
troll_policies = reorder_cols(troll_policies_raw, order_idx)             # (Nt, 6)
non_troll_policies = reorder_cols(non_troll_policies_raw, order_idx)     # (Nn, 6)

def row_normalize(M):
    s = M.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return M / s

# before quantiles & before computing the cards’ values:
troll_policies = row_normalize(troll_policies)
non_troll_policies = row_normalize(non_troll_policies)

def plot_policy_radar_plot_multi(dfs: list, labels: list, save_name: str, colors: list, data_legend: str):
    """
    Plot multiple policy radar plots on the same chart.
    
    Parameters
    ----------
    dfs : list of pd.DataFrame or np.ndarray
        Each dataframe/array contains rows of policies with same number of actions.
    labels : list of str
        Labels corresponding to each dataframe.
    save_name : str
        Save file name of the plot.
    
    """
    actions = [
        "New thread",
        "Root comment",
        "Reply comment (agree)",
        "Reply comment (neut)",
        "Reply comment (disagree)",
        "Wait reply"
    ]

    action_labels = [
        r"\textbf{$CT$}",
        r"\textbf{$RC$}",
        r"\textbf{$PR_{+}$}",
        r"\textbf{$PR_{\sim}$}",
        r"\textbf{$PR_{-}$}",
        r"\textbf{$WR$}",
    ]

    num_vars = len(actions)

    def transform_values(arr):
        """Root transformation - makes small values appear larger"""
        return arr ** (1/2)

    def compute_stats(data):
        data = transform_values(data)
        return {
            "min": np.min(data, axis=0),
            "q1": np.percentile(data, 25, axis=0),
            "median": np.percentile(data, 50, axis=0),
            "mean": np.mean(data, axis=0),
            "q3": np.percentile(data, 75, axis=0),
            "max": np.max(data, axis=0),
        }

    # Angles for radar
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    def close_loop(arr):
        return np.concatenate([arr, [arr[0]]])

    # Precompute stats for each dataset
    stats_all = []
    for df in dfs:
        stats = compute_stats(df)
        for k in stats:
            stats[k] = close_loop(stats[k])
        stats_all.append(stats)

    # Plot
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))

    # Store handles for separate legends
    dataset_handles = []  # For dataset color legend
    stat_handles = []     # For statistical measure legend

    # Statistical legend entries (created once, not per dataset)
    stat_handles.append(Line2D([0], [0], color='gray', linewidth=2, label=r'Median'))
    stat_handles.append(Patch(facecolor='gray', alpha=0.15, label=r'IQR (25-75\%)'))
    # stat_handles.append(Patch(facecolor='gray', alpha=0.05, label='0-25% range'))

    # Plot each dataset
    for i, (stats, label, color) in enumerate(zip(stats_all, labels, colors)):
        # Plot with dataset colors
        fill_q1 = ax.fill(angles, stats["q1"], alpha=0.02, color=color)
        fill_iqr = ax.fill_between(angles, stats["q1"], stats["q3"], alpha=0.15, color=color)
        median_line, = ax.plot(angles, stats["median"], color=color, linewidth=2)
        # mean_line, = ax.plot(angles, stats["mean"], color=color, linewidth=2, linestyle=":")
        
        # Store dataset handle for legend (just the median line)
        dataset_handles.append(median_line)

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(action_labels, fontsize=12)

    # Add transformed scale tick labels
    original_ticks = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
    transformed_ticks = [transform_values(x) for x in original_ticks]
    ax.set_yticks(transformed_ticks)
    ax.set_yticklabels([f"{x:.1f}" for x in original_ticks], fontsize=9)
    ax.set_ylim(0, 1.0)
    
    # Add scale transformation note
    # plt.figtext(0.5, 0.02, r"Scale: $y' = \sqrt{y}$", 
    #             ha='center', fontsize=9, style='italic', color='gray')

    # Create two separate legends
    # Legend 1
    if 'Accounts' not in data_legend:  
        legend1 = ax.legend(handles=stat_handles, 
                        loc='upper right',
                        frameon=True,
                        framealpha=1.0,
                        edgecolor='black',
                        handlelength=1.0,
                        bbox_to_anchor=(1.5, 1.05),
                        title=r"\textbf{Action Distribution}",
                        fontsize=11,
                        title_fontsize=11)
        legend1.get_frame().set_linewidth(0.75)
        
        # Add the first legend to the axes
        ax.add_artist(legend1)
    
    # Legend 2
    legend2 = ax.legend(handles=dataset_handles,
                       labels=labels,
                       frameon=True,
                       framealpha=1.0,
                       edgecolor='black',
                       loc='upper right',
                       handlelength=1.0,
                       bbox_to_anchor=(1.5, 0.3),
                       title=data_legend,
                       fontsize=11,
                       title_fontsize=11)
    legend2.get_frame().set_linewidth(0.75)

    fig.savefig(f"figures/{save_name}.pdf", bbox_inches='tight')

class_colors = ['#006DFF', '#FF6B6B']
plot_policy_radar_plot_multi(
    [non_troll_policies, troll_policies],
    ["Organics", "Trolls"],
    "radar-gail-classes", 
    colors=class_colors,
    data_legend=r"\textbf{User Groups}"
)

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
klabels = kmeans.fit_predict(troll_policies)   # cluster trolls ONLY (cleanest)

# Reorder cluster labels so that the largest cluster becomes label 0
_, counts = np.unique(klabels, return_counts=True)
# Get the sorted indices from largest to smallest cluster
sorted_indices = np.argsort(-counts)  # Negative for descending order

# Create a mapping from old labels to new labels
label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}

# Apply the mapping to reorder labels
klabels = np.array([label_mapping[label] for label in klabels])

trolls_df["cluster"] = klabels

_, cluster_counts = np.unique(klabels, return_counts=True)

cluster_dfs = []

for c in range(n_clusters):
    print("cluster", c, cluster_counts[c])
    flattened_policies = trolls_df[trolls_df.cluster == c].flattened_policies.values
    flattened_policies = np.vstack(flattened_policies)
    flattened_policies = reorder_cols(flattened_policies, order_idx)
    flattened_policies = row_normalize(flattened_policies)
    cluster_dfs.append(flattened_policies)

troll_cluster_colors = [
    "#FF6B6B",  # Your red (keep middle cluster)
    "#FF8E00",  # Orange (distinct but warm)
    "#9C27B0",  # Purple (cool contrast)
]

plot_policy_radar_plot_multi(
    cluster_dfs,
    ["Cluster 1", "Cluster 2", "Cluster 3"],
    save_name="radar-gail-troll-clusters",
    colors=troll_cluster_colors,
    data_legend=r"\textbf{Troll Clusters}"
)

evaders_list = ["petouchoque", "TojatMalaron", "xameg"]
evaders_labels = ['A', 'B', 'C']

evadors_colors = ["#E41A1C", "#1B9E77", "#17BECF"]

evaders_policy_list = []

for e in evaders_list:
    flattened_policies = trolls_df[trolls_df.user.isin([e])].flattened_policies.values
    flattened_policies = np.vstack(flattened_policies)
    flattened_policies = reorder_cols(flattened_policies, order_idx)
    flattened_policies = row_normalize(flattened_policies)
    evaders_policy_list.append(flattened_policies)

plot_policy_radar_plot_multi(
    evaders_policy_list,
    evaders_labels,
    save_name="radar-gail-evaders",
    colors=evadors_colors,
    data_legend=r"\textbf{Evader Accounts}"
)