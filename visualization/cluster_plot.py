import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D

from cluster_utils import plot_gap_statistic, plot_silhouette_score, plot_elbow_method
from cluster_utils import calc_gap_statistic, calc_silhouette_scores, calc_elbow_inertia

plt.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.sans-serif": "Libertine",
})

plt.rc('text.latex', preamble=r'\usepackage{amsmath, bm, mathrsfs, mathtools, color}')

save_folder = 'figures/'
os.makedirs(save_folder, exist_ok=True)

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
    "../data-analysis/sampled_matched_perturbed_df_w_gail_opt.pkl"
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

k_clusters = range(1, 11)
gap_values, optimal_k = calc_gap_statistic(X=troll_policies, k_clusters=k_clusters)
sil_scores = calc_silhouette_scores(X=troll_policies, k_clusters=range(2, 11))
inertia = calc_elbow_inertia(X=troll_policies, k_clusters=k_clusters)

plot_gap_statistic(gap_values=gap_values, k_clusters=k_clusters, optimal_k=optimal_k)
plot_silhouette_score(silhouette_scores=sil_scores, k_clusters=range(2, 11))
plot_elbow_method(elbow_inertia=inertia, k_clusters=k_clusters)

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
klabels = kmeans.fit_predict(troll_policies)   # cluster trolls ONLY (cleanest)
trolls_df["cluster"] = klabels

# Means per cluster (already in new_order columns)
cluster_means = []
cluster_sizes = []
for c in range(n_clusters):
    M = troll_policies[klabels == c]
    if M.size == 0:
        cluster_means.append(np.zeros(6))
        cluster_sizes.append(0)
    else:
        cluster_means.append(M.mean(axis=0))
        cluster_sizes.append(M.shape[0])

cluster_means = np.vstack(cluster_means)   # (n_clusters, 6)

# Organics mean (same order)
non_troll_mean = non_troll_policies.mean(axis=0)  # (6,)

name_map = {
    "New thread": "New thread",
    "Root comment": "Root comment",
    "Reply comment (agree)": "Reply (agree)",
    "Reply comment (neut)": "Reply (neut)",
    "Reply comment (disagree)": "Reply (disagree)",
    "Wait reply": "Wait reply",
}

def vec_to_named_dict(vec, order_names, name_map, renorm=True):
    v = np.asarray(vec, float)
    if renorm:
        s = v.sum()
        if s > 0:
            v = v / s
    return {name_map[k]: float(x) for k, x in zip(order_names, v)}

cards = []
# Organics first
cards.append((
    r"\textbf{Sample of Organics}",
    int(non_troll_policies.shape[0]),
    vec_to_named_dict(non_troll_mean, new_order, name_map, renorm=True)
))

display_order = [2, 1, 0]  # swap (1↔3) in display: 2,1,0 (labels are 0-based)

# Troll clusters in display order
for rank, lbl in enumerate(display_order, start=1):
    comp = vec_to_named_dict(cluster_means[lbl], new_order, name_map, renorm=True)   # ✅ use lbl
    cards.append((
        fr"\textbf{{Troll Cluster {rank}}}",
        int((klabels == lbl).sum()),                                                 # ✅ use lbl
        comp
    ))

# Build dataframe for plotting from cards
rows = []
for name, n, comp in cards:
    for cat, v in comp.items():
        rows.append({"cluster": name, "n": n, "category": cat, "value": v})
df = pd.DataFrame(rows)

num_clusters = len(cards)
quarter_span = 2 * np.pi / num_clusters

cat_order = [name_map[a] for a in new_order]

cat_to_color = {
    "New thread":       "#4e79a7",  # steel blue
    "Root comment":     "#d55e00",  # muted burnt orange
    "Reply (agree)":    "#009e73",  # muted green-teal
    "Reply (neut)":     "#8073ac",  # soft muted violet
    "Reply (disagree)": "#999999",  # elegant neutral grey
    "Wait reply":       "#cc79a7",  # muted pink-magenta
}

df["category"] = pd.Categorical(df["category"], categories=cat_order, ordered=True)

# Top Right, Bottom Right, Bottom Left, Top Left
corner_xy = [(0.8, 0.85), (0.8, 0.25), (-0.3, 0.25), (-0.3, 0.85)]

# Annular ring (constant radii)
inner_r = 0.5
outer_r = 3.0
thickness = outer_r - inner_r

# Choose a *linear* gap width (same at inner & outer). Tune this:
gap_linear = 0.05  # in radius units (same units as inner_r/outer_r)

# Outer radial bars
bar_base = outer_r + 0.10
bars_max_len = 4.0
bar_pad = 0.12

def add_annular_slice_equal_gap(ax, theta0, theta1, r0, r1, gap_width, facecolor, edgecolor='white', lw=1.0, n=64, zorder=2):
    """
    Draw an annular slice between angles [theta0, theta1] and radii [r0, r1],
    using *linear* gap width `gap_width` at both edges (so inner/outer gaps look equal).
    Angles in radians, r in data units. Works on a polar Axes (theta,r) via ax.transData.
    """
    if r1 <= r0:  # nothing to draw
        return

    # half-gap angles at inner/outer edges
    d_out = gap_width / (2.0 * r1)
    d_in  = gap_width / (2.0 * r0)

    th_out0 = theta0 + d_out
    th_out1 = theta1 - d_out
    th_in1  = theta1 - d_in
    th_in0  = theta0 + d_in

    if th_out1 <= th_out0 or th_in1 <= th_in0:  # gap too large for this slice
        return

    # build outer arc (forward) and inner arc (reverse)
    th_out = np.linspace(th_out0, th_out1, n)
    th_in  = np.linspace(th_in1,  th_in0,  n)

    # vertices in polar data coords: (theta, r)
    thetas = np.r_[th_out, th_in]
    radii  = np.r_[np.full_like(th_out, r1), np.full_like(th_in,  r0)]
    verts  = np.column_stack([thetas, radii])

    # path codes: MOVETO then LINETOs
    codes = np.r_[ [Path.MOVETO], np.full(len(verts)-1, Path.LINETO) ]

    patch = PathPatch(Path(verts, codes),
                      transform=ax.transData,  # interpret as (theta, r)
                      facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=zorder)
    ax.add_patch(patch)

# ----- Figure (polar) -----
fig = plt.figure(figsize=(8, 4))
ax = plt.subplot(111, polar=True)
ax.set_theta_zero_location('N')   # 0° at top
ax.set_theta_direction(-1)        # clockwise
ax.set_yticklabels([])
# ax.set_ylim(0, bar_base + bars_max_len + 0.3)

# ----- Draw ring quarters (stacked, equal-chord gaps) -----
for i, (cluster_name, n, _) in enumerate(cards):
    # data for this quarter, sorted for consistent stacking order
    sub = df[df["cluster"] == cluster_name].sort_values("category")

    theta0 = i * quarter_span
    theta1 = (i + 1) * quarter_span
    theta_center = 0.5 * (theta0 + theta1)

    # normalize values within the quarter (so they fill the annulus)
    vals = np.clip(sub["value"].to_numpy(dtype=float), 0, None)
    s = vals.sum()
    if s <= 0:
        continue
    fracs = vals / s

    # stack radially from inner_r to outer_r
    r_bottom = inner_r
    for frac, (_, row) in zip(fracs, sub.iterrows()):
        r_top = r_bottom + frac * (outer_r - inner_r)

        # draw one annular slice with equal-chord gap at this slice's inner/outer radii
        add_annular_slice_equal_gap(
            ax,
            theta0, theta1,
            r_bottom, r_top,
            gap_width=gap_linear,
            facecolor=cat_to_color[row["category"]],
            edgecolor='white',
            lw=1.0,
            n=96,
            zorder=2
        )
        r_bottom = r_top

    x, y = corner_xy[i]

    ax.text(
        x, y, f"{cluster_name}",
        transform=ax.transAxes, ha='left', va='top', fontsize=14,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9),
        zorder=5
    )

    ax.text(
        x, y - 0.08, f"($n={n}$)",
        transform=ax.transAxes, ha='left', va='top', fontsize=14,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9),
        zorder=5
    )

q_stats = {}  # (cluster_name, category) -> (q05,q25,q50,q75,q95)

for i, (cluster_name, n, _) in enumerate(cards):
    if i == 0:
        M = non_troll_policies  # columns already in new_order
    else:
        label_id = display_order[i - 1]            # ✅ map display slot to KMeans label
        M = troll_policies[klabels == label_id]

    if M.size == 0:
        for cat_short in cat_order:
            q_stats[(cluster_name, cat_short)] = (np.nan,)*5
        continue

    qs = np.percentile(M, [5, 25, 50, 75, 95], axis=0)  # (5, 6)
    for j, cat_short in enumerate(cat_order):
        q_stats[(cluster_name, cat_short)] = tuple(map(float, qs[:, j]))


print("\n=== Action-share quantiles by cluster (Median, 5th, 95th) ===")
for cluster_name, n, _ in cards:
    print(f"\n--- {cluster_name} (n={n}) ---")
    for cat in cat_order:
        q05, q25, q50, q75, q95 = q_stats[(cluster_name, cat)]
        if not np.isfinite(q50):
            print(f"{cat:24s}  Median:   NA     5th:   NA     95th:   NA")
            continue
        # values are probabilities (0..1); print as percentages
        print(f"{cat:24s}  Median: {q50*100:5.1f}%   5th: {q05*100:5.1f}%   95th: {q95*100:5.1f}%")

# ----- Add outer radial bars per quadrant -----
for i, (cluster_name, n, _) in enumerate(cards):
    sub = df[df["cluster"] == cluster_name].sort_values("category")
    K = len(sub)

    theta_start = i*quarter_span + gap_linear/2
    theta_end   = (i+1)*quarter_span - gap_linear/2
    usable_span = theta_end - theta_start

    bar_width = usable_span / K * (1 - bar_pad)
    centers = theta_start + (np.arange(K) + 0.5) * (usable_span / K)

    heights = np.array([q_stats[(cluster_name, cat)][2]        # ✅ q50/median
                    for cat in sub["category"]]) * bars_max_len
    ax.bar(
        centers, heights, width=bar_width, bottom=bar_base, align='center',
        color=[cat_to_color[c] for c in sub["category"]],
        edgecolor='white', linewidth=0.8
    )

    # Optional small labels above bars (comment out if cluttered)
    # for theta, h, cat in zip(centers, heights, sub["category"]):
    #     if h > 0.06 * bars_max_len:  # skip tiny ones
    #         ax.text(theta, bar_base + h + 0.03, str(cat),
    #                 rotation=np.degrees(-theta) + 90, ha='center', va='center', fontsize=8)

     # ---- Radial boxplot overlay (median, IQR, 5–95 whiskers) ----
    box_w = bar_width * 0.55     # angular width of the box (narrower than bars)
    cap_w = bar_width * 0.35     # width of whisker caps

    for theta, cat_short in zip(centers, sub["category"]):
        q05, q25, q50, q75, q95 = q_stats[(cluster_name, cat_short)]
        if not np.isfinite(q50):
            continue

        # map quantiles (0..1) to radial coords, same scale as bars
        r05 = bar_base + q05 * bars_max_len
        r25 = bar_base + q25 * bars_max_len
        r50 = bar_base + q50 * bars_max_len
        r75 = bar_base + q75 * bars_max_len
        r95 = bar_base + q95 * bars_max_len

        # IQR box as a rectangle in (theta, r)
        verts = [
            (theta - box_w/2, r25),
            (theta + box_w/2, r25),
            (theta + box_w/2, r75),
            (theta - box_w/2, r75),
            (theta - box_w/2, r25),
        ]
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]  # CLOSEPOLY closes shape
        # NB: correct code list (typo above): 
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]

        patch = PathPatch(
            Path(verts, codes),
            transform=ax.transData,          # interpret as polar data (theta, r)
            facecolor="white",               # or "none" to only show outline
            edgecolor="black",
            linewidth=0.9,
            zorder=6,
            alpha=0.9,
            clip_on=False
        )
        ax.add_patch(patch)

        # median line
        ax.plot([theta - box_w/2, theta + box_w/2], [r50, r50],
                color="black", lw=1.2, zorder=7, solid_capstyle='butt')

        # whiskers
        ax.plot([theta, theta], [r05, r25], color="black", lw=0.9, zorder=6)
        ax.plot([theta, theta], [r75, r95], color="black", lw=0.9, zorder=6)

        # whisker caps
        ax.plot([theta - cap_w/2, theta + cap_w/2], [r05, r05], color="black", lw=0.9, zorder=6)
        ax.plot([theta - cap_w/2, theta + cap_w/2], [r95, r95], color="black", lw=0.9, zorder=6)

# Legend
proxies = [Line2D([0],[0], marker='s', linestyle='None', markersize=10,
                  markerfacecolor=cat_to_color[c], markeredgecolor='none')
           for c in cat_order]

legend = ax.legend(proxies, cat_order, ncol=1, frameon=False, fontsize=14, title_fontsize=16,
          loc='lower center', bbox_to_anchor=(1.8, 0.2), title=r"\textbf{Actions}",
          handletextpad=0.05, labelspacing=0.2, columnspacing=0.4, alignment='left')
legend.get_title().set_position((9, 0))


ax.spines['polar'].set_visible(False)

# Cosmetics
ax.set_ylim(0, bar_base + bars_max_len)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(False)

save_name = os.path.join(save_folder, 'clusters.pdf')
# Show the plot
plt.savefig(save_name, bbox_inches='tight', pad_inches=0.02)
# plt.show()
plt.close(fig)