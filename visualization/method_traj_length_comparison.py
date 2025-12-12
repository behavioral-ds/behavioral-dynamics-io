import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.sans-serif": "Libertine",
})

plt.rc('text.latex', preamble=r'\usepackage{amsmath, bm, mathrsfs, mathtools, color}')

save_folder = 'figures/'
# ---------- load ----------
experiment_output_dir = "../final-experiment-results/first_n_classification/"

state_count_results = pd.read_pickle(os.path.join(experiment_output_dir, "emp_policies_results.pkl"))
gail_policies_results = pd.read_pickle(os.path.join(experiment_output_dir, "gail_policies_results.pkl"))
irl_policies_results = pd.read_pickle(os.path.join(experiment_output_dir, "irl_policies_results.pkl"))
embed_results = pd.read_pickle(os.path.join(experiment_output_dir, "embed_results.pkl"))

def to_dict(lst): return {rec["n"]: rec for rec in lst}

D = {
    "state_rf": to_dict(state_count_results),
    "embed_rf": to_dict(embed_results),
    "gail_rf":  to_dict(gail_policies_results),
    "irl_rf":   to_dict(irl_policies_results),
}

# Which N levels to show as ridgeline rows
N_LIST = [3, 5, 10, 20, 50, -1]  # -1 == "Full"
Y_LABELS = [str(n) if n != -1 else "Full" for n in N_LIST]

# Build per-run arrays
DATA = {
    m: [np.asarray(D[m][n]["f1_list_rf"], dtype=float) * 100.0 for n in N_LIST]
    for m in D
}

print("\n=== F1 Summary Statistics by N (Median, 5th, 95th) ===")
methods = ["state_rf", "gail_rf", "irl_rf", "embed_rf"]

for idx, n in enumerate(N_LIST):
    n_label = "Full" if n == -1 else str(n)
    print(f"\n--- N = {n_label} ---")
    for m in methods:
        vals = np.asarray(DATA[m][idx], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            # nothing recorded for this (method, N); skip or mark as NA
            print(f"{m:30s}  Median:   NA    5th:   NA    95th:   NA")
            continue

        q50 = float(np.nanmedian(vals))
        q05 = float(np.nanpercentile(vals, 5))
        q95 = float(np.nanpercentile(vals, 95))

        print(f"{m:30s}  Median: {q50:.1f}   5th: {q05:.1f}   95th: {q95:.1f}")

# --------------------
# Simple 1D Gaussian KDE (no external deps)
# --------------------
def silverman_bandwidth(x):
    x = np.asarray(x)
    n = len(x)
    if n < 2: return 0.02  # fallback
    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(std, iqr / 1.349) if iqr > 0 else std
    return 0.9 * sigma * n**(-1/5) or 0.02

def kde_gaussian(x, grid, h=None):
    x = np.asarray(x)
    if h is None: h = silverman_bandwidth(x)
    h = max(h, 1e-3)  # guardrail
    z = (grid[:, None] - x[None, :]) / h
    phi = np.exp(-0.5 * z**2) / np.sqrt(2*np.pi)
    dens = phi.mean(axis=1) / h
    return dens

methods = ["state_rf", "gail_rf", "irl_rf", "embed_rf"]  # column order

display_name = {
    "state_rf": r"\textbf{Empirical}",
    "gail_rf":  r"\textbf{GAIL}",
    "irl_rf":   r"\textbf{MaxEnt Deep IRL}",
    "embed_rf": r"\textbf{Embedding}",
}

palette = {
    r"\textbf{Empirical}":       "#3b7a75",
    r"\textbf{GAIL}":            "#2a9d8f",
    r"\textbf{MaxEnt Deep IRL}": "#264653",
    r"\textbf{Embedding}":       "#d4a373",
}

marker_map = {
    r"\textbf{Empirical}": "o",
    r"\textbf{GAIL}": "s",
    r"\textbf{MaxEnt Deep IRL}": "D",
    r"\textbf{Embedding}": "^",
}

colors    = {m: palette[display_name[m]] for m in methods}
title_map = {m: display_name[m]         for m in methods}

# --------------------
# Common x-limits and grid
# --------------------
pooled = np.concatenate([np.asarray(v).ravel() for m in methods for v in DATA[m] if len(v)>0])
x_min = 60
x_max = 100
x_grid = np.linspace(x_min, x_max, 600)

# Vertical geometry
row_gap = 0.7
amp     = 0.4

def method_xlim(method_values, wide_lo=60, tight_lo=80, hi=100, thresh=80):
    pooled_m = np.concatenate([np.asarray(v).ravel() for v in method_values if len(v) > 0])
    if pooled_m.size == 0:
        return (tight_lo, hi)
    lo = np.nanpercentile(pooled_m, 1)  # robust min
    return (wide_lo, hi) if lo < thresh else (tight_lo, hi)

# --------------------
# Plot (no sharex so each column can choose its own xlim)
# --------------------
fig, axes = plt.subplots(1, 4, figsize=(8, 4), sharex=False, sharey=True)
fig.subplots_adjust(wspace=0.01)

cap_h   = 0.1                # whisker cap height in y-units
iqr_lw  = 4.2                # thickness of IQR bar
whisk_lw= 1.2                # thickness of q5–q95 whisker
whisk_c = '#0f0f0f'        # neutral whisker color

for c, m in enumerate(methods):
    ax   = axes[c]
    col  = colors[m]
    name = title_map[m]

    # decide x-limits per column
    x_lo, x_hi = method_xlim(DATA[m], wide_lo=60, tight_lo=80, hi=100, thresh=80)
    x_grid = np.linspace(x_lo, x_hi, 600)

    max_dens_overall = 1e-12
    dens_rows = []

    # precompute KDEs with boundary reflection
    for r, vals in enumerate(DATA[m]):
        vals = np.asarray(vals)
        if vals.size < 2:
            dens_rows.append(None)
            continue
        h = max(silverman_bandwidth(vals) * 1.2, 1e-3)
        lo_b, hi_b = 0.0, 100
        left  = vals[vals - lo_b < 3*h]
        right = vals[hi_b - vals < 3*h]
        x_aug = np.concatenate([vals, 2*lo_b - left, 2*hi_b - right])
        dens  = kde_gaussian(x_aug, x_grid, h=h)
        dens_rows.append(dens)
        max_dens_overall = max(max_dens_overall, dens.max())

    # draw rows (top→bottom)
    for r in reversed(range(len(N_LIST))):
        vals = np.asarray(DATA[m][r])
        base_y = r * row_gap
        if vals.size == 0 or dens_rows[r] is None:
            continue

        # KDE half-violin
        dens = dens_rows[r] / max_dens_overall
        y = base_y + amp * dens
        ax.fill_between(x_grid, base_y, y, color=col, alpha=0.20, linewidth=0)
        if m == 'embed_rf':
            ax.fill_between(
                x_grid, base_y, y,
                facecolor='none',
                edgecolor='black',
                hatch='//',
                alpha=1.0,
                linewidth=0.1
            )

        ax.plot(x_grid, y, color=col, lw=1.0, alpha=0.9)

        # ---- Whiskers + IQR + median (no jitter) ----
        vals_clean = vals[np.isfinite(vals)]
        if vals_clean.size:
            q05 = float(np.nanpercentile(vals_clean, 5))
            q25 = float(np.nanpercentile(vals_clean, 25))
            q50 = float(np.nanmedian(vals_clean))
            q75 = float(np.nanpercentile(vals_clean, 75))
            q95 = float(np.nanpercentile(vals_clean, 95))

            # outer whisker q5–q95 (thin)
            ax.hlines(base_y, q05, q95, color=whisk_c, lw=whisk_lw, alpha=0.9, zorder=3)
            # small vertical caps
            ax.vlines([q05, q95], base_y - cap_h/2, base_y + cap_h/2,
                      color=whisk_c, lw=whisk_lw, alpha=0.9, zorder=3)

            # IQR bar q25–q75 (thicker, in method color)
            ax.hlines(base_y, q25, q75, color=col, lw=iqr_lw, alpha=0.95, zorder=4)

            # median marker
            mk = marker_map[name]
            ax.scatter([q50], [base_y], s=45, marker=mk, color=col,
                       edgecolors='white', linewidths=0.6, zorder=5)
            
            # Add text
            ax.text(
                q50,                     # ~0.8% offset to the right
                base_y -0.3,
                fr'$\tilde{{x}}\!=\!{q50:.1f}$', # median with percent sign
                fontsize=11,
                va='bottom', ha='left',
                color='k',
                clip_on=False,             # ✅ allows it to spill over next subplot
                in_layout=False,
                zorder=6
            )

    # cosmetics
    # ax.set_title(name, fontsize=10, pad=6, color=col)
    ax.grid(True, axis='x', alpha=0.18)
    ax.set_xlim(x_lo, x_hi)

    # y-axis only on first column
    if c == 0:
        ax.spines['left'].set_visible(True)
        ax.tick_params(axis='y', labelsize=12, length=0)
    else:
        ax.spines['left'].set_visible(False)
        ax.set_yticklabels([])

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

# y ticks/labels + margins
axes[0].set_yticks([i*row_gap for i in range(len(N_LIST))])
axes[0].set_yticklabels(Y_LABELS, fontsize=12)
axes[0].set_ylim(-0.5*row_gap, row_gap*(len(N_LIST)-1) + 0.85*row_gap)

# reference at F1=90 for every column
for ax in axes:
    ax.axvline(90, ls='--', lw=0.8, color='0.45', alpha=0.3)
    ax.tick_params(axis='x', labelsize=12)

# labels
axes[0].set_ylabel(r"\textbf{Observed Steps ($N$)}", fontsize=14)
fig.supxlabel(r'$\boldsymbol{F_1}$\textbf{--Scores (\%)}', fontsize=14, y=0.04)

# Custom positions
x_offsets = [0.135, 0.375, 0.54, 0.81]
y_position = 0.95

for i, method in enumerate(methods):
    x_pos = x_offsets[i]
    name = title_map[method]
    
    fig.add_artist(Line2D(
        [x_pos], [y_position],
        marker=marker_map[name],
        color='none',
        markerfacecolor=colors[method],
        markeredgecolor='white',
        markeredgewidth=0.6,
        markersize=10,
        transform=fig.transFigure,
        linestyle='None',
        figure=fig
    ))
    
    fig.text(x_pos + 0.02, y_position - 0.005, name,
             fontsize=12, transform=fig.transFigure,
             verticalalignment='center', horizontalalignment='left')

plt.tight_layout(rect=(0,0,1,0.96))
os.makedirs(save_folder, exist_ok=True)
save_name = os.path.join(save_folder, 'method-traj-length-comparison.pdf')
plt.savefig(save_name, bbox_inches="tight")
plt.show()