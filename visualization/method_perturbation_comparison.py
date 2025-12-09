"""
Figure 5
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle

plt.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.sans-serif": "Libertine",
})

plt.rc('text.latex', preamble=r'\usepackage{amsmath, bm, mathrsfs, mathtools, color}')

def compute_dodge(methods, x_base, frac=0.30):
    """
    Returns per-method x-offsets in DATA units.
    'frac' is the total half-band as a fraction of the median x-step.
    - Smaller frac (e.g., 0.20) -> more overlap
    - Larger frac (e.g., 0.40) -> less overlap
    """
    x_steps = np.diff(x_base)
    step = np.median(x_steps) if x_steps.size else 0.1
    band = step * frac
    centers = np.arange(len(methods)) - (len(methods) - 1) / 2.0
    denom = max(1, (len(methods) - 1))
    return (centers / denom) * band

def load_results(base_path, experiments, percents=None, key='f1_list_rf'):
    """
    Load results for classifier and each experiment in percent.
    Returns:
      data_dict: {exp_label: {percent: np.array([...])}}
      color_dict: {exp_label: color_hex}
      percents: sorted list of percents actually used
    """
    data_dict = {}
    color_dict = {}

    for exp_name, exp_label in experiments.items():
        per_exp = {}
        for p in percents:
            file_path = os.path.join(base_path, exp_name, f'{p}_results.pkl')
            with open(file_path, 'rb') as f:
                res = pickle.load(f)
            per_exp[p] = np.array(res[key]) * 100

        data_dict[exp_label] = per_exp

    return data_dict, color_dict, percents

if __name__ == "__main__":
    save_folder = 'figures/'
    classifier = 'rf' # 'rf' or 'xgb'
    base_path = '../../final-experiment-results'
    os.makedirs(save_folder, exist_ok=True)

    experiments = {
        'normalised_traj_count': 'State visitation',
        'gail_policies': 'GAIL',
        'irl_policies': 'MaxEntDeepIRL'
    }

    cl_dict = {
        'xgb': 'f1_list_xgb',
        'rf': 'f1_list_rf'
    }

    percents = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    palette = {
        'State visitation': '#3b7a75',
        'GAIL': '#2a9d8f',
        'MaxEntDeepIRL': '#264653',
        'Embed': '#d4a373',
    }

    data_dict, color_dict, percents = load_results(
        base_path=base_path,
        experiments=experiments,
        percents=percents,
        key=cl_dict[classifier],  # RF only
    )

    print("\n=== F1 Summary Statistics by perturbation p (Median, 5th, 95th) ===")

    for p in percents:
        print(f"\n--- Perturbation p = {p:.1f} ({p*100:.0f}%) ---")
        for label in experiments.values():  # 'State visitation', 'GAIL', 'MaxEntDeepIRL'
            arr = np.asarray(data_dict[label].get(p, []), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                print(f"{label:25s}  Median:   NA    5th:   NA    95th:   NA")
                continue

            q50 = float(np.nanmedian(arr))
            q05 = float(np.nanpercentile(arr, 5))
            q95 = float(np.nanpercentile(arr, 95))

            print(f"{label:25s}  Median: {q50:.1f}   5th: {q05:.1f}   95th: {q95:.1f}")

    print("\n==============================================================\n")

    pretty_name = {
        'State visitation': r'\textbf{Empirical}',
        'GAIL': r'\textbf{GAIL}',
        'MaxEntDeepIRL': r'\textbf{MaxEnt Deep IRL}',
    }

    markers = {
        'State visitation': 'o',
        'GAIL': 's',
        'MaxEntDeepIRL': 'D',
    }

    methods = list(experiments.values())
    x_base = np.array(percents, dtype=float) * 100

    dodge = compute_dodge(methods, x_base, frac=0.32) if len(methods) > 1 else np.array([0.0])

    show_outliers = True       # set False to hide q<5% / q>95% dots
    outer_alpha   = 0.85       # opacity of q5–q95 whisker
    inner_lw      = 4.0        # IQR line width
    outer_lw      = 1.2        # q5–q95 line width
    cap_width     = 0.8        # in data x-units; small horizontal cap
    median_ms     = 7.0        # median marker size

    fig, ax = plt.subplots(figsize=(8, 4))
    y_all = []

    for i, label in enumerate(methods):
        color = palette.get(label, None)
        mk    = markers.get(label, 'o')
        x     = x_base + dodge[i]

        med, q25, q75, q05, q95, valid_x, per_p_vals = [], [], [], [], [], [], []

        for xb, p in zip(x, percents):
            arr = np.asarray(data_dict[label].get(p, []), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                med.append(float(np.nanmedian(arr)))
                q25.append(float(np.nanpercentile(arr, 25)))
                q75.append(float(np.nanpercentile(arr, 75)))
                q05.append(float(np.nanpercentile(arr, 5)))
                q95.append(float(np.nanpercentile(arr, 95)))
                valid_x.append(xb)
                per_p_vals.append(arr)

        med  = np.array(med); q25 = np.array(q25); q75 = np.array(q75)
        q05  = np.array(q05); q95 = np.array(q95); valid_x = np.array(valid_x)

        # --- Outer whisker: q5–q95 (thin) ---
        ax.vlines(valid_x, q05, q95, color='#151515', linewidth=outer_lw, alpha=outer_alpha, zorder=2)

        # # optional small caps on q5 and q95 to make each dodge "special"
        for xv, lo, hi in zip(valid_x, q05, q95):
            ax.plot([xv - cap_width/2, xv + cap_width/2], [lo, lo], color='#151515', lw=outer_lw, alpha=outer_alpha, zorder=2)
            ax.plot([xv - cap_width/2, xv + cap_width/2], [hi, hi], color='#151515', lw=outer_lw, alpha=outer_alpha, zorder=2)

        # # --- Inner whisker: IQR (q25–q75) thicker to pop ---
        ax.vlines(valid_x, q25, q75, color=color, linewidth=inner_lw, alpha=0.95, zorder=3)

        # --- Optional: show outliers beyond 95% (and below 5%) lightly ---
        if show_outliers:
            for xv, arr, lo, hi in zip(valid_x, per_p_vals, q05, q95):
                if arr.size:
                    top = arr[arr > hi]
                    bot = arr[arr < lo]
                    if top.size:
                        ax.scatter(np.full(top.size, xv), top, s=16, color=color, alpha=0.6, edgecolor='none', zorder=1)
                    if bot.size:
                        ax.scatter(np.full(bot.size, xv), bot, s=16, color=color, alpha=0.6, edgecolor='none', zorder=1)

        # --- Median dot (white edge for readability over overlap) ---
        ax.plot(valid_x, med, linestyle='none', marker=mk, markersize=median_ms,
                markeredgewidth=0.5, markeredgecolor='white',
                color=color, label=pretty_name.get(label, label), zorder=3)

        # track ranges
        y_all.extend(med[np.isfinite(med)])
        y_all.extend(q25[np.isfinite(q25)]); y_all.extend(q75[np.isfinite(q75)])
        y_all.extend(q05[np.isfinite(q05)]); y_all.extend(q95[np.isfinite(q95)])

    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    x_vals = np.array(percents, dtype=float) * 100

    stats_med = {}

    for label in methods:
        m_arr = []
        for p in percents:
            arr = np.asarray(data_dict[label].get(p, []), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                m_arr.append(np.nanmedian(arr))
            else:
                m_arr.append(np.nan)
        stats_med[label]  = np.array(m_arr, dtype=float)

    # Create inset
    axins_med = inset_axes(ax, width="50%", height="100%",
                        bbox_to_anchor=(0.05, 0.1, 0.5, 0.5),
                        bbox_transform=ax.transAxes, loc="lower left", borderpad=0.0)
    
    # --- Inset 1: Median lines (no fills, no legend) ---
    for label in methods:
        y = stats_med[label]
        if np.all(np.isnan(y)): 
            continue
        axins_med.plot(x_vals, y, linewidth=1.6, marker='.', markersize=2.5,
                    color=palette.get(label, None))
    axins_med.set_title(r'$\textbf{(A) Median }\boldsymbol{F_1}$\textbf{--Scores}')
    axins_med.set_xlim(x_vals.min(), x_vals.max())

    # Minimal ticks
    axins_med.tick_params(axis='both', labelsize=7, length=2)
    axins_med.grid(True, axis='both', linestyle='--', alpha=0.6)
    axins_med.set_yticks([60, 70, 80, 90, 100])
    axins_med.set_xticks([0, 20, 40, 60, 80, 90])

    # Axes & labels
    ax.set_xlabel(r'\textbf{Perturbation (\%)}', labelpad=4, fontsize=14)
    ax.set_ylabel(r'$\boldsymbol{F_1}$\textbf{--Scores (\%)}', fontsize=14)
    
    ax.set_ylim(50, 100)
    ax.set_xticks(x_base)
    
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
    
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(
        frameon=False, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.1), 
        ncol=len(methods),
        fontsize=12, 
        handletextpad=0.1, 
        columnspacing=0.5
    )

    save_name = os.path.join(save_folder, f'method-comparison-perturbation-{classifier}.pdf')
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()