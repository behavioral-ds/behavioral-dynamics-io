import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.lines import Line2D

plt.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.sans-serif": "Libertine",
})

plt.rc('text.latex', preamble=r'\usepackage{amsmath, bm, mathrsfs, mathtools, color}')

def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"[warn] Missing file: {path}")
    except Exception as e:
        print(f"[warn] Could not load {path}: {e}")
    return None

if __name__ == "__main__":
    save_folder = 'figures/'
    base_path = '../final-experiment-results'
    data_dir = os.path.join(base_path, "detection_evade")
    os.makedirs(save_folder, exist_ok=True)

    classifier = 'rf' # 'xgb' or 'rf'
    
    cl_dict = {
        'xgb': 'f1_list_xgb',
        'rf': 'f1_list_rf'
    }

    # Percentages
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    # Data containers
    data_map = {(p, "State"): [] for p in percentages}
    data_map.update({(p, "Content"): [] for p in percentages})

    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        state_path = os.path.join(data_dir, f"results_mixed_emp_policy_{p}.pkl")
        content_path = os.path.join(data_dir, f"results_mixed_content_{p}.pkl")

        d_state = load_pickle(state_path)
        if d_state and cl_dict[classifier] in d_state:
            data_map[(p, "State")].extend(d_state[cl_dict[classifier]])

        d_content = load_pickle(content_path)
        if d_content and cl_dict[classifier] in d_content:
            data_map[(p, "Content")].extend(d_content[cl_dict[classifier]])

    # -----------------------------
    # Plot: side-by-side violins per x
    # -----------------------------
    # fig, ax = plt.subplots(figsize=(8, 5))  # similar aspect to method_comparison.py

    # Colors (consistent, restrained)
    color_state = "#6495ED"   # cornflower-ish (State)
    color_content = "#F08080" # light salmon (Content)
    alpha_fill = 0.8

    rows = []
    for (p, typ), arr in data_map.items():
        for v in arr:
            rows.append({
                "percent_non_troll": int(round(p * 100)),  # 0..100
                "F1": float(v) * 100.0,                    # 0..100 if original in 0..1
                "Type": typ
            })
    df = pd.DataFrame(rows)

    order = [int(p*100) for p in percentages]
    hue_order = ["State", "Content"]

    print("\n=== F1 Summary Statistics by hijacking % ===")
    print("  - Median + 5th/95th = central 90% coverage interval of F1 across runs")
    print("  - Mean + 50% CI     = normal-approximate 95% CI for the mean F1\n")

    for p in order:
        print(f"\n--- Hijacking = {p}% ---")
        for typ in hue_order:
            vals = df.loc[
                (df["percent_non_troll"] == p) & (df["Type"] == typ),
                "F1"
            ].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                print(f"{typ:10s}  Median:   NA    5th:   NA    95th:   NA    "
                    f"Mean:   NA    95% CI: NA–NA")
                continue

            # Coverage interval (what you already had)
            q05, q50, q95 = np.percentile(vals, [5, 50, 95])

            # 90% CI for the *mean* F1 (normal approximation)
            mean = np.mean(vals)
            if vals.size > 1:
                std = np.std(vals, ddof=1)
                se = std / np.sqrt(vals.size)
                z = 1.96  # 95% two-sided normal quantile
                ci_low = mean - z * se
                ci_high = mean + z * se
            else:
                # With only one sample, CI collapses to the point estimate
                std = 0.0
                ci_low = ci_high = mean

            print(
                f"{typ:10s}  "
                f"Median: {q50:5.1f}   5th: {q05:2.1f}   95th: {q95:2.1f}   "
                f"Mean: {mean:5.1f}   95% CI: {ci_low:2.1f}–{ci_high:2.1f}"
            )

    print("\n==============================================================\n")

    palette = {"State": "#2a9d8f", "Content": "#d4a373"}
    # plt.figure(figsize=(8, 4))
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.violinplot(
        data=df,
        x="percent_non_troll", 
        y="F1", 
        hue="Type",
        order=order,
        hue_order=hue_order,
        split=True,
        inner="quartile",
        palette=palette  # punchy blue/salmon
    )

    # PolyCollection objects correspond to the violin bodies
    for pc in ax.collections:
        pc.set_edgecolor("black")     # violin edge color
        pc.set_linewidth(1.0)         # edge thickness
        pc.set_alpha(0.7)             # face alpha (transparency)

    # Lines correspond to median/IQR lines when inner="quartile"
    for line in ax.lines:
        line.set_color("black")       # quartile line color
        line.set_linewidth(1.5)       # thicker quartile lines
        line.set_alpha(0.8)

    # palette = {"State": "#006DFF", "Content": "#FF5A5F"}
    # Optional: make median lines thicker than quartiles
    # Usually: Seaborn draws 3 lines per violin when inner="quartile" (Q1, median, Q3)
    for i, line in enumerate(ax.lines):
        if (i % 3) == 1:  # median line
            line.set_linewidth(2)
            line.set_color("black")

    handles = [
        mpatches.Patch(
            facecolor=palette["State"],
            edgecolor="black",
            linewidth=1.0,
            alpha=0.7,
            label=r"\textbf{Poliy-based (Empirical)}"
        ),
        mpatches.Patch(
            facecolor=palette["Content"],
            edgecolor="black",
            linewidth=1.0,
            alpha=0.7,
            label=r"\textbf{Content-based (Embedding)}"
        )
    ]

    median_handle = Line2D(
        [0], [0],
        color="black",
        lw=2,
        linestyle='--',
        label=r"\textbf{Median}"
    )

    quartile_handle = Line2D(
        [0], [0],
        color="black",
        lw=1.5,
        linestyle=':',
        alpha=0.8,
        label=r"\textbf{Quartiles ($Q_{25}$, $Q_{75}$)}"
    )

    # Combine them
    all_handles = handles + [median_handle, quartile_handle]

    # Custom legend
    ax.legend(
        handles=all_handles,
        fontsize=12,
        title_fontsize=12,
        frameon=False,
        loc="best",
        handlelength=1.6
    )

    ax.tick_params(axis='both', labelsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel(r"\textbf{Percentage of Account Hijacking}", fontsize=14)
    plt.ylabel(r"$\boldsymbol{F_1}$\textbf{--Scores (\%)}", fontsize=14)
    plt.ylim(60, 100)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(save_folder, exist_ok=True)
    save_name = os.path.join(save_folder, f"mixed-policy-{classifier}.pdf")
    plt.savefig(save_name, bbox_inches="tight")
    # plt.show()
    plt.close(fig)