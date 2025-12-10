import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase

plt.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.sans-serif": "Libertine",
})

plt.rc('text.latex', preamble=r'\usepackage{amsmath, bm, mathrsfs, mathtools, color}')

 # Custom multi-patch legend handler
class HandlerMultiPatch(HandlerBase):
    def __init__(self, colors, alpha=1.0, y_offset=0.0, hatch_style='', **kwargs):
        super().__init__(**kwargs)
        self.colors = colors
        self.alpha = alpha
        self.y_offset = y_offset
        self.hatch_style = hatch_style

    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        n = len(self.colors)
        w = width / n
        dy = self.y_offset * height
        artists = []
        for i, c in enumerate(self.colors):
            rect = mpatches.Rectangle(
                (x0 + i * w, y0 - dy), w, height,
                facecolor=c, edgecolor='black', linewidth=0.8, hatch=self.hatch_style,
                alpha=self.alpha, transform=trans
            )
            artists.append(rect)
        return artists

if __name__ == "__main__":
    save_folder = 'figures/'
    base_path = '../final-experiment-results'
    os.makedirs(save_folder, exist_ok=True)

    zero_percent = 0.0

    experiments = {
        'empirical_policies': 'Empirical',
        'gail_policies': 'GAIL', 
        'irl_policies': 'MaxEntDeepIRL',
        'modernbert_embeds_full': 'Embed'
    }

    label_map = {
        'Empirical': 'Empirical',
        'GAIL': 'GAIL',
        'MaxEntDeepIRL': 'MaxEnt Deep IRL',
        'Embed': 'Embedding'
    }

    palette = {
        'Empirical': '#3b7a75',
        'GAIL': '#2a9d8f',
        'MaxEntDeepIRL': '#264653',
        'Embed': '#d4a373',
    }

    data_dict = {}
    color_dict = {}

    for exp_name, exp_label in experiments.items():
        if exp_name == 'modernbert_embeds_full':
            file_path = f'{base_path}/{exp_name}/full_results.pkl'
        else:
            file_path = f'{base_path}/{exp_name}/{zero_percent}_results.pkl'
        
        with open(file_path, 'rb') as f:
            res = pickle.load(f)
            
        # Add RF and XGB results
        data_dict[f"{exp_label} RF"] = np.array(res["f1_list_rf"]) * 100
        data_dict[f"{exp_label} XG"] = np.array(res["f1_list_xgb"]) * 100
        
        # Assign color from palette
        color = palette[exp_label]
        color_dict[f"{exp_label} RF"] = color
        color_dict[f"{exp_label} XG"] = color
    
    methods = [key for key in data_dict.keys() if key.endswith('RF')] + \
          [key for key in data_dict.keys() if key.endswith('XG')]
    
    adjusted_labels = []
    for m in methods:
        # extract the base experiment name (strip " RF" or " XG")
        base = m.replace(" RF", "").replace(" XG", "")
        # look up friendly name (fallback to base if not found)
        adjusted_labels.append(label_map.get(base, base))
    
    data = [data_dict[method] for method in methods]
    colours = [color_dict[method] for method in methods]

    # Calculate median and 5th-95th percentiles
    medians = [np.median(lst) for lst in data]
    lower_bounds = [np.percentile(lst, 5) for lst in data]  # 5th percentile
    upper_bounds = [np.percentile(lst, 95) for lst in data]  # 95th percentile

    lower_errors = [medians[i] - lower_bounds[i] for i in range(len(medians))]
    upper_errors = [upper_bounds[i] - medians[i] for i in range(len(medians))]
    error_bars = [lower_errors, upper_errors]  # Format for xerr

    print("\n=== F1 Summary Statistics (Median ± CI) ===")
    for method, median, lb, ub in zip(methods, medians, lower_bounds, upper_bounds):
        print(f"{method:30s}  Median: {median:.1f}   5th: {lb:.1f}   95th: {ub:.1f}")
    print("============================================\n")

    # Bar plot with error bars
    y = [1, 2, 3, 4, 5.2, 6.2, 7.2, 8.2] 

    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.barh(y, medians, edgecolor='black', color=colours, alpha=0.7, height=0.6, linewidth=1.0)

    hatches = ['', '', '', '//',   # RF methods
           '', '', '', '//']    # XG methods (repeat pattern)

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # Annotate each bar with the median value
    for bar, median in zip(bars, medians):
        width = bar.get_width()
        ax.text(
            width + 0.2,                              # horizontal offset
            bar.get_y() + bar.get_height() * 1.07,      # vertical position
            rf"$\tilde{{x}}\!=\!{median:.1f}$",   # percentage with one decimal
            ha='left', va='center',
            fontsize=12, fontweight='medium'
        )
    
    # Add elegant capsule-style error ranges
    for i, (y_pos, median, lower, upper, color) in enumerate(zip(y, medians, lower_bounds, upper_bounds, colours)):
        # Shadow effect
        plt.hlines(y_pos+0.02, lower, upper, color='black', linewidth=5.4, alpha=0.3, zorder=1)
        plt.hlines(y_pos-0.02, lower, upper, color='black', linewidth=5.4, alpha=0.3, zorder=1)
        # Main line
        plt.hlines(y_pos, lower, upper, color=color, linewidth=5, alpha=1.0, zorder=2)
        # End caps
        plt.scatter([lower, upper], [y_pos, y_pos], color=color, s=60, 
                    alpha=1.0, edgecolor='#151515', linewidth=1, zorder=3)

    # Add a vertical line to separate the classifiers
    plt.axhline(y=4.6, color='black', linestyle='--', alpha=0.7, linewidth=1.5)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add labels
    plt.text(98.5, 2, r"$\textbf{\parbox{2.2cm}{Random\\Forest}}$",
        ha='left', va='bottom', fontsize=14, color='black')
    plt.text(98.5, 6.2, r"$\textbf{\parbox{2.2cm}{Gradient\\Boosting}}$", 
        ha='left', va='bottom', fontsize=14, color='black')


    # Adding labels and title
    plt.xlabel(r'$\boldsymbol{F_1}$\textbf{--Scores (\%)}', fontsize=14)
    plt.xlim(75, 100)

    ax.set_yticks(y)
    ax.set_yticklabels(adjusted_labels, fontsize=12)

    # ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(axis='x', labelsize=12)

    plt.grid(axis='x', alpha=0.6, linestyle='--', linewidth=0.5, zorder=0)

    # Colors
    policy_colors = [palette['Empirical'], palette['GAIL'], palette['MaxEntDeepIRL']] # Empirical, GAIL, MaxEnt Deep IRL
    embed_color   = palette['Embed'] # Embedding
    patch_alpha = 0.7 

    # Dummy handle placeholders for the legend
    class PolicyHandle: pass
    class EmbedHandle: pass

    policy_handle = PolicyHandle()
    embed_handle  = EmbedHandle()

    # Colors
    policy_colors = [palette['Empirical'], palette['GAIL'], palette['MaxEntDeepIRL']]
    embed_color   = palette['Embed']
    patch_alpha   = 0.7

    legend = ax.legend(
        handles=[policy_handle, embed_handle],
        labels=[r"\textbf{Policy-based}", r"\textbf{Content-based}"],
        handler_map={
            PolicyHandle: HandlerMultiPatch(policy_colors, alpha=patch_alpha, y_offset=0.15),
            EmbedHandle:  HandlerMultiPatch([embed_color], alpha=patch_alpha, y_offset=0.15, hatch_style='//'),
        },
        handlelength=3.0,
        handleheight=0.9,
        borderpad=0.05,
        frameon=False,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        columnspacing=1.4,
        fontsize=12,
        alignment='center',     # <- keeps entries vertically centered in the row (Matplotlib >=3.6)
        handletextpad=0.6,      # (optional) fine-tune spacing
        labelspacing=0.6        # (optional) row spacing
    )

    save_name = os.path.join(save_folder, 'method-comparison.pdf')
    # Show the plot
    plt.savefig(save_name, bbox_inches='tight')
    # plt.show()
    plt.close(fig)