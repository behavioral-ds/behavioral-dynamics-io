import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mpltern

plt.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "sans-serif",
    "font.sans-serif": "Times New Roman",
})

plt.rc('text.latex', preamble=r'\usepackage{amsmath, bm, mathrsfs, mathtools, color}')

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================
print("Loading and preparing data...")

# Define file paths
troll_accounts_path = '../data-analysis/sampled_matched_perturbed_df.pkl'
active_content_path = '../data-analysis/data-analysis-timestamps/all_user_active_content_df.pkl'

# Load the data
try:
    df_trolls_all = pd.read_pickle(troll_accounts_path)
    df_trolls = df_trolls_all[(df_trolls_all['perturb_percent'] == 0) & (df_trolls_all['russian'] == 1)]
    trolls = set(df_trolls['user'])

    df_active = pd.read_pickle(active_content_path)
    df_active['is_troll'] = df_active['author'].isin(trolls)

    print("Data loaded successfully.")

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("\nPlease make sure the file paths are correct.")
    exit()

# =============================================================================
# 2. DATA PROCESSING FUNCTION (Unchanged)
# =============================================================================

def user_ternary_coords(df, usernames):
    """
    For each username in `usernames`, compute (t, l, r) = normalized
    shares of actions {1,2,3}. Returns a list of dicts:
    [{'user': name, 't': t, 'l': l, 'r': r}]
    """
    if df.empty or not usernames:
        return []

    df_filtered = df[df['action'].isin([1, 2, 3])]
    # counts per user/action
    ct = df_filtered.groupby(['author', 'action']).size().unstack(fill_value=0)

    out = []
    for u in usernames:
        if u not in ct.index:
            continue
        a1 = ct.loc[u][1] if 1 in ct.columns else 0
        a2 = ct.loc[u][2] if 2 in ct.columns else 0
        a3 = ct.loc[u][3] if 3 in ct.columns else 0
        s = a1 + a2 + a3
        if s == 0:
            continue
        t = a1 / s  # action 1 share
        l = a2 / s  # action 2 share
        r = a3 / s  # action 3 share
        out.append({'user': u, 't': t, 'l': l, 'r': r})
    return out

def process_user_actions(df):
    """
    Transforms a dataframe of user actions into percentages for ternary plotting.
    This version correctly filters for actions 1, 2, and 3.
    """
    if df.empty:
        return []

    df_filtered = df[df['action'].isin([1, 2, 3])]

    # Calculate percentages on the filtered data
    action_counts = df_filtered.groupby('author')['action'].value_counts(normalize=True).unstack(fill_value=0)
    
    # Ensure all three action columns (1, 2, 3) exist
    for i in range(1, 4): # Use range(1, 4) for 1, 2, 3
        if i not in action_counts.columns:
            action_counts[i] = 0
            
    # Select the columns in the correct order: 1, 2, 3
    action_percentages = action_counts[[1, 2, 3]]
    
    return action_percentages.values.tolist()

# =============================================================================
# 3. PLOTTING FUNCTION (Updated for mpltern)
# =============================================================================

def create_ternary_plot_mpltern(points, flag, save_dir='figures/', highlights=None, highlight_colors=None):
    """
    Generates a ternary hexbin plot using mpltern with final descriptive labels.
    """
    if not points:
        print(f"Skipping plot for {'trolls' if flag else 'organics'} as there is no data.")
        return

    t, l, r = np.array(points).T

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection="ternary")

    pc = ax.hexbin(t, l, r, gridsize=25, cmap='plasma', norm=colors.LogNorm())

    counts = pc.get_array()
    probs = counts / len(points)
    # print(f'Max prob.:{max(probs)}')
    pc.set_array(probs)
    
	# non_zero_probs = probabilities[probabilities > 0]
    vmin = 0.01 if flag else 0.0001
    vmax = 0.1
    pc.set_clim(vmin=vmin, vmax=vmax)
    

    cax = ax.inset_axes([1.05, 0.1, 0.05, 0.8])
    colorbar = fig.colorbar(pc, cax=cax)
    colorbar.set_label(r"\textbf{User Density}", rotation=90, va="baseline", fontsize=12, labelpad=12)

    # --- FINAL LABELS ---
    ax.set_tlabel(r"\textbf{← Create Thread (\%)}", fontsize=12, labelpad=0)  # Left-side axis (Action 1)
    ax.set_llabel(r"\textbf{← Root Comment (\%)}", fontsize=12, labelpad=0)   # Right-side axis (Action 2)
    ax.set_rlabel(r"\textbf{Reply Comment (\%) →}", fontsize=12, labelpad=0)   # Bottom axis (Action 3)

    ax.taxis.set_label_position('tick1')
    ax.raxis.set_label_position('tick1')
    ax.laxis.set_label_position('tick1')
    
    # Customize ticks and grid
    ax.taxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x * 100:.0f}'))
    ax.laxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x * 100:.0f}'))
    ax.raxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x * 100:.0f}'))
    
    ax.grid(linestyle='--', linewidth=0.8)

    if highlights:
        # Plot X markers
        hx = [d['t'] for d in highlights]
        hy = [d['l'] for d in highlights]
        hz = [d['r'] for d in highlights]
        ax.scatter(hx, hy, hz, marker='x', s=90, c=highlight_colors, linewidths=3, zorder=5)
	
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f'{save_dir}/ternary-{'trolls' if flag else 'organics'}.pdf', bbox_inches='tight')
    # plt.show()
    plt.close(fig)

# =============================================================================
# 4. SCRIPT EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Separate the data
    df_trolls_actions = df_active[df_active['is_troll'] == True]
    df_organics_actions = df_active[df_active['is_troll'] == False]
    
    # Process each group
    print("Processing data for plotting...")
    troll_points = process_user_actions(df_trolls_actions)
    organics_points = process_user_actions(df_organics_actions)

    highlight_users = ['petouchoque', 'TojatMalaron', 'xameg']
    highlight_colors = ["#E41A1C", "#1B9E77", "#17BECF"]

    troll_highlights = user_ternary_coords(df_trolls_actions, highlight_users)
    print(f'Troll highlights: {troll_highlights}')

    # Create the two plots
    print(f"\nGenerating plot for {len(troll_points)} trolls...")
    create_ternary_plot_mpltern(troll_points, True, highlights=troll_highlights, highlight_colors=highlight_colors)
    
    print(f"\nGenerating plot for {len(organics_points)} organics...")
    create_ternary_plot_mpltern(organics_points, False, highlights=None)
    
    print("\nDone.")