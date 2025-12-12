import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from scipy.stats import norm
from scipy.stats import anderson_ksamp, fisher_exact
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

URL_RE = re.compile(r'https?://[^\s)>\]]+', flags=re.IGNORECASE)

plt.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "sans-serif",
    "font.sans-serif": "Times New Roman",
})

plt.rc('text.latex', preamble=r'\usepackage{amsmath, bm, mathrsfs, mathtools, color}')

def prepare_data(df, troll_usernames):
    # 1) load & clean
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    df['action']      = df['action'].astype('category')
    df['is_troll']      = df['author'].isin(troll_usernames)

    # 2) reference‐day for daily collapse
    df['time_of_day'] = df['created_utc'].apply(
        lambda t: t.replace(year=2000, month=1, day=1)
    )

    # 3) weekly features
    df['week_start'] = (
        df['created_utc'].dt.normalize()
        - pd.to_timedelta(df['created_utc'].dt.weekday, unit='d')
    )
    wd = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df['weekday'] = pd.Categorical(df['created_utc'].dt.day_name(),
                                   categories=wd, ordered=True)
    df['hour'] = df['created_utc'].dt.hour

    return df

def plot_heatmaps(df, percent=True, cmap='plasma'):
    """
    Plot hourly activity heatmaps for troll vs organics, optionally as percentage of total activity.

    Parameters:
    - df: DataFrame with ['created_utc','is_troll','week_start','weekday','hour']
    - percent: bool, if True shows percentage of total activity; otherwise normalized [0,1]
    - cmap: colormap name
    """
    # Ensure hour column exists
    if 'hour' not in df.columns:
        df['hour'] = df['created_utc'].dt.hour

    # Aggregate counts per group/hour
    counts = (
        df.groupby(['is_troll','weekday','hour'])
          .size()
          .reset_index(name='count')
    )
    wkdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    pivs = {}
    for b in [False, True]:
        # Raw pivot table
        raw = (
            counts[counts['is_troll']==b]
                  .pivot(index='weekday', columns='hour', values='count')
                  .reindex(wkdays)
                  .fillna(0)
        )
        pivs[b] = raw / raw.values.max()

    # Set up figure and subplots
    fig, axes = plt.subplots(
        1, 2,
        figsize=(8, 3),
        sharey=True,
        gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.1}
    )

    # Plot each heatmap
    for ax, b, title in zip(axes, [False, True], [r'\textbf{Organics}', r'\textbf{Trolls}']):
        X, Y = np.meshgrid(np.arange(25), np.arange(8))
        pcm = ax.pcolormesh(
            X, Y, pivs[b].values,
            shading='auto', cmap=cmap,
            vmin=0,
            vmax=1
        )
        ax.set_title(title, fontsize=14)
        ax.set_yticks(np.arange(0.5, 7.5))
        ax.set_yticklabels(wkdays)
        ax.set_xticks(np.arange(0, 24, 1) + 0.5)
        tick_labels = ['0', '', '', '', '', '', '6', '', '', '', '', '', '12', '', '', '', '', '', '18', '', '', '', '', '23']
        ax.set_xticklabels(tick_labels)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_xlabel(r'\textbf{Hour of Day}', fontsize=14)

    axes[0].set_ylabel(r'\textbf{Weekday}', fontsize=14)

    # Colorbar axis anchored to first subplot
    pos = axes[1].get_position()
    cax = fig.add_axes([
        pos.x1 + 0.02, pos.y0,
        0.02, pos.height
    ])
    cbar = fig.colorbar(pcm, cax=cax)
    label = r'\textbf{Percentage of Activity}' if percent else r'\textbf{Normalized mean count}'
    cbar.set_label(label, fontsize=14, rotation=90, labelpad=3)
    cbar.ax.tick_params(labelsize=12)
    # format ticks
    if percent:
        cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    else:
        cbar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    # Final layout
    plt.savefig('figures/heatmaps-plot.pdf', bbox_inches='tight')
    plt.close(fig)

def plot_polar(
    df,
    action_col='action',
    ci=0.95,
    smooth_window=3,
    colors=('#006DFF', '#FF6B6B'),
    linestyles=('-', '--'),
    titles={1: r"\textbf{Create Thread ($CT$)}", 2: r"\textbf{Root Comment ($RC$)}", 3: r"\textbf{Post Reply ($PR$)}"}
):
    """
    Polar plot of normalized rhythmic activity by action type and is_troll, with confidence intervals and optional smoothing.

    Parameters:
    - df: DataFrame containing ['is_troll','week_start','weekday','hour', action_col]
    - action_col: column name for action type
    - ci: confidence level (e.g., 0.95)
    - smooth_window: integer window size for moving-average smoothing (odd integer >=1)
    - colors: tuple of two matplotlib-compatible colors for Organics and Trolls
    - legend_loc: location string within bbox for legend placement
    - legend_bbox: (x, y) tuple in axes coordinates for legend box anchor
    """
    # 1. Per-week counts
    counts = (
        df.groupby(['is_troll', action_col, 'week_start', 'weekday', 'hour'], observed=False)
          .size().reset_index(name='count')
    )

    # 2. Compute mean and SEM per weekday/hour
    stats = (
        counts.groupby(['is_troll', action_col, 'weekday', 'hour'], observed=False)['count']
              .agg(['mean', 'sem']).reset_index()
    )
    z = norm.ppf(1 - (1 - ci) / 2)

    # Ordering and theta
    wd = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    wd_labels = ['Mon.','Tue.','Wed.','Thu.','Fri.','Sat.','Sun.']
    labels= [r'\textbf{Organics}', r'\textbf{Trolls}']
    total_bins = len(wd) * 24
    theta = np.linspace(0, 2 * np.pi, total_bins + 1)

    actions = sorted(df[action_col].unique())
    n = len(actions)

    width = 2 * np.pi / len(wd)
    angles = np.arange(len(wd)) * width
    angles_mid = angles + width/2

    fig, axes = plt.subplots(
        1, n,
        subplot_kw={'projection': 'polar'},
        figsize=(8, 4)
    )

    fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05, wspace=0.25)

    if n == 1:
        axes = [axes]

    dy = -0.3  # negative moves down middle
    
    if n % 2 == 1:
        mid = n // 2
        p = axes[mid].get_position()
        axes[mid].set_position([p.x0, p.y0 + dy, p.width, p.height])
        axes[mid].set_zorder(10)  # draw on top when overlapping
    else:
        mids = [n // 2 - 1, n // 2]
        for m in mids:
            p = axes[m].get_position()
            axes[m].set_position([p.x0, p.y0 + dy, p.width, p.height])
            axes[m].set_zorder(10)

    # smoothing helper\    
    def smooth(arr, window):
        if window < 2:
            return arr
        pad = window // 2
        arr_p = np.concatenate([arr[-pad:], arr, arr[:pad]])
        kernel = np.ones(window) / window
        return np.convolve(arr_p, kernel, mode='valid')

    for ax, action in zip(axes, actions):
        for (is_troll, label, color, linestyle) in zip([False, True], labels, colors, linestyles):
            mask = (stats['is_troll']==is_troll) & (stats[action_col]==action)
            subset = stats.loc[mask].copy()
            subset['wd_idx'] = subset['weekday'].map({d:i for i,d in enumerate(wd)})
            subset = subset.sort_values(['wd_idx','hour'])
            
            mean_vals = subset['mean'].values
            sem_vals = subset['sem'].values

            # 3. Normalize
            max_m = mean_vals.max() or 1
            r_mean = mean_vals / max_m
            r_sem = sem_vals / max_m

            # 4. CI bounds
            r_lower = np.clip(r_mean - z * r_sem, 0, None)
            r_upper = np.clip(r_mean + z * r_sem, 0, None)

            # 5. Clip to [0,1]
            r_mean = np.clip(r_mean, 0, 1)
            r_lower = np.clip(r_lower, 0, 1)
            r_upper = np.clip(r_upper, 0, 1)

            # Close loop
            r_mean = np.append(r_mean, r_mean[0])
            r_lower = np.append(r_lower, r_lower[0])
            r_upper = np.append(r_upper, r_upper[0])

            # smoothing
            r_mean_sm = smooth(r_mean, smooth_window)
            r_lower_sm = smooth(r_lower, smooth_window)
            r_upper_sm = smooth(r_upper, smooth_window)

            ax.plot(theta, r_mean_sm, label=label, color=color, linestyle=linestyle, linewidth=1.5)
            ax.fill_between(theta, r_lower_sm, r_upper_sm, alpha=0.3, color=color)

        ax.set_xticks(angles_mid)
        ax.set_xticklabels(wd_labels, fontsize=12)
        ax.tick_params(axis='x', pad=5)

        # set minor ticks at boundaries (no labels)
        ax.xaxis.set_ticks(angles, minor=True)
        ax.xaxis.set_ticklabels([], minor=True)
        ax.set_axisbelow(False) # Draw on top
        ax.set_rlabel_position(0) 
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([r'0\%', r' 100\%'], fontsize=12)
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        # ax.tick_params(axis='y', labelsize=12, pad=12)
        ax.set_title(titles[action], fontsize=14, pad=8)
        ax.grid(True)
        ax.grid(which='minor', axis='x', color='k', linestyle=':', linewidth=0.5)
        # disable grid for major
        ax.grid(which='major', axis='x', linestyle='', linewidth=0)
        ax.yaxis.grid(True, color='k', linestyle=':', linewidth=0.5)  # radial (r) grid lines

    # axes[-1].legend(loc=legend_loc, bbox_to_anchor=legend_bbox, handlelength=1.8, bbox_transform=axes[-1].transAxes, fontsize=16)

    handles = [
        Line2D([0], [0], color=colors[0], linestyle=linestyles[0], lw=2,
            label='Organics'),
        Line2D([0], [0], color=colors[1], linestyle=linestyles[1], lw=2,
            label='Trolls')
    ]

    # Figure-level legend; coords are in figure space
    leg = fig.legend(handles=handles,
        loc='lower right',           # corner
        bbox_to_anchor=(0.98, -0.15), # nudge from edges
        frameon=True,
        handlelength=2.0,
        fontsize=14
    )
    frame = leg.get_frame()
    frame.set_edgecolor("black") # black border
    plt.savefig('figures/polar-ci-smooth-plot.pdf', bbox_inches='tight')
    plt.close(fig)

def plot_interevent_pmf_cdf_custom_bins(
    df,
    time_col='created_utc',
    user_col='author',
    troll_flag_col='is_troll',
    colors=('#006DFF', '#FF6B6B'),
    figsize=(8, 3)
):
    """
    Main axes: PMF (probability mass per custom bin)
    Inset (on the right): Empirical CDF (log-x)
    """
    # inset_cfg = inset_cfg or dict(width="40%", height="40%", loc="upper right", borderpad=1.0)

    # 1) compute inter-event times in minutes
    df_sorted = df.sort_values([user_col, time_col])
    # ensure datetime
    if not np.issubdtype(df_sorted[time_col].dtype, np.datetime64):
        df_sorted[time_col] = pd.to_datetime(df_sorted[time_col], utc=True, errors='coerce')
    deltas = (
        df_sorted
        .groupby(user_col, sort=False)[time_col]
        .diff()
        .dt.total_seconds()
    ).dropna() / 60.0

    # 2) split troll vs organics (align indices)
    is_troll = df_sorted.loc[deltas.index, troll_flag_col].astype(bool)
    troll_deltas = deltas[is_troll].to_numpy()
    human_deltas = deltas[~is_troll].to_numpy()

    # --- Global distributional difference (tail-sensitive) ---
    ad_res = anderson_ksamp([human_deltas, troll_deltas])  # ad_res.statistic, ad_res.pvalue
    print(f"[AD test] A^2={ad_res.statistic:.3f}, p={ad_res.pvalue:.3e}")

    # 3) custom bins (minutes)
    bins = [0, 1, 5, 15, 60, 240, 720, 1440, 4320, np.inf]
    bin_labels = [
        r'$<$1 min', '1-5 min', '5-15 min', '15-60 min',
        '1-3 hrs', '3-12 hrs', '12-24 hrs', '24-72 hrs', r'$>$72 hrs'
    ]

    # 4) figure + main axis (PMF)
    fig, ax_pmf = plt.subplots(figsize=figsize)

    # --- PMF ---
    counts_h, _ = np.histogram(human_deltas, bins=bins)
    counts_b, _ = np.histogram(troll_deltas, bins=bins)

    n_h = max(len(human_deltas), 1)  # avoid div-by-zero
    n_b = max(len(troll_deltas), 1)

    # Early bin: "< 1 min" is index 0
    early_idx = 0
    # Tail bin: "> 72 hrs" is the last bin (since last label corresponds to [4320, inf))
    tail_idx = -1

    # 2x2 contingency for Fisher's exact
    def fisher_for_bin(idx):
        table = np.array([
            [counts_h[idx], n_h - counts_h[idx]],   # Organics: in-bin vs out-of-bin
            [counts_b[idx], n_b - counts_b[idx]]    # Trolls:    in-bin vs out-of-bin
        ])
        odds, p = fisher_exact(table, alternative='two-sided')
        ph = counts_h[idx] / n_h
        pb = counts_b[idx] / n_b
        return odds, p, ph, pb

    odds_e, p_e, ph_e, pb_e = fisher_for_bin(early_idx)
    odds_t, p_t, ph_t, pb_t = fisher_for_bin(tail_idx)

    print(f"[Early <1min] p={p_e:.3e}, odds={odds_e:.2f}, "
        f"Organics={ph_e:.3%}, Trolls={pb_e:.3%}")
    print(f"[Tail >72h]  p={p_t:.3e}, odds={odds_t:.2f}, "
        f"Organics={ph_t:.3%}, Trolls={pb_t:.3%}")

    pmf_h = counts_h / n_h
    pmf_b = counts_b / n_b

    x = np.arange(len(bin_labels))
    width = 0.38
    edgeprops = dict(edgecolor='black', linewidth=0.5)  # thin black edges
    ax_pmf.bar(x - width/2, pmf_h, width, label='Organics', alpha=1.0, color=colors[0], **edgeprops)
    ax_pmf.bar(x + width/2, pmf_b, width, label='Trolls',     alpha=1.0, color=colors[1], **edgeprops)

    ax_pmf.set_xticks(x)
    ax_pmf.set_xticklabels(bin_labels, rotation=40, ha='right')
    ax_pmf.set_ylabel(r'\textbf{Probability}', fontsize=14)
    ax_pmf.set_xlabel(r'\textbf{Inter-event Times}', fontsize=14)
    ax_pmf.set_ylim(0, 0.35)
    # ax_pmf.set_ylim(0, max((pmf_h.max(), pmf_b.max(), 0.3)) * 1.15)
    ax_pmf.legend(loc="upper left", frameon=False, fontsize=12, ncol=2)
    ax_pmf.grid(True, ls='--', alpha=0.35, axis='y')
    ax_pmf.tick_params(axis='both', labelsize=12)

    # Remove top and right spines
    ax_pmf.spines['top'].set_visible(False)
    ax_pmf.spines['right'].set_visible(False)

    # 5) inset axis for the CDF on the right
    ax_cdf = inset_axes(ax_pmf, width="50%", height="100%",
                        bbox_to_anchor=(0.75, 0.4, 0.5, 0.5),
                        bbox_transform=ax_pmf.transAxes, loc="lower left", borderpad=0.0)

    def ecdf(data):
        if len(data) == 0:
            return np.array([]), np.array([])
        x = np.sort(data)
        y = np.arange(1, len(x)+1) / len(x)
        return x, y

    x_h, y_h = ecdf(human_deltas)
    x_b, y_b = ecdf(troll_deltas)

    if len(x_h):
        ax_cdf.plot(x_h, y_h, drawstyle='steps-post', lw=2.0, label='Organics', color=colors[0])
    if len(x_b):
        ax_cdf.plot(x_b, y_b, drawstyle='steps-post', lw=2.0, label='Trolls', color=colors[1])

    # bin boundary guides (exclude 0 and inf)
    for boundary in bins[1:-1]:
        ax_cdf.axvline(boundary, color='gray', linestyle=':', alpha=0.4, lw=0.8)

    ax_cdf.set_xscale('log')
    ax_cdf.set_xlim(max(bins[0], 1e-3), bins[-2]*5)  # show up to 5 * last finite bin
    ax_cdf.set_ylim(0, 1.0)

    # inset cosmetics
    ax_cdf.set_title('(A) Empirical CDF', fontsize=12, pad=2)
    ax_cdf.tick_params(axis='both', labelsize=11)
    ax_cdf.grid(True, which='both', ls='--', alpha=0.25)

    plt.savefig('figures/interevent-pmf-cdf-custom-bins.pdf', bbox_inches='tight')
    plt.close(fig)
    
# ── Main ──────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    os.makedirs('figures/', exist_ok=True)
    # Load data
    df_trolls = pd.read_csv('../data-analysis/data-trolls/reddit-suspicious-accounts.csv')
    df_trolls['Username'] = df_trolls['Username'].str.replace(r'^u/','',regex=True)
    trolls = set(df_trolls['Username'])
    df_active = pd.read_pickle('../data-analysis/data-analysis-timestamps/all_user_active_content_df.pkl')
    df = prepare_data(df_active, trolls)
    
    # Generate activity patterns plots
    plot_heatmaps(df)
    plot_polar(df)
    plot_interevent_pmf_cdf_custom_bins(df)