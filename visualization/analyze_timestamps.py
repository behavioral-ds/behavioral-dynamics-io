import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
import datetime as dt
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

def plot_daily_collapsed_minute(df, action_col='action', window='15min'):
    """
    Stacked-area plots of mean per-day activity by action for:
      - All users
      - Organics
      - Trolls
    Allows custom time-binning via the `window` parameter (e.g. '1min', '5min').
    """
    # Prepare DataFrame and time bins
    df2 = df.copy()
    # floor to interval
    df2['time_slot'] = df2['created_utc'].dt.floor(window)
    df2['minute_of_day'] = df2['time_slot'].dt.hour * 60 + df2['time_slot'].dt.minute
    df2['date'] = df2['created_utc'].dt.date

    # Determine interval in minutes from window
    slot = pd.Timedelta(window)
    interval = int(slot.total_seconds() // 60)
    # full index of minute bins
    minute_index = np.arange(0, 1440, interval)

    # hourly ticks for x-axis
    tick_locs   = np.arange(0, 1440, 60)
    tick_labels = [f"{h:02d}" for h in range(24)]

    # Define groups to plot
    groups = [
        ('All Users', df2),
        ('Organics', df2[df2['is_troll']==False]),
        ('Trolls',     df2[df2['is_troll']==True]),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for ax, (label, subdf) in zip(axes, groups):
        # 1) Count per day × action × minute bin
        per_day_act = (
            subdf.groupby(['date', 'minute_of_day', action_col], observed=False)
                 .size()
                 .reset_index(name='count')
        )
        # 2) Mean across days by minute and action
        mean_act = (
            per_day_act.groupby(['minute_of_day', action_col], observed=False)['count']
                       .mean()
                       .unstack(action_col)
                       .reindex(index=minute_index, fill_value=0)
        )
        # 3) Stacked area
        mean_act.plot.area(ax=ax, stacked=True)

        ax.set_title(f'Mean per-day activity by action: {label}')
        ax.set_ylabel('Mean count per day')
        ax.legend(title='Action', loc='upper right')
        ax.set_xlim(0, minute_index[-1])
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, rotation=90)
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    axes[-1].set_xlabel('Time of day')
    plt.tight_layout()
    plt.savefig('figures/daily_collapsed_by_action.pdf', bbox_inches='tight')
    # plt.show()

# ── Plot 2: Weekly Rhythm Line Subplots ────────────────────────────────────────
def plot_weekly_rhythm(
    df,
    window='15min',
    normalize=True,
    ci=True,
    save_dir='figures'
):
    """
    Plot the weekly posting rhythm collapsed into a single-day time axis,
    separately for trolls and organics, with optional global normalization.
    """
    os.makedirs(save_dir, exist_ok=True)
    wd = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    base = dt.datetime(2000, 1, 1)
    df2 = df.copy()

    # Map timestamps into one dummy day
    df2['time_slot_dt'] = (
        df2['created_utc'].dt.floor(window)
             .apply(lambda t: base + dt.timedelta(
                 hours=t.hour, minutes=t.minute)
             )
    )

    # Count per week × weekday × time_slot
    counts = (
        df2.groupby(
            ['is_troll','week_start','weekday','time_slot_dt'],
            observed=True
        ).size().reset_index(name='count')
    )

    # Fill missing slots robustly
    slot_delta = pd.Timedelta(window)
    periods = int(pd.Timedelta('1d').total_seconds() / slot_delta.total_seconds())
    all_slots = pd.date_range(start=base, periods=periods, freq=window)

    full_idx = pd.MultiIndex.from_product(
        [[False, True], df['week_start'].unique(), wd, all_slots],
        names=['is_troll','week_start','weekday','time_slot_dt']
    )
    counts = (
        counts.set_index(['is_troll','week_start','weekday','time_slot_dt'])
              .reindex(full_idx, fill_value=0)
              .reset_index()
    )

    # Aggregate: mean, std, sem, optional 95% CI
    agg = (
        counts.groupby(['is_troll','weekday','time_slot_dt'], observed=False)['count']
              .agg(['mean','count','std'])
              .rename(columns={'count':'n_weeks'})
              .reset_index()
    )
    agg['sem'] = agg['std'] / np.sqrt(agg['n_weeks'])
    if ci:
        z = 1.96
        agg['ci_lower'] = agg['mean'] - z * agg['sem']
        agg['ci_upper'] = agg['mean'] + z * agg['sem']

    # Global normalization
    if normalize:
        m = agg['mean'].max() or 1
        cols = ['mean','ci_lower','ci_upper'] if ci else ['mean']
        agg[cols] = agg[cols] / m

    # Plotting
    hour_locator = mdates.HourLocator(interval=4)
    hour_fmt = mdates.DateFormatter('%H')
    start, end = base, base + dt.timedelta(days=1)

    for flag, title in [(False,'Organics'), (True,'Trolls')]:
        sub = agg[agg['is_troll']==flag]
        fig, axes = plt.subplots(1, 7, figsize=(22, 3), sharex=True, sharey=True)
        for ax, day in zip(axes, wd):
            day_df = sub[sub['weekday']==day].sort_values('time_slot_dt')
            ax.plot(day_df['time_slot_dt'], day_df['mean'])
            if ci:
                ax.fill_between(
                    day_df['time_slot_dt'], day_df['ci_lower'], day_df['ci_upper'], alpha=0.2
                )
            ax.set_title(day[:3])
            ax.xaxis.set_major_locator(hour_locator)
            ax.xaxis.set_major_formatter(hour_fmt)
            ax.tick_params(axis='x', rotation=45, labelsize=6)

        axes[0].set_ylabel('Normalized count' if normalize else 'Mean count')
        fig.suptitle(f'Weekly rhythm ({window}) — {title}')
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        fig.savefig(f'{save_dir}/weekly-rhythm-{'trolls' if flag else 'organics'}.pdf')
        # plt.show()

def plot_weekly_rhythm_action(
    df,
    window='15min',
    normalize=True,
    ci=True,
    save_dir='figures'
):
    """
    Plot the weekly posting rhythm by action, collapsed into a single-day
    time axis, separately for trolls and organics, with optional global normalization.
    """
    wd = ['Monday','Tuesday','Wednesday','Thursday',
          'Friday','Saturday','Sunday']
    base = dt.datetime(2000, 1, 1)

    # Determine all slots in the day at given window
    slot_delta = pd.Timedelta(window)
    periods = int(pd.Timedelta('1d').total_seconds() / slot_delta.total_seconds())
    all_slots = pd.date_range(start=base, periods=periods, freq=window)

    # Map timestamps into dummy day
    df2 = df.copy()
    df2['time_slot_dt'] = (
        df2['created_utc'].dt.floor(window)
             .apply(lambda t: base + dt.timedelta(hours=t.hour, minutes=t.minute))
    )

    # Count per week × weekday × slot × action × is_troll
    counts = (
        df2.groupby(
            ['is_troll','week_start','weekday','time_slot_dt','action'],
            observed=True
        )
        .size()
        .reset_index(name='count')
    )

    # Fill missing combinations
    actions = counts['action'].unique()
    full_idx = pd.MultiIndex.from_product(
        [[False, True],
         counts['week_start'].unique(),
         wd,
         all_slots,
         actions],
        names=['is_troll','week_start','weekday','time_slot_dt','action']
    )
    counts = (
        counts
        .set_index(['is_troll','week_start','weekday','time_slot_dt','action'])
        .reindex(full_idx, fill_value=0)
        .reset_index()
    )

    # Aggregate across weeks
    agg = (
        counts
        .groupby(['is_troll','weekday','time_slot_dt','action'], observed=False)['count']
        .agg(['mean','std','count'])
        .rename(columns={'count':'n_weeks'})
        .reset_index()
    )
    agg['sem'] = agg['std'] / np.sqrt(agg['n_weeks'])
    if ci:
        z = 1.96
        agg['ci_lower'] = agg['mean'] - z * agg['sem']
        agg['ci_upper'] = agg['mean'] + z * agg['sem']

    # Global normalization
    if normalize:
        m = agg['mean'].max() or 1
        cols = ['mean','ci_lower','ci_upper'] if ci else ['mean']
        agg[cols] = agg[cols] / m

    # Plotting setup
    hour_locator = mdates.HourLocator(interval=2)
    hour_fmt = mdates.DateFormatter('%H')
    start, end = base, base + dt.timedelta(days=1)

    for flag, title in [(False, 'Organics'), (True, 'Trolls')]:
        sub = agg[agg['is_troll'] == flag]
        fig, axes = plt.subplots(1, len(wd), figsize=(22, 3),
                                 sharex=True, sharey=True)
        for ax, day in zip(axes, wd):
            day_df = sub[sub['weekday'] == day].sort_values('time_slot_dt')
            for action in actions:
                sel = day_df[day_df['action'] == action]
                ax.plot(sel['time_slot_dt'], sel['mean'], label=action)
                if ci:
                    ax.fill_between(
                        sel['time_slot_dt'],
                        sel['ci_lower'],
                        sel['ci_upper'],
                        alpha=0.2
                    )
            ax.set_title(day[:3])
            ax.xaxis.set_major_locator(hour_locator)
            ax.xaxis.set_major_formatter(hour_fmt)
            ax.tick_params(axis='x', rotation=45, labelsize=6)
            ax.set_xlim(start, end)
            ax.set_ylabel('Normalized count' if normalize else 'Mean count')

        # Legend above panels
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            title='Action',
            loc='upper center',
            ncol=len(actions),
            bbox_to_anchor=(0.5, 1.02)
        )
        fig.suptitle(f'Weekly rhythm by action ({window}) — {title}', y=1.04)
        plt.tight_layout(rect=[0, 0, 1, 0.9])

        suffix = 'trolls' if flag else 'organics'
        fig.savefig(f'{save_dir}/weekly_rhythm_{suffix}_by_action.pdf',
                    bbox_inches='tight')
        # plt.show()

# ── Plot 3: Side-by-Side Heatmaps (Hourly) ────────────────────────────────────
def plot_heatmaps(df, percent=True, cmap='plasma'):
    """
    Plot hourly activity heatmaps for troll vs  non-troll, optionally as percentage of total activity.

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
    # plt.show()

# ── Plot 4: Polar Plot ─────────────────────────────────────────────────
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
    plt.savefig('figures/polar_ci_smooth_plot.pdf', bbox_inches='tight')

def plot_weekly_rhythm_by_user_combined(
    df,
    action_col='action',
    window='15min',
    ci=True,
    normalize=True,
    colors=('#3EB489','#FF6F61'),
    linestyles=('-', '--'),
    figsize=(18, 3)
):
    """
    Overlay trolls vs. organics: average per-user weekly rhythm ±95% CI, faceted by action.
    """
    # 1) Map timestamps into a dummy day
    base = dt.datetime(2000,1,1)
    # avoid full DataFrame copy for memory savings
    df2 = df  # operate in place or assume user handles copies
    df2['time_slot_dt'] = (
        df2['created_utc'].dt.floor(window)
             .apply(lambda t: base + dt.timedelta(hours=t.hour, minutes=t.minute))
    )

    # 2) Count per user × action × slot × weekday
    counts = (
        df2.groupby(
            ['is_troll', action_col, 'author', 'weekday', 'time_slot_dt'],
            observed=False
        )
        .size().reset_index(name='count')
    )

    # 3) Aggregate across users
    agg = (
        counts.groupby(
            ['is_troll', action_col, 'weekday', 'time_slot_dt'],
            observed=False
        )['count']
        .agg(['mean','std','count'])
        .rename(columns={'count':'n_users'})
        .reset_index()
    )
    agg['sem'] = agg['std'] / np.sqrt(agg['n_users'])
    if ci:
        z = norm.ppf(0.975)
        agg['ci_lower'] = agg['mean'] - z * agg['sem']
        agg['ci_upper'] = agg['mean'] + z * agg['sem']

    # 4) Optional normalization
    if normalize:
        for flag in [False, True]:
            for action in agg[action_col].unique():
                mask = (agg['is_troll']==flag) & (agg[action_col]==action)
                m = agg.loc[mask, 'mean'].max() or 1
                cols = ['mean'] + (['ci_lower','ci_upper'] if ci else [])
                agg.loc[mask, cols] = agg.loc[mask, cols] / m    # 3.5) Fill missing slots to zero across full grid
    slot = pd.Timedelta(window)
    interval = int(slot.total_seconds() // 60)
    n_slots = 1440 // interval
    time_slots = [base + dt.timedelta(minutes=interval * i) for i in range(n_slots)]
    weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    all_actions = sorted(df2[action_col].unique())
    full_idx = pd.MultiIndex.from_product(
        [[False, True], all_actions, weekdays, time_slots],
        names=['is_troll', action_col, 'weekday', 'time_slot_dt']
    )
    agg = (
        agg.set_index(['is_troll', action_col, 'weekday', 'time_slot_dt'])
           .reindex(full_idx, fill_value=0)
           .reset_index()
    )

    # 5) Prepare x-limits for full day
    slot = pd.Timedelta(window)
    start = base
    end = base + pd.Timedelta('1d')

    # 6) Plot grid of [actions × weekdays]
    actions = sorted(agg[action_col].unique())
    weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    fig, axes = plt.subplots(
        len(actions), len(weekdays),
        figsize=(figsize[0], figsize[1] * len(actions)),
        sharex=True, sharey=True
    )
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    # Locators & formatters
    hour_locator = mdates.HourLocator(interval=4)
    hour_fmt = mdates.DateFormatter('%H')
    start, end = base, base + dt.timedelta(days=1)

    for i, action in enumerate(actions):
        for j, wd in enumerate(weekdays):
            ax = axes[i, j]
            for (flag, label, color, ls) in zip(
                [False, True], ['Organics','Trolls'], colors, linestyles
            ):
                df_day = (
                    agg[(agg['is_troll']==flag) &
                        (agg[action_col]==action) &
                        (agg['weekday']==wd)]
                    .sort_values('time_slot_dt')
                )
                ax.plot(
                    df_day['time_slot_dt'], df_day['mean'],
                    label=label, color=color, linestyle=ls
                )
                if ci:
                    ax.fill_between(
                        df_day['time_slot_dt'],
                        df_day['ci_lower'],
                        df_day['ci_upper'],
                        color=color, alpha=0.2
                    )
            if i == 0:
                ax.set_title(wd[:3])
                ax.xaxis.set_major_locator(hour_locator)
                ax.xaxis.set_major_formatter(hour_fmt)
                ax.tick_params(axis='x', rotation=90, labelsize=6)
            if j == 0:
                ylabel = 'Mean per-user count'
                if normalize:
                    ylabel += ' (norm.)'
                ax.set_ylabel(ylabel)
            ax.set_xlim(start, end)
            if normalize:
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(0, None)
                ax.set_xlim(start, end)

    # Single legend
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper left', bbox_to_anchor=(1.02, 1),
        title='User Type'
    )

    fig.suptitle('Weekly rhythm by user and action: Trolls vs Organics', y=1.03)
    plt.tight_layout()
    plt.savefig('figures/weekly_rhythm_by_user_and_action.pdf', bbox_inches='tight')
    # plt.show()

# def plot_interevent_pmf_cdf(
#     df,
#     time_col='created_utc',
#     user_col='author',
#     troll_flag_col='is_troll',
#     xmin=0.1,      # in minutes
#     xmax=1e4,      # in minutes (~1 week)
#     n_bins=60
# ):
#     """
#     Left:  PMF (probability mass per log-spaced bin)
#     Right: Empirical CDF
#     of inter-event times (in minutes) for trolls vs organics.
#     """
#     # 1) compute inter-event times in minutes
#     df_sorted = df.sort_values([user_col, time_col])
#     deltas = (
#         df_sorted
#         .groupby(user_col)[time_col]
#         .diff()
#         .dt.total_seconds()
#         .dropna()
#     ) / 60.0

#     # 2) split troll vs  non-troll
#     is_troll = df_sorted[troll_flag_col].reindex(deltas.index)
#     troll_deltas   = deltas[is_troll]
#     human_deltas = deltas[~is_troll]

#     # 3) create log-spaced bins
#     bins = np.logspace(np.log10(xmin), np.log10(xmax), n_bins)

#     # 4) set up subplots
#     fig, (ax_pmf, ax_cdf) = plt.subplots(1, 2, figsize=(14,6))

#     # --- PMF: probability mass per bin ---
#     w_h = np.ones_like(human_deltas) / human_deltas.size
#     w_b = np.ones_like(troll_deltas)   / troll_deltas.size
#     ax_pmf.hist(human_deltas, bins=bins, weights=w_h,
#                 histtype='step', label='Organics', linewidth=1.5)
#     ax_pmf.hist(troll_deltas,   bins=bins, weights=w_b,
#                 histtype='step', label='Trolls',    linewidth=1.5)
#     ax_pmf.set_xscale('log')
#     ax_pmf.set_xlim(xmin, xmax)
#     ax_pmf.set_xlabel('Inter-event time (minutes)')
#     ax_pmf.set_ylabel('Probability per bin (PMF)')
#     ax_pmf.set_title('PMF of Inter-event Times')
#     ax_pmf.legend()
#     ax_pmf.grid(True, which='both', ls='--', alpha=0.4)
#     ax_pmf.xaxis.set_major_locator(mtick.LogLocator(base=10))
#     ax_pmf.xaxis.set_major_formatter(mtick.LogFormatter())

#     # --- CDF: empirical cumulative distribution ---
#     def ecdf(data):
#         x = np.sort(data)
#         y = np.arange(1, len(x)+1) / len(x)
#         return x, y

#     x_h, y_h = ecdf(human_deltas)
#     x_b, y_b = ecdf(troll_deltas)
#     ax_cdf.plot(x_h, y_h, drawstyle='steps-post', label='Organics', linewidth=1.5)
#     ax_cdf.plot(x_b, y_b, drawstyle='steps-post', label='Trolls',    linewidth=1.5)
#     ax_cdf.set_xscale('log')
#     ax_cdf.set_xlim(xmin, xmax)
#     ax_cdf.set_xlabel('Inter-event time (minutes)')
#     ax_cdf.set_ylabel('Cumulative probability (CDF)')
#     ax_cdf.set_title('Empirical CDF of Inter-event Times')
#     ax_cdf.legend()
#     ax_cdf.grid(True, which='both', ls='--', alpha=0.4)
#     ax_cdf.xaxis.set_major_locator(mtick.LogLocator(base=10))
#     ax_cdf.xaxis.set_major_formatter(mtick.LogFormatter())

#     plt.tight_layout()
#     plt.savefig('figures/interevent_pmf_cdf.pdf', bbox_inches='tight')
#     plt.show()

# def plot_interevent_pmf_cdf(
#     df,
#     time_col='created_utc',
#     user_col='author',
#     troll_flag_col='is_troll',
#     xmin=0.0,      # in minutes
#     xmax=1000.0,   # in minutes
#     n_bins=60
# ):
#     """
#     Left:  PMF (probability mass per linear-spaced bin)
#     Right: Empirical CDF
#     of inter-event times (in minutes) for trolls vs organics.
#     """
#     # 1) compute inter-event times in minutes
#     df_sorted = df.sort_values([user_col, time_col])
#     deltas = (
#         df_sorted
#         .groupby(user_col)[time_col]
#         .diff()
#         .dt.total_seconds()
#         .dropna()
#     ) / 60.0

#     # 2) split troll vs  non-troll
#     is_troll = df_sorted[troll_flag_col].reindex(deltas.index)
#     troll_deltas   = deltas[is_troll]
#     human_deltas = deltas[~is_troll]

#     # 3) create linear-spaced bins
#     bins = np.linspace(xmin, xmax, n_bins + 1)

#     # 4) set up subplots
#     fig, (ax_pmf, ax_cdf) = plt.subplots(1, 2, figsize=(14,6))

#     # --- PMF: probability mass per bin ---
#     w_h = np.ones_like(human_deltas) / human_deltas.size
#     w_b = np.ones_like(troll_deltas)   / troll_deltas.size
#     ax_pmf.hist(
#         human_deltas, bins=bins, weights=w_h,
#         histtype='step', label='Organics', linewidth=1.5
#     )
#     ax_pmf.hist(
#         troll_deltas,   bins=bins, weights=w_b,
#         histtype='step', label='Trolls',     linewidth=1.5
#     )
#     ax_pmf.set_xlim(xmin, xmax)
#     ax_pmf.set_xlabel('Inter-event time (minutes)')
#     ax_pmf.set_ylabel('Probability per bin (PMF)')
#     ax_pmf.set_title('PMF of Inter-event Times')
#     ax_pmf.legend()
#     ax_pmf.grid(True, ls='--', alpha=0.4)

#     # --- CDF: empirical cumulative distribution ---
#     def ecdf(data):
#         x = np.sort(data)
#         y = np.arange(1, len(x)+1) / len(x)
#         return x, y

#     x_h, y_h = ecdf(human_deltas)
#     x_b, y_b = ecdf(troll_deltas)
#     ax_cdf.plot(
#         x_h, y_h, drawstyle='steps-post',
#         label='Organics', linewidth=1.5
#     )
#     ax_cdf.plot(
#         x_b, y_b, drawstyle='steps-post',
#         label='Trolls',    linewidth=1.5
#     )
#     ax_cdf.set_xlim(xmin, xmax)
#     ax_cdf.set_xlabel('Inter-event time (minutes)')
#     ax_cdf.set_ylabel('Cumulative probability (CDF)')
#     ax_cdf.set_title('Empirical CDF of Inter-event Times')
#     ax_cdf.legend()
#     ax_cdf.grid(True, ls='--', alpha=0.4)

#     plt.tight_layout()
#     plt.savefig('figures/interevent_pmf_cdf_linear.pdf', bbox_inches='tight')
#     plt.show()

# def plot_interevent_pmf_cdf_custom_bins(
#     df,
#     time_col='created_utc',
#     user_col='author',
#     troll_flag_col='is_troll',
#     colors=('#3EB489','#FF6F61')
# ):
#     """
#     Left:  PMF (probability mass per custom bin)
#     Right: Empirical CDF
#     of inter-event times (in minutes) for trolls vs organics.
#     Uses custom bins: <1, 1-10, 10-60, 60-240, 240-1440 (1 day), >1440 minutes
#     """
#     # 1) compute inter-event times in minutes
#     df_sorted = df.sort_values([user_col, time_col])
#     deltas = (
#         df_sorted
#         .groupby(user_col)[time_col]
#         .diff()
#         .dt.total_seconds()
#         .dropna()
#     ) / 60.0

#     # 2) split troll vs  non-troll
#     is_troll = df_sorted[troll_flag_col].reindex(deltas.index)
#     troll_deltas = deltas[is_troll]
#     human_deltas = deltas[~is_troll]

#     # 3) create custom bins (in minutes)
#     bins = [0, 1, 5, 15, 60, 240, 720, 1440, 4320, np.inf]
#     bin_labels = [
#         r'$<$1 min',
#         r'1-5 min',
#         r'5-15 min', 
#         r'15-60 min', 
#         r'1-3 hrs', 
#         r'3-12 hrs',
#         r'12-24 hrs', 
#         r'24-72 hrs',
#         r'$>$72 hrs'
#     ]

#     # 4) set up subplots
#     fig, (ax_pmf, ax_cdf) = plt.subplots(1, 2, figsize=(14, 6))

#     # --- PMF: probability mass per bin ---
#     # Calculate histograms
#     counts_h, _ = np.histogram(human_deltas, bins=bins)
#     counts_b, _ = np.histogram(troll_deltas, bins=bins)
    
#     # Convert to probabilities
#     pmf_h = counts_h / human_deltas.size
#     pmf_b = counts_b / troll_deltas.size
    
#     # Plot as bar chart
#     x = np.arange(len(bin_labels))
#     width = 0.35
#     ax_pmf.bar(x - width/2, pmf_h, width, label='Organics', alpha=0.7, color=colors[0])
#     ax_pmf.bar(x + width/2, pmf_b, width, label='Trolls', alpha=0.7, color=colors[1])
    
#     ax_pmf.ylim(0, 0.3)
#     ax_pmf.set_xticks(x)
#     ax_pmf.set_xticklabels(bin_labels, rotation=45, ha='right')
    
#     ax_pmf.set_xlabel('Inter-event time', fontsize=14)
#     ax_pmf.set_ylabel('Probability (PMF)', fontsize=14)
#     ax_pmf.set_title('PMF of Inter-event Times (Custom Bins)')
#     ax_pmf.legend(fontsize=14)
#     ax_pmf.grid(True, ls='--', alpha=0.4)

#     # --- CDF: empirical cumulative distribution ---
#     def ecdf(data):
#         x = np.sort(data)
#         y = np.arange(1, len(x)+1) / len(x)
#         return x, y

#     x_h, y_h = ecdf(human_deltas)
#     x_b, y_b = ecdf(troll_deltas)
    
#     ax_cdf.plot(x_h, y_h, drawstyle='steps-post', label='Organics', linewidth=1.5, color=colors[0])
#     ax_cdf.plot(x_b, y_b, drawstyle='steps-post', label='Trolls', linewidth=1.5, color=colors[1])
    
#     # Add vertical lines for bin boundaries
#     for boundary in bins[1:-1]:
#         ax_cdf.axvline(boundary, color='gray', linestyle=':', alpha=0.5)
    
#     ax_cdf.set_xscale('log')
#     ax_cdf.set_xlabel('Inter-event time (minutes)', fontsize=14)
#     ax_cdf.set_ylabel('Cumulative probability (CDF)', fontsize=14)
#     ax_cdf.set_title('Empirical CDF of Inter-event Times', fontsize=14)
#     ax_cdf.legend(fontsize=14)
#     ax_cdf.grid(True, which='both', ls='--', alpha=0.4)
    
#     # Set x-axis limits to show all bins
#     ax_cdf.set_xlim(bins[0], bins[-2]*5)  # Exclude infinity from display
#     ax_cdf.set_ylim(0, 1.0)

#     plt.tight_layout()
#     plt.savefig('figures/interevent_pmf_cdf_custom_bins.pdf', bbox_inches='tight')
#     plt.show()

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

    # 2) split troll vs non-troll (align indices)
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
    # # a light frame helps separate inset visually
    # for spine in ax_cdf.spines.values():
    #     spine.set_alpha(0.6)

    # plt.tight_layout()
    plt.savefig('figures/interevent_pmf_cdf_custom_bins.pdf', bbox_inches='tight')
    plt.show()
    
# ── Main ──────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    os.makedirs('figures/', exist_ok=True)
    # Load data
    df_trolls = pd.read_csv('../data-analysis/data-trolls/reddit-suspicious-accounts.csv')
    df_trolls['Username'] = df_trolls['Username'].str.replace(r'^u/','',regex=True)
    trolls = set(df_trolls['Username'])
    df_active = pd.read_pickle('../data-analysis/data-analysis-timestamps/all_user_active_content_df.pkl')
    df = prepare_data(df_active, trolls)
    
    # # Draw all plots
    # plot_daily_collapsed_minute(df)
    # plot_weekly_rhythm(df)
    # plot_weekly_rhythm_action(df)
    # plot_weekly_rhythm_by_user_combined(df)
    plot_heatmaps(df)
    plot_polar(df)
    plot_interevent_pmf_cdf_custom_bins(df)