import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from variables import action_labels
from variables import state_labels

plt.rcParams.update({
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.sans-serif": "Libertine",
})

plt.rc('text.latex', preamble=r'\usepackage{amsmath, bm, mathrsfs, mathtools, color}')

def generate_frequency_plots(
    s_counts_arr: np.ndarray,
    a_counts_arr: np.ndarray,
    name: str
):
    # Create the plot for trolls
    plt.figure(figsize=(4, 7))

    # Create positions for the bars
    y_pos = np.arange(len(state_labels) + len(action_labels))
    
    # Combine counts and labels
    combined_counts = np.concatenate([s_counts_arr, a_counts_arr])
    combined_labels = state_labels + action_labels

    # Create different colors for states and actions
    state_colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(state_labels)))
    action_colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(action_labels)))
    all_colors = list(state_colors) + list(action_colors)

    # Create the bar plot
    bars = plt.barh(y_pos, combined_counts, color=all_colors, edgecolor='black', linewidth=0.5)

    # Set y-axis labels
    plt.yticks(y_pos, combined_labels, fontsize=12)
    
    plt.xticks(fontsize=12)
    # Customize the plot
    plt.xlabel(r"\textbf{Frequency}", fontsize=14)
    # plt.ylabel(r"\textbf{States and Actions}", fontsize=14)

    # Add value labels on bars
    for bar, count in zip(bars, combined_counts):
        if count > 0:  # Only label bars with non-zero values
            width = bar.get_width()
            plt.text(width + max(combined_counts)*0.02, bar.get_y() + bar.get_height()/2, 
                    f'{int(count):,}', ha='left', va='center', fontsize=11)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust x-axis limit to accommodate labels
    plt.xlim(0, max(combined_counts) * 1.15)

    delta = 0.75
    plt.ylim(min(y_pos) - delta, max(y_pos) + delta)
    
    # Add a vertical line to separate states from actions
    plt.axhline(y=len(state_labels)-0.5, color='black', linestyle='--', alpha=0.7, linewidth=1)
    # plt.text(max(combined_counts)*0.8, len(state_labels)-0.2, 'States → Actions', 
    #          ha='center', va='bottom', fontsize=10, color='gray')

    # Add labels
    plt.text(max(combined_counts) * 0.85, 5.3, r"\textbf{States}", 
        ha='left', va='bottom', fontsize=14, color='black')
    plt.text(max(combined_counts) * 0.85, 14.3, r"\textbf{Actions}", 
        ha='left', va='bottom', fontsize=14, color='black')

    plt.tight_layout()
    plt.savefig(f'figures/states-actions-frequencies-{name}.pdf', 
                facecolor='white', bbox_inches='tight')
    # plt.show()
    plt.close()


def get_state_action_counts(df: pd.DataFrame, column: str):
    # Extract all state-action pairs into a flat list
    all_pairs = [pair for traj in df[column] for pair in traj]

    # Separate states and actions
    all_states = [pair[0] for pair in all_pairs]
    all_actions = [pair[1] for pair in all_pairs]

    # Count frequencies
    state_counts = Counter(all_states)
    action_counts = Counter(all_actions)
    return state_counts, action_counts

def counter_to_array(counter_obj, num_items):
    """Convert Counter object to numpy array with counts in correct order"""
    # Create array with zeros for all possible indices
    arr = np.zeros(num_items)
    for idx, count in counter_obj.items():
        arr[idx] = count
    return arr

if __name__ == "__main__":
    save_folder = 'figures/'
    folder_path = '../data-analysis/'
    file_name_organics = 'organics_df_full_traj.pkl'
    file_name_trolls = 'sampled_matched_perturbed_df.pkl'
    path_trolls = folder_path + file_name_trolls
    path_organics = folder_path + file_name_organics
    
    # Including perturbed versions
    df_trolls_p = pd.read_pickle(path_trolls)
    df_organics = pd.read_pickle(path_organics)
    
    df_trolls = df_trolls_p[(df_trolls_p.perturb_percent == 0.0) & (df_trolls_p.russian == 1) & (df_trolls_p.run == 0)]
    df_organics = pd.read_pickle(path_organics)

    s_counts_trolls, a_counts_trolls = get_state_action_counts(df=df_trolls, column='traj')
    s_counts_organics, a_counts_organics = get_state_action_counts(df=df_organics, column='traj')

    # Prints
    # print("Trolls - States:", s_counts_trolls)
    # print("Trolls - Actions:", a_counts_trolls)
    # print("Organics - States:", s_counts_organics)
    # print("Organics - Actions:", a_counts_organics)

    # Convert Counter objects to arrays
    num_states = len(state_labels)  # Should be 12
    num_actions = len(action_labels)  # Should be 6
    
    s_counts_trolls_arr = counter_to_array(s_counts_trolls, num_states)
    a_counts_trolls_arr = counter_to_array(a_counts_trolls, num_actions)
    
    s_counts_organics_arr = counter_to_array(s_counts_organics, num_states)
    a_counts_organics_arr = counter_to_array(a_counts_organics, num_actions)

    combined_counts_trolls = np.concatenate([s_counts_trolls_arr, a_counts_trolls_arr])
    combined_labels = state_labels + action_labels

    num_states = len(state_labels)  # Should be 12
    num_actions = len(action_labels)  # Should be 6
    
    s_counts_trolls_arr = counter_to_array(s_counts_trolls, num_states)
    a_counts_trolls_arr = counter_to_array(a_counts_trolls, num_actions)

    s_counts_organics_arr = counter_to_array(s_counts_organics, num_states)
    a_counts_organics_arr = counter_to_array(a_counts_organics, num_actions)
    
    generate_frequency_plots(
        s_counts_arr=s_counts_trolls_arr, 
        a_counts_arr=a_counts_trolls_arr,
        name='trolls'
    )
    
    generate_frequency_plots(
        s_counts_arr=s_counts_organics_arr, 
        a_counts_arr=a_counts_organics_arr, 
        name='organics'
    )
