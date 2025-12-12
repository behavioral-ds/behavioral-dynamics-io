import os
import pandas as pd

if __name__ == "__main__":
    os.makedirs('exports/', exist_ok=True)
    df_trolls_p = pd.read_pickle('../data-analysis/sampled_matched_perturbed_df.pkl')
    df_trolls = df_trolls_p[(df_trolls_p.perturb_percent == 0.0) & (df_trolls_p.russian == 1) & (df_trolls_p.run == 0)]
    trolls = set(df_trolls['user'].values)

    df_active = pd.read_pickle('../data-analysis/data-analysis-timestamps/all_user_active_content_df.pkl')
    df_passive = pd.read_pickle('../data-analysis/data-analysis-timestamps/all_user_passive_content_df.pkl')

    df_trolls_active = df_active[df_active.author.isin(trolls)]
    df_trolls_passive = df_passive[df_passive.author.isin(trolls)]

    # Concatenate the active and passive dataframes
    df_trolls_combined = pd.concat([df_trolls_active, df_trolls_passive], ignore_index=True)

    # Find most common subreddits
    most_common_subreddits = df_trolls_combined['subreddit'].value_counts()

    print("Top 20 most common subreddits for trolls:")
    print(most_common_subreddits.head(20))

    print(f"Found {len(df_trolls_active)} active posts from trolls")
    print(f"Found {len(df_trolls_passive)} passive posts from trolls")

    # Create DataFrame for LaTeX
    top_n = 20
    df_table = pd.DataFrame({
        'Subreddit': most_common_subreddits.head(top_n).index,
        'Count': most_common_subreddits.head(top_n).values,
        'Percentage': (most_common_subreddits.head(top_n).values / most_common_subreddits.sum() * 100).round(2)
    })
    
    latex_code = df_table.to_latex(
        index=False,  # Don't show pandas index
        caption='Top 20 Most Frequent Subreddits for Troll Accounts',
        label='tab:troll_subreddits',
        position='htbp'
    )
    
    # Save to file
    with open('exports/troll-subreddits.tex', 'w') as f:
        f.write(latex_code)
    print("\nSaved to 'troll-subreddits.tex'")
