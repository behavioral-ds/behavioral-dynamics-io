import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def calc_silhouette_scores(X: np.ndarray, k_clusters: range):
    silhouette_scores = []

    for k in k_clusters:
        # Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=1234)
        kmeans.fit(X)
        
        # Calculate silhouette score
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)

    # Print the best k value
    best_k = k_clusters[np.argmax(silhouette_scores)]
    print(f"The optimal number of clusters based on silhouette score is: {best_k}")
    return silhouette_scores

def calc_gap_statistic(X: np.ndarray, k_clusters: range, n_repeats=10):
    """
    Calculate the Gap Statistic for K-means clustering.
    
    Parameters:
    - X: input data
    - k_clusters: range of k values to test
    - n_repeats: number of reference datasets to generate
    
    Returns:
    - gap_values: calculated Gap Statistic values
    - optimal_k: optimal number of clusters
    """
    gap_values = []

    # Generate reference datasets
    for k in k_clusters:
        # Fit KMeans on the original data
        kmeans = KMeans(n_clusters=k, random_state=1234)
        kmeans.fit(X)
        orig_inertia = kmeans.inertia_

        # Generate reference datasets and calculate their inertia
        reference_inertia = []
        for _ in range(n_repeats):
            # Create a random uniform distribution dataset
            random_data = np.random.random_sample(size=X.shape)
            kmeans_random = KMeans(n_clusters=k, random_state=42)
            kmeans_random.fit(random_data)
            reference_inertia.append(kmeans_random.inertia_)

        # Calculate the gap statistic
        mean_ref_inertia = np.mean(reference_inertia)
        gap = np.log(mean_ref_inertia) - np.log(orig_inertia)
        gap_values.append(gap)

    # Find the optimal number of clusters
    optimal_k = k_clusters[np.argmax(gap_values)]

    return gap_values, optimal_k

def calc_elbow_inertia(
    X: np.ndarray,
    k_clusters: range
):
    inertia = []
    
    for k in k_clusters:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    
    return inertia

def plot_elbow_method(
    elbow_inertia: list, 
    k_clusters: range
):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(k_clusters, elbow_inertia, marker='o', c='blue')
    ax.set_xlabel(r'\textbf{Number of clusters ($k$)}', fontsize=14)
    ax.set_ylabel(r'\textbf{Inertia}', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(min(k_clusters), max(k_clusters))
    ax.set_ylim(0, 25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.25, zorder=1)
    fig.savefig('figures/k-means/elbow-method.pdf', facecolor='white', bbox_inches='tight')
    plt.close(fig)

def plot_silhouette_score(
    silhouette_scores: list,
    k_clusters: range
):
    # Plot the Silhouette Score
    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot the Silhouette Score
    ax.plot(k_clusters, silhouette_scores, marker='o', c='blue')
    ax.set_xlabel(r'\textbf{Number of clusters ($k$)}', fontsize=14)
    ax.set_ylabel(r'\textbf{Silhouette score}', fontsize=14)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(min(k_clusters), max(k_clusters))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    # Find the best k value based on the silhouette score
    best_k = k_clusters[np.argmax(silhouette_scores)]
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.25, zorder=1)
    ax.axvline(best_k, linestyle='--', color='green', label=f'Optimal $k$: {best_k}')
    ax.legend(frameon=True, framealpha=1.0, edgecolor='black', handlelength=1, fontsize=14, loc="upper right")

    fig.savefig('figures/k-means/silhouette-score.pdf', facecolor='white', bbox_inches='tight')    
    plt.close(fig)

    print(f"The optimal number of clusters based on silhouette score is: {best_k}")

def plot_gap_statistic(
    gap_values: list,
    k_clusters: range,
    optimal_k: int
):
    # Plot the Gap Statistic
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(k_clusters, gap_values, marker='o', c='blue')
    ax.set_xlabel(r'\textbf{Number of clusters ($k$)}', fontsize=14)
    ax.set_ylabel(r'\textbf{Gap statistic}', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim(0, 4)
    ax.set_xlim(min(k_clusters), max(k_clusters))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.25, zorder=1)
    ax.axvline(optimal_k, linestyle='--', color='green', label=f'Optimal $k$: {optimal_k}')
    ax.legend(frameon=True, framealpha=1.0, edgecolor='black', handlelength=1, fontsize=14, loc="upper left")

    fig.savefig('figures/k-means/gap-statistic.pdf', facecolor='white', bbox_inches='tight') 
    plt.close(fig)   
    # Print the optimal number of clusters
    print(f"The optimal number of clusters based on gap statistic is: {optimal_k}")