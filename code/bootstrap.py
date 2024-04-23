import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def DK_metric(A_L, A_R, B_L, B_R, kernel_case):
    # Initialize DK
    DK = 0
    try:
        # Case 1: Kernel K is not positive definite, only midpoints matter
        if kernel_case == 1:
            A_M = (A_L + A_R) / 2
            B_M = (B_L + B_R) / 2
            DK = (A_M - B_M) ** 2

    # Case 2: Kernel K values that focus on the endpoints
        elif kernel_case == 2:
            DK = ((A_R - B_R) ** 2 + (A_L - B_L) ** 2) / 2

    # Case 3: Kernel K with a specific weight distribution
        elif kernel_case == 3:
            a = c = 2.0
            b = -1.9
            DK = ((a + b) / 2) * (A_R - B_R) ** 2 + 2 * (a - b) * ((A_R + A_L) / 2 - (B_R + B_L) / 2) ** 2

    # Case 4: Kernel K focusing on separate left and right boundaries
        elif kernel_case == 4:
            a = 0.7000000000000001
            c = 0.29999999999999993
            DK = a * (A_R - B_R) ** 2 + c * (A_L - B_L) ** 2
    # Case 5: Kernel K with a comprehensive consideration of boundaries
        elif kernel_case == 5:
            a = 0.8999999999999999
            b = 0.8485281374238569
            c = 0.7999999999999999
            DK = ((a * (A_R - B_R) ** 2 + c * (A_L - B_L) ** 2) - 2 * b * (A_R - B_R) * (A_L - B_L)) / (2 + 2 * b + c)
    # Case 5*: Kernel K with a comprehensive consideration of boundaries
        elif kernel_case == 6:
            a = 1.3
            b = -0.29999999999999993
            c = 0.1
            DK = ((a * (A_R - B_R) ** 2 + c * (A_L - B_L) ** 2) - 2 * b * (A_R - B_R) * (A_L - B_L)) / (2 + 2 * b + c)

        else:
            raise ValueError("Invalid kernel case.")
            # Your cases...
            
            
        # Ensure DK is non-negative before taking square root
        if DK < 0:
            raise ValueError("Negative value under square root")
        return np.sqrt(DK)
    except ValueError as e:
        print(f"Error in DK_metric with kernel case {kernel_case}: {e}")
        return None

from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

def perform_clustering_and_analysis(data, kernel_case, n_clusters=5):
    """Perform hierarchical clustering and calculate quality metrics."""
    distance_matrix = np.array([[DK_metric(A_L, A_R, B_L, B_R, kernel_case) for (A_L, A_R) in data] for (B_L, B_R) in data])
    labels = fcluster(linkage(distance_matrix, 'ward'), n_clusters, criterion='maxclust')
    
    if len(set(labels)) < 2:  # 检查是否有多于一个聚类
        print(f"Warning: Only one cluster detected in Kernel Case {kernel_case}.")
        silhouette_avg = None  # 不能计算轮廓系数
        db_score = None  # 不能计算Davies-Bouldin指数
    else:
        silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")
        db_score = davies_bouldin_score(distance_matrix, labels)
    
    return silhouette_avg, db_score, labels


def bootstrap_samples(data, n_samples):
    """ Generate bootstrap samples from the dataset. """
    # Create an empty list to hold the bootstrap samples
    bootstrap_samples = []
    # Generate n_samples bootstrap samples
    for _ in range(n_samples):
        # Randomly choose indices with replacement
        indices = np.random.randint(0, len(data), len(data))
        # Append the sampled data to the list of bootstrap samples
        bootstrap_samples.append([data[i] for i in indices])
    return bootstrap_samples

def generate_intervals(num_intervals, length_min, length_max, start_min, start_max):
    """
    Generate a list of intervals.

    Args:
    num_intervals : int
        Number of intervals to generate.
    length_min, length_max : int
        Minimum and maximum possible length of intervals.
    start_min, start_max : int
        Minimum and maximum starting points of intervals.

    Returns:
    intervals : list of tuples
        Generated intervals represented as (start, end).
    """
    starts = np.random.randint(start_min, start_max - length_max, num_intervals)
    lengths = np.random.randint(length_min, length_max, num_intervals)
    ends = starts + lengths
    return list(zip(starts, ends))

def main():
    num_intervals = 500
    length_min, length_max = 1, 10
    start_min, start_max = 0, 100
    n_clusters = 5
    kernel_cases = 6
    n_samples = 30

    intervals = generate_intervals(num_intervals, length_min, length_max, start_min, start_max)
    print("Generated Intervals:", len(intervals))

    for kernel_case in range(1, 7):
        print(f"Analyzing Kernel Case {kernel_case}")
        silhouette_scores = []
        db_scores = []

        samples = bootstrap_samples(intervals, n_samples)
        for sample in samples:
            distance_matrix = np.array([[DK_metric(A_L, A_R, B_L, B_R, kernel_case) for A_L, A_R in sample] for B_L, B_R in sample])
            if np.isnan(distance_matrix).any():
                print("NaN found in distance matrix, skipping sample")
                continue
            labels = fcluster(linkage(distance_matrix, 'ward'), n_clusters, criterion='maxclust')
            silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")
            db_score = davies_bouldin_score(distance_matrix, labels)
            
            silhouette_scores.append(silhouette_avg)
            db_scores.append(db_score)

        # Print statistics for current kernel case
        if silhouette_scores and db_scores:  # Check lists are not empty
            silhouette_mean = np.mean(silhouette_scores)
            silhouette_std = np.std(silhouette_scores)
            db_mean = np.mean(db_scores)
            db_std = np.std(db_scores)
            print(f"Silhouette Mean = {silhouette_mean}, Std Dev = {silhouette_std}")
            print(f"Davies-Bouldin Mean = {db_mean}, Std Dev = {db_std}\n")

if __name__ == "__main__":
    main()

