import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Define the DK metric with different choices of the kernel K
def DK_metric(A_L, A_R, B_L, B_R, kernel_case):
    """
    Calculate the DK metric for intervals A and B with different kernel cases.

    Args:
    A_L, A_R : float
        Left and right bounds of interval A.
    B_L, B_R : float
        Left and right bounds of interval B.
    kernel_case : int
        The case of the kernel function to use.

    Returns:
    DK : float
        The calculated DK metric.
    """
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

    return np.sqrt(DK)

# Example intervals
A_L, A_R = 2, 5  # Interval A: [2, 5]
B_L, B_R = 3, 6  # Interval B: [3, 6]

# Calculate DK metric for different kernel cases
dk_case_1 = DK_metric(A_L, A_R, B_L, B_R, kernel_case=1)
dk_case_2 = DK_metric(A_L, A_R, B_L, B_R, kernel_case=2)
dk_case_3 = DK_metric(A_L, A_R, B_L, B_R, kernel_case=3)
dk_case_4 = DK_metric(A_L, A_R, B_L, B_R, kernel_case=4)
dk_case_5 = DK_metric(A_L, A_R, B_L, B_R, kernel_case=5)
dk_case_6 = DK_metric(A_L, A_R, B_L, B_R, kernel_case=6)

print(dk_case_1, dk_case_2, dk_case_3, dk_case_4, dk_case_5, dk_case_6)


import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

import numpy as np

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

# Parameters
num_intervals = 20  # Number of intervals
length_min, length_max = 1, 10  # Min and max length of intervals
start_min, start_max = 0, 100  # Min and max start of intervals

# Generate intervals
intervals = generate_intervals(num_intervals, length_min, length_max, start_min, start_max)

# Number of clusters (for example purposes)
k = 3

print("Generated Intervals:", intervals)

# Results storage
silhouette_scores = []

for kernel_case in range(1, 7):
    n = len(intervals)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                A_L, A_R = intervals[i]
                B_L, B_R = intervals[j]
                distance_matrix[i, j] = DK_metric(A_L, A_R, B_L, B_R, kernel_case)
            else:
                distance_matrix[i, j] = 0  # Distance to self is zero

    # Hierarchical clustering
    Z = linkage(distance_matrix, method='ward')
    labels = fcluster(Z, k, criterion='maxclust')
'''
    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")
    silhouette_scores.append(silhouette_avg)

    # Plot the dendrogram
    plt.figure(figsize=(10, 8))
    dendrogram(Z, labels=[f"Interval {chr(65+i)}" for i in range(n)])
    plt.title(f"Kernel Case {kernel_case} - Silhouette Score: {silhouette_avg:.6f}")
    plt.ylabel("Distance")
    plt.show()
'''
# Print the silhouette scores for comparison
#print("Silhouette Scores by Kernel Case:", silhouette_scores)


def calculate_cluster_quality(distance_matrix, labels):
    """
    Calculate various cluster quality metrics.

    Args:
    distance_matrix : ndarray
        The precomputed distance matrix.
    labels : ndarray
        The labels of the clusters.

    Returns:
    quality_metrics : dict
        A dictionary containing various quality metrics.
    """
    silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")
    db_score = davies_bouldin_score(distance_matrix, labels)
    
    # 计算组内平均距离（内聚度）
    intra_cluster_distances = [distance_matrix[labels == i][:, labels == i] for i in np.unique(labels)]
    intra_avg_distance = np.mean([np.mean(distances) for distances in intra_cluster_distances if distances.size > 0])

    # 将所有指标存储在字典中
    quality_metrics = {
        'silhouette_score': silhouette_avg,
        'davies_bouldin_score': db_score,
        'intra_cluster_distance': intra_avg_distance
    }
    return quality_metrics

# Results storage
all_quality_metrics = []

for kernel_case in range(1, 7):
    n = len(intervals)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                A_L, A_R = intervals[i]
                B_L, B_R = intervals[j]
                distance_matrix[i, j] = DK_metric(A_L, A_R, B_L, B_R, kernel_case)
            else:
                distance_matrix[i, j] = 0  # Distance to self is zero

    # Hierarchical clustering
    Z = linkage(distance_matrix, method='ward')
    labels = fcluster(Z, k, criterion='maxclust')

    # Calculate various cluster quality metrics
    quality_metrics = calculate_cluster_quality(distance_matrix, labels)
    all_quality_metrics.append(quality_metrics)

    # Plot the dendrogram
    plt.figure(figsize=(10, 8))
    dendrogram(Z, labels=[f"Interval {chr(65+i)}" for i in range(n)])
    plt.title(f"Kernel Case {kernel_case} - Silhouette Score: {quality_metrics['silhouette_score']:.6f}")
    plt.ylabel("Distance")
    plt.show()

# Print the quality metrics for comparison
for idx, quality_metrics in enumerate(all_quality_metrics, 1):
    print(f"Kernel Case {idx}:")
    print(f"Silhouette Score: {quality_metrics['silhouette_score']}")
    print(f"Davies-Bouldin Score: {quality_metrics['davies_bouldin_score']}")
    print(f"Average Intra-Cluster Distance: {quality_metrics['intra_cluster_distance']}\n")
