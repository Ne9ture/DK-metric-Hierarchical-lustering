import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

def DK_metric(A_L, A_R, B_L, B_R, kernel_case):
    if kernel_case != 5:
        raise ValueError("Invalid kernel case. Only case 5 is supported.")
    a = 1.3
    b = -0.29999999999999993
    c = 0.1
    DK = ((a * (A_R - B_R) ** 2 + c * (A_L - B_L) ** 2) - 2 * b * (A_R - B_R) * (A_L - B_L)) / (2 + 2 * b + c)
    return np.sqrt(DK)

def generate_intervals(num_intervals, length_min, length_max, start_min, start_max):
    starts = np.random.randint(start_min, start_max - length_max, num_intervals)
    lengths = np.random.randint(length_min, length_max, num_intervals)
    ends = starts + lengths
    return list(zip(starts, ends))

def calculate_cluster_quality(distance_matrix, labels):
    silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")
    db_score = davies_bouldin_score(distance_matrix, labels)
    intra_cluster_distances = [distance_matrix[labels == i][:, labels == i] for i in np.unique(labels)]
    intra_avg_distance = np.mean([np.mean(distances) for distances in intra_cluster_distances if distances.size > 0])
    quality_metrics = {
        'silhouette_score': silhouette_avg,
        'davies_bouldin_score': db_score,
        'intra_cluster_distance': intra_avg_distance
    }
    return quality_metrics

# Parameters
num_intervals = 500
length_min, length_max = 1, 10
start_min, start_max = 0, 100
k = 3

# Generate intervals
intervals = generate_intervals(num_intervals, length_min, length_max, start_min, start_max)

n = len(intervals)
distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            A_L, A_R = intervals[i]
            B_L, B_R = intervals[j]
            distance_matrix[i, j] = DK_metric(A_L, A_R, B_L, B_R, kernel_case=5)
        else:
            distance_matrix[i, j] = 0  # Distance to self is zero

# Hierarchical clustering
Z = linkage(distance_matrix, method='ward')
labels = fcluster(Z, k, criterion='maxclust')

# Calculate various cluster quality metrics
quality_metrics = calculate_cluster_quality(distance_matrix, labels)

# Plot the dendrogram
plt.figure(figsize=(10, 8))
dendrogram(Z, labels=[f"Interval {chr(65+i)}" for i in range(n)])
plt.title(f"Kernel Case 5 - Silhouette Score: {quality_metrics['silhouette_score']:.6f}")
plt.ylabel("Distance")
plt.show()

# Print the quality metrics
print(f"Silhouette Score: {quality_metrics['silhouette_score']}")
print(f"Davies-Bouldin Score: {quality_metrics['davies_bouldin_score']}")
print(f"Average Intra-Cluster Distance: {quality_metrics['intra_cluster_distance']}")
