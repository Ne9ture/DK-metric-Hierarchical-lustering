import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

def DK_metric(A_L, A_R, B_L, B_R, kernel_case):
    if kernel_case != 5:
        raise ValueError("Invalid kernel case. Only case 6 is supported.")
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

def monte_carlo_simulation(num_simulations, num_intervals, length_min, length_max, start_min, start_max, k):
    results = []
    for _ in range(num_simulations):
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
                    distance_matrix[i, j] = 0

        Z = linkage(distance_matrix, method='ward')
        labels = fcluster(Z, k, criterion='maxclust')
        quality_metrics = calculate_cluster_quality(distance_matrix, labels)
        results.append(quality_metrics)
    
    return results

# Run the simulation
num_simulations = 100
results = monte_carlo_simulation(num_simulations, 500, 1, 10, 0, 100, 5)

# Analyze results
silhouettes = [result['silhouette_score'] for result in results]
db_scores = [result['davies_bouldin_score'] for result in results]
intra_distances = [result['intra_cluster_distance'] for result in results]

print("Average Silhouette Score:", np.mean(silhouettes))
print("Average Davies-Bouldin Score:", np.mean(db_scores))
print("Average Intra-Cluster Distance:", np.mean(intra_distances))
