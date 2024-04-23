import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Define the DK metric for case 5 with the kernel K
def DK_metric(A_L, A_R, B_L, B_R, a, b, c):
    DK = ((a * (A_R - B_R) ** 2 + c * (A_L - B_L) ** 2) - 2 * b * (A_R - B_R) * (A_L - B_L)) / (a + 2 * b + c)
    return np.sqrt(max(DK, 0))  # Ensure non-negative result

# Generate random intervals
def generate_intervals(num_intervals, length_min, length_max, start_min, start_max):
    starts = np.random.randint(start_min, start_max - length_max, num_intervals)
    lengths = np.random.randint(length_min, length_max, num_intervals)
    ends = starts + lengths
    return list(zip(starts, ends))

# Parameters for interval generation
num_intervals = 50
length_min, length_max = 1, 10
start_min, start_max = 0, 100
intervals = generate_intervals(num_intervals, length_min, length_max, start_min, start_max)

# Cluster count
k = 5

# Storage for the best silhouette score and corresponding parameters
best_score = -1
best_params = None

# Iterate over possible values for a, b, and c
for a in np.linspace(0.1, 2, 20):
    for c in np.linspace(0.1, 2, 20):
        b = (a + c - 2) / 2  # Derived from a + c = 2 + 2b
        # Now we ensure that a * c > b^2 for the positive definiteness
        if a * c > b ** 2:
            n = len(intervals)
            distance_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):  # Compute upper triangle only, since the matrix is symmetric
                    distance_matrix[i, j] = DK_metric(*intervals[i], *intervals[j], a, b, c)
                    distance_matrix[j, i] = distance_matrix[i, j]  # Mirror the value for the lower triangle

            # Hierarchical clustering using the 'ward' linkage method
            Z = linkage(distance_matrix, 'ward')
            labels = fcluster(Z, k, criterion='maxclust')

            # Calculate the silhouette score
            silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")

            # Check if we got a better silhouette score
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_params = (a, b, c)

# Display the best parameters and the corresponding silhouette score
print("Best parameters (a, b, c):", best_params)
print("Best Silhouette Score:", best_score)

# Optional: Plot the dendrogram for the best clustering
plt.figure(figsize=(10, 8))
dendrogram(Z, labels=[f"Interval {chr(65 + i)}" for i in range(n)])
plt.title(f"Dendrogram with a={best_params[0]:.2f}, b={best_params[1]:.2f}, c={best_params[2]:.2f}, Silhouette Score: {best_score:.6f}")
plt.ylabel("Distance")
plt.show()


#Best parameters (a, b, c): (1.3, -0.29999999999999993, 0.1)
#Best Silhouette Score: 0.6366687282911612