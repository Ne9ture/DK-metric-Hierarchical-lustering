import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Adjusted DK_metric function for case 5, considering the conditions for kernel K
def DK_metric(A_L, A_R, B_L, B_R, a, b, c):
    numerator = a * (A_R - B_R) ** 2 + c * (A_L - B_L) ** 2 - 2 * b * (A_R - B_R) * (A_L - B_L)
    denominator = a + 2 * b + c
    # Ensure that the denominator is positive to avoid invalid sqrt operation
    if denominator <= 0:
        raise ValueError("Denominator is non-positive, leading to an invalid DK value.")
    DK = numerator / denominator
    return np.sqrt(max(DK, 0))  # Ensure the square root of a non-negative value

def generate_intervals(num_intervals, length_min, length_max, start_min, start_max):
    starts = np.random.randint(start_min, start_max - length_max, num_intervals)
    lengths = np.random.randint(length_min, length_max, num_intervals)
    ends = starts + lengths
    return list(zip(starts, ends))

# Parameters
num_intervals = 50  # Number of intervals
length_min, length_max = 1, 10  # Min and max length of intervals
start_min, start_max = 0, 100  # Min and max start of intervals

# Generate intervals
intervals = generate_intervals(num_intervals, length_min, length_max, start_min, start_max)

# Number of clusters (for example purposes)
k = 5

print("Generated Intervals:", intervals)

# Results storage
best_score = -1
best_params = (None, None, None)

# Iterate over a range of positive values for a and c, ensuring that a * c > b^2
a_values = np.linspace(0.1, 2, 20)
c_values = np.linspace(0.1, 2, 20)

for a in a_values:
    for c in c_values:
        for b in np.linspace(-np.sqrt(a * c), np.sqrt(a * c), 20):  # b^2 must be less than a*c
            if a * c > b**2:  # This ensures K(1,1)K(-1, -1) > K(1, -1)^2
                # Compute the silhouette score using these parameters
                n = len(intervals)
                distance_matrix = np.zeros((n, n))

                for i in range(n):
                    for j in range(n):
                        if i != j:
                            A_L, A_R = intervals[i]
                            B_L, B_R = intervals[j]
                            distance_matrix[i, j] = DK_metric(A_L, A_R, B_L, B_R, a, b, c)
                        else:
                            distance_matrix[i, j] = 0  # Distance to self is zero

                # Hierarchical clustering
                Z = linkage(distance_matrix, method='ward')
                labels = fcluster(Z, k, criterion='maxclust')

                # Calculate Silhouette Score
                silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")

                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    best_params = (a, b, c)

# After finding the best parameters, you can output them and/or use them for further clustering
print("Best parameters (a, b, c):", best_params)
print("Best Silhouette Score:", best_score)

# Optional: Plot dendrogram or other relevant clustering visuals using best_params
plt.figure(figsize=(10, 8))
dendrogram(Z, labels=[f"Interval {chr(65+i)}" for i in range(n)])
plt.title(f"Best Kernel Parameters - Silhouette Score: {best_score:.6f}")
plt.ylabel("Distance")
plt.show()

#Best parameters (a, b, c): (0.8999999999999999, 0.8485281374238569, 0.7999999999999999)
#Best Silhouette Score: 0.6772590462298276