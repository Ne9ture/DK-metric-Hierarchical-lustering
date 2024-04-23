import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Adjusted DK metric function
def DK_metric(A_L, A_R, B_L, B_R, a, c):
    DK = a * (A_L - B_L) ** 2 + c * (A_R - B_R) ** 2  # Distance formula for case 4
    return np.sqrt(DK)

# Generate intervals function
def generate_intervals(num_intervals, length_min, length_max, start_min, start_max):
    starts = np.random.randint(start_min, start_max - length_max, num_intervals)
    lengths = np.random.randint(length_min, length_max, num_intervals)
    ends = starts + lengths
    return list(zip(starts, ends))

# Parameters
num_intervals = 50
intervals = generate_intervals(num_intervals, 1, 10, 0, 100)
n = len(intervals)
best_score = -1
best_params = None

# Iterate over possible values of a (c will be 1 - a)
for a in np.linspace(0.1, 0.9, 9):  # Iterate between 0.1 to 0.9 for a
    c = 1 - a
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                A_L, A_R = intervals[i]
                B_L, B_R = intervals[j]
                distance_matrix[i, j] = DK_metric(A_L, A_R, B_L, B_R, a, c)

    Z = linkage(distance_matrix, 'ward')
    labels = fcluster(Z, 5, criterion='maxclust')
    silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")

    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_params = (a, c)

# Display best clustering result
plt.figure(figsize=(10, 8))
dendrogram(Z, labels=[f"Interval {chr(65+i)}" for i in range(n)])
plt.title(f"Best Parameters a={best_params[0]:.2f}, c={best_params[1]:.2f}, Silhouette Score: {best_score:.6f}")
plt.ylabel("Distance")
plt.show()

print("Best parameters:", best_params)
print("Best silhouette score:", best_score)

#Best parameters: (0.7000000000000001, 0.29999999999999993)
#Best silhouette score: 0.6445726808024677