import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def DK_metric(A_L, A_R, B_L, B_R, a, b):
    c = a  # Ensure c = a
    DK = ((a + b) / 2) * (A_R - B_R) ** 2 + 2 * (a - b) * ((A_R + A_L) / 2 - (B_R + B_L) / 2) ** 2
    return np.sqrt(DK)

def generate_intervals(num_intervals, length_min, length_max, start_min, start_max):
    starts = np.random.randint(start_min, start_max - length_max, num_intervals)
    lengths = np.random.randint(length_min, length_max, num_intervals)
    ends = starts + lengths
    return list(zip(starts, ends))

intervals = generate_intervals(50, 1, 10, 0, 100)  # Generated intervals

best_score = -1
best_params = None
best_labels = None
best_distance_matrix = None

# Iterate over possible values of a and b
for a in np.linspace(0.1, 2, 20):
    for b in np.linspace(-a + 0.1, a - 0.1, 10):
        n = len(intervals)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    A_L, A_R = intervals[i]
                    B_L, B_R = intervals[j]
                    distance_matrix[i, j] = DK_metric(A_L, A_R, B_L, B_R, a, b)
                else:
                    distance_matrix[i, j] = 0
        
        Z = linkage(distance_matrix, method='ward')
        labels = fcluster(Z, 5, criterion='maxclust')
        silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")
        
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_params = (a, b)
            best_labels = labels
            best_distance_matrix = distance_matrix

# Display best clustering result
plt.figure(figsize=(10, 8))
dendrogram(Z, labels=[f"Interval {chr(65+i)}" for i in range(n)])
plt.title(f"Best Parameters a={best_params[0]}, b={best_params[1]}, Silhouette Score: {best_score:.6f}")
plt.ylabel("Distance")
plt.show()

print("Best parameters:", best_params)
print("Best silhouette score:", best_score)


#Best parameters: (2.0, -1.9)
#Best silhouette score: 0.6506865068532518