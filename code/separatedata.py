# Data provided by the user
intervals_data = {
    "A": {"dim1": [-5, 0], "dim2": [1, 3]},
    "B": {"dim1": [-6, -1], "dim2": [1.5, 4]},
    "C": {"dim1": [-7, 0], "dim2": [2, 5]},
    "D": {"dim1": [7, 12], "dim2": [5, 9]},
    "E": {"dim1": [8, 11], "dim2": [6, 8]},
    "F": {"dim1": [1, 5], "dim2": [-5, -1]},
    "G": {"dim1": [2, 4], "dim2": [-9, 0]},
    "H": {"dim1": [2, 6], "dim2": [-8, -0.5]}
}

# Extracting interval data into a list for easier manipulation
interval_list = list(intervals_data.values())

# Calculate the distance matrix for the interval data
n = len(interval_list)
distance_matrix_example = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j:
            # Compute the distance using the DK_metric_fixed function for both dimensions
            dim1_distance = DK_metric_fixed(interval_list[i]["dim1"][0], interval_list[i]["dim1"][1], interval_list[j]["dim1"][0], interval_list[j]["dim1"][1], kernel_case=6)
            dim2_distance = DK_metric_fixed(interval_list[i]["dim2"][0], interval_list[i]["dim2"][1], interval_list[j]["dim2"][0], interval_list[j]["dim2"][1], kernel_case=6)
            # Combine the distances from both dimensions using Pythagorean theorem
            distance_matrix_example[i, j] = np.sqrt(dim1_distance**2 + dim2_distance**2)
        else:
            distance_matrix_example[i, j] = 0  # Distance to self is zero

# Apply hierarchical clustering to the distance matrix
Z_example = linkage(distance_matrix_example, method='ward')

# Here we assume the user wants to examine clustering with varying numbers of clusters as before
# For simplicity, let's use 3 clusters which can be changed if needed
k_example = 3
labels_example = fcluster(Z_example, k_example, criterion='maxclust')

# Calculate various cluster quality metrics using the distance matrix
quality_metrics_example = calculate_cluster_quality(distance_matrix_example, labels_example)

# Plot the dendrogram for the clustering result
plt.figure(figsize=(10, 8))
dendrogram(Z_example, labels=list(intervals_data.keys()), leaf_rotation=90, leaf_font_size=12)
plt.title(f"Example Data Clustering - Kernel Case 6 - Silhouette Score: {quality_metrics_example['silhouette_score']:.6f}")
plt.ylabel("Distance")
plt.show()

# Output the quality metrics and cluster labels for verification
quality_metrics_example, labels_example

