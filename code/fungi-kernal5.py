# Fix the DK_metric function to support the correct case check and adapt it for fungal data clustering
def DK_metric_fixed(A_L, A_R, B_L, B_R, kernel_case):
    if kernel_case != 6:
        raise ValueError("Invalid kernel case. Only case 6 is supported.")
    a = 1.3
    b = -0.29999999999999993
    c = 0.1
    DK = ((a * (A_R - B_R) ** 2 + c * (A_L - B_L) ** 2) - 2 * b * (A_R - B_R) * (A_L - B_L)) / (2 + 2 * b + c)
    return np.sqrt(DK)

# Prepare to calculate the distance matrix using the corrected metric for the fungi dataset
# Assume A_L and A_R correspond to the lower and upper bounds of a particular feature, e.g., 'spores 1d lower' and 'spores 1d upper'
n = len(fungi_data)
distance_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j:
            A_L, A_R = fungi_data.loc[i, ['spores 1d lower', 'spores 1d upper']]
            B_L, B_R = fungi_data.loc[j, ['spores 1d lower', 'spores 1d upper']]
            distance_matrix[i, j] = DK_metric_fixed(A_L, A_R, B_L, B_R, kernel_case=6)
        else:
            distance_matrix[i, j] = 0  # Distance to self is zero

# Proceed with hierarchical clustering using the modified distance matrix
Z = linkage(distance_matrix, method='ward')
k = 10  # Using 10 clusters as the dataset suggests
labels = fcluster(Z, k, criterion='maxclust')

# Calculate various cluster quality metrics using the custom distance matrix
quality_metrics = calculate_cluster_quality(distance_matrix, labels)

# Plot the dendrogram
plt.figure(figsize=(10, 8))
dendrogram(Z, labels=[fungi_data.loc[i, 'name'] for i in range(n)], leaf_rotation=90, leaf_font_size=8)
plt.title(f"Fungi Clustering - Kernel Case 6 - Silhouette Score: {quality_metrics['silhouette_score']:.6f}")
plt.ylabel("Distance")
plt.show()

# Return the quality metrics
quality_metrics

# Fix the DK_metric function to support the correct case check and adapt it for fungal data clustering
def DK_metric_fixed(A_L, A_R, B_L, B_R, kernel_case):
    if kernel_case != 6:
        raise ValueError("Invalid kernel case. Only case 6 is supported.")
    a = 1.3
    b = -0.29999999999999993
    c = 0.1
    DK = ((a * (A_R - B_R) ** 2 + c * (A_L - B_L) ** 2) - 2 * b * (A_R - B_R) * (A_L - B_L)) / (2 + 2 * b + c)
    return np.sqrt(DK)

# Prepare to calculate the distance matrix using the corrected metric for the fungi dataset
# Assume A_L and A_R correspond to the lower and upper bounds of a particular feature, e.g., 'spores 1d lower' and 'spores 1d upper'
n = len(fungi_data)
distance_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j:
            A_L, A_R = fungi_data.loc[i, ['spores 1d lower', 'spores 1d upper']]
            B_L, B_R = fungi_data.loc[j, ['spores 1d lower', 'spores 1d upper']]
            distance_matrix[i, j] = DK_metric_fixed(A_L, A_R, B_L, B_R, kernel_case=6)
        else:
            distance_matrix[i, j] = 0  # Distance to self is zero

# Proceed with hierarchical clustering using the modified distance matrix
Z = linkage(distance_matrix, method='ward')
k = 10  # Using 10 clusters as the dataset suggests
labels = fcluster(Z, k, criterion='maxclust')

# Calculate various cluster quality metrics using the custom distance matrix
quality_metrics = calculate_cluster_quality(distance_matrix, labels)

# Plot the dendrogram
plt.figure(figsize=(10, 8))
dendrogram(Z, labels=[fungi_data.loc[i, 'name'] for i in range(n)], leaf_rotation=90, leaf_font_size=8)
plt.title(f"Fungi Clustering - Kernel Case 6 - Silhouette Score: {quality_metrics['silhouette_score']:.6f}")
plt.ylabel("Distance")
plt.show()

# Return the quality metrics
quality_metrics

# Create a DataFrame showing the fungus names and their corresponding cluster labels
fungi_cluster_assignments = pd.DataFrame({
    'Name': fungi_data['name'],
    'Cluster': labels
})

# Sort the DataFrame by cluster for better visualization
fungi_cluster_assignments_sorted = fungi_cluster_assignments.sort_values(by='Cluster')

# Display the sorted DataFrame
fungi_cluster_assignments_sorted

# Calculate the distance matrix using only the 'pileus width lower' and 'pileus width upper' features
distance_matrix_pileus = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j:
            A_L, A_R = fungi_data.loc[i, ['pileus width lower', 'pileus width upper']]
            B_L, B_R = fungi_data.loc[j, ['pileus width lower', 'pileus width upper']]
            distance_matrix_pileus[i, j] = DK_metric_fixed(A_L, A_R, B_L, B_R, kernel_case=6)
        else:
            distance_matrix_pileus[i, j] = 0  # Distance to self is zero

# Hierarchical clustering using the pileus width distance matrix
Z_pileus = linkage(distance_matrix_pileus, method='ward')
labels_pileus = fcluster(Z_pileus, 10, criterion='maxclust')

# Calculate various cluster quality metrics using the pileus width distance matrix
quality_metrics_pileus = calculate_cluster_quality(distance_matrix_pileus, labels_pileus)

# Plot the dendrogram for the new clustering
plt.figure(figsize=(10, 8))
dendrogram(Z_pileus, labels=[fungi_data.loc[i, 'name'] for i in range(n)], leaf_rotation=90, leaf_font_size=8)
plt.title(f"Pileus Width Clustering - Kernel Case 6 - Silhouette Score: {quality_metrics_pileus['silhouette_score']:.6f}")
plt.ylabel("Distance")
plt.show()

# Return the new quality metrics
quality_metrics_pileus

# Create a DataFrame showing the fungus names, their pileus width, and their corresponding cluster labels
fungi_pileus_cluster_assignments = pd.DataFrame({
    'Name': fungi_data['name'],
    'Pileus Width Lower': fungi_data['pileus width lower'],
    'Pileus Width Upper': fungi_data['pileus width upper'],
    'Cluster': labels_pileus
})

# Define the output file path for the pileus width cluster assignments
output_file_path_pileus = '/mnt/data/Fungi_Pileus_Width_Cluster_Assignments.csv'

# Save the DataFrame to CSV
fungi_pileus_cluster_assignments.to_csv(output_file_path_pileus, index=False)

output_file_path_pileus

import seaborn as sns

# Set the plot size
plt.figure(figsize=(12, 8))

# Create a scatter plot colored by cluster labels
sns.scatterplot(x='Pileus Width Lower', y='Pileus Width Upper', hue='Cluster', palette='viridis', data=fungi_pileus_cluster_assignments, s=100)

# Set plot title and labels
plt.title('Scatter Plot of Fungi Clustering by Pileus Width')
plt.xlabel('Pileus Width Lower (mm)')
plt.ylabel('Pileus Width Upper (mm)')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()

# Set up the plot
plt.figure(figsize=(12, 8))

# Plot Silhouette Score
plt.subplot(311)
plt.plot(cluster_results_df['k'], cluster_results_df['silhouette_score'], marker='o')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

# Plot Davies-Bouldin Score
plt.subplot(312)
plt.plot(cluster_results_df['k'], cluster_results_df['davies_bouldin_score'], marker='o', color='red')
plt.title('Davies-Bouldin Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Score')

# Plot Intra Cluster Distance
plt.subplot(313)
plt.plot(cluster_results_df['k'], cluster_results_df['intra_cluster_distance'], marker='o', color='green')
plt.title('Average Intra-Cluster Distance vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Average Intra-Cluster Distance')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

# Setting up the plot
plt.figure(figsize=(16, 12))

# Plotting spore diameter intervals
plt.subplot(221)
plt.boxplot([fungi_data['spores 1d lower'], fungi_data['spores 1d upper']], positions=[1, 2], widths=0.6)
plt.xticks([1, 2], ['Spores 1D Lower', 'Spores 1D Upper'])
plt.title('Distribution of Spore Diameter')
plt.ylabel('Diameter (mm)')

# Plotting pileus width intervals
plt.subplot(222)
plt.boxplot([fungi_data['pileus width lower'], fungi_data['pileus width upper']], positions=[1, 2], widths=0.6)
plt.xticks([1, 2], ['Pileus Width Lower', 'Pileus Width Upper'])
plt.title('Distribution of Pileus Width')
plt.ylabel('Width (mm)')

# Plotting stipes length intervals
plt.subplot(223)
plt.boxplot([fungi_data['stipes long lower'], fungi_data['stipes long upper']], positions=[1, 2], widths=0.6)
plt.xticks([1, 2], ['Stipes Long Lower', 'Stipes Long Upper'])
plt.title('Distribution of Stipes Length')
plt.ylabel('Length (mm)')

# Plotting stipes thickness intervals
plt.subplot(224)
plt.boxplot([fungi_data['stipes thick lower'], fungi_data['stipes thick upper']], positions=[1, 2], widths=0.6)
plt.xticks([1, 2], ['Stipes Thick Lower', 'Stipes Thick Upper'])
plt.title('Distribution of Stipes Thickness')
plt.ylabel('Thickness (mm)')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()
