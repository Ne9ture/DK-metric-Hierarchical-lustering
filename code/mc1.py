import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

def adaptive_distance(xi, center, weights):
    """根据加权距离计算间隔数据点和中心之间的距离。"""
    return np.sum(weights * ((xi[:, 0] - center[:, 0])**2 + (xi[:, 1] - center[:, 1])**2))

def generate_intervals(num_intervals, num_features):
    """生成随机间隔作为聚类特征。"""
    starts = np.random.rand(num_intervals, num_features)
    lengths = np.random.rand(num_intervals, num_features) * 0.5  # 确保间隔长度在0到0.5之间
    ends = starts + lengths
    return np.stack((starts, ends), axis=-1)

def kmeans_plus_plus(X, k):
    """使用类似K-means++的方法初始化聚类中心。"""
    num_samples, num_features, _ = X.shape
    indices = np.random.choice(num_samples, 1)
    centers = X[indices]
    for _ in range(1, k):
        dist_sq = np.min([np.sum((X - center)**2, axis=(1, 2)) for center in centers], axis=0)
        probs = dist_sq / dist_sq.sum()
        cumulative_probs = np.cumsum(probs)
        r = np.random.rand()
        i = np.where(cumulative_probs >= r)[0][0]
        centers = np.vstack([centers, X[i:i+1]])
    return centers

def update_centers(X, labels, k, num_features):
    """根据当前聚类标签重新计算中心。"""
    centers = np.zeros((k, num_features, 2))
    for i in range(k):
        cluster_data = X[labels == i]
        if len(cluster_data) > 0:
            centers[i, :, 0] = np.mean(cluster_data[:, :, 0], axis=0)  # 计算下界的均值
            centers[i, :, 1] = np.mean(cluster_data[:, :, 1], axis=0)  # 计算上界的均值
    return centers

def clustering(X, k):
    """使用初始化权重和通过迭代更新的自适应聚类方法。"""
    num_intervals, num_features, _ = X.shape
    centers = kmeans_plus_plus(X, k)  # 初始化中心
    weights = np.random.rand(num_features)  # 随机初始化权重

    for _ in range(10):  # 迭代固定次数或直到收敛
        labels = np.array([np.argmin([adaptive_distance(X[i], centers[j], weights) for j in range(k)]) for i in range(num_intervals)])
        centers = update_centers(X, labels, k, num_features)
        weights = adaptive_weights(X, labels, k, num_features)  # 更新权重

    return labels, centers

def adaptive_weights(X, labels, k, num_features):
    """根据每个聚类中的特征方差调整权重。"""
    weights = np.zeros(num_features)
    for j in range(num_features):
        variances = np.array([np.var(X[labels == i, j, :]) for i in range(k)])
        weights[j] = 1.0 / (variances.mean() + 1e-8)
    return weights / weights.sum()

def calculate_cluster_quality(X, labels):
    """计算聚类质量指标。"""
    if len(np.unique(labels)) > 1:
        silhouette_avg = silhouette_score(X.reshape(X.shape[0], -1), labels, metric='euclidean')
        db_score = davies_bouldin_score(X.reshape(X.shape[0], -1), labels)
    else:
        silhouette_avg = -1  # 聚类数不足，无法计算
        db_score = -1
    return silhouette_avg, db_score

def monte_carlo_simulation(num_simulations, num_intervals, num_features, k):
    silhouette_scores = []
    db_scores = []
    for _ in range(num_simulations):
        X = generate_intervals(num_intervals, num_features)
        labels, centers = clustering(X, k)
        silhouette_avg, db_score = calculate_cluster_quality(X, labels)
        silhouette_scores.append(silhouette_avg)
        db_scores.append(db_score)
    return silhouette_scores, db_scores

# 模拟参数
num_simulations = 100
num_intervals = 500
num_features = 5
k = 5

# 运行模拟
silhouettes, db_scores = monte_carlo_simulation(num_simulations, num_intervals, num_features, k)
print("Average Silhouette Score:", np.mean([s for s in silhouettes if s != -1]))  # 过滤无效得分
print("Average Davies-Bouldin Score:", np.mean([d for d in db_scores if d != -1]))  # 过滤无效得分


