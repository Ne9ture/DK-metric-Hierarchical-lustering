import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# 假设data是一个包含不同核情况和不同指标平均值的DataFrame
data = {
    'Kernel Case': ['Case 1', 'Case 2', 'Case 3', 'Case 4', 'Case 5'],
    'Silhouette': [0.5547, 0.5314, 0.5498, 0.5362, 0.5639],
    'DB Index': [0.5365, 0.5288, 0.5314, 0.5322, 0.5310]
}
df = pd.DataFrame(data)
df = df.set_index('Kernel Case')
sns.heatmap(df, annot=True, cmap='coolwarm')
plt.title('Performance Metrics Heatmap')
plt.show()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 示例数据
# 假设每行是一个数据点，每列是一个特征
X = np.random.rand(100, 5)  # 100个数据点，每个数据点5个特征

# 初始化 t-SNE 模型
tsne = TSNE(n_components=2, random_state=42)

# 使用 t-SNE 进行降维
X_reduced = tsne.fit_transform(X)

# 绘制 t-SNE 结果的散点图
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7)
plt.title('t-SNE Projection')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
