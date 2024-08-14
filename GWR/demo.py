import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 样例数据
data = {
    'Location1': [1.0, 2.1, 3.2, 4.3, 5.4],
    'Location2': [2.0, 3.1, 4.2, 5.3, 6.4],
    'Location3': [3.0, 4.1, 5.2, 6.3, 7.4]
}

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 定义点之间的距离
distances = np.array([
    [0, 1, 2],
    [1, 0, 1],
    [2, 1, 0]
])

# 构建基于反距离的空间权重矩阵
W = 1 / (distances + np.eye(distances.shape[0]))


# 计算空间滞后变量
def weighted_variable(Y, W):
    return np.dot(W, Y)


# 计算空间相关系数
def spatial_correlation(X, Y, W):
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    WY = weighted_variable(Y, W)
    WY_mean = np.mean(WY)

    numerator = np.sum((X - X_mean) * (WY - WY_mean))
    denominator = np.sqrt(np.sum((X - X_mean) ** 2) * np.sum((WY - WY_mean) ** 2))

    return numerator / denominator


# 计算空间相关系数矩阵
n = df.shape[0]  # 每个位置的观测值数量
correlation_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j:
            X = df.iloc[:, i].values
            Y = df.iloc[:, j].values
            correlation_matrix[i, j] = spatial_correlation(X, Y, W)
        else:
            correlation_matrix[i, j] = 1

# 将相关系数矩阵转换为DataFrame
correlation_df = pd.DataFrame(correlation_matrix, index=df.columns, columns=df.columns)

# 打印相关系数矩阵
print(correlation_df)

# 可视化相关系数矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0)
plt.title('Adjusted Spatial Correlation Matrix')
plt.show()
