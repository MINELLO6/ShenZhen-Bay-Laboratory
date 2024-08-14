import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import glob

# 定义读取文件的函数
def load_data_from_folder(folder_path):
    data_dict = {}
    for file_path in glob.glob(os.path.join(folder_path, '*.txt')):
        df = pd.read_csv(file_path, sep='\t', header=0, names=['x', 'y', 'z'])
        variable_name = os.path.basename(file_path).split('.')[0]
        data_dict[variable_name] = df['z'].astype(float).values
    return data_dict

# 读取数据
folder_path = 'coordinate_31'
data_dict = load_data_from_folder(folder_path)

# 加载 Abat.txt 文件的数据
Abat_df = pd.read_csv('Abat.txt', sep='\t', header=0, names=['x', 'y', 'z'])
data_dict['Abat'] = Abat_df['z'].astype(float).values

# 构建数据框
df = pd.DataFrame(data_dict)

# 假设Abat是目标变量
target = 'Abat'

# 检查是否包含目标变量
if target not in df.columns:
    raise ValueError(f"Target variable '{target}' not found in data")

# 分割特征和目标变量
X = df.drop(columns=[target])
y = df[target]

# 确保 X 中不包含 'Abat' 列
if 'Abat' in X.columns:
    X = X.drop(columns=['Abat'])

# 训练随机森林模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# 使用 permutation importance 进行特征选择
result = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
importances = result.importances_mean

# 获取选择的特征名称
indices = np.argsort(importances)[::-1]
important_features = X.columns[indices]

# 绘制前10个最重要特征的特征重要性图
plt.figure(figsize=(10, 6))
plt.title("Top 10 Feature Importances")
plt.bar(range(10), importances[indices[:10]], align="center")
plt.xticks(range(10), X.columns[indices[:10]], rotation=90)
plt.xlim([-1, 10])
plt.tight_layout()
plt.show()

# 打印最重要的特征
print("Most important features:")
for feature in important_features[:10]:
    print(feature)
