import os
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from esda.moran import Moran
from libpysal.weights import KNN, W

# 目标文件夹
folder_path = 'coordinate_31'

# 获取文件夹中所有 txt 文件的列表
file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# 初始化最大和最小莫兰指数及对应文件名
max_moran_index = -float('inf')
max_moran_p_value = None
max_moran_file = None
min_moran_index = float('inf')
min_moran_file = None

# 设置要尝试的K值范围
k_values = range(5, 21)

# 打开文件准备写入最佳k值和基因莫兰指数
best_k_file = 'best_k_values_31_2.txt'
gene_moran_file = 'gene_moran_values_31_2.csv'

with open(best_k_file, 'w') as k_file, open(gene_moran_file, 'w') as gene_file:
    k_file.write("File Name\tBest K\n")
    gene_file.write("File Name,Moran Index,p-value\n")

    # 处理每个文件
    for file_name in tqdm(file_list, desc="Processing files"):
        file_path = os.path.join(folder_path, file_name)

        # 读取当前 txt 文件
        data_df = pd.read_csv(file_path, sep='\t')

        # 创建 GeoDataFrame
        gdf = gpd.GeoDataFrame(data_df, geometry=gpd.points_from_xy(data_df['x'], data_df['y']))

        best_moran = -float('inf')
        best_k = None

        for k in tqdm(k_values, desc=f"Selecting K for {file_name}", leave=False):
            # 计算空间权重矩阵
            knn = KNN.from_dataframe(gdf, k=k)

            # 检查是否有孤立点
            if not knn.islands:
                # 基于距离计算反距离权重
                neighbors = knn.neighbors
                distances = knn.weights
                weights = {}
                for i in neighbors:
                    weights[i] = [1 / d for d in distances[i]]

                w = W(neighbors, weights)

                # 标准化权重矩阵
                w.transform = 'r'

                # 计算全局莫兰指数
                moran = Moran(data_df['z'], w)

                if moran.I > best_moran:
                    best_moran = moran.I
                    best_k = k

        # 使用最佳K值计算全局莫兰指数
        knn = KNN.from_dataframe(gdf, k=best_k)
        neighbors = knn.neighbors
        distances = knn.weights
        weights = {}
        for i in neighbors:
            weights[i] = [1 / d for d in distances[i]]

        w = W(neighbors, weights)
        w.transform = 'r'
        moran = Moran(data_df['z'], w)

        # 更新全局最大和最小莫兰指数
        if moran.I > max_moran_index:
            max_moran_index = moran.I
            max_moran_p_value = moran.p_sim
            max_moran_file = file_name

        if moran.I < min_moran_index:
            min_moran_index = moran.I
            min_moran_file = file_name

        # 写入最佳k值
        k_file.write(f"{file_name}\t{best_k}\n")

        # 写入基因名称和莫兰指数
        gene_file.write(f"{file_name},{moran.I},{moran.p_sim}\n")

# 输出最小和最大莫兰指数及对应的文件名
print(f"最大莫兰指数出现在文件: {max_moran_file}，值为: {max_moran_index}，p值为: {max_moran_p_value}")
print(f"最小莫兰指数出现在文件: {min_moran_file}，值为: {min_moran_index}")
