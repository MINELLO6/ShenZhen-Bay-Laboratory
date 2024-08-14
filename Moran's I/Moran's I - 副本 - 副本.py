import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from esda.moran import Moran, Moran_Local
from libpysal.weights import KNN, W
from matplotlib.colors import LinearSegmentedColormap, Normalize

# 强制使用 'Agg' 后端
matplotlib.use('Agg')

# 创建保存图像和局部莫兰指数结果的新文件夹
output_folder = 'negative_output_31'
local_moran_folder = 'negative_local_moran_results_31'
histogram_folder = 'negative_histograms_31'
origin_folder = 'negative_origin_31'
global_moran_folder = 'negative_global_moran_31'
best_k_file = 'negative_best_k_values_31.txt'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(local_moran_folder):
    os.makedirs(local_moran_folder)
if not os.path.exists(histogram_folder):
    os.makedirs(histogram_folder)
if not os.path.exists(origin_folder):
    os.makedirs(origin_folder)
if not os.path.exists(global_moran_folder):
    os.makedirs(global_moran_folder)

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

# 打开文件准备写入最佳k值
with open(best_k_file, 'w') as k_file:
    k_file.write("File Name\tBest K\n")

    # 处理每个文件
    for file_name in tqdm(file_list, desc="Processing files"):
        file_path = os.path.join(folder_path, file_name)

        # 读取当前 txt 文件
        data_df = pd.read_csv(file_path, sep='\t')

        # 对 z 值进行 log2(z+1) 变换
        data_df['log_z'] = data_df['z']

        # 创建 log2(z+1) 值的六边形栅格图
        plt.figure(dpi=300)  # 提高分辨率
        colors = ["#cfd1d0", "#db2a2b"]
        cmap = LinearSegmentedColormap.from_list("custom_red", colors)
        norm = Normalize(vmin=0, vmax=2 * data_df['z'].sum() / 3805)
        hb = plt.hexbin(data_df['x'], data_df['y'], C=data_df['z'], gridsize=45, cmap=cmap, norm=norm, reduce_C_function=np.mean)
        plt.colorbar(hb, label='z value')
        plt.title(f'Hexbin Plot of z values for {file_name}')
        plt.xlabel('x')
        plt.ylabel('y')

        # 保存六边形栅格图到 origin 文件夹中
        origin_path = os.path.join(origin_folder, f'{file_name}_hexbin.png')
        plt.savefig(origin_path, bbox_inches='tight')
        plt.close()

        # 创建基因表达量的直方图
        plt.figure(dpi=300)  # 提高分辨率
        plt.hist(np.log2(data_df['z']+1), bins=50, color='blue', edgecolor='black')
        plt.title(f'Histogram of Gene Expression for {file_name}')
        plt.xlabel(r'$\log_2(\text{Gene Expression} \, z + 1)$')
        plt.ylabel('Frequency')

        # 保存直方图到新的文件夹中
        histogram_path = os.path.join(histogram_folder, f'{file_name}_histogram.png')
        plt.savefig(histogram_path)
        plt.close()

        # 创建 GeoDataFrame
        gdf = gpd.GeoDataFrame(data_df, geometry=gpd.points_from_xy(data_df['x'], data_df['y']))

        best_moran = float('inf')  # 初始化为正无穷大
        best_k = None
        best_moran_loc = None

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

                if moran.I < best_moran:  # 寻找最小的莫兰指数
                    best_moran = moran.I
                    best_k = k
                    best_moran_loc = Moran_Local(data_df['z'], w)

        # 使用最佳K值计算全局莫兰指数和局部莫兰指数
        knn = KNN.from_dataframe(gdf, k=best_k)
        neighbors = knn.neighbors
        distances = knn.weights
        weights = {}
        for i in neighbors:
            weights[i] = [1 / d for d in distances[i]]

        w = W(neighbors, weights)
        w.transform = 'r'
        moran = Moran(data_df['z'], w)
        moran_loc = best_moran_loc

        # 更新全局最大和最小莫兰指数
        if moran.I > max_moran_index:
            max_moran_index = moran.I
            max_moran_p_value = moran.p_sim
            max_moran_file = file_name

        if moran.I < min_moran_index:
            min_moran_index = moran.I
            min_moran_file = file_name

        # 保存局部莫兰指数结果到文件
        local_moran_result = pd.DataFrame({
            'x': data_df['x'],
            'y': data_df['y'],
            'local_moran_I': moran_loc.Is,
            'p_value': moran_loc.p_sim,
            'significant': moran_loc.p_sim < 0.05
        })
        local_moran_result_path = os.path.join(local_moran_folder, f'{file_name}_local_moran.csv')
        local_moran_result.to_csv(local_moran_result_path, index=False)

        # 写入最佳k值
        k_file.write(f"{file_name}\t{best_k}\n")

        # 绘制局部莫兰指数的六边形栅格图
        plt.figure(dpi=300)  # 提高分辨率
        edgecolors = ['k' if sig else 'none' for sig in local_moran_result['significant']]
        hb = plt.hexbin(data_df['x'], data_df['y'], C=moran_loc.Is, gridsize=30, cmap='coolwarm', edgecolors=edgecolors, reduce_C_function=np.mean)
        plt.colorbar(hb, label='Local Moran I')
        plt.title(f'Local Moran I for {file_name} with k={best_k}')
        plt.xlabel('x')
        plt.ylabel('y')

        # 保存六边形栅格图到新的文件夹中
        output_path = os.path.join(output_folder, f'{file_name}_local_moran.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        # 绘制全局莫兰指数的六边形栅格图
        plt.figure(dpi=300)  # 提高分辨率
        hb_global = plt.hexbin(data_df['x'], data_df['y'], C=data_df['z'], gridsize=45, cmap=cmap, norm=norm, reduce_C_function=np.mean)
        plt.colorbar(hb_global, label='z value')
        plt.title(f'Global Moran I for {file_name} (I={moran.I:.4f}, p={moran.p_sim:.4f})')
        plt.xlabel('x')
        plt.ylabel('y')

        # 保存全局莫兰指数的六边形栅格图到新的文件夹中
        global_moran_path = os.path.join(global_moran_folder, f'{file_name}_global_moran.png')
        plt.savefig(global_moran_path, bbox_inches='tight')
        plt.close()

        # 绘制LISA聚集图
        sig = moran_loc.p_sim < 0.05
        hotspots = (moran_loc.q == 1) & sig
        coldspots = (moran_loc.q == 3) & sig
        doughnut = (moran_loc.q == 2) & sig
        diamond = (moran_loc.q == 4) & sig

        plt.figure(dpi=300)  # 提高分辨率
        plt.scatter(data_df['x'], data_df['y'], c='lightgrey', label='Not significant', s=10)
        plt.scatter(data_df['x'][hotspots], data_df['y'][hotspots], c='red', label='Hot spot', s=10)
        plt.scatter(data_df['x'][coldspots], data_df['y'][coldspots], c='blue', label='Cold spot', s=10)
        plt.scatter(data_df['x'][doughnut], data_df['y'][doughnut], c='purple', label='Low-High', s=10)
        plt.scatter(data_df['x'][diamond], data_df['y'][diamond], c='pink', label='High-Low', s=10)
        plt.legend()
        plt.title(f'LISA Cluster Map for {file_name} with k={best_k}')
        plt.xlabel('x')
        plt.ylabel('y')

        # 保存LISA聚集图到新的文件夹中
        lisa_output_path = os.path.join(output_folder, f'{file_name}_lisa_cluster.png')
        plt.savefig(lisa_output_path, bbox_inches='tight')
        plt.close()

# 输出最小和最大莫兰指数及对应的文件名
print(f"最大莫兰指数出现在文件: {max_moran_file}，值为: {max_moran_index}，p值为: {max_moran_p_value}")
print(f"最小莫兰指数出现在文件: {min_moran_file}，值为: {min_moran_index}")
