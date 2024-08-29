import pandas as pd
import numpy as np
import itertools
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import random
import warnings


#用GWR预测coordinate31中所有基因两两关系

# 忽略可能不影响结果的警告
warnings.filterwarnings("ignore")

# 读取莫兰指数数据
file_path = 'gene_moran_values_31_2.csv'
data = pd.read_csv(file_path)

# 筛选代码，修改变成对于所有基因
filtered_genes = data[data['Moran Index'] > -1]

# 获取所有基因的名称和路径
gene_files = filtered_genes['File Name'].tolist()

# 随机筛选版--随机选择30个两两组合，确保没有重复
#random_combinations = random.sample(list(itertools.combinations(gene_files, 2)), 10)


# 初始化结果列表
results = []

# 对所有基因两两组合进行GWR建模，记录结果
for gene1, gene2 in tqdm(itertools.combinations(gene_files, 2), total=len(gene_files)*(len(gene_files)-1)//2):
#for gene1, gene2 in tqdm(random_combinations, total=10): #随机筛选版

    try:
        # 加载两个基因的表达数据
        gene1_data = pd.read_csv(f'coordinate_31/{gene1}', sep='\t', header=0)
        gene2_data = pd.read_csv(f'coordinate_31/{gene2}', sep='\t', header=0)
        
        # 重命名列
        gene1_data.columns = ['x', 'y', 'x_expression']
        gene2_data.columns = ['x', 'y', 'y_expression']
        
        # 合并数据
        merged_data = pd.merge(gene1_data, gene2_data, on=['x', 'y'])
        merged_data = merged_data.apply(pd.to_numeric)
        
        # 提取坐标和表达数据
        coords = merged_data[['x', 'y']].values
        X = merged_data[['x_expression']].values
        y = merged_data[['y_expression']].values
        
        # 10折交叉验证
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        predictions = np.zeros_like(y)
        
        for train_index, test_index in kf.split(X):
            coords_train, coords_test = coords[train_index], coords[test_index]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # 选择最佳带宽
            bw = Sel_BW(coords_train, y_train, X_train).search()
            
            # 拟合GWR模型
            gwr_model = GWR(coords_train, y_train, X_train, bw)
            gwr_results = gwr_model.fit()
            
            # 预测
            gwr_pred = gwr_model.predict(coords_test, X_test)
            predictions[test_index] = gwr_pred.predictions
            
        # 计算评价指标
        pcc, _ = pearsonr(predictions.flatten(), y.flatten())
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mse)
        n = len(merged_data)
        p = X.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        moran_gene1 = filtered_genes.loc[filtered_genes['File Name'] == gene1, 'Moran Index'].values[0]
        moran_gene2 = filtered_genes.loc[filtered_genes['File Name'] == gene2, 'Moran Index'].values[0]
        
        # 将结果保存到列表
        results.append({
            'Gene 1': gene1,
            'Moran Index Gene 1': moran_gene1,
            'Gene 2': gene2,
            'Moran Index Gene 2': moran_gene2,
            'PCC': pcc,
            'R^2': r2,
            'Adjusted R²': adj_r2,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse
        })
    
    except np.linalg.LinAlgError:
        # 处理奇异矩阵错误，跳过当前组合
        print(f"Skipping gene pair: {gene1} and {gene2} due to singular matrix error.")
        continue
    except Exception as e:
        # 处理其他可能的错误，并继续处理其他基因对
        print(f"Skipping gene pair: {gene1} and {gene2} due to an unexpected error: {e}")
        continue    

# 将所有结果保存为CSV
results_df = pd.DataFrame(results)
results_df.to_csv('gene_gwr_all_results.csv', index=False)

print("All pairwise GWR results have been saved to gene_gwr_all_results.csv")