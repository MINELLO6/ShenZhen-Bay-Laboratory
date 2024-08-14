import os
import pandas as pd

# 定义文件夹路径和Location2.txt文件路径
prediction_all_idw_folder = "prediction_all_idw"
predictions_based_on_locations_folder = "predictions_based_on_locations"
locations_path = 'location_2.txt'

# 读取location2.txt文件
df_locations = pd.read_csv(locations_path, delimiter='\t')

# 计算原始x和y的最小值和最大值
x_min, x_max = df_locations['x'].min(), df_locations['x'].max()
y_min, y_max = df_locations['y'].min(), df_locations['y'].max()

# 定义映射回原始区间的函数
def denormalize(series, original_min, original_max):
    return series * (original_max - original_min) + original_min

# 创建输出目录，如果不存在的话
reversed_prediction_all_idw_folder = "Reversed_prediction_all_idw"
if not os.path.exists(reversed_prediction_all_idw_folder):
    os.makedirs(reversed_prediction_all_idw_folder)

reversed_predictions_based_on_locations_folder = "Reversed_predictions_based_on_locations"
if not os.path.exists(reversed_predictions_based_on_locations_folder):
    os.makedirs(reversed_predictions_based_on_locations_folder)

# 定义一个通用的处理函数
def process_folder(input_folder, output_folder):
    # 对每个文件进行逆映射操作
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            full_file_name = os.path.join(input_folder, file_name)

            # 读取文件内容
            df_combined = pd.read_csv(full_file_name, delimiter='\t')

            # 对x和y进行反归一化处理
            df_combined['X'] = denormalize(df_combined['x'], x_min, x_max)
            df_combined['Y'] = denormalize(df_combined['y'], y_min, y_max)

            # 删除原始归一化后的x和y列，只保留反归一化后的列
            df_combined = df_combined[['X', 'Y', 'z']]

            # 保存逆映射后的内容到新的文件夹中
            reversed_file_name = os.path.join(output_folder, file_name)
            df_combined.to_csv(reversed_file_name, sep='\t', index=False)

    print(f"Reversed files saved to {output_folder}")

# 处理两个文件夹
process_folder(prediction_all_idw_folder, reversed_prediction_all_idw_folder)
process_folder(predictions_based_on_locations_folder, reversed_predictions_based_on_locations_folder)
