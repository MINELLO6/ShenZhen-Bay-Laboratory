import os
import pandas as pd

# 定义文件夹路径和输出文件名
reversed_prediction_all_idw_folder = "Reversed_prediction_all_idw"
reversed_predictions_based_on_locations_folder = "Reversed_predictions_based_on_locations"
output_file_name_idw = "insitu_count_all_idw.txt"
output_file_name_locations = "insitu_count_all_locations.txt"

# 初始化一个DataFrame来存储所有z列
def combine_files(input_folder, output_file_name):
    all_z = pd.DataFrame()

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('_predict.txt'):
            file_path = os.path.join(input_folder, filename)

            # 读取reversed文件
            df_reversed = pd.read_csv(file_path, delimiter='\t')

            # 获取基因名称，假设文件名格式为"基因_predict.txt"
            gene_name = os.path.splitext(filename)[0].replace('_predict', '')

            # 只保留z列并重命名为文件前缀
            df_z = df_reversed[['z']].rename(columns={'z': gene_name})

            # 将当前文件的z列添加到all_z中
            if all_z.empty:
                all_z = df_z
            else:
                all_z = pd.concat([all_z, df_z], axis=1)

    # 保存所有z列到一个新的txt文件
    all_z.to_csv(output_file_name, sep='\t', index=False)

    # 计算稀疏度
    total_values = all_z.size
    zero_values = (all_z == 0).sum().sum()
    sparsity = zero_values / total_values

    # 输出文件的行和列信息
    rows, columns = all_z.shape

    print(f"All z columns combined and saved to {output_file_name}")
    print(f"The combined file has {rows} rows and {columns} columns.")
    print(f"Sparsity of the combined file is {sparsity:.4f} (zero values / total values).")

# 处理两个文件夹
combine_files(reversed_prediction_all_idw_folder, output_file_name_idw)
combine_files(reversed_predictions_based_on_locations_folder, output_file_name_locations)
