import os
import pandas as pd

# 读取 Locations.txt 文件中的坐标
locations_file = 'Locations.txt'
locations_df = pd.read_csv(locations_file, sep='\t')

# 目标文件夹
folder_path = 'coordinate'

# 获取文件夹中所有 txt 文件的列表
file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# 处理每个文件
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)

    # 读取当前 txt 文件
    data_df = pd.read_csv(file_path, sep='\t')

    # 检查文件中是否有 x 和 y 列
    if 'x' in data_df.columns and 'y' in data_df.columns:
        # 用 Locations.txt 中的值替换 x 和 y 列
        data_df['x'] = locations_df['X']
        data_df['y'] = locations_df['Y']

        # 保存替换后的文件
        data_df.to_csv(file_path, sep='\t', index=False)

print("所有文件已成功更新")
