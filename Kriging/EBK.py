import arcpy
from arcpy import env

# 设置工作空间
env.workspace = "C:/path/to/your/workspace"

# 输入 CSV 文件路径
csv_file = r"E:\Kriging"
shapefile = "pollution_points.shp"

# 将 CSV 文件转换为点要素类
arcpy.management.XYTableToPoint(csv_file, shapefile, "x", "y", coordinate_system=arcpy.SpatialReference(4326))

# 输入参数
in_features = shapefile  # 输入点要素类
z_field = "z"             # 表示元素值的字段
out_ga_layer = "C:/path/to/output.lyr"  # 输出GA图层文件路径
out_raster = "C:/path/to/output.tif"    # 输出栅格文件路径
cell_size = 100                        # 栅格的像元大小
transformation_type = "NONE"           # 变换类型，可选 "NONE", "EMPIRICAL", "LOG", "LOGEMPIRICAL"
max_local_points = 50                  # 每个局部模型使用的最大点数
overlap_factor = 1                     # 重叠因子
number_semivariograms = 100            # 半变异函数数量
search_neighborhood = arcpy.SearchNeighborhoodStandardCircular(1000, 10)  # 搜索邻域
output_type = "PREDICTION"             # 输出类型，可选 "PREDICTION", "QUANTILE", "PROBABILITY", "STANDARD_ERROR"
quantile_value = 0.5                   # 分位数值，仅在 output_type 为 "QUANTILE" 时使用
threshold_type = "EXCEED"              # 阈值类型，仅在 output_type 为 "PROBABILITY" 时使用
probability_threshold = 0.5            # 概率阈值，仅在 output_type 为 "PROBABILITY" 时使用
semivariogram_model_type = "POWER"     # 半变异函数模型类型

# 执行经验贝叶斯克里金插值
arcpy.ga.EmpiricalBayesianKriging(
    in_features,
    z_field,
    out_ga_layer,
    out_raster,
    cell_size,
    transformation_type,
    max_local_points,
    overlap_factor,
    number_semivariograms,
    search_neighborhood,
    output_type,
    quantile_value,
    threshold_type,
    probability_threshold,
    semivariogram_model_type
)
