# 加载必要的包
if (!require("gstat")) {
  install.packages("gstat")
}
if (!require("sp")) {
  install.packages("sp")
}
library(gstat)
library(sp)

# 设置文件路径
unpredicted_genes_file_path <- "unpredicted_genes.txt"
coordinate_folder_path <- "./Coordinate_4"
locations_file_path <- "Locations.txt"

# 读取Locations文件中的坐标
locations <- read.table(locations_file_path, header = TRUE)

# 确保列名是X和Y
colnames(locations) <- c("x", "y")
coordinates(locations) <- ~x + y

# 读取未预测的基因列表
unpredicted_genes <- readLines(unpredicted_genes_file_path)

# 创建输出目录
output_dir <- "predictions_based_on_locations"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# 循环处理每个未预测的基因文件
for (gene_name in unpredicted_genes) {
  # 构建文件路径
  file_path <- file.path(coordinate_folder_path, paste0(gene_name, ".txt"))
  
  # 检查文件是否存在
  if (!file.exists(file_path)) {
    cat("File not found:", file_path, "\n")
    next
  }
  
  # 读取txt文件中的数据
  data <- read.table(file_path, header = TRUE)[, 1:3]
  colnames(data) <- c("x", "y", "z")
  data1 <- data
  
  # 确保数据列是数值类型
  data$x <- as.numeric(data$x)
  data$y <- as.numeric(data$y)
  data$z <- as.numeric(data$z)
  
  # 创建空间点数据框
  coordinates(data) <- ~x + y
  
  # 使用IDW进行插值
  idw_result <- idw(z ~ 1, data, newdata = locations)
  
  # 转换预测结果为数据框，并选择预测数据的前三列
  idw_df <- as.data.frame(idw_result)
  idw_df <- idw_df[, c("x", "y", "var1.pred")]
  colnames(idw_df) <- c("x", "y", "z")
  
  # 保存预测数据到predict文件夹
  output_file_path <- file.path(output_dir, paste0(gene_name, "_predict.txt"))
  write.table(idw_df, output_file_path, sep = "\t", row.names = FALSE, quote = FALSE)
}

print("IDW prediction complete.")
