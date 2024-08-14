# 安装并加载必要的包
library(sp)
library(gstat)
if (!require("readxl")) {
  install.packages("readxl")
}
library(readxl)

# 设置Locations文件路径
locations_file_path <- "Locations.txt"

# 读取Locations文件中的坐标
locations <- read.table(locations_file_path, header = TRUE)

# 获取Coordinate_4文件夹中的所有txt文件
coordinate_files <- list.files("./Coordinate_4_downsample", pattern = "*.txt", full.names = TRUE)

# 读取下采样后的kriging结果
file_path <- "downsample.kriging.result.csv"
kriging_result <- read.csv(file_path)
new_result <- kriging_result

# 用户选择模型参数
model_choice <- as.integer(readline(prompt="请选择模型参数（0：auto_fit_minSSerr 或 1：auto_fit_maxCor）: "))
0
# 根据用户选择设置参数字段
if (model_choice == 0) {
  model_field <- "auto_fit_minSSerr_model"
  psill_field <- "auto_fit_minSSerr_model_psill"
  range_field <- "auto_fit_minSSerr_model_range"
  nugget_field <- "auto_fit_minSSerr_model_nugget"
  filter_condition <- new_result$var_model_minSSerr_cv_cor > 0.15 & new_result$var_model_minSSerr_sserr < 10
} else if (model_choice == 1) {
  model_field <- "auto_fit_maxCor_model"
  psill_field <- "auto_fit_maxCor_model_psill"
  range_field <- "auto_fit_maxCor_model_range"
  nugget_field <- "auto_fit_maxCor_model_nugget"
  filter_condition <- new_result$var_model_maxCor_cv_cor > 0.15 & new_result$var_model_maxCor_sserr < 10
} else {
  stop("无效的模型参数选择。请选择 '0' 或 '1'。")
}

# 创建输出目录
output_dir <- "predictions_based_on_locations"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# 筛选符合条件的基因
filtered_genes <- new_result$Gene[filter_condition]

# 创建一个向量来存储未被预测的基因
unpredicted_genes <- c()

# 初始化被预测的基因计数器
predicted_gene_count <- 0

# 循环处理每个txt文件
for (file in coordinate_files) {
  # 获取文件名（不包含路径和扩展名）
  gene_name <- tools::file_path_sans_ext(basename(file))
  
  # 检查基因是否在筛选的基因列表中
  if (!(gene_name %in% filtered_genes)) {
    cat("Gene does not meet criteria:", gene_name, "\n")
    unpredicted_genes <- c(unpredicted_genes, gene_name)
    next
  }
  
  # 读取txt文件中的数据
  data <- read.table(file, header = TRUE)[, 1:3]
  colnames(data) <- c("x", "y", "z")
  
  # 确保数据列是数值类型
  data$x <- as.numeric(data$x)
  data$y <- as.numeric(data$y)
  data$z <- as.numeric(data$z)
  
  # 提取参数
  model <- as.character(new_result[new_result$Gene == gene_name, model_field])
  psill <- new_result[new_result$Gene == gene_name, psill_field]
  range <- new_result[new_result$Gene == gene_name, range_field]
  nugget <- new_result[new_result$Gene == gene_name, nugget_field]
  
  # 确保range为正值
  if (range < 0) {
    range <- abs(range)
  }
  
  # 打印模型参数
  cat("Processing gene:", gene_name, "\n")
  cat("Model:", model, "\n")
  cat("Psill:", psill, "\n")
  cat("Range:", range, "\n")
  cat("Nugget:", nugget, "\n\n")
  
  # 定义变异函数模型 (variogram model)
  vgm_model <- vgm(psill = psill, model = model, range = range, nugget = nugget)
  
  # 创建gstat对象并进行kriging预测
  gstat_object <- gstat(formula = z ~ 1, locations = ~x + y, data = data, model = vgm_model)
  
  # 使用Locations文件中的坐标进行kriging预测
  kriging_result <- predict(gstat_object, newdata = locations)
  
  # 转换预测结果为数据框，并选择预测数据的前三列
  kriging_df <- as.data.frame(kriging_result)
  kriging_df <- kriging_df[, c("x", "y", "var1.pred")]
  colnames(kriging_df) <- c("x", "y", "z")
  
  # 保存预测数据到predict文件夹
  output_file_path <- file.path(output_dir, paste0(gene_name, "_predict.txt"))
  write.table(kriging_df, output_file_path, sep = "\t", row.names = FALSE, quote = FALSE)
  
  # 增加被预测的基因计数器
  predicted_gene_count <- predicted_gene_count + 1
}

# 将未被预测的基因写入到一个txt文件中
unpredicted_file_path <- file.path(dirname(output_dir), "unpredicted_genes.txt")
writeLines(unpredicted_genes, unpredicted_file_path)

# 打印被预测基因的数量
cat("Number of predicted genes:", predicted_gene_count, "\n")

print("Processing complete.")
