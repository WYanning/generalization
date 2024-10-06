import pandas as pd
import os

# 定义文件夹路径
source_directory = r"D:\dresden\mapgeneralization\dataset\output1\updated_adj_df"
target_directory = r"D:\dresden\mapgeneralization\output0719\adj"
output_directory = r"D:\dresden\mapgeneralization\output0719\adj_cut"

# 确保输出目录存在
os.makedirs(output_directory, exist_ok=True)

# 遍历源目录中的所有文件
for filename in os.listdir(source_directory):
    if filename.startswith("updated_adj_df_"):
        # 获取数字后缀
        suffix = filename.split("_")[-1].split(".")[0]

        # 构建目标文件路径
        target_filename = f"index{suffix}.csv"
        target_file_path = os.path.join(target_directory, target_filename)

        if os.path.exists(target_file_path):
            # 读取源文件和目标文件
            source_df = pd.read_csv(os.path.join(source_directory, filename))
            target_df = pd.read_csv(target_file_path)

            # 获取行和列的数量
            nrows, ncols = source_df.shape

            # 裁剪目标文件
            trimmed_df = target_df.iloc[:nrows, :ncols]

            # 保存裁剪后的文件
            output_file_path = os.path.join(output_directory, target_filename)
            trimmed_df.to_csv(output_file_path, index=False)
