import pandas as pd
import os


# def sort_csv_files(input_dir, output_dir, node_id_column='node_id'):
#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 获取输入目录中的所有CSV文件
#     csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
#
#     for file in csv_files:
#         # 构建完整的文件路径
#         input_file = os.path.join(input_dir, file)
#         output_file = os.path.join(output_dir, file)
#
#         # 读取CSV文件
#         df = pd.read_csv(input_file)
#
#         # 对DataFrame按node_id列排序
#         sorted_df = df.sort_values(by=node_id_column).reset_index(drop=True)
#
#         # 将排序后的DataFrame保存回CSV文件
#         sorted_df.to_csv(output_file, index=False)
#         print(f"Sorted and saved: {output_file}")
#
#
# def copy_columns_from_nodes_movement(nodes_movement_dir, feature_matrix_dir, output_dir):
#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 获取nodes_movement_df目录中的所有CSV文件
#     nodes_movement_files = [f for f in os.listdir(nodes_movement_dir) if f.endswith('.csv')]
#
#     for file in nodes_movement_files:
#         # 获取文件的数字后缀
#         file_suffix = file.split('_')[-1].split('.')[0]
#
#         # 构建完整的文件路径
#         nodes_movement_file = os.path.join(nodes_movement_dir, file)
#         feature_matrix_file = os.path.join(feature_matrix_dir, f'feature_matrix_geb10_{file_suffix}.csv')
#         output_file = os.path.join(output_dir, f'feature_matrix_geb10_{file_suffix}.csv')
#
#         # 读取nodes_movement_df文件
#         nodes_movement_df = pd.read_csv(nodes_movement_file)
#
#         # 提取第三列和第四列
#         columns_to_copy = nodes_movement_df.iloc[:, [2, 3]]
#
#         # 读取feature_matrix_geb10文件
#         feature_matrix_df = pd.read_csv(feature_matrix_file)
#
#         # 合并数据
#         combined_df = pd.concat([feature_matrix_df, columns_to_copy], axis=1)
#
#         # 保存更新后的DataFrame到CSV文件
#         combined_df.to_csv(output_file, index=False)
#         print(f"Updated and saved: {output_file}")
#
#
# # 定义输入和输出目录
# input_dir_geb10 = "D:/dresden/mapgeneralization/dataset/tri/feature_matrix_geb10"
# output_dir_geb10 = "D:/dresden/mapgeneralization/dataset/tri2/feature_matrix_geb10"
#
#
# # 批量排序并保存CSV文件
# sort_csv_files(input_dir_geb10, output_dir_geb10)


# def copy_columns_from_nodes_movement(nodes_movement_dir, feature_matrix_dir):
#     # 获取nodes_movement_df目录中的所有CSV文件
#     nodes_movement_files = [f for f in os.listdir(nodes_movement_dir) if f.endswith('.csv')]
#
#     for file in nodes_movement_files:
#         # 获取文件的数字后缀
#         file_suffix = file.split('_')[-1].split('.')[0]
#
#         # 构建完整的文件路径
#         nodes_movement_file = os.path.join(nodes_movement_dir, file)
#         feature_matrix_file = os.path.join(feature_matrix_dir, f'feature_matrix_geb10_{file_suffix}.csv')
#
#         # 读取nodes_movement_df文件
#         nodes_movement_df = pd.read_csv(nodes_movement_file)
#
#         # 提取第三列和第四列
#         columns_to_copy = nodes_movement_df.iloc[:, [2, 3]]
#         columns_to_copy.columns = ['Column3', 'Column4']  # 可以根据需要调整列名
#
#         # 读取feature_matrix_geb10文件
#         feature_matrix_df = pd.read_csv(feature_matrix_file)
#
#         # 合并数据
#         combined_df = pd.concat([feature_matrix_df, columns_to_copy], axis=1)
#
#         # 保存更新后的DataFrame到CSV文件
#         combined_df.to_csv(feature_matrix_file, index=False)
#         print(f"Updated and saved: {feature_matrix_file}")
#
# input_dir_geb10 = "D:/dresden/mapgeneralization/dataset/tri2/feature_matrix_geb10"
# nodes_movement_dir = "D:/dresden/mapgeneralization/dataset/move/nodes_movement_df"
#
# # 复制columns并更新原文件
# copy_columns_from_nodes_movement(nodes_movement_dir, input_dir_geb10)

# import os
# import re
#
# def extract_number_suffix(filename):
#     # 提取文件名中的数字后缀
#     match = re.search(r'(\d+)(?=\D*$)', filename)
#     return match.group(1) if match else None
#
# # 定义两个文件夹的路径
# folder1 = r"D:\dresden\mapgeneralization\dataset\train"
# folder2 = r"D:\dresden\mapgeneralization\dataset\tri2\feature_matrix_geb10"
#
# # 获取两个文件夹中的所有文件名
# files_in_folder1 = os.listdir(folder1)
# files_in_folder2 = os.listdir(folder2)
#
# # 提取每个文件夹中文件名的数字后缀并存储在集合中
# suffixes_in_folder1 = set(extract_number_suffix(file) for file in files_in_folder1 if extract_number_suffix(file) is not None)
# suffixes_in_folder2 = set(extract_number_suffix(file) for file in files_in_folder2 if extract_number_suffix(file) is not None)
#
# # 找出只存在于folder1中的数字后缀
# unique_suffixes = suffixes_in_folder1 - suffixes_in_folder2
#
# # 打印结果
# print("以下数字后缀存在于 '{}' 中，但不存在于 '{}' 中:".format(folder1, folder2))
# for suffix in unique_suffixes:
#     print(suffix)
#
# import os
# import pandas as pd
# from collections import defaultdict
#
# # 定义文件夹路径
# folder_path = r"D:\dresden\mapgeneralization\dataset\tri2\feature_matrix_geb10"
#
# # 列出文件夹中的所有文件
# files = os.listdir(folder_path)
#
# # 初始化一个字典来统计不同列数的文件数量
# column_count = defaultdict(int)
#
# # 遍历文件夹中的所有文件
# for file in files:
#     if file.endswith('.csv'):  # 确保处理的是 CSV 文件
#         file_path = os.path.join(folder_path, file)
#         try:
#             # 读取文件
#             df = pd.read_csv(file_path)
#             # 更新字典，增加对应列数的文件计数
#             column_count[df.shape[1]] += 1
#         except Exception as e:
#             print(f"Error processing file {file}: {e}")
#
# # 打印每种列数的文件数量
# print("列数统计结果：")
# for columns, count in column_count.items():
#     print(f"列数 {columns}: {count} 个文件")

# import os
# import pandas as pd
#
# # 定义文件夹路径
# folder_path = r"D:\dresden\mapgeneralization\dataset\tri2\feature_matrix_geb10"
#
# # 遍历文件夹中的所有文件
# for file in os.listdir(folder_path):
#     if file.endswith('.csv'):  # 确保处理的是 CSV 文件
#         file_path = os.path.join(folder_path, file)
#         try:
#             # 读取文件
#             df = pd.read_csv(file_path)
#             # 检查文件是否有16列
#             if df.shape[1] == 16:
#                 # 删除最后两列
#                 df_modified = df.iloc[:, :-2]
#                 # 覆盖保存修改后的文件
#                 df_modified.to_csv(file_path, index=False)
#         except Exception as e:
#             print(f"Error processing file {file}: {e}")
#
# print("已处理所有16列文件，删除最后两列，并覆盖原文件。")

# import torch
# print(torch.__version__)
# print(torch.version.cuda)
# import torch_scatter
# print(torch_scatter.__version__)
# import torch
# from torch_scatter import scatter_mean
#
# # 创建一些示例数据
# src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]], dtype=torch.float).t()  # Transpose for correct shape
# index = torch.tensor([0, 1, 0, 1, 0])  # Indices to indicate where to scatter
#
# # 使用 torch_scatter 进行聚合操作
# output = scatter_mean(src, index, dim=0)  # Computing the mean per index
#
# print("Input:")
# print(src)
# print("Scatter index:")
# print(index)
# print("Output (mean per index):")
# print(output)

# import os
# import shutil
#
# # 设置要检查的目录
# base_dir = "D:/dresden/mapgeneralization/dataset/train"
#
# # 设置新文件夹的路径
# new_base_dir = "D:/dresden/mapgeneralization/dataset/missing_files"
#
# # 定义必须存在的文件列表
# required_files = [
#     "updated_adj_df.csv",
#     "nodes_movement_df.csv",
#     "feature_matrix_geb10.csv",
#     "df_adj_matrix_geb10.csv"
# ]
#
# # 用于存储不包含所有必需文件的文件夹
# missing_folders = []
#
# # 遍历目录下的所有子文件夹
# for folder in os.listdir(base_dir):
#     folder_path = os.path.join(base_dir, folder)
#     if os.path.isdir(folder_path):  # 确保是一个文件夹
#         # 检查所有必需的文件是否都存在
#         if not all(os.path.exists(os.path.join(folder_path, file)) for file in required_files):
#             missing_folders.append(folder_path)
#
# # 确保新目录存在，如果不存在则创建
# if not os.path.exists(new_base_dir):
#     os.makedirs(new_base_dir)
#
# # 移动文件夹
# for folder in missing_folders:
#     dest_folder = os.path.join(new_base_dir, os.path.basename(folder))
#     shutil.move(folder, dest_folder)  # 移动文件夹
#     print(f"Moved {folder} to {dest_folder}")
#
# import os
# import pandas as pd
#
#
# def modify_files(directory):
#     # 遍历指定目录下的所有文件
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)
#
#         # 确保只处理CSV文件
#         if file_path.endswith('.csv'):
#             # 读取CSV文件
#             df = pd.read_csv(file_path)
#
#             # 检查列数是否足够
#             if df.shape[1] >= 3:
#                 # 删除第二列
#                 df.drop(df.columns[1], axis=1, inplace=True)
#
#                 # 保存修改后的DataFrame覆盖原文件
#                 df.to_csv(file_path, index=False)
#                 print(f"Processed {file_path}")
#             else:
#                 print(f"Skipped {file_path}, not enough columns.")
# #
# #
# 指定要处理的文件夹路径
# directory_path = r"D:\dresden\mapgeneralization\dataset\feature\feature_matrix_geb10"
# modify_files(directory_path)

# import os
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
#
#
# def standardize_features(directory):
#     scaler = StandardScaler()
#
#     # 遍历指定目录下的所有文件
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)
#
#         # 确保只处理CSV文件
#         if file_path.endswith('.csv'):
#             # 读取CSV文件
#             df = pd.read_csv(file_path)
#
#             # 检查列数是否足够
#             if df.shape[1] >= 9:
#                 # 选择前九列进行标准化
#                 features = df.iloc[:, :9]
#                 scaled_features = scaler.fit_transform(features)
#
#                 # 替换原数据框中的前九列
#                 df.iloc[:, :9] = scaled_features
#
#                 # 保存修改后的DataFrame覆盖原文件
#                 df.to_csv(file_path, index=False)
#                 print(f"Processed and standardized {file_path}")
#             else:
#                 print(f"Skipped {file_path}, not enough columns.")
#
#
# # 指定要处理的文件夹路径
# directory_path = r"D:\dresden\mapgeneralization\dataset\feature\feature_matrix_geb10"
# standardize_features(directory_path)

# import os
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
#
# # 文件夹路径
# folder_path = "D:\\dresden\\mapgeneralization\\dataset\\feature\\feature_matrix_geb10"
#
# # 初始化标准化器
# scaler = StandardScaler()
#
# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     if filename.endswith(".csv"):  # 假设文件格式是CSV
#         file_path = os.path.join(folder_path, filename)
#
#         # 读取文件
#         df = pd.read_csv(file_path)
#
#         # 提取前九列进行标准化
#         features = df.iloc[:, :10]
#         scaled_features = scaler.fit_transform(features)
#
#         # 将标准化后的数据替换原数据
#         df.iloc[:, :10] = scaled_features
#
#         # 保存标准化后的数据回文件
#         df.to_csv(file_path, index=False)
#
# print("标准化完成")

# import os
# import random
# import shutil
# #
# # 定义源文件夹和目标文件夹路径
# src_folder = r"D:\dresden\mapgeneralization\dataset\feature\feature_matrix_geb10"
# dst_folder = "D:\\dresden\\mapgeneralization\\dataset\\feature_valid"
#
# # 获取源文件夹中的所有文件
# files = os.listdir(src_folder)
#
# # 确保源文件夹中至少有1000个文件
# if len(files) < 1000:
#     print("源文件夹中的文件不足1000个")
# else:
#     # 随机选择1000个文件
#     selected_files = random.sample(files, 1000)
#
#     # 将选择的文件移动到目标文件夹
#     for file in selected_files:
#         src_path = os.path.join(src_folder, file)
#         dst_path = os.path.join(dst_folder, file)
#         shutil.move(src_path, dst_path)
#
#     print("文件移动完成")

# import os
# import pandas as pd
#
# # 文件夹路径
# folder_path = "D:\\dresden\\mapgeneralization\\dataset\\feature\\feature_matrix_geb10"
#
# # 初始化一个空列表来存储列数不是12的文件名
# mismatch_files = []
#
# # 遍历文件夹中的所有文件
# for filename in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, filename)
#
#     if os.path.isfile(file_path) and filename.endswith(".csv"):  # 假设文件格式是CSV
#         # 读取文件
#         df = pd.read_csv(file_path, header=None)
#
#         # 检查列数
#         if df.shape[1] != 12:
#             mismatch_files.append((filename, df.shape[1]))
#
# # 打印列数不是12的文件
# if mismatch_files:
#     print("以下文件的列数不是12：")
#     for file, cols in mismatch_files:
#         print(f"{file}: {cols}列")
# else:
#     print("所有文件的列数都是12。")

import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

def standardize_data(train_dir, val_dir, test_dir):
    scaler = StandardScaler()

    # 遍历训练目录以拟合标准化器
    for filename in os.listdir(train_dir):
        if filename.endswith(".csv"):
            train_path = os.path.join(train_dir, filename)
            train_df = pd.read_csv(train_path)

            # 替换无穷大值和NaN为0
            train_df = train_df.replace([np.inf, -np.inf], np.nan)
            train_df = train_df.fillna(0)

            # 只拟合前 10 列
            scaler.partial_fit(train_df.iloc[:, :])

            # 分别标准化训练集、验证集和测试集
    for dir_path in [train_dir, val_dir, test_dir]:
        for filename in os.listdir(dir_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(dir_path, filename)
                df = pd.read_csv(file_path)

                # 替换无穷大值和NaN为0
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.fillna(0)

                # 应用标准化
                features = df.iloc[:, :]
                scaled_features = scaler.transform(features)

                # 替换原数据
                df.iloc[:, :] = scaled_features

                # 保存回文件
                df.to_csv(file_path, index=False)


# 设置路径
train_dir = r"D:\dresden\mapgeneralization\dataset\feature_train"
val_dir = r"D:\dresden\mapgeneralization\dataset\feature_valid"
test_dir = r"D:\dresden\mapgeneralization\dataset\feature_test"

# 调用函数
standardize_data(train_dir, val_dir, test_dir)
# import os
# import shutil
#
# # 源目录
# source_dir = r"D:\dresden\mapgeneralization\dataset\valid"
# # 目标目录
# target_dir = r"D:\dresden\mapgeneralization\dataset\miss"
#
# # 确保目标目录存在
# os.makedirs(target_dir, exist_ok=True)
#
# # 遍历源目录下所有文件夹
# for folder in os.listdir(source_dir):
#     folder_path = os.path.join(source_dir, folder)
#     if os.path.isdir(folder_path) and folder.startswith("index"):  # 确保是目录且名称以'index'开头
#         # 计算文件夹中的文件数量
#         file_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
#         # 如果文件数量少于7
#         if file_count < 7:
#             # 移动文件夹到目标目录
#             shutil.move(folder_path, os.path.join(target_dir, folder))
#             print(f"Moved {folder} to {target_dir} because it has only {file_count} files.")
# #
#
#
# import os
# import pandas as pd
#
# # 基本路径，包含所有的index文件夹
# base_directory = r'D:\dresden\mapgeneralization\dataset\train'
# column_count = 7  # 我们需要查找的列数
#
# # 用于存储含有所需列数CSV文件的文件夹名称
# index_folders_with_match = set()
#
# # 遍历基本路径下的所有文件夹
# for folder in os.listdir(base_directory):
#     if folder.startswith('index'):  # 检查文件夹名是否以 'index' 开头
#         full_path = os.path.join(base_directory, folder)
#         if os.path.isdir(full_path):  # 确保这是一个文件夹
#             # 遍历文件夹中的所有文件
#             for filename in os.listdir(full_path):
#                 filepath = os.path.join(full_path, filename)
#                 if filepath.endswith('.csv'):  # 假设数据文件为CSV格式
#                     try:
#                         df = pd.read_csv(filepath)
#                         if df.shape[1] == column_count:  # 检查列数
#                             index_folders_with_match.add(folder)
#                             break  # 找到一个后就不再检查当前文件夹的其他文件
#                     except Exception as e:
#                         print(f'Error reading {filename} in {folder}: {e}')
#
# # 打印包含符合条件CSV文件的文件夹名称
# for folder in sorted(index_folders_with_match):
#     print(f'Folder {folder} contains a CSV file with {column_count} columns')

# import os
# import shutil
#
# # 基本路径，包含所有的index文件夹
# base_directory = r'D:\dresden\mapgeneralization\dataset\train'
# target_directory = r'D:\dresden\mapgeneralization\dataset\miss2'
# column_count = 7  # 我们需要查找的列数
#
# # 创建目标目录（如果不存在）
# if not os.path.exists(target_directory):
#     os.makedirs(target_directory)
#
# # 用于存储含有所需列数CSV文件的文件夹名称
# index_folders_with_match = set()
#
# # 遍历基本路径下的所有文件夹
# for folder in os.listdir(base_directory):
#     if folder.startswith('index'):  # 检查文件夹名是否以 'index' 开头
#         full_path = os.path.join(base_directory, folder)
#         if os.path.isdir(full_path):  # 确保这是一个文件夹
#             # 遍历文件夹中的所有文件
#             for filename in os.listdir(full_path):
#                 filepath = os.path.join(full_path, filename)
#                 if filepath.endswith('.csv'):  # 假设数据文件为CSV格式
#                     try:
#                         df = pd.read_csv(filepath)
#                         if df.shape[1] == column_count:  # 检查列数
#                             index_folders_with_match.add(full_path)
#                             break  # 找到一个后就不再检查当前文件夹的其他文件
#                     except Exception as e:
#                         print(f'Error reading {filename} in {folder}: {e}')
#
# # 移动文件到目标目录
# for folder in index_folders_with_match:
#     folder_name = os.path.basename(folder)
#     target_path = os.path.join(target_directory, folder_name)
#     shutil.move(folder, target_path)
#     print(f'Moved {folder} to {target_path}')
#
# print(f'Total folders moved: {len(index_folders_with_match)}')
