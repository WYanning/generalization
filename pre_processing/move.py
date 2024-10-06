import os, shutil
import re

# def move_folder(base_folder=r"D:\dresden\mapgeneralization\dataset\nodeandadj1",
#                 save_folder=r"D:\dresden\mapgeneralization\dataset\trigraph"):
#     sources = ["Tri"]
#     for source in sources:
#         os.makedirs(os.path.join(save_folder, source), exist_ok=True)
#     for str_index in os.listdir(base_folder):
#         index = int(re.sub("index", "", str_index))
#         for source in sources:
#             source_path = os.path.join(base_folder, str_index, source + ".csv")
#             target_path = os.path.join(save_folder, source, source + "_" + str(index) + ".csv")
#             try:
#                 shutil.copy(source_path, target_path)
#             except:
#                 pass


#
def back_move(base_folder=r"D:\dresden\mapgeneralization\dataset\tri_valid",
              target_folder=r"D:\dresden\mapgeneralization\dataset\valid"):
    sources = ["feature_matrix_geb10"]
    for source in sources:
        source_folder = os.path.join(base_folder, source)
        for str_index_file in os.listdir(source_folder):
            # 使用更精确的正则表达式从文件名中提取数字
            match = re.search(r"(\d+)(?=\.\w+$)", str_index_file)  # 查找文件扩展名前的数字
            if match:
                index = int(match.group(1))
                # 创建目标目录，确保它存在
                target_index_folder = os.path.join(target_folder, "index" + str(index))
                os.makedirs(target_index_folder, exist_ok=True)

                source_path = os.path.join(source_folder, str_index_file)
                target_path = os.path.join(target_index_folder, "df_adj_matrix_tri.csv")

                # 尝试复制文件，并捕捉可能的错误
                try:
                    shutil.copy(source_path, target_path)
                except Exception as e:
                    print(f"Error copying {source_path} to {target_path}: {e}")
            else:
                print(f"Failed to extract index from file name: {str_index_file}")



if __name__ == '__main__':
    # move_folder()
    back_move()