import os
import re

def read_file(filepath):
    with open(filepath, 'r') as file:
        return file.readlines()

def write_file(filepath, lines):
    with open(filepath, 'w') as file:
        file.writelines(lines)

def trim_lines(lines, target_length):
    return lines[:target_length]

def extract_suffix(filename, pattern):
    match = re.search(pattern, filename)
    return match.group(1) if match else None

def process_files(dir1, dir2):
    pattern1 = r'index(\d+)\.csv'
    pattern2 = r'nodes_movement_df_(\d+)\.csv'

    files1 = {extract_suffix(f, pattern1): os.path.join(dir1, f) for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))}
    files2 = {extract_suffix(f, pattern2): os.path.join(dir2, f) for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))}

    for suffix in files1:
        if suffix in files2:
            filepath1 = files1[suffix]
            filepath2 = files2[suffix]

            lines1 = read_file(filepath1)
            lines2 = read_file(filepath2)

            target_length = len(lines2)
            lines1_trimmed = trim_lines(lines1, target_length)

            write_file(filepath1, lines1_trimmed)

if __name__ == "__main__":
    dir1 = r"D:\dresden\mapgeneralization\output0719\movement" # 替换为第一个目录的路径
    dir2 = r"D:\dresden\mapgeneralization\dataset\output1\nodes_movement_df" # 替换为第二个目录的路径
    process_files(dir1, dir2)
