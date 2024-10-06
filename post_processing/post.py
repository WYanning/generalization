import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

def find_matching_files(pred_file_name,original_dir, adj_matrix_dir, updated_adj_matrix_dir,updated_adj_matrix_new_dir):
    match = re.search(r'index(\d+).csv', pred_file_name)
    if match:
        identifier = match.group(1)
        original_file_name = f"nodes_movement_df_{identifier}.csv"
        adj_matrix_file_name = f"df_adj_matrix_geb10_{identifier}.csv"
        updated_adj_matrix_file_name = f"updated_adj_df_{identifier}.csv"
        # updated_adj_matrix_file_new_name = f"Hamiltonian_Matrix_{identifier}.csv"
        updated_adj_matrix_file_new_name = f"index{identifier}.csv"


        original_file_path = os.path.join(original_dir, original_file_name)
        adj_matrix_file_path = os.path.join(adj_matrix_dir, adj_matrix_file_name)
        updated_adj_matrix_file_path = os.path.join(updated_adj_matrix_dir, updated_adj_matrix_file_name)
        updated_adj_matrix_file_new_path = os.path.join(updated_adj_matrix_new_dir, updated_adj_matrix_file_new_name)

        if (os.path.exists(original_file_path) and
            os.path.exists(adj_matrix_file_path) and
            os.path.exists(updated_adj_matrix_file_path)and
            os.path.exists(updated_adj_matrix_file_new_path)):
            return original_file_path, adj_matrix_file_path, updated_adj_matrix_file_path, updated_adj_matrix_file_new_path

    return None, None, None,None

def read_adj_matrix(file_path):
    return pd.read_csv(file_path, header=0, index_col=0)


def read_adj_matrix_new(file_path):
    return pd.read_csv(file_path, header=0)
def read_orginal_file(file_path):
    return pd.read_csv(file_path,index_col=0)

def process_files(pred_file,original_file, adj_matrix_file, updated_adj_matrix_file, updated_adj_matrix_file_new, save_dir):
    pred_data = pd.read_csv(pred_file)
    original_data = read_orginal_file(original_file)
    adj_matrix = read_adj_matrix(adj_matrix_file)
    updated_adj_matrix = read_adj_matrix(updated_adj_matrix_file)
    updated_adj_matrix_new = read_adj_matrix(updated_adj_matrix_file_new)
    identifier = os.path.basename(updated_adj_matrix_file_new).split('.')[0]
    # print("Original Data Shape:", original_data.shape)
    # print("Prediction Data Shape:", pred_data.shape)
    # print("Adjacency Matrix Shape:", adj_matrix.shape)
    # print("Updated Adjacency Matrix Shape:", updated_adj_matrix.shape)
    # print( updated_adj_matrix_new.shape)
    original_data['X_Final'] = original_data.iloc[:, 1] + original_data.iloc[:, 3]
    original_data['Y_Final'] = original_data.iloc[:, 2] + original_data.iloc[:, 4]

    pred_data['Pred_X_Final'] = original_data.iloc[:, 1] + pred_data.iloc[:, 1]
    pred_data['Pred_Y_Final'] = original_data.iloc[:, 2] + pred_data.iloc[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Plot 1: Original positions with original adjacency matrix connections
    axes[0].scatter(original_data.iloc[:, 1], original_data.iloc[:, 2], color='green', label='Original Position',
                    alpha=0.6, s=100)  # Increased point size
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix.iloc[i, j] == 1:
                axes[0].plot([original_data.iloc[i, 1], original_data.iloc[j, 1]],
                             [original_data.iloc[i, 2], original_data.iloc[j, 2]], 'k-', lw=2)
    axes[0].set_aspect('equal')  # Keep the aspect ratio of the axes
    axes[0].set_xlabel('X Coordinate')
    axes[0].set_ylabel('Y Coordinate')
    axes[0].set_title('Original Positions with Connections')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Original to moved positions with updated adjacency matrix connections
    axes[1].scatter(original_data.iloc[:, 1], original_data.iloc[:, 2], color='green', label='Original Position',
                    alpha=0.6, s=100)  # Increased point size
    axes[1].scatter(original_data['X_Final'], original_data['Y_Final'], color='blue', label='Moved Position', alpha=0.6,
                    s=100)  # Increased point size
    for i in range(len(updated_adj_matrix)):
        for j in range(len(updated_adj_matrix)):
            if updated_adj_matrix.iloc[i, j] == 1:
                axes[1].plot([original_data['X_Final'][i], original_data['X_Final'][j]],
                             [original_data['Y_Final'][i], original_data['Y_Final'][j]], 'r-', lw=2)
    for i in range(len(original_data)):
        axes[1].arrow(original_data.iloc[i, 1], original_data.iloc[i, 2],
                      original_data['X_Final'][i] - original_data.iloc[i, 1],
                      original_data['Y_Final'][i] - original_data.iloc[i, 2],
                      head_width=0.2, head_length=0.2, fc='orange', ec='orange', lw=2)
    axes[1].set_aspect('equal')  # Keep the aspect ratio of the axes
    axes[1].set_xlabel('X Coordinate')
    axes[1].set_ylabel('Y Coordinate')
    axes[1].set_title('Original to Moved Positions with Connections')
    axes[1].legend()
    axes[1].grid(True)


    # Plot 4: Final plot modifications
    axes[2].scatter(pred_data['Pred_X_Final'], pred_data['Pred_Y_Final'], color='red', label='Predicted Final Position',
                    alpha=0.6, s=100)  # Increased point size
    for i in range(len(updated_adj_matrix_new)):
        for j in range(len(updated_adj_matrix_new)):
            if updated_adj_matrix_new.iloc[i, j] == 1:
                axes[2].plot([pred_data['Pred_X_Final'][i], pred_data['Pred_X_Final'][j]],
                             [pred_data['Pred_Y_Final'][i], pred_data['Pred_Y_Final'][j]], 'green', lw=2)
    axes[2].set_aspect('equal')  # Keep the aspect ratio of the axes
    axes[2].set_xlabel('X Coordinate')
    axes[2].set_ylabel('Y Coordinate')
    axes[2].set_title('Predicted Positions with Connections')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    # plt.show()

    # 确保保存目录存在

    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f'visualization_{identifier}.png')
    plt.savefig(fig_path)
    plt.close()

# 设置文件目录和处理所有匹配的文件
pred_directory = r"D:\dresden\mapgeneralization\output0719\movement"
# pred_directory = 'D:/dresden/mapgeneralization/dataset/inference'
type_directory = r"D:\dresden\mapgeneralization\output0716\type"
original_directory = r"D:\dresden\mapgeneralization\dataset\output1\nodes_movement_df"
adj_matrix_directory = r"D:\dresden\mapgeneralization\dataset\output1\df_adj_matrix_geb10"
updated_adj_matrix_directory = r"D:\dresden\mapgeneralization\dataset\output1\updated_adj_df"
# updated_adj_matrix_directory_new = r"D:\dresden\mapgeneralization\dataset\adj\Hamiltonian_Matrix"
updated_adj_matrix_directory_new = r"D:\dresden\mapgeneralization\output0719\adj_cut"
save_dir = r"D:\dresden\mapgeneralization\output0719\v18"
# save_directory = r"D:\dresden\mapgeneralization\output0616\visualization"
os.makedirs(save_dir, exist_ok=True)

for pred_file in os.listdir(pred_directory):
    if pred_file.endswith('.csv'):
        full_pred_path = os.path.join(pred_directory, pred_file)
        original_file,adj_matrix_file, updated_adj_matrix_file, updated_adj_matrix_file_new= find_matching_files(pred_file,original_directory, adj_matrix_directory, updated_adj_matrix_directory,updated_adj_matrix_directory_new)
        if original_file and adj_matrix_file and updated_adj_matrix_file and updated_adj_matrix_file_new:
            process_files(full_pred_path, original_file, adj_matrix_file, updated_adj_matrix_file,
                          updated_adj_matrix_file_new, save_dir)


