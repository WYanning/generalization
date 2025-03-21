import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,recall_score,precision_score
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import os,re
import numpy as np
import matplotlib.pyplot as plt
from model.Data_Util.dsutil import *
from edge_model2 import *
import glob
import einops
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
def get_csv_paths(directory_path):
    """Get a list of all CSV file paths in the given directory."""
    return glob.glob(f"{directory_path}/*.csv")

def read_csv_file(path, skiprows=1, usecols=None):
    """Read a CSV file and return its values."""
    df = pd.read_csv(path, header=None, skiprows=skiprows, usecols=usecols)
    return df.values

def collate_fn(batch):
    """Custom collate function to handle batched data."""
    return Batch.from_data_list(batch)

def extract_index_from_path(path):
    """从文件路径中提取索引中的数字"""
    directory_name = os.path.basename(os.path.dirname(path))
    match = re.search(r'\d+', directory_name)
    if match:
        return match.group(0)
    return ''

def save_epoch_data(all_predictions, output_dir):
    """保存每个批次的预测数据到CSV文件中，并在文件名中包含索引。"""
    for idx, data in enumerate(all_predictions):
        if data['file_name']:
            original_file_path = data['file_name']
        else:
            original_file_path = 'unknown'

        index = extract_index_from_path(original_file_path)  # 从路径中提取索引中的数字
        base_name = os.path.basename(original_file_path)
        if index and base_name.startswith("feature_matrix_geb10"):
            base_name = f'feature_matrix_geb10_{index}.csv'

        file_name = f"{base_name}.csv"
        file_path = os.path.join(output_dir, file_name)

        predictions_df = pd.DataFrame(data['predictions'], columns=['Pred_X', 'Pred_Y'])
        truths_df = pd.DataFrame(data['truths'], columns=['True_X', 'True_Y'])
        combined_df = pd.concat([predictions_df, truths_df], axis=1)

        combined_df.to_csv(file_path, index=False)
def fill_nan_with_mean(data):
        # 计算每列的均值
    col_mean = np.nanmean(data, axis=0)
        # 找到 NaN 的索引
    inds = np.where(np.isnan(data))
        # 用均值填充 NaN
    data[inds] = np.take(col_mean, inds[1])
    return data

def to_directed(edge_index):

        # 选择起始节点索引小于结束节点索引的边
    mask = edge_index[0] < edge_index[1]
    directed_edge_index = edge_index[:, mask]
    return directed_edge_index

class MultiTaskMappingDataset(Dataset):
    def __init__(self, base_folder):
        super(MultiTaskMappingDataset, self).__init__()
        self.base_folder = base_folder
        self.check_list = ["feature", "df_adj_matrix_geb10", "nodes_movement_df", "updated_adj_df","df_adj_matrix_tri"]
        self.max_nodes = 40  # 直接设定最大节点数为 40
        self.max_edges = 80  # 直接设定最大边数为 40
        self.max_y_edges = 80
        self.built_dataset()

    def check_illg(self, folder):
        """Check if all necessary files exist and contain valid data."""
        # 检查所有必要文件是否存在
        for check_name in self.check_list:
            if not os.path.exists(os.path.join(folder, check_name + ".csv")):
                return False
        # 加载并检查nodes_movement_df数据
        movement_data = read_csv_file(os.path.join(folder, self.check_list[2] + ".csv"), usecols=[4,5])
        if movement_data.shape[0] > 40:
            return False  # 节点移动数据行数不得超过40
        # 计算并检查距离是否全部小于等于20
        distances = np.sqrt(np.sum(np.square(movement_data), axis=1))
        if not all(distances <= 10):
            return False
        # 加载并检查feature_matrix_geb10数据
        features1 = read_csv_file(os.path.join(folder, self.check_list[0] + ".csv"))
        if features1.shape[0] > 40:
            return False  # 特征数据行数不得超过40
        # 加载并检查df_adj_matrix_geb10数据
        adj_matrix = read_csv_file(os.path.join(folder, self.check_list[1] + ".csv"))
        if adj_matrix.shape[0] > 40 or adj_matrix.shape[1] > 35:
            return False  # 邻接矩阵尺寸不得超过40x40
        # 加载并检查updated_adj_df数据
        updated_adj_matrix = read_csv_file(os.path.join(folder, self.check_list[3] + ".csv"))
        if updated_adj_matrix.shape[0] > 40 or updated_adj_matrix.shape[1] > 40:
            return False  # 更新的邻接矩阵尺寸不得超过40x40

        adj_tri = read_csv_file(os.path.join(folder,self.check_list[4] + ".csv"))
        if adj_tri.shape[0] > 40 or adj_tri.shape[1] > 40:
            return False
        return True

    def built_dataset(self):
        """Build the dataset by checking all folders in the base directory。"""
        self.index_folder = []
        for index_folder in os.listdir(self.base_folder):
            if self.check_illg(os.path.join(self.base_folder, index_folder)):
                self.index_folder.append(os.path.join(self.base_folder, index_folder))

    def get_max_nodes(self):
        """Get the maximum number of nodes in the dataset。"""
        max_nodes = 0
        for folder in self.index_folder:
            features1 = read_csv_file(os.path.join(folder, self.check_list[0] + ".csv"))
            if features1.shape[0] > max_nodes:
                max_nodes = features1.shape[0]
        return max_nodes

    def get_max_edges(self):
        """Get the maximum number of edges in the dataset。"""
        max_edges = 0
        for folder in self.index_folder:
            adjacency_path1 = os.path.join(folder, self.check_list[1] + ".csv")
            adjacencies1 = read_csv_file(adjacency_path1,
                                         usecols=range(1, len(pd.read_csv(adjacency_path1, header=None).columns)))
            adjacency = torch.tensor(adjacencies1, dtype=torch.float)
            adjacency = self.pad_adjacency(adjacency, self.max_nodes)
            edge_index = adjacency.nonzero().t().contiguous()
            if edge_index.shape[1] > max_edges:
                max_edges = edge_index.shape[1]
        return max_edges

    def get_max_y_edges(self):
        """Get the maximum number of edges in the dataset。"""
        max_edges = 0
        for folder in self.index_folder:
            updated_adjacency_path = os.path.join(folder, self.check_list[3] + ".csv")
            updated_adjacencies = read_csv_file(updated_adjacency_path,
                                                usecols=range(1, len(pd.read_csv(updated_adjacency_path, header=None).columns)))
            updated_adjacency = torch.tensor(updated_adjacencies, dtype=torch.float)
            updated_adjacency = self.pad_adjacency(updated_adjacency, self.max_nodes)
            edge_index = updated_adjacency.nonzero().t().contiguous()
            if edge_index.shape[1] > max_edges:
                max_edges = edge_index.shape[1]
        return max_edges

    def pad_features(self, features, max_nodes):
        """Pad the features matrix to have max_nodes rows。"""
        num_nodes, num_features = features.shape
        if num_nodes < max_nodes:
            padding = torch.zeros((max_nodes - num_nodes, num_features))
            features = torch.cat((features, padding), dim=0)
        return features

    def pad_adjacency(self, adjacency, max_nodes):
        """Pad the adjacency matrix to have max_nodes x max_nodes dimensions。"""
        num_nodes = adjacency.shape[0]
        if num_nodes < max_nodes:
            padding = torch.zeros((max_nodes, max_nodes))
            padding[:num_nodes, :num_nodes] = adjacency
            adjacency = padding
        elif num_nodes > max_nodes:
            adjacency = adjacency[:max_nodes, :max_nodes]
        return adjacency

    def adjacency_matrix_to_edge_list(matrix):
        edge_list = []
        n = len(matrix)  # 获取矩阵大小，即顶点数量
        for i in range(n):
            for j in range(n):
                if i != j:  # 排除自环
                    edge_list.append((i, j, matrix[i][j]))
        return edge_list

    def positional_encoding(self, coords, d_model):
        """Generate positional encoding for given coordinates."""
        N = coords.size(0)
        if coords.size(1) != 2:
            raise ValueError("coords must have shape (N, 2) representing (x, y) coordinates")

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(N, d_model)
        pe[:, 0::2] = torch.sin(coords[:, 0].unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(coords[:, 1].unsqueeze(1) * div_term)
        return pe

    def __len__(self):
        return len(self.index_folder)

    def __getitem__(self, item):
        def directed_adjacency(adjacency):
            num_nodes = adjacency.shape[0]  # 获取实际的节点数
            triu_indices = torch.triu_indices(num_nodes, num_nodes, 1)
            directed_adjacency = torch.zeros((self.max_nodes, self.max_nodes), dtype=adjacency.dtype)
            directed_adjacency[triu_indices[0], triu_indices[1]] = adjacency[triu_indices[0], triu_indices[1]]
            return directed_adjacency

        index_folder = self.index_folder[item]

        features1 = read_csv_file(os.path.join(index_folder, self.check_list[0] + ".csv"))

        adjacency_path1 = os.path.join(index_folder, self.check_list[1] + ".csv")
        df_adj_matrix_tri_path = os.path.join(index_folder, self.check_list[4] + ".csv")
        adjacencies1 = read_csv_file(adjacency_path1,
                                     usecols=range(1, len(pd.read_csv(adjacency_path1, header=None).columns)))
        adjacencies_tri = read_csv_file(df_adj_matrix_tri_path,
                                     usecols=range(1, len(pd.read_csv(df_adj_matrix_tri_path, header=None).columns)))
        movement_data = read_csv_file(os.path.join(index_folder, self.check_list[2] + ".csv"), usecols=[4,5])
        pos = read_csv_file(os.path.join(index_folder, self.check_list[2] + ".csv"), usecols=[2,3])
        updated_adjacency_path = os.path.join(index_folder, self.check_list[3] + ".csv")
        updated_adjacencies = read_csv_file(updated_adjacency_path, usecols=range(1, len(pd.read_csv(
            updated_adjacency_path, header=None).columns)))
        features1 = fill_nan_with_mean(features1)
        columns_to_delete = [9,10]
        features1 = np.delete(features1, columns_to_delete, axis=1)

        x = torch.tensor(features1, dtype=torch.float)
        adjacency = torch.tensor(adjacencies1, dtype=torch.float)
        updated_adjacency = torch.tensor(updated_adjacencies, dtype=torch.float)
        adjacencies_tri= torch.tensor( adjacencies_tri, dtype=torch.float)
        # adjacency = directed_adjacency(torch.tensor(adjacencies1, dtype=torch.float))
        # updated_adjacency = directed_adjacency(torch.tensor(updated_adjacencies, dtype=torch.float))
        # adjacencies_tri = directed_adjacency(torch.tensor(adjacencies_tri, dtype=torch.float))
        num_nodes = features1.shape[0]
        x = self.pad_features(x, self.max_nodes)
        adjacency = self.pad_adjacency(adjacency, self.max_nodes)
        updated_adjacency = self.pad_adjacency(updated_adjacency, self.max_nodes)
        adjacencies_tri = self.pad_adjacency(adjacencies_tri, self.max_nodes)
        padded_movement_data= np.pad(movement_data, ((0, self.max_nodes - movement_data.shape[0]), (0,0)),
                                      mode='constant')

        padded_pos = np.pad(pos,
                                           ((0, self.max_nodes - pos.shape[0]), (0, 0)),
                                           mode='constant')
        # if padded_pos.shape != (self.max_nodes, 2):
        #     raise ValueError(f"padded_pos shape is {padded_pos.shape}, but expected ({self.max_nodes}, 2)")
        mask = torch.zeros(self.max_nodes, dtype=torch.bool)
        mask[:num_nodes] = True
        #
        # positional_encoding = self.positional_encoding(torch.tensor(padded_pos, dtype=torch.float), d_model=4)
        # x = torch.cat((x, positional_encoding), dim=1)
        node_movement = torch.tensor(padded_movement_data, dtype=torch.float)

        pos = torch.tensor(padded_pos,dtype=torch.float)

        # edge_index = adjacencies_tri.nonzero().t().contiguous()
        # assert edge_index.shape[1] <= self.max_edges
        # padding = torch.zeros((2, self.max_edges - edge_index.shape[1]))
        # edge_index = torch.cat((edge_index, padding), dim=1).to(torch.int32)

        edge_index = adjacency.nonzero().t().contiguous()
        def check_tensor(tensor, tensor_name="tensor"):
            if torch.isnan(tensor).any():
                print(f"{tensor_name} contains NaN values.")
            if torch.isinf(tensor).any():
                print(f"{tensor_name} contains infinite values.")

        check_tensor(x, "x")
        check_tensor(edge_index, "edge_index")
        check_tensor(updated_adjacency, "updated_adjacency")
        check_tensor(node_movement, "node_movement")
        num_edges = edge_index.shape[1]
        # torch.set_printoptions(threshold=5000, linewidth=200, precision=2, edgeitems=50)
        # print(x.shape,'x')
        # print(edge_index.shape,'edge_index')
        # print(updated_adjacency.shape,'updated_adj')
        # print(node_movement.shape,'movement')
        # print(node_type,'type')
        # print(mask.shape,'mask')
        # print(adjacency)
        # print(pos.shape,'pos')
        return Data(pos=pos, x=x, edge_index=edge_index,adjacency=adjacency, updated_adjacency=updated_adjacency, node_movement=node_movement, mask=mask,adj_tri=adjacencies_tri,
                    file_name=os.path.basename(index_folder),num_edges=num_edges)

if __name__ == '__main__':
    save_loc_adj = r"D:\dresden\mapgeneralization\output0719\adj"
    train_path = r"D:\dresden\mapgeneralization\dataset\train"
    test_path = r"D:\dresden\mapgeneralization\dataset\test"
    valid_path = r"D:\dresden\mapgeneralization\dataset\valid"

    # Create datasets
    train_dataset = MultiTaskMappingDataset(base_folder=train_path)
    print("Number of samples in dataset:", len(train_dataset))
    test_dataset = MultiTaskMappingDataset(base_folder=test_path)
    valid_dataset = MultiTaskMappingDataset(base_folder=valid_path)
    data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, drop_last=True)
    in_channels = train_dataset[0].x.shape[1]
    out_channels_edge = train_dataset.max_nodes
    model = EdgeModel_pos_GCN(in_channels, [64, 64, 64], out_channels_edge).to(device)
    model_file_path = 'D:\\dresden\\mapgeneralization\\output0914\\model\\best_model_gcn.pth'
    model.load_state_dict(torch.load(model_file_path))
    model.eval()
    total_loss = 0
    all_pred_edges = []
    all_true_edges = []
    valid_masks = []  # 用来存储有效的掩码

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            mask = batch.mask
            out_edge = model(batch.x, batch.edge_index, batch.pos, mask, batch.adjacency, batch.adj_tri)

            # Compute final edges
            initial_edges = batch.adjacency.view(64, 40, 40)
            mask_edge = mask.view(64, 40)
            mask = (mask_edge.unsqueeze(2) * mask_edge.unsqueeze(1))# 重新形状掩码以匹配邻接矩阵的维度
            _, predicted_changes = torch.max(out_edge, dim=-1)
            predicted_changes -= 1
            final_edges = initial_edges + predicted_changes

            # 使用掩码来确定有效的改变
            final_edges *= mask
            true_edges = batch.updated_adjacency.view(64, 40, 40) * mask  # 也应用掩码到真实边

            # 转换到CPU并保存
            final_edges_np = final_edges.cpu().detach().numpy()
            true_edges_np = true_edges.cpu().detach().numpy()

            all_pred_edges.append(final_edges_np)
            all_true_edges.append(true_edges_np)
            valid_masks.append(mask.cpu().numpy())

    # Flatten all results for metric calculation
    all_pred_edges_flat = np.concatenate([fe[mask == 1].flatten() for fe, mask in zip(all_pred_edges, valid_masks)])
    all_true_edges_flat = np.concatenate([te[mask == 1].flatten() for te, mask in zip(all_true_edges, valid_masks)])

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_true_edges_flat, all_pred_edges_flat)
    f1 = f1_score(all_true_edges_flat, all_pred_edges_flat, average='weighted')
    recall = recall_score(all_true_edges_flat, all_pred_edges_flat, average='weighted')
    precision = precision_score(all_true_edges_flat, all_pred_edges_flat, average='weighted')
    conf_matrix = confusion_matrix(all_true_edges_flat, all_pred_edges_flat)

    total_rows = 0  # 总行数
    degree_two_rows = 0  # 度数为2的行数

    for adjacency_matrix in all_true_edges:
        for matrix in adjacency_matrix:
            # 计算每行的度数
            row_degrees = np.sum(matrix, axis=1)  # 沿着列求和得到每行的度数
            # 计算度数为2的行的数量
            degree_two_rows += np.sum(row_degrees == 2)
            total_rows += len(row_degrees)  # 总行数累加

    # 计算度数为2的行的比率
    if total_rows > 0:
        degree_two_ratio_true = degree_two_rows / total_rows
        print("Ratio of rows with degree 2:", degree_two_ratio_true)
    else:
        print("No rows to evaluate.")

    for adjacency_matrix in all_pred_edges:
        for matrix in adjacency_matrix:
            # 计算每行的度数
            row_degrees = np.sum(matrix, axis=1)  # 沿着列求和得到每行的度数
            # 计算度数为2的行的数量
            degree_two_rows += np.sum(row_degrees == 2)
            total_rows += len(row_degrees)  # 总行数累加
    if total_rows > 0:
        degree_two_ratio_pred = degree_two_rows / total_rows
        print("Ratio of rows with degree 2:", degree_two_ratio_pred)
    else:
        print("No rows to evaluate.")

    ratio = degree_two_ratio_pred/degree_two_ratio_true
    print(ratio)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Recall: {recall:.4f}')
    print('Confusion Matrix:')
    print("Precision:", precision)
    print(conf_matrix)


    def plot_confusion_matrix(cm):
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()


    plot_confusion_matrix(conf_matrix)
