from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import os,re
import numpy as np
import matplotlib.pyplot as plt
from model.Data_Util.dsutil import *
from movementmodel2 import *
import glob
import einops
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import sqrt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
def get_csv_paths(directory_path):
    """Get a list of all CSV file paths in the given directory."""
    return glob.glob(f"{directory_path}/*.csv")

def read_csv_file(path, skiprows=1, usecols=None):
    """Read a CSV file and return its values."""
    df = pd.read_csv(path, header=None, skiprows=skiprows, usecols=usecols)
    return df.values



def extract_index_from_path(path):
    """从文件路径中提取索引中的数字"""
    directory_name = os.path.basename(os.path.dirname(path))
    match = re.search(r'\d+', directory_name)
    if match:
        return match.group(0)
    return ''

# def save_epoch_data(all_predictions, output_dir):
#     """保存每个批次的预测数据到CSV文件中，并在文件名中包含索引。"""
#     for idx, data in enumerate(all_predictions):
#         if data['file_name']:
#             original_file_path = data['file_name']
#         else:
#             original_file_path = 'unknown'
#
#         index = extract_index_from_path(original_file_path)  # 从路径中提取索引中的数字
#         base_name = os.path.basename(original_file_path)
#         if index and base_name.startswith("feature_matrix_geb10"):
#             base_name = f'feature_matrix_geb10_{index}.csv'
#
#         file_name = f"{base_name}.csv"
#         file_path = os.path.join(output_dir, file_name)
#
#         predictions_df = pd.DataFrame(data['predictions'], columns=['Pred_X', 'Pred_Y'])
#         truths_df = pd.DataFrame(data['truths'], columns=['True_X', 'True_Y'])
#         combined_df = pd.concat([predictions_df, truths_df], axis=1)
#
#         combined_df.to_csv(file_path, index=False)
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
        self.max_edges = 40  # 直接设定最大边数为 40
        self.max_y_edges = 40
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
        index_folder = self.index_folder[item]

        features1 = read_csv_file(os.path.join(index_folder, self.check_list[0] + ".csv"))

        adjacency_path1 = os.path.join(index_folder, self.check_list[1] + ".csv")
        df_adj_matrix_tri_path = os.path.join(index_folder, self.check_list[4] + ".csv")
        adjacencies1 = read_csv_file(adjacency_path1,
                                     usecols=range(1, len(pd.read_csv(adjacency_path1, header=None).columns)))
        adjacencies_tri = read_csv_file(df_adj_matrix_tri_path,
                                     usecols=range(1, len(pd.read_csv(df_adj_matrix_tri_path, header=None).columns)))
        movement_data = read_csv_file(os.path.join(index_folder, self.check_list[2] + ".csv"), usecols=[4,5])
        updated_adjacency_path = os.path.join(index_folder, self.check_list[3] + ".csv")
        pos = read_csv_file(os.path.join(index_folder, self.check_list[2] + ".csv"), usecols=[2, 3])
        updated_adjacencies = read_csv_file(updated_adjacency_path, usecols=range(1, len(pd.read_csv(
            updated_adjacency_path, header=None).columns)))
        features1 = fill_nan_with_mean(features1)
        columns_to_delete = [9,10]
        features1 = np.delete(features1, columns_to_delete, axis=1)

        x = torch.tensor(features1, dtype=torch.float)
        adjacency = torch.tensor(adjacencies1, dtype=torch.float)
        updated_adjacency = torch.tensor(updated_adjacencies, dtype=torch.float)
        adjacencies_tri = torch.tensor(adjacencies_tri,dtype=torch.float)
        pos = torch.tensor(pos, dtype=torch.float)

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
        mask = torch.zeros(self.max_nodes, dtype=torch.bool)
        mask[:num_nodes] = True

        # positional_encoding = self.positional_encoding(torch.tensor(padded_pos, dtype=torch.float), d_model=128)
        # x = torch.cat((x, positional_encoding), dim=1)
        node_movement = torch.tensor(padded_movement_data, dtype=torch.float)
        edge_index = adjacency.nonzero().t().contiguous()
        edge_index = to_directed(edge_index)
        # edge_index = adjacencies.nonzero().t().contiguous()


        pos = torch.tensor(padded_pos, dtype=torch.float)
        pos_org = torch.tensor(padded_pos,dtype=torch.float)
        edge_index_tri= adjacencies_tri.nonzero().t().contiguous()
        edge_index_tri = to_directed(edge_index_tri)
        assert edge_index.shape[1] <= self.max_edges
        padding = torch.zeros((2, self.max_edges - edge_index.shape[1]))
        edge_index = torch.cat((edge_index, padding), dim=1).to(torch.int32)

        def check_tensor(tensor, tensor_name="tensor"):
            if torch.isnan(tensor).any():
                print(f"{tensor_name} contains NaN values.")
            if torch.isinf(tensor).any():
                print(f"{tensor_name} contains infinite values.")

        check_tensor(x, "x")
        check_tensor(edge_index, "edge_index")
        check_tensor(updated_adjacency, "updated_adjacency")
        check_tensor(node_movement, "node_movement")

        # torch.set_printoptions(threshold=5000, linewidth=200, precision=2, edgeitems=50)
        # print(x.shape,'x')
        # print(edge_index.shape,'edge_index')
        # print(updated_adjacency,'updated_adj')
        # print(node_movement,'movement')
        # print(node_type,'type')
        # print(mask.shape,'mask')
        # print(adjacency)
        return Data(pos_org=pos_org, pos=pos,x=x, edge_index=edge_index,edge_index_tri=edge_index_tri,adjacency=adjacency, updated_adjacency=updated_adjacency, node_movement=node_movement,mask=mask,mask_tri=adjacencies_tri,
                    file_name=os.path.basename(index_folder))

if __name__ == '__main__':
    save_loc_movement = r"D:\dresden\mapgeneralization\output0914\movement_gcn"
    save_loc_type = r"D:\dresden\mapgeneralization\output0719\type"
    train_path = r"D:\dresden\mapgeneralization\dataset\train"
    test_path = r"D:\dresden\mapgeneralization\dataset\test"
    valid_path = r"D:\dresden\mapgeneralization\dataset\valid"

    # Create datasets
    train_dataset = MultiTaskMappingDataset(base_folder=train_path)
    print("Number of samples in dataset:", len(train_dataset))
    test_dataset = MultiTaskMappingDataset(base_folder=test_path)
    valid_dataset = MultiTaskMappingDataset(base_folder=valid_path)
    # data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,drop_last=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, drop_last=True)
    print(train_dataset.max_nodes)
    data=valid_dataset[0]
    # in_channels = train_dataset[0].x.shape[1]
    in_channels= train_dataset[0].x.shape[1]
    out_channels_node = 2
    # model =MovementGATModel_pos(in_channels,  [64,64,64],out_channels_node).to(device)
    model =MovementGCNModel_pos(in_channels, [64, 64, 64], out_channels_node).to(device)
    # model = MovementSageModel_pos(in_channels, [64, 64, 64], out_channels_node).to(device)
    model_path = r"D:\dresden\mapgeneralization\output0914\model\movement_model_gcn.pth" # Path to your trained model file
    model.load_state_dict(torch.load(model_path))
    model.eval()


    def evaluate_model(test_loader, model, device):
        model.eval()  # 切换到评估模式
        mse_criterion = torch.nn.MSELoss(reduction='sum')
        mae_criterion = torch.nn.L1Loss(reduction='sum')

        total_mse = 0
        total_mae = 0
        total_samples = 0

        with torch.no_grad():  # 关闭梯度计算
            for batch in test_loader:
                batch = batch.to(device)
                mask = batch.mask
                mask_tri = batch.mask_tri
                output_node, updated_pos = model(batch.x, batch.edge_index, batch.pos, mask, batch.batch)
                batch.pos = updated_pos.detach()

                # 确保output_node与batch.node_movement形状一致
                if output_node.shape != batch.node_movement.shape:
                    output_node = output_node.view(-1, batch.node_movement.shape[-1])  # 适应目标形状

                # 计算MSE和MAE
                mse = mse_criterion(output_node, batch.node_movement)
                mae = mae_criterion(output_node, batch.node_movement)

                total_mse += mse.item()
                total_mae += mae.item()
                total_samples += batch.node_movement.numel() / batch.node_movement.shape[-1]  # 计算总节点数

        # 计算平均MSE和MAE
        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples
        avg_rmse = torch.sqrt(torch.tensor(avg_mse))

        # 打印结果
        print(f"Test MSE: {avg_mse:.4f}, Test MAE: {avg_mae:.4f}, Test RMSE: {avg_rmse:.4f}")


    # 使用实例
    # 假设model, test_loader和device已经被正确设置和初始化
    evaluate_model(test_loader, model, device)