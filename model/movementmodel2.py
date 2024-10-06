import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv,GCNConv,SAGEConv,GINConv,TopKPooling
import torch
from torch_geometric.nn import BatchNorm
from sklearn.preprocessing import StandardScaler

from torch_scatter import scatter_mean
import numpy as np
import networkx as nx
from scipy.linalg import eigh




class MovementGATModel(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels_node, heads=1):
        super(MovementGATModel, self).__init__()
        self.heads = heads

        # # Shared layers with dropout and alpha for GAT layers
        # self.gat1 = GATConv(in_channels, hidden_dims[0], heads=heads)
        # self.bn1 = BatchNorm(hidden_dims[0] * heads)
        # self.gat2 = GATConv(hidden_dims[0] * heads, hidden_dims[1], heads=heads)
        # self.bn2 = BatchNorm(hidden_dims[1] * heads)
        # # self.gat3 = GATConv(hidden_dims[1] * heads, hidden_dims[2], heads=heads)
        # # self.bn3 = BatchNorm(hidden_dims[2] * heads)
        # self.out_type = GATConv(hidden_dims[1] * heads,out_channels_type, heads=heads)
        # self.fc_type = nn.Linear(hidden_dims[1] * heads, out_channels_type)

        self.move_l1 = GATConv(in_channels, hidden_dims[0])
        self.move_norm1 = BatchNorm(hidden_dims[0])
        self.move_l2 = GATConv(hidden_dims[0], hidden_dims[1])
        self.move_norm2 = BatchNorm(hidden_dims[1])
        self.out_move = GATConv(hidden_dims[1] * heads, out_channels_node, heads=heads)
        # self.move_l3 = GATConv(hidden_dims[1], hidden_dims[2])
        # self.move_norm3 = BatchNorm(hidden_dims[2])
        self.fc_node = nn.Linear(hidden_dims[1], out_channels_node)


        # Task-specific outputs for out_edge and out_type

    def forward(self, x, edge_index, mask):
        # 基础的共享层
        # edge_index = edge_index.long()
        # cl1 = F.relu(self.bn1(self.gat1(x, edge_index)))
        # cl2 = F.relu(self.bn2(self.gat2(cl1, edge_index)))
        # # cl3= F.relu(self.bn3(self.gat3(cl2, edge_index)))
        # type_output = self.out_type(cl2, edge_index)

        # enhanced_features1 = torch.cat([x, type_output], dim=1)
        move_features = F.relu(self.move_norm1(self.move_l1(x, edge_index)))
        move_features = F.relu(self.move_norm2(self.move_l2(move_features, edge_index)))
        # move_features = F.relu(self.move_norm3(self.move_l3(move_features, edge_index)))
        # move_output =self.out_move(move_features,edge_index)


        out_node = self.fc_node(move_features)
        # out_type = self.fc_type(cl2)

        # 应用掩码
        mask = mask.unsqueeze(1)
       # 扩展掩码维度以匹配输出
       #  out_type = out_type * mask
        out_node = out_node * mask
        # out_edge = torch.sigmoid(out_edge)
        # print(out_edge)
        # out_type = out_type.view(64, 40, 1)
        out_node = out_node.view(64, 40, 2)

        # print(out_edge)

        return out_node

    # def compute_tri(self,output_node,target_node,mask,initial_node):
    #
    #     return

    # def compute_loss(self, output_node, target_node,mask):
    #     # 计算节点移动的回归损失
    #     target_node = target_node.view(64, 40, 2)
    #
    #     mask_reshaped = mask.view(64, 40)  # 假设 mask 是一维的
    #     mask_expanded_node = mask_reshaped.unsqueeze(2).expand(-1, -1, 2)
    #     # mask_expanded_type = mask_reshaped.unsqueeze(2).expand(-1, -1, 1)
    #
    #     # 应用掩码
    #     output_node_masked = output_node[mask_expanded_node].view(-1, 2)
    #     target_node_masked = target_node[mask_expanded_node].view(-1, 2)
    #
    #     # 计算节点移动的回归损失
    #     loss_node = F.mse_loss(output_node_masked, target_node_masked)
    #
    #     # # 应用掩码并计算类型损失
    #     # output_type_masked = output_type[mask_expanded_type].view(-1, 1)
    #     # target_type_masked = target_type[mask_expanded_type].view(-1, 1)
    #     # loss_type = F.binary_cross_entropy_with_logits(output_type_masked, target_type_masked)
    #
    #
    #     return loss_node



    def compute_mse(self, output_node, target_node, mask):
        target_node = target_node.view(64, 40, 2)
        mask_reshaped = mask.view(64, 40)  # 假设 mask 是一维的
        mask_expanded_node = mask_reshaped.unsqueeze(2).expand(-1, -1, 2)
        output_node_masked = output_node[mask_expanded_node].view(-1, 2)
        target_node_masked = target_node[mask_expanded_node].view(-1, 2)
        loss_node = F.mse_loss(output_node_masked, target_node_masked)
        return loss_node

def filter_edges(edge_index, mask):
# 扁平化mask并获取有效节点的索引
    node_indices = torch.arange(mask.numel(), device=mask.device)[mask.view(-1)]
# 创建一个映射，将旧索引映射到新索引
    new_indices = torch.full((mask.numel(),), -1, device=mask.device, dtype=torch.long)
    new_indices[node_indices] = torch.arange(node_indices.size(0), device=mask.device)
# 应用映射到edge_index
    masked_edge_index = new_indices[edge_index.view(-1)].view(edge_index.shape)
# 移除包含-1的边，即移除被mask掉的节点的边
    mask_valid_edges = (masked_edge_index >= 0).all(dim=0)
    filtered_edge_index = masked_edge_index[:, mask_valid_edges]
    return filtered_edge_index


def angle_loss(predicted, actual, edge_index):
    # edge_index 是一个 2xN 的张量，其中每列代表一条边的两个节点索引
    row, col = edge_index[0], edge_index[1]
    # 选择实际连接的节点对应的预测和实际向量
    vectors_predicted = predicted[col] - predicted[row]
    vectors_actual = actual[col] - actual[row]
    dot_product = torch.sum(vectors_predicted * vectors_actual, dim=1)
    norms_predicted = torch.norm(vectors_predicted, dim=1)
    norms_actual = torch.norm(vectors_actual, dim=1)
    cos_angle = 1 - dot_product / (norms_predicted * norms_actual + 1e-8)
    return torch.sum(cos_angle)

def edge_length_loss(predicted_pos, actual_pos, edge_index):
    # 从edge_index获取端点
    start_points = edge_index[0]
    end_points = edge_index[1]
    # 计算预测和实际的边长
    predicted_lengths = torch.norm(predicted_pos[end_points] - predicted_pos[start_points], dim=1)
    actual_lengths = torch.norm(actual_pos[end_points] - actual_pos[start_points], dim=1)
    # 计算边长损失
    return F.mse_loss(predicted_lengths, actual_lengths)

class MovementGATModel_pos(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels_node, heads=1):
        super(MovementGATModel_pos, self).__init__()
        self.heads = heads

        self.move_l1 = GATConv(in_channels, hidden_dims[0],heads=heads)
        self.move_norm1 = BatchNorm(hidden_dims[0]*heads)
        self.move_l2 = GATConv(hidden_dims[0]*heads, hidden_dims[1],heads=heads)
        self.move_norm2 = BatchNorm(hidden_dims[1]*heads)
        # self.rnn = nn.GRU(hidden_dims[1] * heads, hidden_dims[1], batch_first=True)
        # self.move_l3 = GATConv(hidden_dims[1] * heads, hidden_dims[1], heads=heads)
        # self.move_norm3 = BatchNorm(hidden_dims[2] * heads)
        self.fc_node = nn.Linear(hidden_dims[1]*heads, out_channels_node)
        self.eigenvecs = None

    def forward(self, x, edge_index, pos, mask,batch):
        edge_index = edge_index.long()
        # x = self.graph_fourier_transform(x, edge_index)
        x = self.recompute_features(x, pos, edge_index)
        # 使用带权重的边进行图卷积
        move_features = F.relu(self.move_norm1(self.move_l1(x, edge_index)))
        move_features = F.relu(self.move_norm2(self.move_l2(move_features, edge_index)))
        # move_features = scatter_mean(move_features, batch, dim=0)
        out_node = self.fc_node(move_features)
        mask = mask.unsqueeze(1)
        out_node = out_node * mask
        pos = pos.view(64,40,2)
        out_node = out_node.view(64, 40, 2)
        updated_pos = pos + out_node
        return out_node,updated_pos

    def recompute_features(self, x, pos, edge_index):
        # 重新计算基于位置的特征
        n = pos.size(0)
        distances = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)  # 边长
        angles = torch.atan2(pos[edge_index[1], 1] - pos[edge_index[0], 1],
                             pos[edge_index[1], 0] - pos[edge_index[0], 0])  # 计算角度
        # 将边长和角度特征连接到原始特征
        edge_features = torch.stack([distances, angles], dim=1)
        new_features = torch.zeros(n, x.size(1) + 2, device=x.device)
        new_features.scatter_add_(0, edge_index[0].repeat(2, 1).t(), edge_features)
        x = torch.cat([x, new_features[:, :2]], dim=1)
        return x

    # def graph_fourier_transform(self, x, edge_index):
    #     # 构建图并计算拉普拉斯矩阵
    #     num_nodes = x.size(0)
    #     edge_list = edge_index.t().tolist()
    #     G = nx.Graph()
    #     G.add_edges_from(edge_list)
    #     L = nx.laplacian_matrix(G).toarray()
    #
    #     # 计算特征值和特征向量
    #     eigenvals, eigenvecs = eigh(L)
    #
    #     # 应用傅里叶变换
    #     transformed_features = torch.matmul(torch.tensor(eigenvecs.T, dtype=torch.float32, device=x.device), x)
    #
    #     return transformed_features
    # def compute_tri(self,output_node,target_node,mask,initial_node):
    #
    #     return

    # def compute_loss(self, output_node, target_node,mask):
    #     # 计算节点移动的回归损失
    #     target_node = target_node.view(64, 40, 2)
    #
    #     mask_reshaped = mask.view(64, 40)  # 假设 mask 是一维的
    #     mask_expanded_node = mask_reshaped.unsqueeze(2).expand(-1, -1, 2)
    #
    #     # 应用掩码
    #     output_node_masked = output_node[mask_expanded_node].view(-1, 2)
    #     target_node_masked = target_node[mask_expanded_node].view(-1, 2)
    #
    #     # 计算节点移动的回归损失
    #     loss_node = F.mse_loss(output_node_masked, target_node_masked)
    #     return loss_node
    def compute_loss(self, output_node, target_node, mask,original_pos,edge_index_tri):
        # 将目标节点和掩码调整形状
        original_pos = original_pos.view(64,40,2)
        target_node = target_node.view(64, 40, 2)
        predicted_pos = output_node + original_pos
        actual_pos = target_node + original_pos
        mask_reshaped = mask.view(64, 40)
        mask_expanded_node = mask_reshaped.unsqueeze(2).expand(-1, -1, 2)
        predicted_pos_masked = predicted_pos[mask_expanded_node].view(-1, 2)
        actual_pos_masked = actual_pos[mask_expanded_node].view(-1, 2)
        # 应用掩码
        output_node_masked = output_node[mask_expanded_node].view(-1, 2)
        target_node_masked = target_node[mask_expanded_node].view(-1, 2)
        # 计算节点移动的回归损失
        loss_node = F.mse_loss(output_node_masked, target_node_masked)
        valid_edge_index = filter_edges(edge_index_tri, mask_reshaped)
        # edge_length_loss_value = edge_length_loss(predicted_pos_masked, actual_pos_masked, valid_edge_index)
        # angle_loss_value = angle_loss(predicted_pos_masked, actual_pos_masked, valid_edge_index)
        total_loss = loss_node
        return total_loss

    def compute_mse(self, output_node, target_node, mask):
        target_node = target_node.view(64, 40, 2)
        mask_reshaped = mask.view(64, 40)  # 假设 mask 是一维的
        mask_expanded_node = mask_reshaped.unsqueeze(2).expand(-1, -1, 2)
        output_node_masked = output_node[mask_expanded_node].view(-1, 2)
        target_node_masked = target_node[mask_expanded_node].view(-1, 2)
        loss_node = F.mse_loss(output_node_masked, target_node_masked)
        return loss_node