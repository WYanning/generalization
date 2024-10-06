import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv,GCNConv,SAGEConv,GINConv,TopKPooling
import torch
from torch_geometric.nn import BatchNorm
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv,GCNConv,SAGEConv,GINConv
import torch
from torch_geometric.nn import BatchNorm
import networkx as nx
from edge_model2 import *
from movementmodel2 import *

class GraphSageModel(nn.Module):
    def __init__(self, in_channels_node, hidden_dims_node, out_channels_node, in_channels_edge,hidden_dims_edge, out_channels_edge):
        super(GraphSageModel, self).__init__()
        # 节点移动模型部分
        self.movement_model = MovementSageModel_pos(in_channels_node, hidden_dims_node, out_channels_node)
        # 边预测模型部分
        self.edge_model = EdgeModel_pos_SAGE(in_channels_edge, hidden_dims_edge, out_channels_edge)

    def forward(self, x, edge_index, pos, mask, batch, adjacency, tri):
        # 处理节点
        out_node, updated_pos = self.movement_model(x, edge_index, pos, mask, batch)

        # 使用更新后的位置和节点特征处理边
        out_edge = self.edge_model(x, edge_index, updated_pos, mask, adjacency, tri)

        return out_node, updated_pos, out_edge

    def compute_loss(self, output_node, target_node, mask_node, original_pos, edge_index_tri,
                     output_edge, target_edge, mask_edge, initial_edges, tri):
        # 计算节点损失
        loss_node = self.movement_model.compute_loss(output_node, target_node, mask_node, original_pos, edge_index_tri)

        # 计算边损失
        loss_edge = self.edge_model.compute_loss(output_edge, target_edge, mask_edge, initial_edges, tri)

        # 总损失为节点损失和边损失的加权和
        total_loss = loss_node + 100*loss_edge
        return total_loss

    def accuracy(self, output_edge, target_edge, mask, initial_edges):
        target_edge = target_edge.view(64, 40, 40)
        mask_edge = mask.view(64, 40)
        initial_edges = initial_edges.view(64, 40, 40)
        output_edge = output_edge.view(64, 40, 40, 3)  # 确保是四维，最后一个维度是类别概率

        # 从output_edge选择最可能的状态改变
        _, predicted_changes = output_edge.max(dim=-1)  # 形状 [64, 40, 40]
        predicted_changes = predicted_changes - 1  # 转换为 -1, 0, 1
        final_edges = initial_edges + predicted_changes
        # print(predicted_changes[0,:,:])
        # final_edges_1 = torch.clamp(final_edges, 0, 1)  # 确保邻接矩阵值在0和1之间
        # 创建2D掩码
        mask_edge_2d = (mask_edge.unsqueeze(2) * mask_edge.unsqueeze(1))

        # 应用mask_2d来过滤出有效的节点对预测
        correct_predictions = (final_edges == target_edge).float()  # 对比预测和真实标签
        correct_masked = correct_predictions[mask_edge_2d]

        # 计算准确率
        accuracy_edge = correct_masked.sum() / mask_edge_2d.sum()
        return accuracy_edge

    def mse(self,output_node, target_node, mask):
        target_node = target_node.view(64, 40, 2)
        mask_reshaped = mask.view(64, 40)  # 假设 mask 是一维的
        mask_expanded_node = mask_reshaped.unsqueeze(2).expand(-1, -1, 2)
        output_node_masked = output_node[mask_expanded_node].view(-1, 2)
        target_node_masked = target_node[mask_expanded_node].view(-1, 2)
        loss_node = F.mse_loss(output_node_masked, target_node_masked)
        return loss_node