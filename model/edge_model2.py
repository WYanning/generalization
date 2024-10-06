import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv,GCNConv,SAGEConv,GINConv
import torch
from torch_geometric.nn import BatchNorm
import networkx as nx

class EdgeGATModel(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels_edge,heads=1):
        super(EdgeGATModel, self).__init__()
        self.heads = heads

        self.shared_l1 = GATConv(in_channels, hidden_dims[0],heads=heads)
        self.shared_norm1 = BatchNorm(hidden_dims[0]*heads)
        self.edge_l2 = GATConv(hidden_dims[0] * heads, hidden_dims[1], heads=heads)
        self.edge_norm2 = BatchNorm(hidden_dims[1] * heads)
        # self.edge_l3 = GATConv(hidden_dims[1] * heads, hidden_dims[2], heads=heads)
        # self.edge_norm3 = BatchNorm(hidden_dims[2] * heads)
        self.edge_out = GATConv(hidden_dims[1] * heads,  out_channels_edge, heads=heads)
        self.fc_edge = nn.Linear(hidden_dims[1] * heads, out_channels_edge)

        # Task-specific outputs for out_edge and out_type

    def forward(self, x, edge_index, mask,adjacency):
        # 基础的共享层
        edge_index = edge_index.long()

        shared_features2 = F.relu(self.shared_norm1(self.shared_l1(x, edge_index)))
        edge = F.relu(self.edge_norm2(self.edge_l2(shared_features2, edge_index)))
        # edge = F.relu(self.edge_norm3(self.edge_l3(edge, edge_index)))
        out_edge = self.fc_edge(edge)

        # 应用掩码
        mask = mask.unsqueeze(1)
       # 扩展掩码维度以匹配输出
        out_edge = out_edge * mask
        # out_edge = torch.sigmoid(out_edge)
        out_edge = out_edge.view(64, 40, 40)
        # print(out_edge)

        initial_edges = adjacency.view(64,40,40)
        # print(initial_edges)
        out_edge = out_edge+initial_edges
        # print(out_edge)

        return out_edge

    def compute_loss(self, output_edge, target_edge,mask,initial_edges,alpha=0.25, gamma=2.0, new_edge_weight=5):

        target_edge = target_edge.view(64, 40, 40)

        initial_edges = initial_edges.view(64,40,40)
        mask_edge = mask.view(64, 40)
        # 然后，我们通过 unsqueeze 和广播创建一个适用于每个图的边掩码
        mask_edge_2d = (mask_edge.unsqueeze(2) * mask_edge.unsqueeze(1))

        loss_edge = F.binary_cross_entropy_with_logits(output_edge[mask_edge_2d], target_edge[mask_edge_2d])
        new_edges = (target_edge == 1) & (initial_edges == 0)
        at = alpha * target_edge + (1 - alpha) * (1 - target_edge)
        new_edges_weight = torch.ones_like(target_edge) + (new_edge_weight - 1) * new_edges.float()

        # 计算Focal Loss
        pt = torch.exp(- loss_edge )
        F_loss = at * (1 - pt) ** gamma *  loss_edge  * new_edges_weight

        predictions = torch.sigmoid(output_edge) > 0.5  # 转换 logits 为 binary predictions
        predictions =predictions * mask_edge_2d  # 应用 mask
        row_sums = predictions.sum(dim=1)
        penalty = (row_sums < 2).float() * 1 # 每行少于1个1则计算惩罚
        row_penalty = penalty.sum()

        mask = (initial_edges == 1) & (output_edge < 0)
        penalty = (output_edge[mask] ** 2).sum()

        return F_loss.mean() + 0.1 * penalty+0.001*row_penalty

    def accuracy(self, output_edge, target_edge, mask):
        target_edge = target_edge.view(64, 40, 40)
        mask_edge = mask.view(64, 40)
        # 然后，我们通过 unsqueeze 和广播创建一个适用于每个图的边掩码
        mask_edge_2d = (mask_edge.unsqueeze(2) * mask_edge.unsqueeze(1))
        probs = torch.sigmoid(output_edge)
        predictions = (probs > 0.5).float()  # 将概率转换为二元预测（0或1）
        # 应用mask_2d来过滤出有效的节点对预测
        correct_predictions = (predictions == target_edge).float()  # 对比预测和真实标签
        correct_masked = correct_predictions[mask_edge_2d]
        # 计算准确率
        accuracy_edge = correct_masked.sum() /mask_edge_2d.sum()
        return accuracy_edge

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, MessagePassing
from torch_geometric.utils import add_self_loops

def convexity_loss(vertices, edges):
    convex_loss = 0
    threshold = 0.5  # 这个阈值可以根据模型表现进行调整
    for i in range(vertices.shape[1]):
        prev = (i - 1) % vertices.shape[1]
        next = (i + 1) % vertices.shape[1]
        v1 = vertices[:, i] - vertices[:, prev]
        v2 = vertices[:, next] - vertices[:, i]
        cross_product = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

        include_loss = (torch.sigmoid(edges[:, prev, i]) > threshold) & (torch.sigmoid(edges[:, i, next]) > threshold)
        convex_loss_contrib = F.relu(-cross_product) * include_loss.float()

        convex_loss += convex_loss_contrib.mean()

    return convex_loss

def closure_loss(edges):
    # Ensure the last vertex is connected to the first
    closure_penalty = (edges[:, -1, 0] + edges[:, 0, -1] - 2).abs().mean()
    return closure_penalty

class EdgeGCNModel_pos(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels_edge,heads=1):
        super(EdgeGCNModel_pos, self).__init__()
        self.heads = heads

        self.shared_l1 = GATConv(in_channels, hidden_dims[0], heads=heads)
        self.shared_norm1 = BatchNorm(hidden_dims[0] * heads)
        self.edge_l2 = GATConv(hidden_dims[0] * heads, hidden_dims[1], heads=heads)
        self.edge_norm2 = BatchNorm(hidden_dims[1] * heads)
        self.edge_out = GATConv(hidden_dims[1] * heads, out_channels_edge, heads=heads)
        self.fc_edge = nn.Linear(hidden_dims[1] * heads, out_channels_edge)

    def forward(self, x, edge_index, pos, mask, adjacency):
        # Convert edge_index to long type and move to CUDA
        edge_index = edge_index.long()
        row, col = edge_index
        edge_weight = 1.0 / (torch.norm(pos[row] - pos[col], dim=1) + 1e-6)  # Euclidean distance

        # Shared layers
        shared_features = self.shared_l1(x, edge_index,edge_weight)
        shared_features = self.shared_norm1(shared_features)
        shared_features = F.relu(shared_features)

        # Edge layers
        edge_features = self.edge_l2(shared_features, edge_index,edge_weight)
        edge_features = self.edge_norm2(edge_features)
        edge_features = F.relu(edge_features)

        # Output edge features
        out_edge = self.fc_edge(edge_features)

        # Apply mask
        mask = mask.unsqueeze(1)
        out_edge = out_edge * mask
        out_edge = out_edge.view(64, 40, 40)

        initial_edges = adjacency.view(64, 40, 40)
        out_edge += initial_edges

        return out_edge

    def compute_loss(self, output_edge, target_edge,mask,initial_edges,tri):

        target_edge = target_edge.view(64, 40, 40)
        output_edge = output_edge.view(64,40,40)
        initial_edges = initial_edges.view(64,40,40)
        tri = tri.view(64,40,40)
        mask_edge = mask.view(64, 40)
        # 然后，我们通过 unsqueeze 和广播创建一个适用于每个图的边掩码
        mask_edge_2d = (mask_edge.unsqueeze(2) * mask_edge.unsqueeze(1))
        pos_weight = torch.tensor(3.0)
        loss_edge = F.binary_cross_entropy_with_logits(output_edge[mask_edge_2d], target_edge[mask_edge_2d],
                                                  pos_weight=pos_weight)

        non_existing_penalty_mask = (tri == 0) & (torch.sigmoid(output_edge) > 0.5)
        penalty_non_existing = (output_edge[non_existing_penalty_mask] ** 2).sum()

        predicted_connections = torch.sigmoid(output_edge).sum(dim=2)
        degree_penalty = ((predicted_connections - 2) ** 2).sum()

        # Total loss with an additional penalty term for degree constraint
        total_loss = loss_edge + 0.1*penalty_non_existing


        return total_loss

    def accuracy(self, output_edge, target_edge, mask, initial_edges):
        target_edge = target_edge.view(64, 40, 40)
        mask_edge = mask.view(64, 40)
        initial_edges = initial_edges.view(64, 40, 40)
        output_edge = output_edge.view(64, 40, 40)
        # 然后，我们通过 unsqueeze 和广播创建一个适用于每个图的边掩码
        mask_edge_2d = (mask_edge.unsqueeze(2) * mask_edge.unsqueeze(1))
        probs = torch.sigmoid(output_edge)
        predictions = (probs > 0.5).float()  # 将概率转换为二元预测（0或1）
        # 应用mask_2d来过滤出有效的节点对预测
        correct_predictions = (predictions == target_edge).float()  # 对比预测和真实标签
        correct_masked = correct_predictions[mask_edge_2d]
        # 计算准确率
        accuracy_edge = correct_masked.sum() / mask_edge_2d.sum()
        return accuracy_edge


def to_one_hot(labels, num_classes):
        # labels 的形状应为 [batch_size, height, width]
        # 返回的 one_hot 形状为 [batch_size, height, width, num_classes]
    labels = labels.long()
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).to(torch.float32)

def adjust_predictions(pred_changes, initial_edges):
    # 如果 initial_edges 是 0，不允许预测为 -1
    # 如果 initial_edges 是 1，不允许预测为 1
    adjusted_pred = torch.where((initial_edges == 0) & (pred_changes == -1), torch.zeros_like(pred_changes), pred_changes)
    adjusted_pred = torch.where((initial_edges == 1) & (pred_changes == 1), torch.zeros_like(pred_changes), adjusted_pred)
    return adjusted_pred


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha if alpha is not None else torch.tensor([1.0, 1.0, 1.0])
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (self.alpha.to(inputs.device) * ((1 - pt) ** self.gamma)) * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class EdgeModel_pos(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels_edge,heads=1):
        super(EdgeModel_pos, self).__init__()
        self.heads = heads

        self.shared_l1 = GATConv(in_channels, hidden_dims[0], heads=heads)
        self.shared_norm1 = BatchNorm(hidden_dims[0] * heads)
        self.edge_l2 = GATConv(hidden_dims[0] * heads, hidden_dims[1], heads=heads)
        self.edge_norm2 = BatchNorm(hidden_dims[1] * heads)
        self.edge_l3 = GATConv(hidden_dims[1] * heads, hidden_dims[2], heads=heads)
        self.edge_norm3 = BatchNorm(hidden_dims[2] * heads)
        # self.edge_l4 = GATConv(hidden_dims[2] * heads, hidden_dims[2], heads=heads)
        # self.edge_norm4 = BatchNorm(hidden_dims[3] * heads)
        self.fc_edge = nn.Linear(hidden_dims[2] * heads, 3*out_channels_edge)


    def forward(self, x, edge_index, pos, mask, adjacency):
        # Convert edge_index to long type and move to CUDA
        # edge_index = edge_index.long()
        # row, col = edge_index
        # edge_weight = 1.0 / (torch.norm(pos[row] - pos[col], dim=1) + 1e-6)  # Euclidean distance
        x = self.recompute_features(x, pos, edge_index)
        # Shared layers
        shared_features = self.shared_l1(x, edge_index)
        shared_features = self.shared_norm1(shared_features)
        shared_features = F.relu(shared_features)
        # Edge layers
        edge_features = self.edge_l2(shared_features, edge_index)
        edge_features = self.edge_norm2(edge_features)
        edge_features = F.relu(edge_features)
        edge_features = self.edge_l3(edge_features, edge_index)
        edge_features = self.edge_norm3(edge_features)
        edge_features = F.relu(edge_features)

        out_edge = self.fc_edge(edge_features)

        # Apply mask
        mask = mask.unsqueeze(1)
        out_edge = out_edge * mask
        out_edge = out_edge.view(64, 40, 40, 3)
        # out_edge = F.softmax(out_edge, dim=-1)
        initial_edges = adjacency.view(64, 40, 40)
        out_edge_new = out_edge.clone()  # 克隆out_edge避免原地操作
        out_edge_new[:, :, :, 0] = torch.where(initial_edges == 0, torch.zeros_like(out_edge[:, :, :, 0]),
                                               out_edge[:, :, :, 0])
        out_edge_new[:, :, :, 2] = torch.where(initial_edges == 1, torch.zeros_like(out_edge[:, :, :, 2]),
                                               out_edge[:, :, :, 2])
        # out_edge_new = out_edge_new / out_edge_new.sum(dim=-1, keepdim=True)

        return out_edge_new

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

    def compute_loss(self, output_edge, target_edge,mask,initial_edges,tri):
        target_edge = target_edge.view(64, 40, 40)
        initial_edges = initial_edges.view(64,40,40)
        tri = tri.view(64,40,40)
        mask_edge = mask.view(64, 40)
        edge_delta = target_edge - initial_edges
        edge_delta += 1
        edge_delta = edge_delta.long()
        target_edge_one_hot = to_one_hot(edge_delta, 3)
        mask_edge_2d = (mask_edge.unsqueeze(2) * mask_edge.unsqueeze(1))
        focal_loss_fn = FocalLoss(alpha=torch.tensor([1.0, 1.0, 1.0]), gamma=2.0, reduction='mean')
        loss_edge = focal_loss_fn(output_edge, target_edge_one_hot)
        loss_edge = (loss_edge * mask_edge_2d.view(-1)).mean()
        # 计算总损失
        total_loss = loss_edge
        return total_loss

    # def accuracy(self, output_edge, target_edge, mask, initial_edges):
    #     target_edge = target_edge.view(64, 40, 40)
    #     mask_edge = mask.view(64, 40)
    #     initial_edges = initial_edges.view(64, 40, 40)
    #     mask_edge_2d = (mask_edge.unsqueeze(2) * mask_edge.unsqueeze(1))
    #     edge_delta = target_edge - initial_edges
    #     edge_delta += 1  # 类别标签转换为 0, 1, 2
    #     print(torch.max(edge_delta))
    #     # 计算准确度
    #     _, predictions = output_edge.max(dim=-1)# 获取预测类别
    #     predictions
    #     correct_predictions = (predictions == edge_delta).float()  # 对比预测和真实标签
    #     correct_masked = correct_predictions[mask_edge_2d]
    #     accuracy_edge = correct_masked.sum() / mask_edge_2d.sum()
    #     return accuracy_edge

    def accuracy(self, output_edge, target_edge, mask, initial_edges):
        target_edge = target_edge.view(64, 40, 40)
        mask_edge = mask.view(64, 40)
        initial_edges = initial_edges.view(64, 40, 40)
        output_edge = output_edge.view(64, 40, 40, 3)  # 确保是四维，最后一个维度是类别概率

        # 从output_edge选择最可能的状态改变
        _, predicted_changes = output_edge.max(dim=-1)  # 形状 [64, 40, 40]
        predicted_changes = predicted_changes - 1  # 转换为 -1, 0, 1
        predicted_changes
        # 使用 torch.where 调整那些 initial_edges 为零的 predicted_changes
        # 假设我们不允许这些边减少，只允许增加或不变
        # 如果预测的改变是 -1，我们将其设置为 0 (不改变)
        # 应用预测改变到初始邻接矩阵
        final_edges = initial_edges + predicted_changes
        # print(predicted_changes[0,:,:])
        final_edges_1 = torch.clamp(final_edges, 0, 1)  # 确保邻接矩阵值在0和1之间
        final_edges
        final_edges_1
        # 创建2D掩码
        mask_edge_2d = (mask_edge.unsqueeze(2) * mask_edge.unsqueeze(1))

        # 应用mask_2d来过滤出有效的节点对预测
        correct_predictions = (final_edges == target_edge).float()  # 对比预测和真实标签
        correct_masked = correct_predictions[mask_edge_2d]

        # 计算准确率
        accuracy_edge = correct_masked.sum() / mask_edge_2d.sum()
        return accuracy_edge


def compute_degree_penalty(final_edges, mask_edge_2d, min_degree=2):
    # 度数计算应考虑 mask_edge_2d，避免对虚拟节点计算度数
    masked_final_edges = final_edges * mask_edge_2d
    degree = masked_final_edges.sum(dim=2)  # 计算每个节点的度，即每行的和
    penalty = ((degree < min_degree) & (mask_edge_2d.sum(dim=2) > 0)).float() # 对未达到最小度数的有效节点施加惩罚
    return penalty.sum(dim=1)  # 对每个图的惩罚求和

class EdgeModel_pos_2(nn.Module):
    def __init__(self, in_channels, hidden_dims, out_channels_edge,heads=4):
        super(EdgeModel_pos_2, self).__init__()
        self.heads = heads

        self.shared_l1 = GATConv(in_channels, hidden_dims[0], heads=heads)
        self.shared_norm1 = BatchNorm(hidden_dims[0] * heads)
        self.edge_l2 = GATConv(hidden_dims[0] * heads, hidden_dims[1], heads=heads)
        self.edge_norm2 = BatchNorm(hidden_dims[1] * heads)
        self.edge_l3 = GATConv(hidden_dims[1] * heads, hidden_dims[2], heads=heads)
        self.edge_norm3 = BatchNorm(hidden_dims[2] * heads)
        # self.edge_l4 = GATConv(hidden_dims[2] * heads, hidden_dims[2], heads=heads)
        # self.edge_norm4 = BatchNorm(hidden_dims[3] * heads)
        self.fc_edge = nn.Linear(hidden_dims[2] * heads, 3*out_channels_edge)


    def forward(self, x, edge_index, pos, mask, adjacency,tri):
        # Convert edge_index to long type and move to CUDA
        edge_index = edge_index.long()
        row, col = edge_index
        edge_weight = 1.0 / (torch.norm(pos[row] - pos[col], dim=1) + 1e-6)  # Euclidean distance
        # x = self.recompute_features(x, pos, edge_index)
        # Shared layers
        shared_features = self.shared_l1(x, edge_index, edge_weight)
        shared_features = self.shared_norm1(shared_features)
        shared_features = F.relu(shared_features)

        edge_features = self.edge_l2(shared_features, edge_index, edge_weight)
        edge_features = self.edge_norm2(edge_features)
        edge_features = F.relu(edge_features)

        edge_features = self.edge_l3(edge_features, edge_index, edge_weight)
        edge_features = self.edge_norm3(edge_features)
        edge_features = F.relu(edge_features)
        #
        # edge_features = self.edge_l4(edge_features, edge_index, edge_weight)
        # edge_features = self.edge_norm4(edge_features)
        # edge_features = F.relu(edge_features)

        out_edge = self.fc_edge(edge_features)

        mask = mask.unsqueeze(1)
        out_edge = out_edge * mask
        out_edge = out_edge.view(64, 40, 40, 3)
        tri = tri.view(64, 40, 40).unsqueeze(-1)  # 确保tri是四维的，最后一维为1，与out_edge的类别维匹配
        out_edge = out_edge * tri
        out_edge = F.softmax(out_edge, dim=-1)
        initial_edges = adjacency.view(64, 40, 40)
        out_edge_new = out_edge.clone()  # 克隆out_edge避免原地操作
        out_edge_new[:, :, :, 0] = torch.where(initial_edges == 0, torch.zeros_like(out_edge[:, :, :, 0]),
                                               out_edge[:, :, :, 0])
        out_edge_new[:, :, :, 2] = torch.where(initial_edges == 1, torch.zeros_like(out_edge[:, :, :, 2]),
                                               out_edge[:, :, :, 2])
        out_edge_new = out_edge_new / out_edge_new.sum(dim=-1, keepdim=True)

        return out_edge_new


    def compute_loss(self, output_edge, target_edge, mask, initial_edges, tri):
        tri = tri.view(64, 40, 40)
        target_edge = target_edge.view(64, 40, 40)
        initial_edges = initial_edges.view(64, 40, 40)
        mask_edge = mask.view(64, 40)
        edge_delta = target_edge - initial_edges
        edge_delta += 1
        edge_delta = edge_delta.long()
        target_edge_one_hot = to_one_hot(edge_delta, 3)
        output_edge = output_edge * tri.unsqueeze(3)
        _, predicted_changes = output_edge.max(dim=-1)  # 形状 [64, 40, 40]
        predicted_changes = predicted_changes - 1  # 转换为 -1, 0, 1
        final_edges = initial_edges + predicted_changes
        # class_vectors = [
        #     torch.tensor([1, 0, 0], dtype=torch.float32, device='cuda'),
        #     torch.tensor([0, 1, 0], dtype=torch.float32, device='cuda'),
        #     torch.tensor([0, 0, 1], dtype=torch.float32, device='cuda')
        # ]
        #
        # # 打印每个类别的计数
        # for i, class_vector in enumerate(class_vectors):
        #     class_count = (target_edge_one_hot == class_vector).all(dim=-1).sum().item()
        #     print(f"Count of {class_vector.tolist()} in target_edge_one_hot: {class_count}")
        target_edge_one_hot = target_edge_one_hot * tri.unsqueeze(3)
        mask_edge_2d = (mask_edge.unsqueeze(2) * mask_edge.unsqueeze(1)) * tri
        # degree_penalty = compute_degree_penalty(final_edges, mask_edge_2d, min_degree=2)
        # 使用 Cross Entropy Loss
        # weights = torch.tensor([1.0, 1.0, 2.0], device=output_edge.device)  # 设定权重，并确保权重与output_edge在同一设备
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss_edge = loss_fn(output_edge.view(-1, 3), target_edge_one_hot.view(-1, 3))
        # 仅在mask_edge_2d为1的地方计算损失
        loss_edge = (loss_edge * mask_edge_2d.view(-1)).mean()
        return loss_edge

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
