import pandas as pd
import numpy as np
from deap import creator, base, tools, algorithms
from scipy.spatial import ConvexHull
import networkx as nx
# 加载邻接矩阵，去除第一列
adj_matrix_path = "D:/dresden/mapgeneralization/output0914/adj_cut_sage/index172908.csv"
adj_matrix = pd.read_csv(adj_matrix_path, header=None)
adj_matrix = adj_matrix.iloc[1:, 1:]  # 去除第一列
print(adj_matrix)
# 加载节点位置数据
positions_path = "D:/dresden/mapgeneralization/output0914/movement_sage/index172908.csv"
positions = pd.read_csv(positions_path)
positions = positions.iloc[:, [1, 2]]  # 去掉第一行，选择第二列和第三列

# 加载附加的节点移动数据
movement_path = "D:/dresden/mapgeneralization/dataset/output1/nodes_movement_df/nodes_movement_df_172908.csv"
movements = pd.read_csv(movement_path)
movements = movements.iloc[:, [2, 3]]  # 去掉第一行，选择第三列和第四列

# 数据合并
positions = positions.reset_index(drop=True)
print(positions)
movements = movements.reset_index(drop=True)
print(movements)
new_positions = pd.DataFrame({
    'x': positions.iloc[:, 0] + movements.iloc[:, 0],
    'y': positions.iloc[:, 1] + movements.iloc[:, 1]
})

print(new_positions)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", np.random.permutation, len(new_positions))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalTSP(individual):
    distance = 0
    for i in range(1, len(individual)):
        x1, y1 = new_positions.iloc[individual[i-1]]
        x2, y2 = new_positions.iloc[individual[i]]
        distance += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    # Add distance from last to first to make it a closed loop
    x1, y1 = new_positions.iloc[individual[-1]]
    x2, y2 = new_positions.iloc[individual[0]]
    distance += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return (distance,)

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalTSP)

def main():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    result, log = algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 30, stats=stats, halloffame=hof, verbose=True)
    return hof[0], log

# 运行算法
best_path, log = main()
print("最优路径:", best_path)

# 可视化
import matplotlib.pyplot as plt

def plot_path(path):
    fig, ax = plt.subplots()
    looped_path = np.append(path, path[0])  # Adding the first point at the end to close the loop
    ax.plot(new_positions.iloc[looped_path, 0], new_positions.iloc[looped_path, 1], 'bo-')
    for i, txt in enumerate(looped_path):
        ax.annotate(txt, (new_positions.iloc[looped_path[i], 0], new_positions.iloc[looped_path[i], 1]))
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()

plot_path(best_path)

# G = nx.Graph()
# num_nodes = len(new_positions)
# for i in range(num_nodes):
#     G.add_node(i)
#
# for i in range(num_nodes):
#     for j in range(num_nodes):
#         if i != j and adj_matrix.iloc[i, j] == 1:
#             G.add_edge(i, j)
#
# # 寻找哈密顿环的尝试函数
# def find_hamiltonian_cycle(G):
#     path = []
#
#     def backtrack(current_node):
#         if len(path) == len(G) and path[0] in G[current_node]:
#             return path + [path[0]]  # 返回闭环
#         for neighbor in G[current_node]:
#             if neighbor not in path:
#                 path.append(neighbor)
#                 if backtrack(neighbor):
#                     return path
#                 path.pop()
#         return None
#
#     for start_node in G.nodes():
#         path = [start_node]
#         cycle = backtrack(start_node)
#         if cycle:
#             return cycle
#     return None
#
# # 寻找哈密顿环
# cycle = find_hamiltonian_cycle(G)
# print("哈密顿环:", cycle)
#
#
# def plot_hamiltonian_cycle(G, positions, cycle):
#     pos = {i: (positions.iloc[i]['x'], positions.iloc[i]['y']) for i in G.nodes()}
#     fig, ax = plt.subplots(figsize=(10, 8))
#
#     # 只绘制哈密尔顿环中的节点和边
#     if cycle:
#         # 确保首尾相连
#         cycle.append(cycle[0])  # 添加第一个节点到列表的末尾以关闭环
#
#         # 创建一个环的边列表
#         cycle_edges = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]
#
#         # 绘制哈密尔顿环的节点和边
#         nx.draw_networkx_nodes(G, pos, nodelist=cycle[:-1], node_color='red', node_size=100, ax=ax)
#         nx.draw_networkx_edges(G, pos, edgelist=cycle_edges, edge_color='red', width=2, ax=ax)
#
#     # 显示首尾相连的路径文字标注
#     for i, node in enumerate(cycle):
#         ax.annotate(str(node), (pos[node][0], pos[node][1]), textcoords="offset points", xytext=(0, 10), ha='center')
#
#     ax.set_aspect('equal')  # 设置坐标轴比例为相同
#     ax.axis('off')  # 关闭坐标轴
#     plt.title('Hamiltonian Cycle Only')
#     plt.show()
#
#
# # 假设cycle是之前找到的哈密顿环
# # 调用绘图函数
# plot_hamiltonian_cycle(G, new_positions, cycle)
# def calculate_turn_angles(G, pos):
#     node_angles = []
#     nodes = list(G.nodes())
#     for i, node in enumerate(nodes):
#         if len(G[node]) < 2:  # 检查节点的邻居数量是否足够
#             continue
#
#         neighbors = list(G[node])
#         if len(neighbors) >= 2:
#             pos_prev = np.array(pos[neighbors[0]])  # 第一个邻居
#             pos_next = np.array(pos[neighbors[1]])  # 第二个邻居
#             pos_node = np.array(pos[node])
#
#             # 计算向量
#             vector_to_prev = pos_prev - pos_node
#             vector_to_next = pos_next - pos_node
#
#             # 计算角度
#             angle = np.degrees(np.arccos(
#                 np.clip(np.dot(vector_to_prev, vector_to_next) / (
#                             np.linalg.norm(vector_to_prev) * np.linalg.norm(vector_to_next)),
#                         -1.0, 1.0)))
#             node_angles.append({'node': node, 'angle': angle})
#     return node_angles
#
# # 定义迭代调整函数
# def iterative_adjust_positions(G, pos, min_angle=50, max_angle=140, max_iterations=10):
#     for iteration in range(max_iterations):
#         node_angles = calculate_turn_angles(G, pos)
#         adjustments_needed = False
#
#         for angle_info in node_angles:
#             if angle_info['angle'] < min_angle or angle_info['angle'] > max_angle:
#                 adjustments_needed = True
#                 node = angle_info['node']
#                 neighbors = list(G[node])
#
#                 if len(neighbors) < 2:
#                     continue  # 如果少于两个邻居，则跳过
#
#                 pos_prev = np.array(pos[neighbors[0]])  # 第一个邻居
#                 pos_next = np.array(pos[neighbors[1]])  # 第二个邻居
#
#                 # 将节点移动到两个邻居的中点
#                 new_pos = (pos_prev + pos_next) / 2
#                 pos[node] = tuple(new_pos)
#
#         if not adjustments_needed:  # 如果这一轮没有调整任何节点，则结束循环
#             print(f"All angles adjusted within {iteration + 1} iterations.")
#             break
#     else:
#         print("Reached maximum iterations without satisfying all angle conditions.")
#
#     return pos
#
# # 对找到的哈密尔顿环进行角度调整
# def adjust_hamiltonian_cycle(G, positions, cycle, min_angle=50, max_angle=140, max_iterations=10):
#     # 创建哈密尔顿环的子图
#     cycle_graph = nx.Graph()
#     cycle_edges = [(cycle[i], cycle[i+1]) for i in range(len(cycle) - 1)]
#     cycle_edges.append((cycle[-1], cycle[0]))  # 闭合环
#     cycle_graph.add_edges_from(cycle_edges)
#
#     # 复制环的节点位置
#     cycle_positions = {node: positions[node] for node in cycle}
#
#     # 调整哈密尔顿环的节点位置
#     adjusted_positions = iterative_adjust_positions(cycle_graph, cycle_positions, min_angle, max_angle, max_iterations)
#     return adjusted_positions, cycle_edges
#
# # 假设cycle是找到的哈密尔顿环
# # 将节点位置转为字典形式，以适应迭代调整函数
# positions_dict = {i: (new_positions.iloc[i]['x'], new_positions.iloc[i]['y']) for i in range(len(new_positions))}
# adjusted_cycle_positions, cycle_edges = adjust_hamiltonian_cycle(G, positions_dict, cycle)
#
# # 可视化调整后的哈密尔顿环
# plt.figure(figsize=(8, 8))
# nx.draw_networkx_edges(G, pos=adjusted_cycle_positions, edgelist=cycle_edges, edge_color='red', width=2)
# nx.draw_networkx_nodes(G, pos=adjusted_cycle_positions, nodelist=cycle, node_color='red', node_size=100)
# # 标注节点的序号
# nx.draw_networkx_labels(G, pos=adjusted_cycle_positions, labels={i: i for i in cycle}, font_size=12)
# plt.title("Adjusted Hamiltonian Cycle Visualization")
# plt.axis('equal')
# plt.axis('off')  # 关闭坐标轴
# plt.show()

points = new_positions.values  # 转换为二维坐标数组
hull = ConvexHull(points)  # 计算凸包

# 可视化凸包和节点
plt.figure(figsize=(8, 8))
plt.scatter(points[:, 0], points[:, 1], color='blue', label='Nodes')  # 绘制节点

# 绘制凸包
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'r-')  # 连接凸包边界上的点

# 绘制凸包顶点的序号
for idx in hull.vertices:
    plt.text(points[idx, 0], points[idx, 1], f'{idx}', fontsize=12, ha='center', color='red')

plt.title('Convex Hull of Nodes')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plt.show()