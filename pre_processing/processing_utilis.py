import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
import geopandas as gpd
from shapely.geometry import box
from scipy.spatial import Delaunay
from DataProcess.data_utils import *


def load_data(shapefile_geb10="D:/dresden/mt/MapGeneralizer-main/"
                              "MapGeneralizer-main/Vectors_MapGeneralization_DL/"
                              "stuttgart_change/geb10.shp",
              shapefile_geb25="D:/dresden/mt/MapGeneralizer-main/"
                              "MapGeneralizer-main/Vectors_MapGeneralization_DL/"
                              "stuttgart_change/geb25.shp",
              target_crs='EPSG:25832'):
    gdf_geb10 = gpd.read_file(shapefile_geb10)
    gdf_geb10 = gdf_geb10.to_crs(target_crs)
    gdf_geb25 = gpd.read_file(shapefile_geb25)
    gdf_geb25 = gdf_geb25.to_crs(target_crs)
    return gdf_geb10, gdf_geb25

def bulit_connection(gdf_geb10, gdf_geb25):
    ##寻找一对多的关系
    gdf_geb10['JOINID_geb10'] = gdf_geb10['JOINID']
    gdf_geb25['JOINID_geb25'] = gdf_geb25['JOINID']
    # 执行空间连接
    joined_df = gpd.sjoin(gdf_geb10, gdf_geb25, how="inner", predicate='intersects')
    # 使用新添加的列来分析和筛选一对多的关系
    one_to_many = joined_df.groupby('JOINID_geb25').filter(lambda x: len(x) > 1)
    # 创建一个DataFrame来存储一对多关系的详细信息
    one_to_many_details = one_to_many.groupby('JOINID_geb25')['JOINID_geb10'].apply(list).reset_index()
    one_to_many_details.columns = ['JOINID_geb25', 'JOINID_geb10_list']
    return one_to_many_details

def creat_bounding_box(one_to_many_details, gdf_geb25):
    ##为建筑物构建边框
    one_to_many_geb25_ids = one_to_many_details['JOINID_geb25'].unique()
    # 筛选出具有一对多关系的geb25建筑物
    geb25_one_to_many = gdf_geb25[gdf_geb25['JOINID'].isin(one_to_many_geb25_ids)]
    # 假设 geb25_one_to_many 已经定义并包含正确的数据
    # 计算统一的边界框边长
    max_width = max_height = 0
    for _, row in geb25_one_to_many.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds
        max_width = max(max_width, maxx - minx)
        max_height = max(max_height, maxy - miny)
    uniform_side_length = max(max_width, max_height)
    # 为每个建筑物创建以其中心为基础的正方形边框
    # 确保新创建的 GeoSeries 具有与 gdf_geb25 相同的 CRS
    bounding_boxes = gpd.GeoDataFrame(columns=['JOINID', 'geometry'], crs=geb25_one_to_many.crs)
    for _, row in geb25_one_to_many.iterrows():
        # 获取多边形的几何中心
        center = row.geometry.centroid
        center_x, center_y = center.x, center.y
        # 计算边框的四个角
        half_side = uniform_side_length / 2
        minx = center_x - half_side
        maxx = center_x + half_side
        miny = center_y - half_side
        maxy = center_y + half_side
        # 创建边框并添加到GeoDataFrame
        bounding_box = box(minx, miny, maxx, maxy)
        bounding_boxes = bounding_boxes._append({'JOINID': row['JOINID'], 'geometry': bounding_box}, ignore_index=True)
    return bounding_boxes


def data_selection(one_to_many_details, bounding_boxes, gdf_geb10, gdf_geb25, index):
    selected_geb25_id = one_to_many_details.iloc[index]['JOINID_geb25']
    selected_bounding_box = bounding_boxes[bounding_boxes['JOINID'] == selected_geb25_id].iloc[0].geometry
    # 获取局部坐标系的原点
    origin_x, origin_y = selected_bounding_box.bounds[0], selected_bounding_box.bounds[1]
    # 更新 gdf_geb25 到局部坐标系
    updated_gdf_geb25 = gdf_geb25.copy()
    updated_gdf_geb25['geometry'] = updated_gdf_geb25['geometry'].apply(
        lambda x: to_local_coordinates(x, origin_x, origin_y))
    # 使用与 gdf_geb25 相同的局部坐标原点来转换 gdf_geb10
    updated_gdf_geb10 = gdf_geb10.copy()
    updated_gdf_geb10['geometry'] = updated_gdf_geb10['geometry'].apply(
        lambda x: to_local_coordinates(x, origin_x, origin_y))
    # 接下来可以使用 updated_gdf_geb25 和 updated_gdf_geb10 进行后续处理
    # 修改接口，让其与之前的代码适配
    selected_one_to_many = one_to_many_details.iloc[index]
    target_joinids_geb10 = selected_one_to_many['JOINID_geb10_list']
    target_joinids_geb25 = [selected_one_to_many['JOINID_geb25']]
    return selected_geb25_id, target_joinids_geb10, target_joinids_geb25


def data_selection_v2(one_to_many_details, bounding_boxes, gdf_geb10, gdf_geb25, index, save_folder):
    selected_one_to_many = one_to_many_details[one_to_many_details['JOINID_geb25'] == index].iloc[0]
    selected_geb25_id = None
    target_joinids_geb10 = selected_one_to_many['JOINID_geb10_list']
    target_joinids_geb25 = [selected_one_to_many['JOINID_geb25']]

    selected_bounding_box = bounding_boxes[bounding_boxes['JOINID'] == index].iloc[0].geometry
    origin_x, origin_y = selected_bounding_box.bounds[0], selected_bounding_box.bounds[1]
    # 更新 gdf_geb25 到局部坐标系
    gdf_geb25 = gdf_geb25.copy()
    gdf_geb25['geometry'] = gdf_geb25['geometry'].apply(lambda x: to_local_coordinates(x, origin_x, origin_y))

    # 使用与 gdf_geb25 相同的局部坐标原点来转换 gdf_geb10
    gdf_geb10 = gdf_geb10.copy()
    gdf_geb10['geometry'] = gdf_geb10['geometry'].apply(lambda x: to_local_coordinates(x, origin_x, origin_y))
    # return selected_geb25_id, target_joinids_geb10, target_joinids_geb25



# def creat_Ajc(gdf_geb10, target_joinids_geb10, gdf_geb25, target_joinids_geb25, save_folder):
    # 过滤geb10和geb25，只保留选定的建筑物
    filtered_gdf_geb10 = gdf_geb10[gdf_geb10['JOINID'].isin(target_joinids_geb10)]
    filtered_gdf_geb25 = gdf_geb25[gdf_geb25['JOINID'].isin(target_joinids_geb25)]

    def count_vertices(gdf):
        total_vertices = sum(len(poly.exterior.coords) for poly in gdf.geometry)
        return total_vertices


    # 仅为顶点数小于等于40的GeoDataFrame创建图
    if count_vertices(filtered_gdf_geb10) <= 35:
        G_geb10 = build_graph_from_polygons(filtered_gdf_geb10)
    else:
        G_geb10=None
    if count_vertices(filtered_gdf_geb25) <= 35:
        G_geb25 = build_graph_from_polygons(filtered_gdf_geb25)
    else:
        G_geb25=None
    # 生成邻接矩阵
    # 为G_geb10中的每个节点分配一个编号
    if G_geb10 is not None:
        node_labels_geb10 = {node: i for i, node in enumerate(G_geb10.nodes())}
    # 创建邻接矩阵
        adj_matrix_geb10 = nx.adjacency_matrix(G_geb10)
        adj_matrix_geb25 = nx.adjacency_matrix(G_geb25)
        # 将邻接矩阵转换为DataFrame以便打印
        df_adj_matrix_geb10 = pd.DataFrame(adj_matrix_geb10.toarray())
        df_adj_matrix_geb25 = pd.DataFrame(adj_matrix_geb25.toarray())
        # TODO df_adj_matrix_geb10 df_adj_matrix_geb25
        df_adj_matrix_geb10.to_csv(os.path.join(save_folder, "df_adj_matrix_geb10.csv"))

        # 将节点ID映射到其索引
        node_id_map_geb10 = {node: i for i, node in enumerate(G_geb10.nodes())}
        node_id_map_geb25 = {node: i for i, node in enumerate(G_geb25.nodes())}
        # 更新DataFrame以使用节点ID作为索引和列名
        df_adj_matrix_geb10.columns = [node_id_map_geb10[n] for n in G_geb10.nodes()]
        df_adj_matrix_geb10.index = [node_id_map_geb10[n] for n in G_geb10.nodes()]
        df_adj_matrix_geb25.columns = [node_id_map_geb25[n] for n in G_geb25.nodes()]
        df_adj_matrix_geb25.index = [node_id_map_geb25[n] for n in G_geb25.nodes()]

        original_points = []
        projected_points = []
        lines = []

        # 遍历所有G_geb10中的节点，计算投影点
        for node_10 in G_geb10.nodes():
            point_10 = Point(node_10)
            proj_point = find_nearest_projection(point_10, filtered_gdf_geb25)
            if proj_point is not None:
                original_points.append(point_10)
                projected_points.append(proj_point)
                lines.append(LineString([point_10, proj_point]))

        # 创建GeoDataFrame
        original_gdf = gpd.GeoDataFrame(geometry=original_points)
        projected_gdf = gpd.GeoDataFrame(geometry=projected_points)

        # 创建一个新的GeoDataFrame来存储移动后的projected点
        moved_projected_gdf = projected_gdf.copy()

        # 遍历每个geb25的节点
        for node in G_geb25.nodes():
            point_25 = Point(node)
            # 找到最近的projected_gdf点及其索引
            closest_projected_point, closest_idx = find_closest_point(point_25, projected_gdf)

            # 移动最近的projected_gdf点到geb25节点的位置
            if closest_projected_point is not None and closest_idx != -1:
                moved_projected_gdf.at[closest_idx, 'geometry'] = point_25

        moved_projected_gdf = prevent_overlap(moved_projected_gdf)
        # 使用 networkx 的最大权重匹配算法找到最佳匹配
        pos_geb10_updated = {}
        G_geb10_updated = nx.Graph()
        for idx, point in moved_projected_gdf.iterrows():
            coords = (point.geometry.x, point.geometry.y)
            G_geb10_updated.add_node(idx, pos=coords)
            pos_geb10_updated[idx] = coords

        for edge in G_geb25.edges():
            node1, node2 = edge
            point1 = Point(node1)
            point2 = Point(node2)

            closest_node1 = None
            closest_node2 = None
            min_dist1 = float('inf')
            min_dist2 = float('inf')

            # 找到G_geb10_updated中与node1和node2最近的点
            for idx, point in moved_projected_gdf.iterrows():
                coords = (point.geometry.x, point.geometry.y)
                dist1 = Point(coords).distance(point1)
                dist2 = Point(coords).distance(point2)

                if dist1 < min_dist1:
                    min_dist1 = dist1
                    closest_node1 = idx

                if dist2 < min_dist2:
                    min_dist2 = dist2
                    closest_node2 = idx

            # 添加边，并避免自连接
            # if closest_node1 is not None and closest_node2 is not None and closest_node1 != closest_node2:
            G_geb10_updated.add_edge(closest_node1, closest_node2)

        def connect_and_cleanup(G, pos):
            all_edges = list(G.edges())
            for u, v in all_edges:
                line = LineString([pos[u], pos[v]])
                nodes_on_line = []
                for idx in G.nodes():
                    if line.distance(Point(pos[idx])) < 1e-8:
                        nodes_on_line.append((idx, line.project(Point(pos[idx]))))

                if len(nodes_on_line) > 2:
                    nodes_on_line.sort(key=lambda x: x[1])  # 按线上位置排序
                    for i in range(len(nodes_on_line) - 1):
                        G.add_edge(nodes_on_line[i][0], nodes_on_line[i + 1][0])

                    # 移除原始边
                    G.remove_edge(u, v)

        connect_and_cleanup(G_geb10_updated, pos_geb10_updated)


        updated_adj_matrix = nx.adjacency_matrix(G_geb10_updated, nodelist=sorted(G_geb10_updated.nodes()))
        updated_adj_df = pd.DataFrame(updated_adj_matrix.toarray(), index=sorted(G_geb10_updated.nodes()),
                                          columns=sorted(G_geb10_updated.nodes()))

            # 保存邻接矩阵到CSV文件
        updated_adj_df.to_csv(os.path.join(save_folder, "updated_adj_df.csv"))
        #TODO updated

        # 打印更新后的邻接矩阵
        # print("Updated Adjacency Matrix for G_geb10_updated with Node IDs:")
        # print(updated_adj_df)

        # 更新节点移动信息，标记与geb25坐标一致的点
        geb25_nodes = [Point(node) for node in G_geb25.nodes()]
        geb25_nodes_gdf = gpd.GeoDataFrame(geometry=geb25_nodes)
        geb25_coords = set((point.x, point.y) for point in geb25_nodes_gdf.geometry)

        # 初始化存储节点移动信息的列表
        nodes_movement_info = []

        # 遍历每个原始节点
        for idx in original_gdf.index:
            original_point = original_gdf.geometry[idx]
            moved_point = moved_projected_gdf.geometry[idx]

            # 计算位移
            delta_x = moved_point.x - original_point.x
            delta_y = moved_point.y - original_point.y

            # 检查新位置是否在geb25的坐标内
            point_type = 1 if (moved_point.x, moved_point.y) in geb25_coords else 0

            # 添加移动信息到列表
            nodes_movement_info.append({
                'Node ID': idx,
                'Original X': original_point.x,
                'Original Y': original_point.y,
                'Delta X': delta_x,
                'Delta Y': delta_y,
                'Type': point_type
            })

        nodes_movement_df = pd.DataFrame(nodes_movement_info)
        # TODO nodes_movement_df
        nodes_movement_df.to_csv(os.path.join(save_folder, "nodes_movement_df.csv"))
        # 打印节点移动信息
        # print(nodes_movement_df)
        # 将节点移动信息保存到CSV文件

        ##可视化结果
        pos_geb10_updated = {}
        for idx, point in moved_projected_gdf.iterrows():
            coords = (point.geometry.x, point.geometry.y)
            G_geb10_updated.add_node(idx, pos=coords)
            pos_geb10_updated[idx] = coords

        pos_geb25 = {node: (node[0], node[1]) for node in G_geb25.nodes()}
        pos_geb10 = {node: (node[0], node[1]) for node in G_geb10.nodes()}
        x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
        for pos in [pos_geb10, pos_geb10_updated, pos_geb25]:
            xs, ys = zip(*pos.values())
            x_min, x_max = min(x_min, min(xs)), max(x_max, max(xs))
            y_min, y_max = min(y_min, min(ys)), max(y_max, max(ys))

        # 添加一些边距以确保内容完整显示
        x_margin = (x_max - x_min) * 0.05  # x轴边距设为范围的5%
        y_margin = (y_max - y_min) * 0.05  # y轴边距设为范围的5%

        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin

        fig, axes = plt.subplots(1, 4, figsize=(20, 8))
        # 设置子图并统一坐标轴范围
        for i, (graph, pos, title) in enumerate([
            (G_geb10, pos_geb10, 'Original GEB10 Graph'),
            (None, None, 'Point Movements'),  # 留空，用于自定义绘制
            (G_geb10_updated, pos_geb10_updated, 'Updated GEB10 Graph'),
            (G_geb25, pos_geb25, 'Original GEB25 Graph')
        ]):
            ax = axes[i]
            if graph:
                nx.draw(graph, pos=pos, ax=ax, node_size=50, node_color='blue' if i != 3 else 'red', edge_color='black',
                        with_labels=False)
            ax.set_title(title)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal', adjustable='box')  # 保持比例一致，无畸变
        # 第一个子图：原始的GEB10图
        nx.draw(G_geb10, pos=pos_geb10, ax=axes[0], node_size=50, node_color='blue', edge_color='black',
                with_labels=False)
        for node, (x, y) in pos_geb10.items():
            axes[0].text(x, y, s=node_labels_geb10[node], horizontalalignment='center', verticalalignment='center',
                         color='red', fontsize=8)
        axes[0].set_title('Original GEB10 Graph')

        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].spines['bottom'].set_visible(False)
        axes[1].spines['left'].set_visible(False)
        # 绘制geb25的边界（黑色，较粗线）
        for target_joinid in target_joinids_geb25:
            selected_buildings = filtered_gdf_geb25[filtered_gdf_geb25['JOINID'] == target_joinid]
            for geom in selected_buildings.geometry:
                if geom.geom_type in ['Polygon', 'MultiPolygon']:
                    x, y = geom.exterior.xy
                    axes[1].plot(x, y, color='black', linewidth=2, zorder=1)
        # 绘制geb10的边界（绿色）
        for target_joinid in target_joinids_geb10:
            selected_buildings = filtered_gdf_geb10[filtered_gdf_geb10['JOINID'] == target_joinid]
            for geom in selected_buildings.geometry:
                if geom.geom_type in ['Polygon', 'MultiPolygon']:
                    x, y = geom.exterior.xy
                    axes[1].plot(x, y, color='green', linewidth=1, zorder=2)
        original_x = original_gdf.geometry.x
        original_y = original_gdf.geometry.y
        moved_x = moved_projected_gdf.geometry.x
        moved_y = moved_projected_gdf.geometry.y
        # 计算位移向量
        u = moved_x - original_x
        v = moved_y - original_y
        axes[1].quiver(original_x, original_y, u, v, angles='xy', scale_units='xy', scale=1, color='red', width=0.003)
        # 在原始位置添加蓝色圆点
        axes[1].scatter(original_x, original_y, color='blue', zorder=3)
        axes[1].set_title('Point Movements')

        nx.draw(G_geb10_updated, pos=pos_geb10_updated, ax=axes[2], node_size=50, node_color='blue', edge_color='black',
                with_labels=False)
        # nx.draw_networkx_labels(G_geb10_updated, pos=pos_geb10_updated, font_size=8, font_color='black')
        axes[2].set_title('Updated GEB10 Graph')
        # axes[2].set_aspect('equal', adjustable='datalim')

        # 第四个子图：原始的GEB25图
        nx.draw(G_geb25, pos=pos_geb25, ax=axes[3], node_size=50, node_color='red', edge_color='black',
                with_labels=False)
        for node, (x, y) in pos_geb25.items():
            axes[3].text(x, y, s=node, horizontalalignment='center', verticalalignment='center', color='white',
                         fontsize=0)
        axes[3].set_title('Original GEB25 Graph')
        # axes[3].set_aspect('equal', adjustable='datalim')
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(save_folder, "Graph.jpg"))
        plt.close()
        # TODO



#
#
# def generate_graph(gdf_geb10, target_joinids_geb10, gdf_geb25, target_joinids_geb25):
#     # 过滤geb10和geb25，只保留选定的建筑物
#     filtered_gdf_geb10 = gdf_geb10[gdf_geb10['JOINID'].isin(target_joinids_geb10)]
#     filtered_gdf_geb25 = gdf_geb25[gdf_geb25['JOINID'].isin(target_joinids_geb25)]
#     # 为geb10和geb25创建图
#     G_geb10 = build_graph_from_polygons(filtered_gdf_geb10)
#     G_geb25 = build_graph_from_polygons(filtered_gdf_geb25)
#     original_geb25_nodes = list(G_geb25.nodes())
#     # 为G_geb10中的每个节点分配一个编号
#     node_labels_geb10 = {node: i for i, node in enumerate(G_geb10.nodes())}
#     # 首先获取所有节点的ID
#     node_ids = list(node_labels_geb10.values())
#     # 获取邻接矩阵
#     adj_matrix = nx.adjacency_matrix(G_geb10)
#     # 转换为稠密格式以便打印
#     # 创建邻接矩阵
#     adj_matrix_geb10 = nx.adjacency_matrix(G_geb10)
#     adj_matrix_geb25 = nx.adjacency_matrix(G_geb25)
#     # 将邻接矩阵转换为DataFrame以便打印
#     df_adj_matrix_geb10 = pd.DataFrame(adj_matrix_geb10.toarray())
#     df_adj_matrix_geb25 = pd.DataFrame(adj_matrix_geb25.toarray())
#     # 将节点ID映射到其索引
#     node_id_map_geb10 = {node: i for i, node in enumerate(G_geb10.nodes())}
#     node_id_map_geb25 = {node: i for i, node in enumerate(G_geb25.nodes())}
#     # 更新DataFrame以使用节点ID作为索引和列名
#     df_adj_matrix_geb10.columns = [node_id_map_geb10[n] for n in G_geb10.nodes()]
#     df_adj_matrix_geb10.index = [node_id_map_geb10[n] for n in G_geb10.nodes()]
#     df_adj_matrix_geb25.columns = [node_id_map_geb25[n] for n in G_geb25.nodes()]
#     df_adj_matrix_geb25.index = [node_id_map_geb25[n] for n in G_geb25.nodes()]
#     return G_geb10, filtered_gdf_geb10, df_adj_matrix_geb10,\
#            G_geb25, filtered_gdf_geb25, df_adj_matrix_geb25,\
#            node_labels_geb10
#
#
# def match_graph(G_geb10, G_geb25, filtered_gdf_geb25):
#     projection_points = {}
#     for node_10 in G_geb10.nodes():
#         point_10 = Point(node_10)
#         proj_point = find_nearest_projection(point_10, filtered_gdf_geb25)
#         projection_points[node_10] = (proj_point.x, proj_point.y)
#     # 创建一个空的加权图
#     G = nx.Graph()
#     # 为 G_geb10 和 G_geb25 中的每个节点对添加边，边的权重是它们之间距离的负值
#     for node_10 in G_geb10.nodes():
#         for node_25 in G_geb25.nodes():
#             weight = -Point(node_10).distance(Point(node_25))
#             G.add_edge(f"10_{node_10}", f"25_{node_25}", weight=weight)
#     # 使用 networkx 的最大权重匹配算法找到最佳匹配
#     matches = nx.max_weight_matching(G, maxcardinality=True)
#     # 初始化 adjusted_points 列表
#     adjusted_points = []
#     # 初始化一个集合来存储已经使用的坐标
#     used_coords = set()
#     for node_10 in G_geb10.nodes():
#         matched = False
#         for match in matches:
#             node_10_str, node_25_str = match
#             if node_10_str == f"10_{node_10}" or node_25_str == f"10_{node_10}":
#                 node_25 = tuple(map(float, node_25_str[3:].strip('()').split(','))) if node_25_str.startswith(
#                     "25_") else tuple(map(float, node_10_str[3:].strip('()').split(',')))
#                 final_coord = adjust_coords(node_25, used_coords=used_coords)
#                 adjusted_points.append((node_10, final_coord))
#                 matched = True
#                 break
#         if not matched:
#             # 如果没有找到匹配，使用调整后的投影点
#             final_coord = adjust_coords(projection_points[node_10], used_coords=used_coords)
#             adjusted_points.append((node_10, final_coord))
#     return projection_points, adjusted_points
#
#
# def update_Ggeb10(adjusted_points, node_labels_geb10, G_geb25):
#     G_geb10_updated = nx.Graph()
#     # 将G_geb10中的节点按照新的位置添加到G_geb10_updated，但使用原始的节点ID
#     for orig, new in adjusted_points:
#         G_geb10_updated.add_node(node_labels_geb10[orig])
#     # 为G_geb10_updated添加边，这些边将基于G_geb25的连接关系
#     for node1, node2 in G_geb25.edges():
#         # 查找与G_geb25中节点相对应的G_geb10节点ID
#         node1_id = next((node_labels_geb10[orig_node] for orig_node, new_node in adjusted_points if new_node == node1),
#                         None)
#         node2_id = next((node_labels_geb10[orig_node] for orig_node, new_node in adjusted_points if new_node == node2),
#                         None)
#         # 如果在G_geb10中找到了对应的节点，则在G_geb10_updated中添加一条边
#         if node1_id is not None and node2_id is not None:
#             G_geb10_updated.add_edge(node1_id, node2_id)
#     return G_geb10_updated
#
#
# def check_G_geb10(adjusted_points, G_geb10_updated, node_labels_geb10, projection_points):
#     for node1, node2 in list(G_geb10_updated.edges()):
#         if node1 == node2:
#             # 跳过自身连接的情况
#             continue
#         # 提取原始坐标
#         orig_coords_node1 = next((new for orig, new in adjusted_points if node_labels_geb10[orig] == node1), None)
#         orig_coords_node2 = next((new for orig, new in adjusted_points if node_labels_geb10[orig] == node2), None)
#
#         if orig_coords_node1 is None or orig_coords_node2 is None:
#             continue
#         # 找到所有在线段node1-node2上的投影点
#         projection_points_on_edge = []
#         for proj_node, proj_point in projection_points.items():
#             if is_point_on_line(proj_point, orig_coords_node1, orig_coords_node2):
#                 projection_points_on_edge.append((node_labels_geb10[proj_node], proj_point))
#
#         # 按距离node1的距离对投影点排序
#         projection_points_on_edge.sort(key=lambda x: np.linalg.norm(np.array(orig_coords_node1) - np.array(x[1])))
#         # 从node1开始依次连接投影点，然后连接到node2
#         last_node = node1
#         for proj_node_id, _ in projection_points_on_edge:
#             if last_node != proj_node_id:  # 避免自连接
#                 G_geb10_updated.add_edge(last_node, proj_node_id)
#             last_node = proj_node_id
#         if last_node != node2:  # 避免自连接
#             G_geb10_updated.add_edge(last_node, node2)
#     # 再次生成更新后的G_geb10_updated的邻接矩阵
#     updated_adj_matrix = nx.adjacency_matrix(G_geb10_updated, nodelist=sorted(G_geb10_updated.nodes()))
#     updated_adj_df = pd.DataFrame(updated_adj_matrix.toarray(), index=sorted(G_geb10_updated.nodes()),
#                                   columns=sorted(G_geb10_updated.nodes()))
#     return updated_adj_matrix, updated_adj_df
#
#
#
# def obtain_node_moving(gdf_geb25, adjusted_points, node_labels_geb10):
#     geb25_coords = set()
#     for _, row in gdf_geb25.iterrows():
#         geom = row.geometry
#         if geom.geom_type == 'Polygon':
#             for x, y in geom.exterior.coords:
#                 geb25_coords.add((x, y))
#         elif geom.geom_type == 'MultiPolygon':
#             for poly in geom:
#                 for x, y in poly.exterior.coords:
#                     geb25_coords.add((x, y))
#
#     # 更新节点移动信息，标记与geb25坐标一致的点
#     nodes_movement_info = []
#     for orig, new in adjusted_points:
#         node_id = node_labels_geb10[orig]
#         delta_x = new[0] - orig[0]
#         delta_y = new[1] - orig[1]
#         # 如果新坐标在geb25_coords中，则标记为1，否则标记为0
#         point_type = 1 if new in geb25_coords else 0
#         nodes_movement_info.append({
#             'Node ID': node_id,
#             'Original X': orig[0],
#             'Original Y': orig[1],
#             'Delta X': delta_x,
#             'Delta Y': delta_y,
#             'Type': point_type
#         })
#     nodes_movement_df = pd.DataFrame(nodes_movement_info)
#     return nodes_movement_df
#
#
# def VisUalize(adjusted_points, node_labels_geb10, G_geb25, G_geb10,
#               target_joinids_geb25, filtered_gdf_geb25, target_joinids_geb10,
#               filtered_gdf_geb10, G_geb10_updated):
#     ##可视化结果
#     pos_geb10_updated = {}
#     for orig_node, new_coords in adjusted_points:
#         node_id = node_labels_geb10[orig_node]  # 获取原始节点的ID
#         pos_geb10_updated[node_id] = new_coords  # 使用新坐标更新位置
#     pos_geb25 = {node: (node[0], node[1]) for node in G_geb25.nodes()}
#     pos_geb10 = {node: (node[0], node[1]) for node in G_geb10.nodes()}
#     # 创建一个1行4列的子图布局
#     fig, axes = plt.subplots(1, 4, figsize=(20, 8))
#     # 第一个子图：原始的GEB10图
#     nx.draw(G_geb10, pos=pos_geb10, ax=axes[0], node_size=50, node_color='blue', edge_color='black', with_labels=False)
#     for node, (x, y) in pos_geb10.items():
#         axes[0].text(x, y, s=node_labels_geb10[node], horizontalalignment='center', verticalalignment='center',
#                      color='red', fontsize=8)
#     axes[0].set_title('Original GEB10 Graph')
#     # 第二个子图：点的移动
#     # 对于移动点后的结果，并移除坐标系
#     axes[1].set_xticks([])
#     axes[1].set_yticks([])
#     axes[1].spines['top'].set_visible(False)
#     axes[1].spines['right'].set_visible(False)
#     axes[1].spines['bottom'].set_visible(False)
#     axes[1].spines['left'].set_visible(False)
#     # 绘制geb25的边界（黑色，较粗线）
#     for target_joinid in target_joinids_geb25:
#         selected_buildings = filtered_gdf_geb25[filtered_gdf_geb25['JOINID'] == target_joinid]
#         for geom in selected_buildings.geometry:
#             if geom.geom_type in ['Polygon', 'MultiPolygon']:
#                 x, y = geom.exterior.xy
#                 axes[1].plot(x, y, color='black', linewidth=2, zorder=1)
#     # 绘制geb10的边界（绿色）
#     for target_joinid in target_joinids_geb10:
#         selected_buildings = filtered_gdf_geb10[filtered_gdf_geb10['JOINID'] == target_joinid]
#         for geom in selected_buildings.geometry:
#             if geom.geom_type in ['Polygon', 'MultiPolygon']:
#                 x, y = geom.exterior.xy
#                 axes[1].plot(x, y, color='green', linewidth=1, zorder=2)
#     # 绘制点的移动
#     for point, new_position in adjusted_points:
#         axes[1].arrow(point[0], point[1], new_position[0] - point[0], new_position[1] - point[1], head_width=0.2,
#                       head_length=0.3, fc='red', ec='red', zorder=3)
#         axes[1].plot(point[0], point[1], 'o', color='blue', zorder=3)
#     axes[1].set_title('Point Movements')
#
#     # 第三个子图：更新后的GEB10图
#     nx.draw(G_geb10_updated, pos=pos_geb10_updated, ax=axes[2], node_size=50, node_color='blue', edge_color='black',
#             with_labels=False)
#     for orig_node, new_node in adjusted_points:
#         node_id = node_labels_geb10[orig_node]  # 使用原始节点ID
#         x, y = new_node  # 新节点的坐标
#         axes[2].text(x, y, s=node_id, horizontalalignment='center', verticalalignment='center', color='red', fontsize=8)
#     axes[2].set_title('Updated GEB10 Graph')
#     # 第四个子图：原始的GEB25图
#     nx.draw(G_geb25, pos=pos_geb25, ax=axes[3], node_size=50, node_color='red', edge_color='black', with_labels=False)
#     for node, (x, y) in pos_geb25.items():
#         axes[3].text(x, y, s=node, horizontalalignment='center', verticalalignment='center', color='white', fontsize=0)
#     axes[3].set_title('Original GEB25 Graph')
#     plt.tight_layout()
#     plt.show()
#
#
# def save_data(output_dir, selected_geb25_id,
#               df_adj_matrix_geb10, df_adj_matrix_geb25,
#               feature_matrix_geb10, feature_matrix_geb25,
#               updated_adj_df, nodes_movement_df):
#     adj_matrix_geb10_dir = f"{output_dir}/adj_matrix_geb10"
#     adj_matrix_geb25_dir = f"{output_dir}/adj_matrix_geb25"
#     feature_matrix_geb10_dir = f"{output_dir}/feature_matrix_geb10"
#     feature_matrix_geb25_dir = f"{output_dir}/feature_matrix_geb25"
#     updated_adj_matrix_dir = f"{output_dir}/updated_adj_matrix_geb10"
#     nodes_movement_info_dir = f"{output_dir}/nodes_movement_info"
#
#     # 创建这些目录（如果它们不存在）
#     os.makedirs(adj_matrix_geb10_dir, exist_ok=True)
#     os.makedirs(adj_matrix_geb25_dir, exist_ok=True)
#     os.makedirs(feature_matrix_geb10_dir, exist_ok=True)
#     os.makedirs(feature_matrix_geb25_dir, exist_ok=True)
#     os.makedirs(updated_adj_matrix_dir, exist_ok=True)
#     os.makedirs(nodes_movement_info_dir, exist_ok=True)
#
#     # 将文件保存到相应的目录
#     df_adj_matrix_geb10.to_csv(f"{adj_matrix_geb10_dir}/adj_matrix_geb10_{selected_geb25_id}.csv")
#     df_adj_matrix_geb25.to_csv(f"{adj_matrix_geb25_dir}/adj_matrix_geb25_{selected_geb25_id}.csv")
#     feature_matrix_geb10.to_csv(f"{feature_matrix_geb10_dir}/feature_matrix_geb10_{selected_geb25_id}.csv", index=False)
#     feature_matrix_geb25.to_csv(f"{feature_matrix_geb25_dir}/feature_matrix_geb25_{selected_geb25_id}.csv", index=False)
#     updated_adj_df.to_csv(f"{updated_adj_matrix_dir}/updated_adj_matrix_geb10_{selected_geb25_id}.csv")
#     nodes_movement_df.to_csv(f"{nodes_movement_info_dir}/nodes_movement_info_{selected_geb25_id}.csv", index=False)
#
#
