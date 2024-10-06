from shapely.geometry import box
from shapely.affinity import translate
from shapely.geometry import MultiPolygon
import os
import math
import numpy as np
from shapely.geometry import LineString, Polygon, Point
import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely.ops import nearest_points
from shapely.geometry import Point
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 加载地理空间数据
shapefile_geb10 = "D:/dresden/mt/MapGeneralizer-main/MapGeneralizer-main/Vectors_MapGeneralization_DL/stuttgart_change/geb10.shp"
gdf_geb10 = gpd.read_file(shapefile_geb10)
target_crs = 'EPSG:25832'
gdf_geb10 = gdf_geb10.to_crs(target_crs)
shapefile_geb25 = "D:/dresden/mt/MapGeneralizer-main/MapGeneralizer-main/Vectors_MapGeneralization_DL/stuttgart_change/geb25.shp"
gdf_geb25 = gpd.read_file(shapefile_geb25)
gdf_geb25 = gdf_geb25.to_crs(target_crs)

# 添加JOINID列
gdf_geb10['JOINID_geb10'] = gdf_geb10['JOINID']
gdf_geb25['JOINID_geb25'] = gdf_geb25['JOINID']
# 执行空间连接
joined_df = gpd.sjoin(gdf_geb10, gdf_geb25, how="inner", predicate='intersects')
# 分析一对一和一对多的关系
# 一对一关系
# one_to_one = joined_df.groupby('JOINID_geb25').filter(lambda x: len(x) == 1)
# 一对多关系
one_to_many = joined_df.groupby('JOINID_geb25').filter(lambda x: len(x) > 1)
# 创建一个DataFrame来存储一对一关系的详细信息
# one_to_one_details = one_to_one.groupby('JOINID_geb25')['JOINID_geb10'].apply(list).reset_index()
# one_to_one_details.columns = ['JOINID_geb25', 'JOINID_geb10_list']
# 创建一个DataFrame来存储一对多关系的详细信息
one_to_many_details = one_to_many.groupby('JOINID_geb25')['JOINID_geb10'].apply(list).reset_index()
one_to_many_details.columns = ['JOINID_geb25', 'JOINID_geb10_list']
# 打印一对一和一对多的详细信息
# print("一对一关系:")
# print(len(one_to_one_details))
# print("\n一对多关系:")
# print(len(one_to_many_details))

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

def to_local_coordinates(geometry, origin_x, origin_y):
    # 平移几何对象使得边框的左下角成为新的原点
    translated_geometry = translate(geometry, xoff=-origin_x, yoff=-origin_y)
    if translated_geometry.geom_type == 'Polygon':
        local_coords = [(round(x, 3), round(y, 3)) for x, y in translated_geometry.exterior.coords]
        return Polygon(local_coords)
    elif translated_geometry.geom_type == 'MultiPolygon':
        # 处理多边形的情况
        local_polygons = []
        for polygon in translated_geometry:
            local_coords = [(round(x, 3), round(y, 3)) for x, y in polygon.exterior.coords]
            local_polygons.append(Polygon(local_coords))
        return MultiPolygon(local_polygons)
    return translated_geometry


# 指定的 JOINID_geb25
selected_geb25_id = 66
# 获取与特定 JOINID_geb25 关联的数据
selected_one_to_many = one_to_many_details[one_to_many_details['JOINID_geb25'] == selected_geb25_id].iloc[0]
target_joinids_geb10 = selected_one_to_many['JOINID_geb10_list']
target_joinids_geb25 = [selected_one_to_many['JOINID_geb25']]
# 获取对应的边界框
selected_bounding_box = bounding_boxes[bounding_boxes['JOINID'] == selected_geb25_id].iloc[0].geometry
origin_x, origin_y = selected_bounding_box.bounds[0], selected_bounding_box.bounds[1]
# 更新 gdf_geb25 到局部坐标系
gdf_geb25 = gdf_geb25.copy()
gdf_geb25['geometry'] = gdf_geb25['geometry'].apply(lambda x: to_local_coordinates(x, origin_x, origin_y))
# 使用与 gdf_geb25 相同的局部坐标原点来转换 gdf_geb10
gdf_geb10 = gdf_geb10.copy()
gdf_geb10['geometry'] = gdf_geb10['geometry'].apply(lambda x: to_local_coordinates(x, origin_x, origin_y))
selected_geb10 = gdf_geb10[gdf_geb10['JOINID'].isin(target_joinids_geb10)]
selected_geb25 = gdf_geb25[gdf_geb25['JOINID'].isin(target_joinids_geb25)]

def build_graph_from_gdf(gdf):
    G = nx.Graph()
    for _, row in gdf.iterrows():
        # 获取多边形的顶点
        coords = list(row['geometry'].exterior.coords)

        # 添加多边形的顶点到图中，并添加位置属性
        for i in range(len(coords) - 1):  # 避免在最后重新添加第一个点
            G.add_node(coords[i], pos=coords[i])  # 添加位置信息
            if i == len(coords) - 1:
                # 连接最后一个顶点与第一个顶点，完成多边形的闭合
                G.add_edge(coords[i], coords[0])
            else:
                # 连接每个顶点与下一个顶点
                G.add_edge(coords[i], coords[i + 1])

        # 如果有内部边界（洞），也处理
        if 'interiors' in dir(row['geometry']):
            for interior in row['geometry'].interiors:
                interior_coords = list(interior.coords)
                for j in range(len(interior_coords) - 1):
                    G.add_node(interior_coords[j], pos=interior_coords[j])  # 添加位置信息
                    if j == len(interior_coords) - 1:
                        G.add_edge(interior_coords[j], interior_coords[0])
                    else:
                        G.add_edge(interior_coords[j], interior_coords[j + 1])
    return G

def remove_self_loops(G):
    loops = list(nx.selfloop_edges(G))
    G.remove_edges_from(loops)
    return G

# 过滤geb10和geb25，只保留选定的建筑物
G_geb10 = build_graph_from_gdf(selected_geb10)
G_geb10 = remove_self_loops(G_geb10)
#
# pos = nx.get_node_attributes(G_geb10, 'pos')
# x_coords, y_coords = zip(*pos.values())
# top_point = max(pos.values(), key=lambda x: x[1])
# bottom_point = min(pos.values(), key=lambda x: x[1])
# left_point = min(pos.values(), key=lambda x: x[0])
# right_point = max(pos.values(), key=lambda x: x[0])
# boundary_points = [top_point, bottom_point, left_point, right_point]
#
# # 构建由四个边界点构成的多边形
# boundary_polygon = Polygon([top_point, right_point, bottom_point, left_point])
#
#
# # 计算每个节点到最近边界点的距离、构成线段与水平线的夹角以及到边界的距离
# def calculate_features(G, boundary_points, boundary_polygon):
#     features = {}
#     for node, position in G.nodes(data='pos'):
#         min_distance = float('inf')
#         closest_point = None
#         angle = None
#
#         for boundary_point in boundary_points:
#             distance = np.sqrt((position[0] - boundary_point[0]) ** 2 + (position[1] - boundary_point[1]) ** 2)
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_point = boundary_point
#
#         if closest_point:
#             dx = closest_point[0] - position[0]
#             dy = closest_point[1] - position[1]
#             angle = math.atan2(dy, dx) * 180 / math.pi
#
#         # 计算到边界的最短距离
#         point = Point(position)
#         boundary_distance = point.distance(boundary_polygon)
#
#         features[node] = (min_distance, angle, boundary_distance)
#     return features
#
# features = calculate_features(G_geb10, boundary_points, boundary_polygon)
#
# # 创建DataFrame
# df = pd.DataFrame.from_dict(features, orient='index', columns=['Distance to nearest boundary point', 'Angle with horizontal', 'Distance to boundary'])
#
# # 输出DataFrame
# print(df)

num_polygons = len(selected_geb10['JOINID'].unique())
selected_geb10.loc[:, 'centroid'] = selected_geb10['geometry'].centroid
selected_geb10.loc[:, 'centroid_x'] = selected_geb10['geometry'].centroid.x
selected_geb10.loc[:, 'centroid_y'] = selected_geb10['geometry'].centroid.y

# 根据中心点的x和y坐标排序
sorted_geb10 = selected_geb10.sort_values(by=['centroid_x', 'centroid_y'])

# 创建图
G_centroid = nx.Graph()

# 添加节点
for index, row in sorted_geb10.iterrows():
    G_centroid.add_node(row['JOINID'], pos=(row['centroid_x'], row['centroid_y']))

# 添加边：连接每个节点至下一个节点
sorted_ids = sorted_geb10['JOINID'].tolist()
for i in range(len(sorted_ids) - 1):
    G_centroid.add_edge(sorted_ids[i], sorted_ids[i + 1])

# 分析节点的邻居
neighbors = {}
for node in G_centroid.nodes:
    neighbors[node] = list(G_centroid.neighbors(node))

# 打印邻居关系
for node, nbs in neighbors.items():
    print(f"多边形 {node} 的邻居有: {nbs}")

vertex_distances = {}

for node in G_centroid.nodes:
    # 获取当前多边形的几何形状
    current_polygon = selected_geb10.loc[selected_geb10['JOINID'] == node, 'geometry'].values[0]
    vertex_distances[node] = {}

    # 获取节点的邻居节点
    neighbors = list(G_centroid.neighbors(node))

    # 遍历当前多边形的每个顶点
    for vertex in current_polygon.exterior.coords:
        vertex_point = Point(vertex)
        min_distance = float('inf')  # 初始化最小距离

        # 对每个邻居多边形计算最短距离
        for neighbor in neighbors:
            neighbor_polygon = selected_geb10.loc[selected_geb10['JOINID'] == neighbor, 'geometry'].values[0]
            # 计算到邻居多边形的最短距离
            distance = vertex_point.distance(neighbor_polygon)
            min_distance = min(min_distance, distance)

        # 存储当前顶点到所有邻居的最小距离
        vertex_distances[node][vertex] = min_distance

# 打印结果
for poly_id, distances in vertex_distances.items():
    print(f"多边形 {poly_id} 的顶点到邻居多边形的最短距离:")
    for vertex, distance in distances.items():
        print(f"  顶点 {vertex} 的最短距离: {distance}")


def calculate_features_new(selected_geb10):
    # 计算中心点并排序
    selected_geb10['centroid'] = selected_geb10['geometry'].centroid
    selected_geb10['centroid_x'] = selected_geb10['geometry'].centroid.x
    selected_geb10['centroid_y'] = selected_geb10['geometry'].centroid.y
    sorted_geb10 = selected_geb10.sort_values(by=['centroid_x', 'centroid_y'])

    # 创建图
    G_centroid = nx.Graph()

    # 添加节点
    for index, row in sorted_geb10.iterrows():
        G_centroid.add_node(row['JOINID'], pos=(row['centroid_x'], row['centroid_y']))

    # 添加边：连接每个节点至下一个节点
    sorted_ids = sorted_geb10['JOINID'].tolist()
    for i in range(len(sorted_ids) - 1):
        G_centroid.add_edge(sorted_ids[i], sorted_ids[i + 1])

    # 准备数据列表
    vertex_distances = []

    for node in G_centroid.nodes:
        # 获取当前多边形的几何形状
        current_polygon = selected_geb10.loc[selected_geb10['JOINID'] == node, 'geometry'].values[0]

        # 获取节点的邻居节点
        neighbors = list(G_centroid.neighbors(node))

        # 获取多边形的所有顶点，并去除重复的起始/终止点
        vertices = list(current_polygon.exterior.coords)[:-1]  # Removes duplicate last point

        # 遍历当前多边形的每个顶点
        for vertex in vertices:
            vertex_point = Point(vertex)
            min_distance = float('inf')  # 初始化最小距离

            # 对每个邻居多边形计算最短距离
            for neighbor in neighbors:
                neighbor_polygon = selected_geb10.loc[selected_geb10['JOINID'] == neighbor, 'geometry'].values[0]
                # 计算到邻居多边形的最短距离
                distance = vertex_point.distance(neighbor_polygon)
                min_distance = min(min_distance, distance)

            # 将每个顶点的x、y坐标和它的最短距离加入到数据列表
            vertex_distances.append({
                'x_coord': vertex[0],  # X coordinate
                'y_coord': vertex[1],  # Y coordinate
                'min_distance': min_distance
            })

    # 将数据列表转换为 DataFrame
    feature_df = pd.DataFrame(vertex_distances)
    return feature_df

feature_df= calculate_features_new(selected_geb10)
print(feature_df)

def calculate_features(G):
    all_features = {}

    for node, data in G.nodes(data=True):
        if len(list(G.neighbors(node))) < 2:
            continue

        node_features = []
        pos_node = data['pos']
        pos = nx.get_node_attributes(G, 'pos')
        top_point = max(pos.values(), key=lambda x: x[1])
        bottom_point = min(pos.values(), key=lambda x: x[1])
        left_point = min(pos.values(), key=lambda x: x[0])
        right_point = max(pos.values(), key=lambda x: x[0])
        boundary_points = [top_point, bottom_point, left_point, right_point]

        # 构建由四个边界点构成的多边形
        boundary_polygon = Polygon([top_point, right_point, bottom_point, left_point])
        neighbors = list(G.neighbors(node))
        pos_prev = G.nodes[neighbors[0]]['pos']
        pos_next = G.nodes[neighbors[1]]['pos']

        # Local features calculations
        prev_vector = np.array(pos_prev) - np.array(pos_node)
        next_vector = np.array(pos_next) - np.array(pos_node)

        # Turn angle
        angle = np.degrees(np.arccos(
            np.clip(np.dot(prev_vector, next_vector) / (np.linalg.norm(prev_vector) * np.linalg.norm(next_vector)),
                    -1.0, 1.0)))
        node_features.append(angle)

        # Convexity
        cross_product = np.cross(prev_vector, next_vector)
        convexity = np.sign(cross_product)
        node_features.append(convexity)

        # Segment lengths
        pre_length = np.linalg.norm(prev_vector)
        next_length = np.linalg.norm(next_vector)
        node_features.append(pre_length)
        node_features.append(next_length)

        # Triangle area
        area = 0.5 * np.abs(np.cross(prev_vector, next_vector))
        node_features.append(area)

        # Local segment length
        loc_seg_length = (pre_length + next_length) / 2
        node_features.append(loc_seg_length)

        # Regional features calculations
        min_distance, angle_to_horizon, boundary_distance = calculate_node_distances(pos_node, boundary_points, boundary_polygon)
        node_features.extend([min_distance, angle_to_horizon, boundary_distance])

        # Add position info
        node_features.extend([pos_node[0], pos_node[1]])  # X and Y coordinates

        # Save all features for this node
        all_features[node] = node_features

    # Convert the dictionary to a DataFrame
    feature_columns = ['loc_turn_angle', 'loc_convexity', 'pre_seg_length', 'next_seg_length',
                       'loc_tri_area', 'loc_seg_length', 'distance_to_nearest_boundary',
                       'angle_with_horizontal', 'distance_to_boundary', 'pos_x', 'pos_y']
    feature_df_old = pd.DataFrame.from_dict(all_features, orient='index', columns=feature_columns)
    return feature_df_old

def calculate_node_distances(pos_node, boundary_points, boundary_polygon):
    # Calculation of distance to the nearest boundary point
    min_distance = float('inf')
    closest_point = None
    angle_to_horizon = None

    for boundary_point in boundary_points:
        distance = np.linalg.norm(np.array(pos_node) - np.array(boundary_point))
        if distance < min_distance:
            min_distance = distance
            closest_point = boundary_point
            dx = closest_point[0] - pos_node[0]
            dy = closest_point[1] - pos_node[1]
            angle_to_horizon = np.degrees(np.arctan2(dy, dx))

    # Calculation of the distance to the boundary polygon
    point = Point(pos_node)
    boundary_distance = point.distance(boundary_polygon)

    return min_distance, angle_to_horizon, boundary_distance

feature_df_old = calculate_features(G_geb10)


# 确保坐标数据类型一致
feature_df['x_coord'] = feature_df['x_coord'].round(2)  # 调整为保留两位小数
feature_df['y_coord'] = feature_df['y_coord'].round(2)
feature_df_old['pos_x'] = feature_df_old['pos_x'].round(2)
feature_df_old['pos_y'] = feature_df_old['pos_y'].round(2)

# 合并 DataFrame
feature_df_old = feature_df_old.merge(
    feature_df[['x_coord', 'y_coord', 'min_distance']],
    left_on=['pos_x', 'pos_y'],
    right_on=['x_coord', 'y_coord'],
    how='left'
)

# 删除不需要的合并产生的额外列
feature_df_old.drop(['x_coord', 'y_coord'], axis=1, inplace=True)

# 查看合并结果
print(feature_df_old)
