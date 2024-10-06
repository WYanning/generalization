from shapely.geometry import LineString
from scipy.spatial import Delaunay
import numpy as np
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('TkAgg')  # or use 'Agg' for non-interactive (file-based) output
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.affinity import translate
import networkx as nx
from shapely.geometry import Polygon,MultiPolygon
import os
import triangle as tr
from triangle import triangulate
from scipy.spatial import distance_matrix
import random
from scipy.spatial.distance import pdist, squareform
import networkx as nx


output_dir = "D:/dresden/mapgeneralization/dataset/nodeandadj"
os.makedirs(output_dir, exist_ok=True)
shapefile_geb10 = "D:/dresden/mt/MapGeneralizer-main/MapGeneralizer-main/Vectors_MapGeneralization_DL/stuttgart_change/geb10.shp"
gdf_geb10 = gpd.read_file(shapefile_geb10)
# original_crs = gdf_geb10.crs
target_crs = 'EPSG:25832'
gdf_geb10 = gdf_geb10.to_crs(target_crs)
shapefile_geb25 = "D:/dresden/mt/MapGeneralizer-main/MapGeneralizer-main/Vectors_MapGeneralization_DL/stuttgart_change/geb25.shp"
gdf_geb25 = gpd.read_file(shapefile_geb25)
# original_crs = gdf_geb25.crs
target_crs = 'EPSG:25832'
gdf_geb25 =gdf_geb25.to_crs(target_crs)

# 添加JOINID列
gdf_geb10['JOINID_geb10'] = gdf_geb10['JOINID']
gdf_geb25['JOINID_geb25'] = gdf_geb25['JOINID']
# 执行空间连接
joined_df = gpd.sjoin(gdf_geb10, gdf_geb25, how="inner", predicate='intersects')
# 分析一对一和一对多的关系
# 一对一关系
one_to_one = joined_df.groupby('JOINID_geb25').filter(lambda x: len(x) == 1)
# 一对多关系
one_to_many = joined_df.groupby('JOINID_geb25').filter(lambda x: len(x) > 1)
# 创建一个DataFrame来存储一对一关系的详细信息
one_to_one_details = one_to_one.groupby('JOINID_geb25')['JOINID_geb10'].apply(list).reset_index()
one_to_one_details.columns = ['JOINID_geb25', 'JOINID_geb10_list']
# 创建一个DataFrame来存储一对多关系的详细信息
one_to_many_details = one_to_many.groupby('JOINID_geb25')['JOINID_geb10'].apply(list).reset_index()
one_to_many_details.columns = ['JOINID_geb25', 'JOINID_geb10_list']
# 打印一对一和一对多的详细信息
print("一对一关系:")
print(len(one_to_one_details))
print("\n一对多关系:")
print(len(one_to_many_details))

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

selected_geb25_id = 853

# 获取与特定 JOINID_geb25 关联的数据
selected_one_to_many = one_to_many_details[one_to_many_details['JOINID_geb25'] == selected_geb25_id].iloc[0]
target_joinids_geb10 = selected_one_to_many['JOINID_geb10_list']
target_joinids_geb25 = [selected_one_to_many['JOINID_geb25']]

# 获取对应的边界框
selected_bounding_box = bounding_boxes[bounding_boxes['JOINID'] == selected_geb25_id].iloc[0].geometry
origin_x, origin_y = selected_bounding_box.bounds[0], selected_bounding_box.bounds[1]
gdf_geb25 = gdf_geb25.copy()
gdf_geb10 = gdf_geb10.copy()
# 更新 gdf_geb25 到局部坐标系
gdf_geb25['geometry'] = gdf_geb25['geometry'].apply(lambda geom: to_local_coordinates(geom, origin_x, origin_y))
gdf_geb10['geometry'] = gdf_geb10['geometry'].apply(lambda geom: to_local_coordinates(geom, origin_x, origin_y))

# 筛选特定的 JOINID
filtered_gdf_geb25 = gdf_geb25[gdf_geb25['JOINID'].isin(target_joinids_geb25)]
filtered_gdf_geb10 = gdf_geb10[gdf_geb10['JOINID'].isin(target_joinids_geb10)]
print("Coordinates for filtered_gdf_geb25:")
for index, row in filtered_gdf_geb25.iterrows():
    print(f"JOINID {row['JOINID']}:")
    for coord in list(row['geometry'].exterior.coords):
        print(coord)

# 打印 filtered_gdf_geb10 中的坐标
print("\nCoordinates for filtered_gdf_geb10:")
for index, row in filtered_gdf_geb10.iterrows():
    print(f"JOINID {row['JOINID']}:")
    for coord in list(row['geometry'].exterior.coords):
        print(coord)
local_bounding_box = to_local_coordinates(selected_bounding_box, origin_x, origin_y)
fig, ax = plt.subplots(figsize=(10, 10))

local_bounding_box_series = gpd.GeoSeries([local_bounding_box.boundary])

# 绘制局部坐标系下的 bounding box
local_bounding_box_series.plot(ax=ax, color='blue', linewidth=2, label='Bounding Box (Local Coordinates)')
# 绘制局部坐标系下的 selected geb25 buildings
filtered_gdf_geb25 .plot(ax=ax, color='red', alpha=0.5, label='GEB25 Buildings (Local Coordinates)')
filtered_gdf_geb10.plot(ax=ax, color='green', alpha=0.5, label='GEB10 Buildings (Local Coordinates)')
# ax.set_title('Local Coordinates System Bounding Box and Buildings')
# ax.legend()
plt.show()

def extract_polygon_data(gdf):
    points = []
    segments = []
    point_index_map = {}
    point_idx = 0

    for geom in gdf.geometry:
        polygon_points = list(geom.exterior.coords[:-1])  # 忽略闭合点
        for pt in polygon_points:
            if pt not in point_index_map:
                points.append(pt)
                point_index_map[pt] = point_idx
                point_idx += 1

        num_points = len(polygon_points)
        for i in range(num_points):
            start_idx = point_index_map[polygon_points[i]]
            end_idx = point_index_map[polygon_points[(i + 1) % num_points]]
            segments.append((start_idx, end_idx))

    return np.array(points), np.array(segments)


def extract_graph_from_triangulation(triangulation, bbox_indices):
    G_tri = nx.Graph()
    # 添加节点
    for idx, point in enumerate(triangulation['vertices']):
        if idx not in bbox_indices:
            G_tri.add_node(idx, pos=(point[0], point[1]))

    # 添加边
    for triangle in triangulation['triangles']:
        # 只有当三角形不包括边界框的顶点时，才添加边
        if not any(v in bbox_indices for v in triangle):
            edges = [(triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])]
            G_tri.add_edges_from(edges)

    return G_tri

# 计算包围框
bounds = filtered_gdf_geb10.total_bounds  # 返回[minx, miny, maxx, maxy]
bounding_box = box(*bounds)  # 使用Shapely的box创建包围框

points, segments = extract_polygon_data(filtered_gdf_geb10)

# 添加包围框的点和边界
bbox_corners = list(bounding_box.exterior.coords[:-1])
bbox_idx_start = len(points)
bbox_points = bbox_corners
bbox_segments = [
    (bbox_idx_start, bbox_idx_start + 1),
    (bbox_idx_start + 1, bbox_idx_start + 2),
    (bbox_idx_start + 2, bbox_idx_start + 3),
    (bbox_idx_start + 3, bbox_idx_start)
]

points = np.vstack([points, bbox_points])
segments = np.vstack([segments, bbox_segments])

# 进行三角剖分
triangulation = tr.triangulate({
    'vertices': points,
    'segments': segments
}, 'p')

bbox_indices = set(range(bbox_idx_start, bbox_idx_start + 4))
G_tri = extract_graph_from_triangulation(triangulation, bbox_indices)

plt.figure(figsize=(10, 8))
tr.plot(ax=plt.gca(), **triangulation)

# 绘制原始建筑物
for geom in filtered_gdf_geb10.geometry:
    x, y = geom.exterior.xy
    plt.plot(x, y, 'k-', linewidth=1)  # 黑色线条表示原始建筑

plt.legend()
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Constrained Delaunay Triangulation with Original Buildings')
plt.grid(True)
plt.axis('equal')
plt.show()

plt.figure(figsize=(8, 6))
pos = nx.get_node_attributes(G_tri, 'pos')
nx.draw(G_tri,pos, with_labels=False, node_color='skyblue', node_size=50, edge_color='k')
plt.title('Graph Representation of Triangulation (G_tri)')
plt.grid(True)
plt.axis('equal')
plt.show()
plt.figure(figsize=(8, 6))
nx.draw(G_tri, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title('Updated Graph without Self-Connections')
plt.show()
# 生成邻接矩阵
adj_matrix = nx.adjacency_matrix(G_tri)
print(adj_matrix.todense())

# G_tri = nx.Graph()
# # 添加节点
# for i in range(len(points_geb10)):
#     G_tri.add_node(i)
# # 添加边，每个三角形的顶点对应的索引
# for simplex in tri.simplices:
#     G_tri.add_edge(simplex[0], simplex[1])
#     G_tri.add_edge(simplex[1], simplex[2])
#     G_tri.add_edge(simplex[2], simplex[0])
# adj_matrix_tri = nx.adjacency_matrix(G_tri)


##图构建
def build_graph_from_polygons(gdf):
    G = nx.Graph()
    for _, row in gdf.iterrows():
        geom = row['geometry']
        if geom.geom_type == 'Polygon':
            process_polygon(G, geom)
        elif geom.geom_type == 'MultiPolygon':
            for polygon in geom:
                process_polygon(G, polygon)
    return G

def process_polygon(G, polygon):
    # 添加多边形的外部边界到图中
    exterior_nodes = list(polygon.exterior.coords)
    for i in range(len(exterior_nodes) - 1):
        G.add_edge(exterior_nodes[i], exterior_nodes[i + 1])

    # 添加多边形的内部边界（如果有的话）到图中
    for interior in polygon.interiors:
        interior_nodes = list(interior.coords)
        for i in range(len(interior_nodes) - 1):
            G.add_edge(interior_nodes[i], interior_nodes[i + 1])

# 过滤geb10和geb25，只保留选定的建筑物
filtered_gdf_geb10 = gdf_geb10[gdf_geb10['JOINID'].isin(target_joinids_geb10)]
filtered_gdf_geb25 = gdf_geb25[gdf_geb25['JOINID'].isin(target_joinids_geb25)]
# 为geb10和geb25创建图
G_geb10 = build_graph_from_polygons(filtered_gdf_geb10)
G_geb25 = build_graph_from_polygons(filtered_gdf_geb25)

for node in G_geb10.nodes():
    print(node, "has", len(list(G_geb10.neighbors(node))), "neighbors")
import matplotlib.pyplot as plt

pos = {node: (node[0], node[1]) for node in G_geb10.nodes()}
nx.draw(G_geb10, pos, node_size=5, edge_color="b", node_color="g", with_labels=False)
plt.title("Graph Visualization of Selected GEB10")
plt.show()
original_geb25_nodes = list(G_geb25.nodes())
# 为G_geb10中的每个节点分配一个编号
node_labels_geb10 = {node: i for i, node in enumerate(G_geb10.nodes())}
# 首先获取所有节点的ID
node_ids = list(node_labels_geb10.values())

adj_matrix_geb10 = nx.adjacency_matrix(G_geb10)
adj_matrix_geb25 = nx.adjacency_matrix(G_geb25)
# 将邻接矩阵转换为DataFrame以便打印
df_adj_matrix_geb10 = pd.DataFrame(adj_matrix_geb10.toarray())
df_adj_matrix_geb25 = pd.DataFrame(adj_matrix_geb25.toarray())
# 将节点ID映射到其索引
node_id_map_geb10 = {node: i for i, node in enumerate(G_geb10.nodes())}
node_id_map_geb25 = {node: i for i, node in enumerate(G_geb25.nodes())}
# 更新DataFrame以使用节点ID作为索引和列名
df_adj_matrix_geb10.columns = [node_id_map_geb10[n] for n in G_geb10.nodes()]
df_adj_matrix_geb10.index = [node_id_map_geb10[n] for n in G_geb10.nodes()]
df_adj_matrix_geb25.columns = [node_id_map_geb25[n] for n in G_geb25.nodes()]
df_adj_matrix_geb25.index = [node_id_map_geb25[n] for n in G_geb25.nodes()]

def find_nearest_projection(point, gdf):
    nearest_point = None
    min_distance = np.inf
    for geom in gdf.geometry:
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            lines = [geom.exterior] + list(geom.interiors)
            for line in lines:
                line_string = LineString(line)
                projected_point = line_string.interpolate(line_string.project(point))
                distance = point.distance(projected_point)
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = projected_point
    return nearest_point
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
lines_gdf = gpd.GeoDataFrame(geometry=lines)

# 可视化
fig, ax = plt.subplots(figsize=(12, 12))

# 绘制geb10和geb25的多边形边框
filtered_gdf_geb10.boundary.plot(ax=ax, color='blue', label='GEB10 Boundaries', linewidth=1.5, alpha=0.6)
filtered_gdf_geb25.boundary.plot(ax=ax, color='green', label='GEB25 Boundaries', linewidth=1.5, alpha=0.6)

# 绘制原始点和投影点
original_gdf.plot(ax=ax, marker='o', color='blue', label='Original Points', markersize=10)
projected_gdf.plot(ax=ax, marker='x', color='red', label='Projected Points', markersize=10)

# 绘制连接线
for line in lines_gdf.geometry:
    xs, ys = line.xy
    ax.plot(xs, ys, color='gray', linestyle='--', label='Projection Lines' if 'Projection Lines' not in ax.get_legend_handles_labels()[1] else "")

# 添加图例
ax.legend()

# 设置图表标题和坐标轴标签
plt.title('Projection of GEB10 Nodes onto GEB25 Polygons and Their Boundaries')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()


def find_closest_point(point, other_gdf):
    """查找给定点与其他GeoDataFrame中点的最近点及其索引"""
    min_dist = np.inf
    closest_point = None
    closest_idx = -1
    for idx, other_point in enumerate(other_gdf.geometry):
        dist = point.distance(other_point)
        if dist < min_dist:
            min_dist = dist
            closest_point = other_point
            closest_idx = idx
    return closest_point, closest_idx


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

def generate_offset(distance=0.001):
    angle = np.random.uniform(0, 2 * np.pi)
    return distance * np.cos(angle), distance * np.sin(angle)

# 防止点重合的逻辑
def prevent_overlap(gdf, distance=0.001):
    moved_points = {}
    for idx, point in enumerate(gdf.geometry):
        while point in moved_points:
            offset = generate_offset(distance)
            point = Point(point.x + offset[0], point.y + offset[1])
        moved_points[point] = idx
        gdf.at[idx, 'geometry'] = point
    return gdf

moved_projected_gdf = prevent_overlap(moved_projected_gdf)

fig, ax = plt.subplots(figsize=(10, 10))

# 绘制geb10和geb25的轮廓
filtered_gdf_geb10.boundary.plot(ax=ax, color='blue', label='GEB10 Boundaries', linewidth=1.5, alpha=0.6)
filtered_gdf_geb25.boundary.plot(ax=ax, color='green', label='GEB25 Boundaries', linewidth=1.5, alpha=0.6)

# 绘制geb25的节点
geb25_nodes = [Point(node) for node in G_geb25.nodes()]
geb25_nodes_gdf = gpd.GeoDataFrame(geometry=geb25_nodes)
geb25_nodes_gdf.plot(ax=ax, color='red', marker='o', label='GEB25 Nodes', markersize=50, alpha=0.6)

# 绘制原始的projected_gdf节点
projected_gdf.plot(ax=ax, color='yellow', marker='x', label='Original Projected Nodes', markersize=50, alpha=0.6)

# 绘制移动后的projected_gdf节点
moved_projected_gdf.plot(ax=ax, color='purple', marker='+', label='Moved Projected Nodes', markersize=50, alpha=0.6)

# 添加图例
ax.legend()

# 设置坐标轴标签和标题
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Visualization of GEB10, GEB25, and Projected Points')

# 显示图形
plt.show()
print(moved_projected_gdf)

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


plt.figure(figsize=(10, 10))
nx.draw_networkx_nodes(G_geb10_updated, pos=pos_geb10_updated, node_size=50, node_color='blue')
nx.draw_networkx_edges(G_geb10_updated, pos=pos_geb10_updated, alpha=0.5)
plt.axis('off')
plt.show()
adj_matrix_df = nx.to_pandas_adjacency(G_geb10_updated)
print(adj_matrix_df)

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

# 可视化
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G_geb10_updated, pos=pos_geb10_updated, node_size=50, node_color='blue', alpha=0.6)
nx.draw_networkx_edges(G_geb10_updated, pos=pos_geb10_updated, edge_color='gray', alpha=0.5)
nx.draw_networkx_labels(G_geb10_updated, pos=pos_geb10_updated, font_size=8, font_color='black')
plt.title('Visualizing the Updated Graph')
plt.axis('off')
plt.show()

adj_matrix = nx.adjacency_matrix(G_geb10_updated)
# 将稀疏矩阵转换为常规数组格式进行显示
dense_adj_matrix = adj_matrix.todense()
updated_adj_matrix = nx.adjacency_matrix(G_geb10_updated, nodelist=sorted(G_geb10_updated.nodes()))
updated_adj_df = pd.DataFrame(updated_adj_matrix.toarray(), index=sorted(G_geb10_updated.nodes()),
                              columns=sorted(G_geb10_updated.nodes()))
print(updated_adj_df)
# 打印邻接矩阵
# print("Adjacency Matrix of G_geb10_updated:")
# print(dense_adj_matrix)

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
# 打印节点移动信息
print(nodes_movement_df)

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
        nx.draw(graph, pos=pos, ax=ax, node_size=50, node_color='blue' if i != 3 else 'red', edge_color='black', with_labels=False)
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box') # 保持比例一致，无畸变
# 第一个子图：原始的GEB10图
nx.draw(G_geb10, pos=pos_geb10, ax=axes[0], node_size=50, node_color='blue', edge_color='black', with_labels=False)
# for node, (x, y) in pos_geb10.items():
#     axes[0].text(x, y, s=node_labels_geb10[node], horizontalalignment='center', verticalalignment='center', color='red', fontsize=8)
axes[0].set_title('G-geb10')

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
            axes[1].plot(x, y, color='green', linewidth=2, zorder=1)
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
axes[1].set_title('Node Movements')

nx.draw(G_geb10_updated, pos=pos_geb10_updated, ax=axes[2], node_size=50, node_color='blue', edge_color='black', with_labels=False)
axes[2].set_title('Updated G-geb10')
# axes[2].set_aspect('equal', adjustable='datalim')

# 第四个子图：原始的GEB25图
nx.draw(G_geb25, pos=pos_geb25, ax=axes[3], node_size=50, node_color='red', edge_color='black', with_labels=False)
for node, (x, y) in pos_geb25.items():
    axes[3].text(x, y, s=node, horizontalalignment='center', verticalalignment='center', color='red', fontsize=0)
axes[3].set_title('G-geb25')
# axes[3].set_aspect('equal', adjustable='datalim')
plt.tight_layout()
plt.show()

