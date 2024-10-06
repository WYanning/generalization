import triangle as tr
import os
import matplotlib
matplotlib.use('Agg')
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

    filtered_gdf_geb10 = gdf_geb10[gdf_geb10['JOINID'].isin(target_joinids_geb10)]
    filtered_gdf_geb25 = gdf_geb25[gdf_geb25['JOINID'].isin(target_joinids_geb25)]

    bounds = filtered_gdf_geb10.total_bounds  # 返回[minx, miny, maxx, maxy]
    bounding_box_1 = box(*bounds)  # 使用Shapely的box创建包围框

    points, segments = extract_polygon_data(filtered_gdf_geb10)

    # 添加包围框的点和边界
    bbox_corners = list(bounding_box_1.exterior.coords[:-1])
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
    plt.figure(figsize=(8, 6))
    pos = nx.get_node_attributes(G_tri, 'pos')
    nx.draw(G_tri, pos, with_labels=True, node_color='skyblue', node_size=50, edge_color='k')
    plt.title('Graph Representation of Triangulation (G_tri)')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(save_folder, "Tri.jpg"))
    plt.close()
    #TODO

    adj_matrix_tri = nx.adjacency_matrix(G_tri)
    df_adj_matrix_tri = pd.DataFrame(adj_matrix_tri.toarray())
    df_adj_matrix_tri.to_csv(os.path.join(save_folder, "df_adj_matrix_tri.csv"))
    #TODO

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
        node_id_map_geb10 = {node: i for i, node in enumerate(G_geb10.nodes())}
        node_id_map_geb25 = {node: i for i, node in enumerate(G_geb25.nodes())}
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
            move_or_not = 1 if delta_x != 0 or delta_y != 0 else 0
            # 添加移动信息到列表
            nodes_movement_info.append({
                'Node ID': idx,
                'Original X': original_point.x,
                'Original Y': original_point.y,
                'Delta X': delta_x,
                'Delta Y': delta_y,
                'Type': point_type,
                'Type2':move_or_not
            })

        nodes_movement_df = pd.DataFrame(nodes_movement_info)
        # TODO nodes_movement_df
        nodes_movement_df.to_csv(os.path.join(save_folder, "nodes_movement_df.csv"))