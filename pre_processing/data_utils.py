import math
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.affinity import translate
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box




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

# 生成更新后的G_geb10_updated的邻接矩阵
def is_point_on_line(point, line_start, line_end):
    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    line_len = np.linalg.norm(line_vec)
    projected_length = np.dot(point_vec, line_vec) / line_len
    return 0 <= projected_length <= line_len and np.isclose(
        np.linalg.norm(np.cross(line_vec / line_len, point_vec)), 0)


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


