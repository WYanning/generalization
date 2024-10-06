import math, os
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from scipy.spatial import Delaunay
from shapely.geometry import Point
from shapely.affinity import translate
from shapely.geometry import Polygon, MultiPolygon


from TriDataProcess.util import *


output_dir = "D:/dresden/mapgeneralization/dataset/feature2"
os.makedirs(output_dir, exist_ok=True)


gdf_geb10, gdf_geb25 = load_data()
one_to_many_details = bulit_connection(gdf_geb10, gdf_geb25)
bounding_boxes = creat_bounding_box(one_to_many_details, gdf_geb25)
num_list = one_to_many_details["JOINID_geb25"].values.reshape(-1)


def Run_Single_Index(index, gdf_geb10, gdf_geb25, plot=False):
    # 获取与特定 JOINID_geb25 关联的数据
    selected_one_to_many = one_to_many_details[one_to_many_details['JOINID_geb25'] == index].iloc[0]
    target_joinids_geb10 = selected_one_to_many['JOINID_geb10_list']

    # 获取对应的边界框
    selected_bounding_box = bounding_boxes[bounding_boxes['JOINID'] == index].iloc[0].geometry
    origin_x, origin_y = selected_bounding_box.bounds[0], selected_bounding_box.bounds[1]

    # 更新 gdf_geb25 到局部坐标系

    # 使用与 gdf_geb25 相同的局部坐标原点来转换 gdf_geb10
    gdf_geb10 = gdf_geb10.copy()
    gdf_geb10['geometry'] = gdf_geb10['geometry'].apply(lambda x: to_local_coordinates(x, origin_x, origin_y))
    selected_geb10 = gdf_geb10[gdf_geb10['JOINID'].isin(target_joinids_geb10)]

    # 为geb10和geb25创建图
    G_geb10 = build_graph_from_polygons(selected_geb10)
    G_geb10 = remove_self_loops(G_geb10)
    feature_matrix_geb10 = calculate_features(G_geb10)
    feature_df= calculate_features_new(selected_geb10)
    feature_df['x_coord'] = feature_df['x_coord'].round(2)  # 调整为保留两位小数
    feature_df['y_coord'] = feature_df['y_coord'].round(2)
    feature_matrix_geb10['pos_x'] = feature_matrix_geb10['pos_x'].round(2)
    feature_matrix_geb10['pos_y'] = feature_matrix_geb10['pos_y'].round(2)

    # 合并 DataFrame
    feature_matrix_geb10 = feature_matrix_geb10.merge(
        feature_df[['x_coord', 'y_coord', 'min_distance']],
        left_on=['pos_x', 'pos_y'],
        right_on=['x_coord', 'y_coord'],
        how='left'
    )

    # 删除不需要的合并产生的额外列
    feature_matrix_geb10.drop(['x_coord', 'y_coord'], axis=1, inplace=True)

    # feature_matrix_geb25 = create_feature_matrix(node_local_features_geb25, global_features_geb25, node_labels_geb25,
    #                                              angles_geb25, prefix="")

    feature_matrix_geb10_dir = f"{output_dir}/feature_matrix_geb10"
    # feature_matrix_geb25_dir = f"{output_dir}/feature_matrix_geb25"
    os.makedirs(feature_matrix_geb10_dir, exist_ok=True)
    # os.makedirs(feature_matrix_geb25_dir, exist_ok=True)
    feature_matrix_geb10.to_csv(f"{feature_matrix_geb10_dir}/feature_matrix_geb10_{index}.csv", index=False)
    # feature_matrix_geb25.to_csv(f"{feature_matrix_geb25_dir}/feature_matrix_geb25_{index}.csv", index=False)



if __name__ == '__main__':
    from tqdm import tqdm
    START = 7000
    END = 8000
    for i in range(START, END):
        try:
            index = num_list[i]
            Run_Single_Index(index, gdf_geb10, gdf_geb25, plot=False)
        except Exception as e:
            print(i)
            print(e)


    # for index in tqdm(num_list):
    #     try:
    #         Run_Single_Index(index, gdf_geb10, gdf_geb25, plot=False)
    #     except Exception as e:
    #         print(index)
    #         print(e)





