import os
import matplotlib
matplotlib.use('Agg')

import multiprocessing


from DataProcess.data_utils import *
from DataProcess.processing_utilis import *




Nums = 1



save_branch = r"D:/dresden/mapgeneralization/dataset/nodeandadj"
# save_branch = r"D:/dresden/mapgeneralization/dataset/savefolder1"

gdf_geb10, gdf_geb25 = load_data()
one_to_many_details = bulit_connection(gdf_geb10, gdf_geb25)
bounding_boxes = creat_bounding_box(one_to_many_details, gdf_geb25)
num_list = one_to_many_details["JOINID_geb25"].values.reshape(-1)


def Run_Single_Time_v2(index):
    index = num_list[index]
    if not os.path.exists(os.path.join(save_branch, "index" + str(index))):
        os.makedirs(os.path.join(save_branch, "index" + str(index)))
    save_folder = os.path.join(save_branch, "index" + str(index))
    data_selection_v2(one_to_many_details, bounding_boxes, gdf_geb10,
                      gdf_geb25, index, save_folder=save_folder)

    # creat_Ajc(gdf_geb10, target_joinids_geb10, gdf_geb25, target_joinids_geb25,
    #           save_folder=save_folder)



# if __name__ == '__main__':
#     START = 7870
#     END = 7875
#     # multiprocessing.freeze_support()
#     # pool = multiprocessing.Pool(processes=8)
#     for index in range(START, END):
#         Run_Single_Time_v2(index)
#         print(f"处理索引 {index} 完成")
#     print("所有运行结束")
if __name__ == '__main__':
    # 设置开始和结束的索引
    start_index = 7800  # 包括这个索引
    end_index = 7900   # 包括这个索引

    for index in range(start_index, end_index + 1):
        Run_Single_Time_v2(index)
        print(f"处理索引 {index} 完成")
    print("所有运行结束")
        # pool.apply_async(Run_Single_Time_v2, args=(index,))
    # pool.close()
    # pool.join()

