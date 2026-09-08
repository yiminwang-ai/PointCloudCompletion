import argparse
import os
import re
import random

import numpy as np
import torch
import open3d as o3d

from split.split import  extract_border_from_partial
from unified_engine.data_processor import generate_normal_gap_shape, find_nearest_points_kdtree, \
    split_process_data_optimized

complete_data_source="shape_data"

#1.遍历shape_data，找到目前pts文件。
#2.为其找到合适的储存路径，
#3.设计随机裁剪
print(os.listdir(complete_data_source))
parser = argparse.ArgumentParser()
parser.add_argument('--pnum', type=int, default=4096, help='the point number of a sample')
parser.add_argument('--crop_point_num', type=int, default=512, help='0 means do not use else use with this weight')
opt = parser.parse_args()
def main():

    for file in os.listdir(complete_data_source):
        if re.match(r"\d{8}",file):
            complete_dir=os.path.join(complete_data_source,file,"points") #shape_data\02691156\points
            patch_pts_root = complete_dir.replace("points", "patch_points")

            dict = {"complete": [], "patch": []}

            for pts in os.listdir(complete_dir):
                complete_pts_path=os.path.join(complete_dir,pts)

                # dict["complete"].append(complete_pts_path)
                # dict["patch"].append(patch_pts_path)

                patch_pc=make_patch(complete_pts_path)

                if not os.path.exists(patch_pts_root):
                    os.makedirs(patch_pts_root, exist_ok=True)
                new_file_path = os.path.join(patch_pts_root, pts)
                print(new_file_path)
                with open(new_file_path, 'w') as f:
                    for points in patch_pc:
                        f.write(f"{points[0]} {points[1]} {points[2]}\n")

        break

def make_patch(file_path:str):
    real_point=np.loadtxt(file_path).astype(np.float32)
    # real_point=random_downsample(real_point)
    normalized_point=pc_normalize(real_point)

    real_point=torch.Tensor(real_point)
    normalized_point = torch.Tensor(normalized_point)

    # choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
    #               torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]
    # index = random.sample(choice, 1)  # Random choose one of the viewpoint
    # p_center = index[0]
    # # 计算每个点与p_center之间的距离
    # # 使用向量化操作计算距离
    # distances = torch.sum((normalized_point - p_center) ** 2, dim=1)
    # distance_list = distances.tolist()
    # 随机选择一个中心点
    random_center_index = np.random.choice(normalized_point.shape[0])
    p_center = normalized_point[random_center_index]
    normalized_point=normalized_point.unsqueeze(0)
    # 生成不规则的边缘点，确保不重复选择
    #gap_indices = generate_normal_gap_shape(p_center, normalized_point, opt.crop_point_num)
    labels, input_cropped, nearest_indices, edge_indices=split_process_data_optimized((normalized_point,None),opt)


    # 创建一个布尔索引，标识哪些点是需要保留的
    mask = torch.ones(real_point.shape[0], dtype=torch.bool)
    mask[nearest_indices] = False  # 将待丢弃的索引位置设为 False
    # 通过布尔索引保留需要的点
    input_cropped = real_point[mask]
    print(input_cropped.shape)
    #second_indices = find_nearest_points_kdtree(input_cropped,512)
    second_indices= edge_indices
    print(len(second_indices))
    colors = np.zeros((real_point.shape[0], 3))  # 创建颜色数组

    # 为部分点赋予不同的颜色
    # 例如，我们将前100个点染成红色
    colors[:] = [0, 1, 0]
    colors[nearest_indices] = [1, 0, 0]  # red
    colors[second_indices] = [0, 0, 1]  # blue
    print('s',input_cropped.shape)


    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(real_point)
    print(colors.shape)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # 可视化点云
    o3d.visualization.draw_geometries([pcd], window_name='Colored Point Cloud')
    # 将 corresponding points in input_cropped 设置为零，形成裁剪
    #input_cropped[gap_indices] = torch.zeros(opt.crop_point_num, 3, dtype=input_cropped.dtype)


    print('input_cropped',input_cropped.shape)
    return input_cropped
    # # 存储最近的点到real_center
    # real_center.data[m, 0, :] = real_point[m, 0, nearest_indices]

def random_downsample(point_cloud):
    choice=np.random.choice(len(point_cloud),opt.pnum,replace=True)
    return point_cloud[choice,:]


def pc_normalize(pc):
    """ pc: NxC, return NxC """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

if __name__=="__main__":
    make_patch(r'D:\Training\PointCloudCompletion\shape_data\02691156\points\baoyingchen_u_label_7.pts')