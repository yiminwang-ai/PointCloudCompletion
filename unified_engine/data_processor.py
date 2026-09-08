import random
import numpy as np
import torch


def find_nearest_index(point_cloud, point):
    distances = np.linalg.norm(point_cloud - np.array(point), axis=1)
    return np.argmin(distances)


def generate_normal_gaps(real_point, batch_size, pnum, crop_point_num, is_simple=False):
    input_cropped = torch.FloatTensor(batch_size, pnum, 3)
    input_cropped = input_cropped.data.copy_(real_point)  # 48,4096,3
    patch = torch.FloatTensor(batch_size, crop_point_num, 3)
    # Set viewpoints
    choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
              torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]
    # 存储每个pc的被裁剪的点的位置
    all_cropped_indices = []
    for m in range(batch_size):
        if not is_simple: choice = real_point[m]
        index = random.sample(choice, 1)  # Random choose one of the viewpoint
        p_center = index[0]
        # 生成不规则的边缘点，确保不重复选择
        gap_indices = generate_normal_gap_shape(p_center, real_point[m], crop_point_num)

        # 将input_cropped对应点设置为零
        input_cropped.data[m, gap_indices] = torch.zeros(crop_point_num, 3,
                                                         dtype=input_cropped.dtype)
        # patch是残缺的部分，input_cropped是被裁剪后的部分
        patch.data[m] = real_point[m, gap_indices]
        all_cropped_indices.append(gap_indices)
    return input_cropped, patch, all_cropped_indices


def generate_normal_gap_shape(center, point_cloud, crop_point_num):
    # 计算每个点与p_center之间的距离
    # 使用向量化操作计算距离
    distances = torch.sum((point_cloud - center) ** 2, dim=1)
    distance_list = distances.tolist()
    # (pnum,2)
    distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])
    gap_indices = [index for index, value in distance_order[:crop_point_num]]
    return gap_indices


def process_data(data, opt,  second_nearest_points_len: int = 0):
    global sencond_points
    # 目的是输入原始数据，得到残缺的数据
    real_point, target = data  # 48,4096,3
    print(real_point.shape)
    batch_size = real_point.size()[0]  # 48
    patch = torch.FloatTensor(batch_size, opt.crop_point_num, 3)
    if second_nearest_points_len >0:
        sencond_points = torch.FloatTensor(batch_size, second_nearest_points_len, 3)
        print(sencond_points.shape)
    input_cropped = torch.FloatTensor(batch_size, opt.pnum, 3)
    input_cropped = input_cropped.data.copy_(real_point)  # 48,4096,3

    # Set viewpoints
    choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
              torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]
    # 存储每个pc的被裁剪的点的位置
    all_cropped_indices = []
    for m in range(batch_size):
        index = random.sample(choice, 1)  # Random choose one of the viewpoint
        p_center = index[0]
        # 计算每个点与p_center之间的距离
        # 使用向量化操作计算距离
        distances = torch.sum((real_point[m] - p_center) ** 2, dim=1)
        distance_list = distances.tolist()
        # (pnum,2)
        distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])

        # 获取最近的点的索引,(crop_point_num),dim=1
        nearest_indices = [distance_order[sp][0] for sp in range(opt.crop_point_num)]
        second_nearest_indices = [distance_order[sp][0] for sp in
                                  range(opt.crop_point_num, opt.crop_point_num + second_nearest_points_len)]
        all_cropped_indices.append(nearest_indices)
        # 将input_cropped对应点设置为零
        input_cropped.data[m, nearest_indices] = torch.zeros(opt.crop_point_num, 3,
                                                             dtype=input_cropped.dtype)
        # patch是残缺的部分，input_cropped是被裁剪后的部分
        patch.data[m] = real_point[m, nearest_indices]
        if second_nearest_points_len>0:sencond_points.data[m] = real_point[m, second_nearest_indices]
    all_cropped_indices = torch.tensor(all_cropped_indices)
    # real_point (batch,pnum,3) 真实的点 gt
    # input_cropped (batch,pnum,3) 被裁剪后的部分,裁去部分用0代替
    # patch (batch,crop_num,3) 残缺的部分，补丁
    # second_points 是最接近被裁剪掉部分的点
    # real_point=input_cropped与patch的结合
    if second_nearest_points_len>0:
        return real_point, input_cropped, patch, sencond_points, all_cropped_indices
    else:
        return real_point, input_cropped, patch, all_cropped_indices


def split_process_data_optimized(data, opt, k2=256, device="cuda"):
    real_point, _ = data  # [B, N, 3]
    B, N, _ = real_point.shape

    # 1. 随机选择视点(向量化)
    choices = torch.tensor([
        [1, 0, 0], [0, 0, 1], [1, 0, 1], [-1, 0, 0], [-1, 1, 0]
    ], device=real_point.device, dtype=real_point.dtype)

    rand_idx = torch.randint(0, len(choices), (B,))
    p_centers = choices[rand_idx]  # [B, 3]

    # 2. 向量化距离计算
    diff = real_point - p_centers.view(B, 1, 3)
    distances = torch.sum(diff ** 2, dim=-1)  # [B, N]

    # 3. 使用topk代替排序
    k1 = opt.crop_point_num
    _, topk_indices = torch.topk(distances, k=k1 + k2, largest=False)  # [B, k1+k2]

    nearest_indices = topk_indices[:, :k1]  # [B, k1]
    edge_indices = topk_indices[:, k1:k1 + k2]  # [B, k2]

    # 4. 创建标签(向量化)
    labels = torch.zeros((B, N),
                         dtype=torch.float32,
                         device=real_point.device)

    # 设置边缘点标签为1
    batch_indices = torch.arange(B).view(B, 1)
    labels[batch_indices, edge_indices] = 1

    # 5. 生成裁剪后输入
    input_cropped = real_point.clone()
    input_cropped[batch_indices, nearest_indices] = 0

    # 6. 移除未使用的变量
    return labels, input_cropped, nearest_indices, edge_indices


from sklearn.neighbors import KDTree


def identify_sparse_regions(point_cloud, k=10, density_threshold=0.5):
    """
    识别稀疏区域。

    :param point_cloud: 原始点云, numpy array shape (N, 3)
    :param k: 最近邻个数，用于密度估计
    :param density_threshold: 用于判断是否稀疏的密度阈值
    :return: 稀疏点的索引
    """
    tree = KDTree(point_cloud)
    sparse_indices = []

    # 计算每个点的最近 k 个邻居
    for i in range(len(point_cloud)):
        distances, indices = tree.query(point_cloud[i].reshape(1, -1), k=k + 1)  # k + 1 to include the point itself
        density = np.sum(distances[0][1:])  # 排除自身
        if density > density_threshold:  # 如果总距离大于阈值，则认为是稀疏
            sparse_indices.append(i)

    return sparse_indices


def find_nearest_points_kdtree(point_cloud, n):
    """
    使用 KDTree 找到稀疏区域的最近 n 个点。

    :param point_cloud: 原始点云, numpy array shape (N, 3)
    :param n: 要找到的最近点的数量
    :return: 稀疏点和它们最近的 n 个点的索引的列表
    """
    tree = KDTree(point_cloud)
    nearest_points_index = []
    sparse_indices = identify_sparse_regions(point_cloud)

    # 使用集合来确保唯一性
    unique_nearest_indices = set()

    for index in sparse_indices:
        # 查询最近的 n + 1 个点
        distances, nearest_indices = tree.query(point_cloud[index].reshape(1, -1), k=n + 1)
        nearest_indices = nearest_indices[nearest_indices != index]  # 排除自身

        # 添加到 unique_nearest_indices 中，只有在其长度小于 n 时才添加
        for ni in nearest_indices:
            if len(unique_nearest_indices) < n:
                unique_nearest_indices.add(ni)
            else:
                break  # 如果已经满足 n，则可以提前退出

    nearest_points_index = list(unique_nearest_indices)[:n]  # 确保返回的长度为 n
    return nearest_points_index
