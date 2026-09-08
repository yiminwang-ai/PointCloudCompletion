#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import torch

def array2samples_distance(array1, array2):
    #  compute number of points and features
    num_point1, num_features1 = array1.shape
    num_point2, num_features2 = array2.shape

    # compute distance between each point in array2 to all points in array1
    distances = torch.cdist(array2, array1)  # Efficiently computes pairwise distances
    min_distances = torch.min(distances, dim=1)[0]  # (num_point2)

    return torch.mean(min_distances)  # Return the mean of minimum distances

def chamfer_distance_numpy(array1, array2):
    batch_size, num_point, num_features = array1.shape

    total_dist = 0  # Initialize total distance

    # Loop through each sample in the batch
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])  # Distance from array1[i] to array2[i]
        av_dist2 = array2samples_distance(array2[i], array1[i])  # Distance from array2[i] to array1[i]
        total_dist += av_dist1 + av_dist2  # Accumulate the distances

    chamfer_distance = total_dist / batch_size  # Average over the batch
    return chamfer_distance * 100  # Scale the distance if necessary

# def array2samples_distance(array1, array2):
#     """
#     arguments:
#         array1: the array, size: (num_point, num_feature)
#         array2: the samples, size: (num_point, num_feature)
#     returns:
#         distances: each entry is the distance from a sample to array1
#     """
#     num_point1, num_features1 = array1.shape
#     num_point2, num_features2 = array2.shape
#     expanded_array1 = array1.repeat(num_point2, 1)
#     expanded_array2 = torch.reshape(
#         torch.unsqueeze(array2, 1).repeat(1, num_point1, 1),
#         (-1, num_features2))
#     distances = (expanded_array1 - expanded_array2) * (expanded_array1 - expanded_array2)
#     #    distances = torch.sqrt(distances)
#     distances = torch.sum(distances, dim=1)
#     distances = torch.reshape(distances, (num_point2, num_point1))
#     distances = torch.min(distances, dim=1)[0]
#     distances = torch.mean(distances)
#     return distances
#
#
# def chamfer_distance_numpy(array1, array2):
#     batch_size, num_point, num_features = array1.shape
#     dist = 0
#     for i in range(batch_size):
#         av_dist1 = array2samples_distance(array1[i], array2[i])
#         av_dist2 = array2samples_distance(array2[i], array1[i])
#         dist = dist + (0.5 * av_dist1 + 0.5 * av_dist2) / batch_size
#     return dist * 100


def chamfer_distance_numpy_test(array1, array2):
    batch_size, num_point, num_features = array1.shape
    dist_all = 0
    dist1 = 0
    dist2 = 0
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
        dist_all = dist_all + (av_dist1 + av_dist2) / batch_size
        dist1 = dist1 + av_dist1 / batch_size
        dist2 = dist2 + av_dist2 / batch_size
    return dist_all, dist1, dist2


class PointLoss(nn.Module):
    def __init__(self):
        super(PointLoss, self).__init__()

    def forward(self, array1, array2):
        return chamfer_distance_numpy(array1, array2)


class PointLoss_test(nn.Module):
    def __init__(self):
        super(PointLoss_test, self).__init__()

    def forward(self, array1, array2):
        return chamfer_distance_numpy_test(array1, array2)

class DistanceSqureLoss(nn.Module):
    def __init__(self):
        super(DistanceSqureLoss, self).__init__()

    def forward(self, p1, p2):
        return distance_squre(p1, p2)

def distance_squre(p1, p2):
    tensor = p1 - p2
    val = tensor.mul(tensor)
    val = val.sum()
    return val


def distance_squre_easy(p1, p2):
    return torch.sum((p1 - p2) ** 2)  # 假设 p1 和 p2 已经是张量类型

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint, RAN=True):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if RAN:
        farthest = torch.randint(0, 1, (B,), dtype=torch.long).to(device)
    else:
        farthest = torch.randint(1, 2, (B,), dtype=torch.long).to(device)

    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


if __name__ == '__main__':
    a = torch.randn(64, 256, 3)
    b = torch.randn(64, 256, 3)
    c = torch.randn(64, 64, 3)
    d = torch.randn(64, 64, 3)
    p = PointLoss()
    print(p(a, b))
    print(p(c, d) * 0.25)
