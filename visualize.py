import open3d as o3d

input_path=r"D:\Training\New Oral Scanner\OrthoCAD_Export_91411131\91411131_lprofile_teethup_l.ply"

import numpy as np

# 读取 PLY 文件
ply_file = input_path
point_cloud = o3d.io.read_point_cloud(ply_file)

# 提取 XYZ 和 RGB 信息
points = np.asarray(point_cloud.points)  # 获取 XYZ 信息
colors = np.asarray(point_cloud.colors)   # 获取 RGB 信息

# 确保 RGB 在 0-255 区间
colors = (colors * 255).astype(np.uint8)

# 生成 PCD 格式的数据
with open("output_file.pcd", "w") as pcd_file:
    # 写入每个点的xyz和rgb值
    for i in range(len(points)):
        x, y, z = points[i]
        r, g, b = colors[i]
        rgb = (r << 16) | (g << 8) | b  # 将 RGB 转换为单一整数
        pcd_file.write(f"{x} {y} {z} {rgb}\n")
