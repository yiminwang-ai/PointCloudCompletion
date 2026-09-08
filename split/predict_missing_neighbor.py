import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import shapenet_part_loader
from opt import get_opt

class DistanceSqureLoss(nn.Module):
    def __init__(self):
        super(DistanceSqureLoss, self).__init__()

    def forward(self, p1, p2):
        tensor = p1 - p2
        val = torch.sum(tensor * tensor, dim=1)
        return torch.mean(val)

class PointMissingNeighborNetwork(nn.Module):
    def __init__(self, pnum):
        super(PointMissingNeighborNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 3))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=1)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 使用自适应平均池化
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def pgl_process_data(opt, data, device, second_nearest_points_len):
    real_point, target = data
    batch_size = real_point.size()[0]
    label = torch.FloatTensor(batch_size).fill_(1)
    patch = torch.FloatTensor(batch_size, opt.crop_point_num, 3)
    second_points = torch.FloatTensor(batch_size, second_nearest_points_len, 3)
    input_cropped = torch.FloatTensor(batch_size, opt.pnum, 3).copy_(real_point)
    p_centers = []

    for m in range(batch_size):
        index = np.random.choice(real_point.shape[1])
        p_center = real_point[m][index]
        p_centers.append(p_center)
        distances = torch.sum((real_point[m] - p_center) ** 2, dim=1)
        distance_list = distances.tolist()
        distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])
        nearest_indices = [distance_order[sp][0] for sp in range(opt.crop_point_num)]
        second_nearest_indices = [distance_order[sp][0] for sp in range(opt.crop_point_num, opt.crop_point_num + second_nearest_points_len)]
        input_cropped.data[m, nearest_indices] = torch.mean(real_point[m], dim=0)  # 用均值填充
        patch.data[m] = real_point[m, nearest_indices]
        second_points.data[m] = real_point[m, second_nearest_indices]

    label = label.to(device)
    real_point = real_point.to(device)
    patch = patch.to(device)
    input_cropped = input_cropped.to(device)
    p_centers_tensor = torch.stack(p_centers).to(device)
    second_points = second_points.to(device)
    return label, real_point, input_cropped, patch, p_centers_tensor, second_points, nearest_indices

def train():
    opt = get_opt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dset = shapenet_part_loader.PartDataset(root='../shape_data/', classification=True, class_choice=None, npoints=opt.pnum, split='train')
    dataloader = DataLoader(dset, batch_size=32, shuffle=True, num_workers=int(opt.workers))
    val_dset = shapenet_part_loader.PartDataset(root='../shape_data/', classification=True, class_choice=None, npoints=opt.pnum, split='val')
    val_dataloader = DataLoader(val_dset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

    model = PointMissingNeighborNetwork(pnum=opt.pnum).to(device)
    criterion = nn.SmoothL1Loss().to(device)  # 使用更鲁棒的损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(100):
        model.train()
        train_loss = 0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            label, real_point, input_cropped, real_center, p_centers, gt_second_points, _ = pgl_process_data(
                opt, data, device, second_nearest_points_len=256)
            input_cropped = Variable(input_cropped, requires_grad=True)
            p_centers = Variable(p_centers, requires_grad=True)
            output_centers = model(input_cropped)
            loss = criterion(output_centers, p_centers)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            if i % 10 == 0:
                print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}, Pred: {output_centers[0]}, GT: {p_centers[0]}')

        print(f'Epoch {epoch}, Train Loss: {train_loss / len(dataloader):.4f}')
        scheduler.step(train_loss / len(dataloader))

        # 验证集评估
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_dataloader:
                label, real_point, input_cropped, real_center, p_centers, gt_second_points, _ = pgl_process_data(
                    opt, data, device, second_nearest_points_len=256)
                output_centers = model(input_cropped)
                val_loss += criterion(output_centers, p_centers).item()
        print(f'Epoch {epoch}, Val Loss: {val_loss / len(val_dataloader):.4f}')
    torch.save(model.state_dict(),'center_point.pth')
if __name__ == '__main__':
    train()
