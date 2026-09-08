import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_engine.data_processor import split_process_data_optimized
from pointnet2_sem_seg_msg import PointNetClassifier
from unified_engine import shapenet_part_loader, data_processor
from unified_engine.opt import get_opt


def train_optimized(model, dataloader, epochs=200, lr=0.002):
    device = torch.device("cuda:0")
    model.to(device).train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # # 假设类别 0 的样本数量为 N0，类别 1 的样本数量为 N1
    # N0 = opt.pnum  # 类别 0 的样本数量
    # N1 = opt.crop_point_num  # 类别 1 的样本数量
    #
    # # 计算类别权重
    # class_weights = torch.tensor([N1 / (N0 + N1), N0 / (N0 + N1)])  # 理论上让类别 1 的权重更高
    # print(class_weights)
    criterion = nn.CrossEntropyLoss().to(device)  # 损失函数也需在GPU

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, data in enumerate(dataloader):
            label, input_cropped, nearest_indices, edge_indices = split_process_data_optimized(data, opt, device=device)
            optimizer.zero_grad()  # 清除梯度
            input_cropped = input_cropped.to(device)

            output = model(input_cropped).float()  # 确保输出为 Float
            probabilities = F.softmax(output, dim=-1)  # 计算每个点属于 0 或 1 的概率

            # 确保标签是 Long 类型
            label = label.long()
            label = label.to(device)  # [B,N,2]
            loss = criterion(output.view(-1, 2), label.view(-1))  # 计算损失


            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 每10个批次输出一次信息
            if (batch_idx + 1) % 20 == 0:
                # 获取预测类
                predicted_classes = torch.argmax(probabilities, dim=-1)
                # 设置打印选项以只显示前5和后5个元素
                torch.set_printoptions(threshold=20, edgeitems=10)
                print(label[0],predicted_classes[0])
        print('loss',running_loss)
    torch.save(model.state_dict(), 'model.pth')
if __name__ == '__main__':
    opt = get_opt()
    dset = shapenet_part_loader.PartDataset(root='../shape_data/', classification=True, class_choice=None,
                                            npoints=opt.pnum, split='train')
    assert dset
    dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    val_dset = shapenet_part_loader.PartDataset(root='../shape_data/', classification=True, class_choice=None,
                                                npoints=opt.pnum, split='val')
    val_dataloader = torch.utils.data.DataLoader(val_dset, batch_size=opt.batchSize,
                                                 shuffle=True, num_workers=int(opt.workers))

    # 2. 创建模型
    input_size = 4096  # 根据你的任务
    hidden_size = 64  # 云点坐标的维度（x, y, z）
    model = PointNetClassifier()

    # 3. 训练模型
    train_optimized(model, dataloader)

    # 4. 测试模型
    #test_model(model, dataloader)
