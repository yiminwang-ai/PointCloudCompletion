import sys, os

from unified_engine.chamfer_distance import ChamferDistanceLoss

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
import random

import shapenet_part_loader

from architecture.PCN import pcn_engine

from opt import get_test_opt


class TestEngine:
    def __init__(self,model_path, test_dataloader: DataLoader, opt=None):
        self.test_dataloader = test_dataloader
        self.opt = opt
        self.model_path=model_path
        if opt is None: get_test_opt()
        # important
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test(self):
        pass

    def process_data(self, data):
        real_label = 1
        fake_label = 0
        label = torch.FloatTensor(self.opt.batchSize)
        # 目的是输入原始数据，得到残缺的数据
        real_point, target = data  #48,4096,3

        batch_size = real_point.size()[0]  #48
        patch = torch.FloatTensor(batch_size, 1, self.opt.crop_point_num, 3)
        input_cropped = torch.FloatTensor(batch_size, self.opt.pnum, 3)
        input_cropped = input_cropped.data.copy_(real_point)  #48,4096,3

        # Set viewpoints
        choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
                  torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]
        for m in range(batch_size):
            index = random.sample(choice, 1)  # Random choose one of the viewpoint
            p_center = index[0]
            # 计算每个点与p_center之间的距离
            # 使用向量化操作计算距离
            distances = torch.sum((real_point[m] - p_center) ** 2, dim=1)
            distance_list = distances.tolist()

            distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])

            # 获取最近的点的索引
            nearest_indices = [distance_order[sp][0] for sp in range(self.opt.crop_point_num)]
            # 将input_cropped1对应点设置为零
            input_cropped.data[m, nearest_indices] = torch.zeros(self.opt.crop_point_num, 3,
                                                                 dtype=input_cropped.dtype)
            # patch是残缺的部分，input_cropped是被裁剪后的部分
            patch.data[m] = real_point[m, nearest_indices]

        label.resize_([batch_size, 1]).fill_(real_label)
        real_point = real_point.to(self.device)
        patch = patch.to(self.device)
        input_cropped1 = input_cropped.to(self.device)
        label = label.to(self.device)

        return label, real_point, input_cropped1, patch
    def cd_loss(self, pred, gt):
        return ChamferDistanceLoss()(pred, gt)

if __name__ == '__main__':

    opt = get_test_opt()

    test_dset = shapenet_part_loader.PartDataset(root='../shape_data/', classification=True, class_choice=None,
                                                npoints=opt.pnum, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchsize,
                                                 shuffle=True, num_workers=int(opt.workers))
    runner = pcn_engine.PCNTestEngine('',test_dataloader)
    runner.opt = opt
    runner.test()
