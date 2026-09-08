import sys, os



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
import random

import shapenet_part_loader
from utils import PointLoss
from unified_engine import data_processor
from unified_engine.chamfer_distance import ChamferDistanceLoss
from opt import *


class TrainEngine:
    def __init__(self, train_dataloader: DataLoader, val_dataloader: DataLoader, opt=None):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.opt = opt

        if opt is None: get_opt()
        # important
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._cd_loss = PointLoss().to(self.device)

    def train(self):
        pass

    def process_data(self, data):
        global all_cropped_indices, patch, input_cropped
        real_label = 1
        fake_label = 0

        # 目的是输入原始数据，得到残缺的数据
        real_point, target = data  #48,4096,3
        batch_size = real_point.size()[0]  #48
        label = torch.FloatTensor(batch_size)

        if self.opt.cropmethod=="random_center":
            input_cropped,patch,all_cropped_indices=data_processor.process_data(data=data,opt=self.opt)
        else :
            print('cropmthod not find')

        label.resize_([batch_size, 1]).fill_(real_label)
        real_point = real_point.to(self.device)
        patch = patch.to(self.device)
        input_cropped = input_cropped.to(self.device)
        label = label.to(self.device)
        # label (batch,1) 全部为1的填充
        # real_point (batch,pnum,3) 真实的点 gt
        # input_cropped (batch,pnum,3) 被裁剪后的部分,裁去部分用0代替
        # patch (batch,crop_num,3) 残缺的部分，补丁
        # real_point=input_cropped与patch的结合

        return label, real_point, input_cropped, patch,all_cropped_indices

    def process_pgl_data(self, data,second_points_num):
        global all_cropped_indices, patch, input_cropped,sencond_points
        real_label = 1
        fake_label = 0

        # 目的是输入原始数据，得到残缺的数据
        real_point, target = data  # 48,4096,3
        batch_size = real_point.size()[0]  # 48
        label = torch.FloatTensor(batch_size)

        if self.opt.cropmethod == "random_center":
            real_point, input_cropped, patch, sencond_points, all_cropped_indices = data_processor.process_data(
                data=data,opt=opt,second_nearest_points_len=second_points_num)
        else:
            print('cropmthod not find')

        label.resize_([batch_size, 1]).fill_(real_label)
        real_point = real_point.to(self.device)
        patch = patch.to(self.device)
        input_cropped = input_cropped.to(self.device)
        label = label.to(self.device)
        # label (batch,1) 全部为1的填充
        # real_point (batch,pnum,3) 真实的点 gt
        # input_cropped (batch,pnum,3) 被裁剪后的部分,裁去部分用0代替
        # patch (batch,crop_num,3) 残缺的部分，补丁
        # real_point=input_cropped与patch的结合

        return label, real_point, input_cropped, patch, sencond_points, all_cropped_indices
    def cd_cuda_loss(self, pred, gt):
        return ChamferDistanceLoss()(pred, gt)
    def cd_loss(self, pred, gt):
        return self._cd_loss(pred,gt)
    def save(self,model_dict,model_folder,name):
        path=self.opt.path
        torch.save(model_dict,
                   f'{path}/{model_folder}/{name}.pth')


if __name__ == '__main__':
    from architecture.GRNet import grnet_engine
    from architecture.PFNet import pf_net_engine
    from architecture.PGLNet import PGLNet_engine
    from architecture.IOSPfnet import ios_pf_net_engine

    # from architecture.PCN import pcn_engine
    # from architecture.TopNet.TopNet import TopNet
    # from unified_engine.general_engine import GeneralTrainEngine

    opt = get_opt()
    dset = shapenet_part_loader.PartDataset(root='../shape_data/', classification=True, class_choice=None,
                                            npoints=opt.pnum, split='train')
    assert dset
    dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    val_dset = shapenet_part_loader.PartDataset(root='../shape_data/', classification=True, class_choice=None,
                                                npoints=opt.pnum, split='val')
    val_dataloader = torch.utils.data.DataLoader(val_dset, opt.batchSize,
                                                 shuffle=True, num_workers=int(opt.workers))
    #runner = pf_net_engine.PFNetEngine(dataloader, val_dataloader)
    #runner = GeneralTrainEngine(TopNet(num_pred=opt.pnum),train_dataloader=dataloader,val_dataloader=val_dataloader)
    runner = ios_pf_net_engine.IOSPFNetEngine(dataloader, val_dataloader)

    runner.opt = opt
    runner.train()
