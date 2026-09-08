#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import time

import numpy as np
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils
from unified_engine.test_engine import TestEngine
from utils import PointLoss_test
from utils import distance_squre
import data_utils as d_utils
import ModelNet40Loader
import shapenet_part_loader
from model_PFNet import _netlocalD,_netG,light_netG
from imodules import *

class PFNetTestEngine(TestEngine):
    def test(self):
        device=self.device
        opt=self.opt
        point_netG = light_netG(opt.num_scales, opt.each_scales_size, opt.point_scales_list, opt.crop_point_num)
        point_netG = torch.nn.DataParallel(point_netG)
        point_netG.to(device)
        point_netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
        point_netG.eval()

        criterion_PointLoss = PointLoss_test().to(device)

        input_cropped1 = torch.FloatTensor(opt.batchSize, 1, opt.pnum, 3)
        errG_min = 100
        n = 0
        CD = 0
        Gt_Pre = 0
        Pre_Gt = 0
        IDX = 1
        for i, data in enumerate(self.test_dataloader, 0):
            label, real_point, input_cropped1, patch=self.process_data(data)
            print("realpointshape", real_point.shape)

            input_cropped2_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[1], RAN=True)
            input_cropped2 = utils.index_points(input_cropped1, input_cropped2_idx)
            input_cropped3_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[2], RAN=False)
            input_cropped3 = utils.index_points(input_cropped1, input_cropped3_idx)
            input_cropped2 = input_cropped2.to(device)
            input_cropped3 = input_cropped3.to(device)
            input_cropped = [input_cropped1, input_cropped2, input_cropped3]

            #    fake,fake_part = point_netG(input_cropped)
            fake_center1, fake_center2, fake = point_netG(input_cropped)
            fake_whole = torch.cat((input_cropped_partial, fake), 1)
            fake_whole = fake_whole.to(device)
            real_point = real_point.to(device)
            real_center = real_center.to(device)
            dist_all, dist1, dist2 = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center,
                                                                                               1))  # +0.1*criterion_PointLoss(torch.squeeze(fake_part,1),torch.squeeze(real_center,1))
            dist_all = dist_all.cpu().detach().numpy()
            dist1 = dist1.cpu().detach().numpy()
            dist2 = dist2.cpu().detach().numpy()
            CD = CD + dist_all / length
            Gt_Pre = Gt_Pre + dist1 / length
            Pre_Gt = Pre_Gt + dist2 / length
            print(CD, Gt_Pre, Pre_Gt)
        print(CD, Gt_Pre, Pre_Gt)
        print("CD:{} , Gt_Pre:{} , Pre_Gt:{}".format(float(CD), float(Gt_Pre), float(Pre_Gt)))
        print(length)
        end_time = time.time()
        print(end_time - start_time)


    def distance_squre1(p1,p2):
        return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2




