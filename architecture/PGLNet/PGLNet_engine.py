import os
import random
from datetime import datetime

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm

import architecture.PFNet.data_utils as d_utils
from architecture.PGLNet.PGLNet import PGLNetG
from architecture.PFNet.model_PFNet import light_netG, _netlocalD, _netG
from unified_engine import utils

from unified_engine.engine import TrainEngine
from unified_engine.utils import PointLoss


class PGLNetEngine(TrainEngine):
    def train(self):

        opt = self.opt
        print(opt)

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        USE_CUDA = True
        device = self.device

        point_netG = PGLNetG(opt.each_scales_size, [opt.point_scales_list[0],opt.crop_point_num], opt.crop_point_num)
        point_netD = _netlocalD(opt.crop_point_num)

        cudnn.benchmark = True
        resume_epoch = 0

        # 获取当前时间并格式化
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # 创建完整的文件名
        file_first_name = opt.save_train_result
        file_name_with_time = f"{file_first_name}_{current_time}.txt"

        def weights_init_normal(m):
            classname = m.__class__.__name__
            if classname.find("Conv2d") != -1:
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find("Conv1d") != -1:
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm2d") != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find("BatchNorm1d") != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

        if USE_CUDA:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            point_netG = torch.nn.DataParallel(point_netG)
            point_netD = torch.nn.DataParallel(point_netD)
            point_netG.to(device)
            point_netG.apply(weights_init_normal)
            point_netD.to(device)
            point_netD.apply(weights_init_normal)
        if opt.netG != '':
            point_netG.load_state_dict(
                torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
            resume_epoch = torch.load(opt.netG)['epoch']
        if opt.netD != '':
            point_netD.load_state_dict(
                torch.load(opt.netD, map_location=lambda storage, location: storage)['state_dict'])
            resume_epoch = torch.load(opt.netD)['epoch']

        if opt.manualSeed is None:
            opt.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", opt.manualSeed)
        random.seed(opt.manualSeed)
        torch.manual_seed(opt.manualSeed)
        if opt.cuda:
            torch.cuda.manual_seed_all(opt.manualSeed)

        data_transforms = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
            ]
        )

        dataloader = self.train_dataloader

        test_dataloader = self.val_dataloader

        print(point_netG)
        print(point_netD)

        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        criterion_PointLoss = PointLoss().to(device)

        # setup optimizer
        optimizerD = torch.optim.Adam(point_netD.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-05,
                                      weight_decay=opt.weight_decay)
        optimizerG = torch.optim.Adam(point_netG.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-05,
                                      weight_decay=opt.weight_decay)
        schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=40, gamma=0.2)
        schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.2)

        real_label = 1
        fake_label = 0

        ###########################
        #  G-NET and T-NET
        ##########################

        if opt.D_choose == 1:
            for epoch in range(resume_epoch, opt.epoch):
                f = open(file_name_with_time, 'a')
                if epoch < 30:
                    alpha1 = 0.01
                    alpha2 = 0.02
                elif epoch < 80:
                    alpha1 = 0.05
                    alpha2 = 0.1
                else:
                    alpha1 = 0.1
                    alpha2 = 0.2
                count = 0
                point_netG = point_netG.train()
                point_netD = point_netD.train()
                for i, data in enumerate(dataloader, 0):
                    count += 1
                    label, real_point, input_cropped, real_center,second_points, all_cropped_indices = self.process_data(data,)
                    #print('second_points',second_points.shape)
                    ############################
                    # (1) data prepare
                    ###########################
                    real_center = Variable(real_center, requires_grad=True)
                    second_points =Variable(second_points, requires_grad=True)
                    input_cropped = Variable(input_cropped, requires_grad=True)


                    # 形成输入
                    input = [input_cropped, second_points]

                    ############################
                    # (2) Update D network
                    # 训练D network 的目的是让D能够判别真伪点
                    ###########################
                    point_netD.zero_grad()
                    real_center = torch.unsqueeze(real_center, 1)
                    # 训练告诉他，这是真实的值
                    output = point_netD(real_center)
                    errD_real = criterion(output, label)
                    errD_real.backward()
                    # 训练告诉他，这是伪造的值
                    # 最核心，生成点
                    fake_center1, fake_center2, fake = point_netG(input)

                    fake = torch.unsqueeze(fake, 1)
                    label.data.fill_(fake_label)
                    output = point_netD(fake.detach())
                    errD_fake = criterion(output, label)
                    errD_fake.backward()
                    errD = errD_real + errD_fake
                    optimizerD.step()
                    ############################
                    # (3) Update G network: maximize log(D(G(z)))
                    ###########################
                    point_netG.zero_grad()
                    label.data.fill_(real_label)
                    output = point_netD(fake)
                    errG_D = criterion(output, label)
                    fake = fake.squeeze(dim=1)
                    errG_l2 = 0

                    CD_LOSS = self.cd_loss(fake, real_center.squeeze(dim=1))
                    errG_l2 = CD_LOSS

                    errG = (1 - opt.wtl2) * errG_D + opt.wtl2 * errG_l2
                    errG.backward()
                    optimizerG.step()
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f/ %.4f'
                          % (epoch + 1, opt.epoch, i + 1, len(dataloader),
                             errD.data, errG_D.data, errG_l2, errG, CD_LOSS))

                    f.write('\n' + '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f /%.4f'
                            % (epoch + 1, opt.epoch, i + 1, len(dataloader),
                               errD.data, errG_D.data, errG_l2, errG, CD_LOSS))

                schedulerD.step()
                schedulerG.step()
                Test_CD_LOSS = 0.0
                point_netG = point_netG.eval()
                with torch.no_grad():
                    for i, data in enumerate(tqdm(test_dataloader)):
                        label, real_point, input_cropped1, real_center,second_points, all_cropped_indices = self.pgl_process_data(data)

                        input_cropped1 = Variable(input_cropped1, requires_grad=False)
                        second_points = Variable(second_points, requires_grad=True)


                        # 形成输入
                        input = [input_cropped1, second_points]

                        fake_center1, fake_center2, fake = point_netG(input)
                        fake = fake.squeeze(dim=1)
                        for m in range(0, len(all_cropped_indices)):
                            input_cropped1.data[m, all_cropped_indices[m]] = fake[m]
                        every_CD_loss = self.cd_loss(input_cropped1, real_point)
                        Test_CD_LOSS += every_CD_loss
                Test_CD_LOSS = Test_CD_LOSS / len(test_dataloader)
                print('test result:', Test_CD_LOSS)
                f.write('\n' + 'test result:  %.4f' % (Test_CD_LOSS))

                f.close()

                if epoch % opt.save_interval == 0:
                    target_root_path = f"./Models/"
                    if not os.path.exists(target_root_path):
                        os.makedirs(target_root_path, exist_ok=True)
                    torch.save({'epoch': epoch + 1,
                                'state_dict': point_netG.state_dict()},
                               f'{target_root_path}/point_netG' + str(epoch) + '.pth')
                    torch.save({'epoch': epoch + 1,
                                'state_dict': point_netD.state_dict()},
                               f'{target_root_path}/point_netD' + str(epoch) + '.pth')

        #
        #############################
        ## ONLY G-NET
        ############################
        else:
            f = open(file_name_with_time, 'a')
            for epoch in range(resume_epoch, opt.epoch):
                if epoch < 30:
                    alpha1 = 0.01
                    alpha2 = 0.02
                elif epoch < 80:
                    alpha1 = 0.05
                    alpha2 = 0.1
                else:
                    alpha1 = 0.1
                    alpha2 = 0.2
                for i, data in enumerate(dataloader, 0):
                    real_point, target = data

                    label, real_point, input_cropped1, real_center = self.process_data(data)
                    ############################
                    # (1) data prepare
                    ###########################
                    real_center = Variable(real_center, requires_grad=True)
                    real_center = torch.squeeze(real_center, 1)
                    real_center_key1_idx = utils.farthest_point_sample(real_center, 64, RAN=False)
                    real_center_key1 = utils.index_points(real_center, real_center_key1_idx)
                    real_center_key1 = Variable(real_center_key1, requires_grad=True)

                    real_center_key2_idx = utils.farthest_point_sample(real_center, 128, RAN=True)
                    real_center_key2 = utils.index_points(real_center, real_center_key2_idx)
                    real_center_key2 = Variable(real_center_key2, requires_grad=True)

                    input_cropped1 = torch.squeeze(input_cropped1, 1)
                    input_cropped2_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[1],
                                                                     RAN=True)
                    input_cropped2 = utils.index_points(input_cropped1, input_cropped2_idx)
                    input_cropped3_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[2],
                                                                     RAN=False)
                    input_cropped3 = utils.index_points(input_cropped1, input_cropped3_idx)
                    input_cropped1 = Variable(input_cropped1, requires_grad=True)
                    input_cropped2 = Variable(input_cropped2, requires_grad=True)
                    input_cropped3 = Variable(input_cropped3, requires_grad=True)
                    input_cropped2 = input_cropped2.to(device)
                    input_cropped3 = input_cropped3.to(device)
                    input_cropped = [input_cropped1, input_cropped2, input_cropped3]
                    point_netG = point_netG.train()
                    point_netG.zero_grad()
                    fake_center1, fake_center2, fake = point_netG(input_cropped)
                    fake = torch.unsqueeze(fake, 1)
                    ############################
                    # (3) Update G network: maximize log(D(G(z)))
                    ###########################

                    CD_LOSS = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center, 1))

                    errG_l2 = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center, 1)) \
                              + alpha1 * criterion_PointLoss(fake_center1, real_center_key1) \
                              + alpha2 * criterion_PointLoss(fake_center2, real_center_key2)

                    errG_l2.backward()
                    optimizerG.step()
                    print('[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
                          % (epoch, opt.batchSize, i, len(dataloader),
                             errG_l2, CD_LOSS))

                    f.write('\n' + '[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
                            % (epoch, opt.batchSize, i, len(dataloader),
                               errG_l2, CD_LOSS))

                schedulerG.step()
                Test_CD_LOSS = 0.0
                with torch.no_grad():
                    point_netG = point_netG.eval()
                    for i, data in enumerate(tqdm(test_dataloader)):
                        label, real_point, input_cropped1, real_center = self.process_data(data)
                        input_cropped2_idx = utils.farthest_point_sample(input_cropped1,
                                                                         opt.point_scales_list[1],
                                                                         RAN=True)
                        input_cropped2 = utils.index_points(input_cropped1, input_cropped2_idx)
                        input_cropped3_idx = utils.farthest_point_sample(input_cropped1,
                                                                         opt.point_scales_list[2],
                                                                         RAN=False)
                        input_cropped3 = utils.index_points(input_cropped1, input_cropped3_idx)
                        input_cropped1 = Variable(input_cropped1, requires_grad=False)
                        input_cropped2 = Variable(input_cropped2, requires_grad=False)
                        input_cropped3 = Variable(input_cropped3, requires_grad=False)
                        input_cropped2 = input_cropped2.to(device)
                        input_cropped3 = input_cropped3.to(device)
                        input_cropped = [input_cropped1, input_cropped2, input_cropped3]

                        fake_center1, fake_center2, fake = point_netG(input_cropped)
                        every_CD_loss = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center, 1))
                        Test_CD_LOSS += every_CD_loss
                Test_CD_LOSS = Test_CD_LOSS / len(test_dataloader)

                f.write('\n' + 'test result:  %.4f' % (Test_CD_LOSS))

                if epoch % opt.save_interval == 0:
                    target_root_path = f"./Models/"
                    if not os.path.exists(target_root_path):
                        os.makedirs(target_root_path, exist_ok=True)
                    torch.save({'epoch': epoch + 1,
                                'state_dict': point_netG.state_dict()},
                               f'{target_root_path}/point_netG' + str(epoch) + '.pth')
                    torch.save({'epoch': epoch + 1,
                                'state_dict': point_netD.state_dict()},
                               f'{target_root_path}/point_netD' + str(epoch) + '.pth')
            f.close()


