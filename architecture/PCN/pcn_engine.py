# author: Vinit Sarode (vinitsarode5@gmail.com) 03/23/2020

import argparse
import os
import sys

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from architecture.PCN.pcn import PCN
from unified_engine.engine import TrainEngine
from unified_engine.test_engine import TestEngine


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


class PCNEngine(TrainEngine):
    def train(self):
        opt = self.opt
        args = self.options()

        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

        boardio = SummaryWriter(log_dir='SaveLog/' + args.exp_name)
        self.init(args)

        textio = IOStream('SaveLog/' + args.exp_name + '/run.txt')
        textio.cprint(str(args))

        # Create PointNet Model.
        model = PCN(emb_dims=args.emb_dims, num_coarse=opt.pnum, detailed_output=args.detailed_output)

        checkpoint = None
        if args.resume:
            assert os.path.isfile(args.resume)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])

        if args.pretrained:
            assert os.path.isfile(args.pretrained)
            model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
        model.to(self.device)

        learnable_params = filter(lambda p: p.requires_grad, model.parameters())
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(learnable_params)
        else:
            optimizer = torch.optim.SGD(learnable_params, lr=0.1)

        if checkpoint is not None:
            min_loss = checkpoint['min_loss']
            optimizer.load_state_dict(checkpoint['optimizer'])

        best_test_loss = np.inf

        for epoch in range(args.start_epoch, opt.epoch):
            train_loss = self.train_one_epoch(model, optimizer)
            test_loss = self.val_one_epoch(model)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                snap = {'epoch': epoch + 1,
                        'model': model.state_dict(),
                        'min_loss': best_test_loss,
                        'optimizer': optimizer.state_dict(), }
                torch.save(snap, f'{opt.save_dir}/%s/models/best_model_snap.t7' % (args.exp_name))
                torch.save(model.state_dict(), f'{opt.save_dir}/%s/models/best_model.t7' % (args.exp_name))

            boardio.add_scalar('Train Loss', train_loss, epoch + 1)
            boardio.add_scalar('Test Loss', test_loss, epoch + 1)
            print('Train Loss', train_loss, epoch + 1, 'Test Loss', test_loss, epoch + 1)
            boardio.add_scalar('Best Test Loss', best_test_loss, epoch + 1)

            textio.cprint('EPOCH:: %d, Traininig Loss: %f, Testing Loss: %f, Best Loss: %f' % (
                epoch + 1, train_loss, test_loss, best_test_loss))

    def init(self, args):
        path = self.opt.save_dir
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(f'{path}/' + args.exp_name):
            os.makedirs(f'{path}/' + args.exp_name)
        if not os.path.exists(f'{path}/' + args.exp_name + '/' + 'models'):
            os.makedirs(f'{path}/' + args.exp_name + '/' + 'models')
        os.system('cp pcn_engine.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')

    def val_one_epoch(self, model):
        model.eval()
        test_loss = 0.0
        pred = 0.0
        count = 0
        for i, data in enumerate(tqdm(self.val_dataloader)):
            label, real_point, input_cropped1, patch,_ = self.process_data(data)

            output = model(input_cropped1)
            loss_val = self.cd_loss(real_point, output['coarse_output'])

            test_loss += loss_val.item()
            count += 1

        test_loss = float(test_loss) / count
        return test_loss

    def val(self, model, textio):
        test_loss = self.val_one_epoch(model)
        textio.cprint('Validation Loss: %f' % (test_loss))

    def train_one_epoch(self, model, optimizer):
        model.train()
        train_loss = 0.0
        pred = 0.0
        count = 0
        for i, data in enumerate(tqdm(self.train_dataloader)):
            label, real_point, input_cropped1, patch,_ = self.process_data(data)

            output = model(input_cropped1)
            loss_val = self.cd_loss(real_point, output['coarse_output'])

            # backward + optimize
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            train_loss += loss_val.item()
            count += 1

        train_loss = float(train_loss) / count
        return train_loss

    def options(self):
        parser = argparse.ArgumentParser(description='Point Completion Network')
        parser.add_argument('--exp_name', type=str, default='exp_pcn', metavar='N',
                            help='Name of the experiment')
        parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

        # settings for PCN
        parser.add_argument('--emb_dims', default=1024, type=int,
                            metavar='K', help='dim. of the feature vector (default: 1024)')
        parser.add_argument('--detailed_output', default=False, type=bool,
                            help='Coarse + Fine Output')

        # settings for on training
        parser.add_argument('--seed', type=int, default=4080)
        parser.add_argument('--start_epoch', default=0, type=int,
                            metavar='N', help='manual epoch number (useful on restarts)')
        parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                            metavar='METHOD', help='name of an optimizer (default: Adam)')
        parser.add_argument('--resume', default='', type=str,
                            metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
        parser.add_argument('--pretrained', default='', type=str,
                            metavar='PATH', help='path to pretrained model file (default: null (no-use))')

        args = parser.parse_args()
        return args


class PCNTestEngine(TestEngine):
    def test(self):
        opt = self.opt
        args = self.options()

        if not torch.cuda.is_available():
            args.device = 'cpu'
        args.device = self.device

        # Create PointNet Model.
        model = PCN(emb_dims=args.emb_dims, detailed_output=args.detailed_output)

        if args.pretrained:
            assert os.path.isfile(args.pretrained)
            model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
        model.to(args.device)

        self.do_test(model)
    def do_test(self,model):
        model.eval()
        test_loss = 0.0
        pred = 0.0
        count = 0
        for i, data in enumerate(tqdm(self.test_dataloader)):
            label, real_point, input_cropped, patch=self.process_data(data)

            output = model(input_cropped)
            loss_val = self.cd_loss(real_point, output['coarse_output'])
            print("Loss Val: ", loss_val)
            #display_open3d(points[0].detach().cpu().numpy(), output['coarse_output'][0].detach().cpu().numpy())

            test_loss += loss_val.item()
            count += 1

        test_loss = float(test_loss) / count
        return test_loss

    def options(self):
        parser = argparse.ArgumentParser(description='Point Completion Network')
        parser.add_argument('--exp_name', type=str, default='exp_pcn', metavar='N',
                            help='Name of the experiment')
        parser.add_argument('--eval', type=bool, default=True, help='Train or Evaluate the network.')

        # settings for PCN
        parser.add_argument('--emb_dims', default=1024, type=int,
                            metavar='K', help='dim. of the feature vector (default: 1024)')
        parser.add_argument('--detailed_output', default=False, type=bool,
                            help='Coarse + Fine Output')

        # settings for on training
        parser.add_argument('--seed', type=int, default=4080)
        parser.add_argument('--start_epoch', default=0, type=int,
                            metavar='N', help='manual epoch number (useful on restarts)')
        parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                            metavar='METHOD', help='name of an optimizer (default: Adam)')
        parser.add_argument('--resume', default='', type=str,
                            metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
        parser.add_argument('--pretrained', default='', type=str,
                            metavar='PATH', help='path to pretrained model file (default: null (no-use))')

        args = parser.parse_args()
        return args


if __name__ == '__main__':
    pass
