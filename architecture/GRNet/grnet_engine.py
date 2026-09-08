# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-07-04 11:01:37
# @Email:  cshzxie@gmail.com

import logging
import os
import torch


from datetime import datetime
from time import time
from tensorboardX import SummaryWriter

from architecture.GRNet import config, utils
from architecture.GRNet.extensions.chamfer_dist import ChamferDistance
from architecture.GRNet.extensions.gridding_loss import GriddingLoss
from architecture.GRNet.grnet import GRNet
from architecture.GRNet.utils import helpers
from architecture.GRNet.utils.average_meter import AverageMeter
from architecture.GRNet.utils.metrics import Metrics
from unified_engine.engine import TrainEngine


class GRNetEngine(TrainEngine):
    def train(self):
        torch.backends.cudnn.benchmark = True
        cfg = config.cfg
        opt = self.opt
        # Set up folders for logs and checkpoints
        cfg.DIR.OUT_PATH=''
        output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
        cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
        cfg.DIR.LOGS = output_dir % 'logs'
        if not os.path.exists(cfg.DIR.CHECKPOINTS):
            os.makedirs(cfg.DIR.CHECKPOINTS)

        # Create tensorboard writers
        train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
        val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))
        # Create tensorboard writers
        train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
        val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

        # Create the networks
        grnet = GRNet()
        grnet.apply(utils.helpers.init_weights)
        logging.debug('Parameters in GRNet: %d.' % utils.helpers.count_parameters(grnet))

        # Move the network to GPU if possible
        if torch.cuda.is_available():
            grnet = torch.nn.DataParallel(grnet).cuda()

        # Create the optimizers
        grnet_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, grnet.parameters()),
                                           lr=cfg.TRAIN.LEARNING_RATE,
                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                           betas=cfg.TRAIN.BETAS)
        grnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(grnet_optimizer,
                                                                  milestones=cfg.TRAIN.LR_MILESTONES,
                                                                  gamma=cfg.TRAIN.GAMMA)

        # Set up loss functions
        chamfer_dist = ChamferDistance()
        gridding_loss = GriddingLoss(  # lgtm [py/unused-local-variable]
            scales=cfg.NETWORK.GRIDDING_LOSS_SCALES,
            alphas=cfg.NETWORK.GRIDDING_LOSS_ALPHAS)

        # Load pretrained model if exists
        init_epoch = 0
        best_metrics = None
        if 'WEIGHTS' in cfg.CONST:
            logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
            checkpoint = torch.load(cfg.CONST.WEIGHTS)
            best_metrics = Metrics(cfg.TEST.METRIC_NAME, checkpoint['best_metrics'])
            grnet.load_state_dict(checkpoint['grnet'])
            logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' % (init_epoch, best_metrics))

        # Training/Testing the network
        for epoch_idx in range(init_epoch, opt.epoch):

            epoch_start_time = time()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter(['SparseLoss', 'DenseLoss'])

            grnet.train()

            batch_end_time = time()
            n_batches = len(self.train_dataloader)
            for batch_idx, data in enumerate(self.train_dataloader):
                label, real_point, input_cropped, patch = self.process_data(data)
                print('label',label.shape)
                data_time.update(time() - batch_end_time)
                # for k, v in data.items():
                #     data[k] = utils.helpers.var_or_cuda(v)

                sparse_ptcloud, dense_ptcloud = grnet(input_cropped)
                sparse_loss = chamfer_dist(sparse_ptcloud, real_point)
                dense_loss = chamfer_dist(dense_ptcloud, real_point)
                _loss = sparse_loss + dense_loss
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])

                grnet.zero_grad()
                _loss.backward()
                grnet_optimizer.step()

                n_itr = (epoch_idx - 1) * n_batches + batch_idx
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

                batch_time.update(time() - batch_end_time)
                batch_end_time = time()
                info='[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s' %(epoch_idx,
                        opt.epoch, batch_idx + 1, n_batches, batch_time.val(), data_time.val(),
                        ['%.4f' % l for l in losses.val()])
                print(info)
                logging.info('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s' %
                             (epoch_idx, opt.epoch, batch_idx + 1, n_batches, batch_time.val(), data_time.val(),
                              ['%.4f' % l for l in losses.val()]))

            grnet_lr_scheduler.step()
            epoch_end_time = time()
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch_idx)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch_idx)
            logging.info(
                '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]))

            # Validate the current model
            #metrics = self.test_net(epoch_idx, val_writer, grnet)

            # Save ckeckpoints
            # if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or metrics.better_than(best_metrics):
            #     file_name = 'ckpt-best.pth' if metrics.better_than(best_metrics) else 'ckpt-epoch-%03d.pth' % epoch_idx
            #     output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            #     print(f'save to{output_path}')
            #     torch.save({
            #         'epoch_index': epoch_idx,
            #         'best_metrics': metrics.state_dict(),
            #         'grnet': grnet.state_dict()
            #     }, output_path)  # yapf: disable
            #
            #     logging.info('Saved checkpoint to %s ...' % output_path)
            #     if metrics.better_than(best_metrics):
            #         best_metrics = metrics

        train_writer.close()
        val_writer.close()

    def test_net(self, epoch_idx=-1, test_writer=None, grnet=None):
        cfg = config.cfg
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Setup networks and initialize networks
        if grnet is None:
            grnet = GRNet()

            if torch.cuda.is_available():
                grnet = torch.nn.DataParallel(grnet).cuda()

            logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
            checkpoint = torch.load(cfg.CONST.WEIGHTS)
            grnet.load_state_dict(checkpoint['grnet'])

        # Switch models to evaluation mode
        grnet.eval()

        # Set up loss functions
        chamfer_dist = ChamferDistance()
        gridding_loss = GriddingLoss(scales=cfg.NETWORK.GRIDDING_LOSS_SCALES,
                                     alphas=cfg.NETWORK.GRIDDING_LOSS_ALPHAS)  # lgtm [py/unused-import]

        # Testing loop
        n_samples = len(self.val_dataloader)
        test_losses = AverageMeter(['SparseLoss', 'DenseLoss'])
        test_metrics = AverageMeter(Metrics.names())
        category_metrics = dict()


        # Testing loop
        for batch_idx, data in enumerate(self.val_dataloader):
            label, real_point, input_cropped, patch = self.process_data(data)
            with torch.no_grad():
                input_cropped=helpers.var_or_cuda(input_cropped)

                sparse_ptcloud, dense_ptcloud = grnet(input_cropped)
                sparse_loss = chamfer_dist(sparse_ptcloud, real_point)
                dense_loss = chamfer_dist(dense_ptcloud, real_point)
                test_losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
                _metrics = Metrics.get(dense_ptcloud, real_point)
                test_metrics.update(_metrics)

                if test_writer is not None and batch_idx < 3:
                    sparse_ptcloud = sparse_ptcloud.squeeze().cpu().numpy()
                    sparse_ptcloud_img = utils.helpers.get_ptcloud_img(sparse_ptcloud)
                    test_writer.add_image('Model%02d/SparseReconstruction' % batch_idx, sparse_ptcloud_img,
                                          epoch_idx)
                    dense_ptcloud = dense_ptcloud.squeeze().cpu().numpy()
                    dense_ptcloud_img = utils.helpers.get_ptcloud_img(dense_ptcloud)
                    test_writer.add_image('Model%02d/DenseReconstruction' % batch_idx, dense_ptcloud_img, epoch_idx)
                    gt_ptcloud = real_point.squeeze().cpu().numpy()
                    gt_ptcloud_img = utils.helpers.get_ptcloud_img(gt_ptcloud)
                    test_writer.add_image('Model%02d/GroundTruth' % batch_idx, gt_ptcloud_img, epoch_idx)

                # logging.info('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                #              (
                #                  model_idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()
                #                                                                    ], ['%.4f' % m for m in _metrics]))

        # Print testing results
        print('============================ TEST RESULTS ============================')
        for metric in test_metrics.items:
            print(metric, end='\t')
        print()

        print('Overall', end='\t\t\t')
        for value in test_metrics.avg():
            print('%.4f' % value, end='\t')
        print('\n')

        # Add testing results to TensorBoard
        if test_writer is not None:
            test_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch_idx)
            test_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(1), epoch_idx)
            for i, metric in enumerate(test_metrics.items):
                test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

        return Metrics(cfg.TEST.METRIC_NAME, test_metrics.avg())
