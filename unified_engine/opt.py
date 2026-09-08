import argparse


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--architecture', type=str, default='pf-net', help='pf-net,pmp-net,grnet')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--pnum', type=int, default=4096, help='the point number of a sample')
    parser.add_argument('--crop_point_num', type=int, default=512,
                        help='0 means do not use else use with this weight')
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--cuda', type=bool, default=False, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
    parser.add_argument('--D_choose', type=int, default=1, help='0 not use D-net,1 use D-net')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int,default=4080, help='manual seed')
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--num_scales', type=int, default=3, help='number of scales')
    parser.add_argument('--point_scales_list', type=list, default=[4096, 2048, 1024],
                        help='number of points in each scales')
    parser.add_argument('--each_scales_size', type=int, default=1, help='each scales size')
    parser.add_argument('--wtl2', type=float, default=0.95, help='0 means do not use else use with this weight')
    parser.add_argument('--cropmethod', default='random_center', help='random_center|narrow')
    parser.add_argument('--save_interval', default=5, help='save the model per interval')
    parser.add_argument('--save_dir', default="Save", help='save the model')
    parser.add_argument('--save_train_result', default="loss_PFNet.txt", help='save the model per interval')

    return parser.parse_args()

def get_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--architecture', type=str, default='pf-net', help='pf-net,pmp-net,grnet')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--pnum', type=int, default=4096, help='the point number of a sample')
    parser.add_argument('--crop_point_num', type=int, default=512,help='0 means do not use else use with this weight')
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--cuda', type=bool, default=False, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
    parser.add_argument('--D_choose', type=int, default=1, help='0 not use D-net,1 use D-net')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int,default=4080, help='manual seed')
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--num_scales', type=int, default=3, help='number of scales')
    parser.add_argument('--point_scales_list', type=list, default=[4096, 2048, 1024],
                        help='number of points in each scales')
    parser.add_argument('--each_scales_size', type=int, default=1, help='each scales size')
    parser.add_argument('--wtl2', type=float, default=0.95, help='0 means do not use else use with this weight')

    return parser.parse_args()