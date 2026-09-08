import torch



import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.GRNet.grnet import GRNet
from architecture.GRNet import utils
from architecture.GRNet.utils import helpers
# 设置均值和标准差
mean = 0.0    # 均值
std = 1.0     # 标准差
size = (2,2 ,3) # 张量的形状
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 创建符合正态分布的张量
tensor_normal = torch.normal(mean=torch.full(size, mean), std=torch.full(size, std))
print(tensor_normal)
tensor_normal=tensor_normal.to(device)
grnet = GRNet()
grnet.apply(utils.helpers.init_weights)
if torch.cuda.is_available():
    grnet = torch.nn.DataParallel(grnet).cuda()
sparse_ptcloud, dense_ptcloud = grnet(tensor_normal)
print(sparse_ptcloud.shape)
print(dense_ptcloud.shape)



print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

