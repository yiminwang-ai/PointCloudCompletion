from thop import profile
from torchvision import models


def calculate(model, input):
    model = model.eval()  # 设置为评估模式，有些层如Dropout和BatchNorm在训练和评估模式下的行为不同。
    flops, params = profile(model, inputs=(input,))  # 注意这里的inputs应该是一个tuple，即使只有一个输入。
    gflops = flops / 1e9  # 使用科学记数法或幂运算符
    mparams = params / 1e6
    # 假设每个参数是float32（4字节）
    model_size_MB = (params * 4) / 1024 / 1024  # 将字节转换为MB
    print(
        f"FLOPs: {flops}, GFLOPs: {gflops:.3f}, Param_num: {int(params)}, MParam_num:{mparams:.3f}M, model_size_MB:{model_size_MB:.3f}MB"
    ) # 这里输出的params实际上是参数的数量，而非大小（以字节为单位）。要获取大小，需要自己转换。
