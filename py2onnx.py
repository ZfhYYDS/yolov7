
# 通道数问题
import torch
import torch.onnx
from nets.yolo import SPPCSPC
# 定义转换函数
def convert_pth_to_onnx(pth_file_path, onnx_file_path, input_size=(1, 3, 640, 640)):
    # 创建SPPCSPC模型实例
    transition_channels  = 16
    model = SPPCSPC(transition_channels * 32, transition_channels * 16)

    # 加载模型权重状态字典
    state_dict = torch.load(pth_file_path)

    # 将状态字典应用到模型实例上
    model.load_state_dict(state_dict)#

    # 设置模型为评估模式
    model.eval()

    # 创建一个虚拟输入张量
    dummy_input = torch.randn(*input_size)

    # 导出模型
    torch.onnx.export(model, dummy_input, onnx_file_path, export_params=True, opset_version=10,
                      do_constant_folding=True, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})


# 指定.pth文件和期望的.onnx文件名
pth_file_path = r'C:\yaogan\yolov7-tiny-pytorch-master\yolov7-tiny-pytorch-master\logs\best_epoch_weights.pth'
onnx_file_path = r'C:\yaogan\yolov7-tiny-pytorch-master\yolov7-tiny-pytorch-master\model_data\yolov7tiny.onnx'

# 调用转换函数
convert_pth_to_onnx(pth_file_path, onnx_file_path)

# 输出成功消息
print(f"成功将 {pth_file_path} 转换为 {onnx_file_path}.")


#
# import torch
# import torch.onnx
# from nets.yolo import SPPCSPC
# import os
#
#
# def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
#     if not onnx_path.endswith('.onnx'):
#         print('Warning! The onnx model name is not correct,\
#               please give a name that ends with \'.onnx\'!')
#         return 0
#
#     model = SPPCSPC(512,256)  # 导入模型
#     model.load_state_dict(torch.load(checkpoint))  # 初始化权重
#     model.eval()
#     # model.to(device)
#
#     torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names,
#                       output_names=output_names)  # 指定模型的输入，以及onnx的输出路径
#     print("Exporting .pth model to onnx model has been successful!")
#
#
# if __name__ == '__main__':
#     os.environ['CUDA_VISIBLE_DEVICES'] = '2'
#     checkpoint = r'C:\yaogan\yolov7-tiny-pytorch-master\yolov7-tiny-pytorch-master\logs\best_epoch_weights.pth'
#     onnx_path = r'C:\yaogan\yolov7-tiny-pytorch-master\yolov7-tiny-pytorch-master\model_data\yolov7tiny.onnx'
#     input = torch.randn(1, 1, 640, 360)
#     # device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
#     pth_to_onnx(input, checkpoint, onnx_path)

# git clone https://github.com/NVIDIA-AI-IOT/torch2trt
# cd torch2trt
# python setup.py install
