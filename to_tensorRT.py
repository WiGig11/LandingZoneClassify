import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化CUDA驱动
import cv2
import numpy as np
"""
change file to tensorRT file for cuda accelaration
"""
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def preprocess_image_opencv(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 确保图像是以彩色模式读入
    if image is None:
        raise FileNotFoundError("Image not found at the specified path.")

    # 调整图像大小到28x28
    image = cv2.resize(image, (28, 28))

    # 转换为灰度图
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 将图像数据转换为float32类型
    image = image.astype(np.float32)

    # 扩展为1个通道（假设模型期待一个单通道的灰度图输入）
    image = np.expand_dims(image, axis=-1)

    # 归一化图像数据
    image = (image / 255.0 - 0.5) / 0.5

    # 添加一个批次维度N，因为模型可能需要NCHW格式
    image = np.expand_dims(image, axis=0)

    # 调整维度顺序从NHWC到NCHW
    image = np.transpose(image, (0, 3, 1, 2))

    return image

def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

trt_runtime = trt.Runtime(TRT_LOGGER)
engine = load_engine(trt_runtime, "model.trt")

# 假设输入数据尺寸和类型已知
input_shape = (1, 1, 28, 28)  # 示例尺寸，根据您的模型调整
dtype = np.float32  # 根据您的模型输入类型调整
processed_image = preprocess_image_opencv('test.png')
# 创建CUDA内存以存储输入输出数据
input_data = processed_image.astype(dtype)
input_d = cuda.mem_alloc(input_data.nbytes)
output_d = cuda.mem_alloc(input_data.nbytes)  # 假设输出数据与输入数据大小相同

# 将输入数据复制到设备
context = engine.create_execution_context()
output_data = np.empty(input_shape, dtype=dtype)  # 为输出数据分配空间

try:
    # 将输入数据复制到设备
    cuda.memcpy_htod(input_d, input_data)
    context.execute(batch_size=1, bindings=[int(input_d), int(output_d)])

    # 将输出数据从设备复制到主机
    cuda.memcpy_dtoh(output_data, output_d)
    print("Output Data:", output_data)
finally:
    input_d.free()
    output_d.free()
    context.destroy()
    engine.destroy()
    trt_runtime.destroy()

# 使用函数预处理图像


#trtexec --onnx=model.onnx --saveEngine=model.trt
