import os, sys,cv2
import numpy as np
import onnxruntime
from PIL import Image

def preprocess_image_opencv(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 确保图像是以彩色模式读入
    if image is None:
        raise FileNotFoundError("Image not found at the specified path.")
    image = cv2.resize(image, (28, 28))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=-1)
    image = (image / 255.0 - 0.5) / 0.5
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    return image

# 模型加载
onnx_model_path = "classifier.onnx"
net_session = onnxruntime.InferenceSession(onnx_model_path)
path = './image_new'
#path = 'subtle_augmentation_images/Z'
for file in os.listdir(path):
    print(file)
    print(os.path.join(path,file))
    inputs = {net_session.get_inputs()[0].name: preprocess_image_opencv(os.path.join(path,file))}
    outs = net_session.run(None, inputs)[0]
    #print("onnx weights", outs)
    print("onnx prediction", outs.argmax(axis=1)[0])
    img = Image.open(os.path.join(path,file))
    output_path = os.path.join(path,f'{file}_predicted_as_{outs.argmax(axis=1)[0]+1}.png') 
    img.save(output_path)
