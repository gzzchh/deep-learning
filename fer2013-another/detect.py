import torch
import DeepEmotionModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
net = DeepEmotionModel.DeepEmotionModel()
# 加载模型
net.load_state_dict(torch.load("DeepEmotionModel.pk1", map_location=device))
net.to(device)

import cv2
import matplotlib.pyplot as plt

frame = cv2.imread('IMG_20190608_171842.jpg')
plt.imshow(frame, cv2.COLOR_BGR2RGB)