import torch
import DeepEmotionModel
import cv2
import matplotlib.pyplot as plt
device = "cuda:0" if torch.cuda.is_available() else "cpu"
net = DeepEmotionModel.DeepEmotionModel()
# 加载模型
net.load_state_dict(torch.load("DeepEmotionModel.pk1", map_location=device))
net.to(device)
