import torch
import DeepEmotionModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"
net = DeepEmotionModel.DeepEmotionModel()
# 加载模型
net.load_state_dict(torch.load("cosann-best-0.pt", map_location=device))
net.to(device)

import cv2
import matplotlib.pyplot as plt

frame = cv2.imread('DSC05404.jpg')
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# 转换成灰度
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, 1.1, 4)
for x, y, w, h in faces:
    roiGray = gray[y:y + h, x:x + w]
    roiColor = frame[y:y + h, x:x + w]
    cv2.rectangle(frame, (x, y), (x + w, y + w), (255, 0, 0), 2)
    facess = faceCascade.detectMultiScale(roiGray)
    if len(facess) == 0:
        print("没有检测到人脸")
    else:
        for (ex, ey, ew, eh) in facess:
            faceRoi = roiColor[ey:ey + eh, ex:ex + ew]

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
