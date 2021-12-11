from matplotlib import pyplot as plt

import pandas as pd



# 把各种参数塞进去画图
fig, ax = plt.subplots()
ax.plot(epochList, lossRateList, color="blue", label="轮数")
ax.plot(epochList, learnRateList, color="orange", label="学习率")
# ax.plot(epochList, trainAccList, color="green", label="训练正确率")
# ax.plot(epochList, testAccList, color="red", label="测试正确率")

plt.savefig(f"{modelNamePrefix}-{i}.png")
plt.show()