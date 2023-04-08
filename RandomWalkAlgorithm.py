import random
import numpy as np
import matplotlib.pyplot as plt

# 一维随机漫步的 Python 代码
# 向上或向下移动的概率
prob = [0.05, 0.95] 
 
# 静态定义起始位置
start = 2 
positions = [start]
 
# 创建随机点
rr = np.random.random(1000)
print(rr)
downp = rr < prob[0]
upp = rr > prob[1]
 
 
for idownp, iupp in zip(downp, upp):
    down = idownp and positions[-1] > 1
    up = iupp and positions[-1] < 4
    positions.append(positions[-1] - down + up)
 
# 绘制一维随机漫步图
plt.plot(positions)
plt.show()