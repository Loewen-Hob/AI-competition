import numpy as np
import matplotlib.pyplot as plt

# 参数初始化
alpha = 0.2
beta = 0.7
gamma = 0.1
K0 = 1
L = 1
sr = 0.15
er = 0.05
N0 = 100
years = 50  # 模拟50年

# 初始化变量
K = [K0]  # 资本
N = [N0]  # 不可再生资源
Y = []  # 收入

# 开始模拟逐年的收入变化
for t in range(years):
    # 计算当年的资源和收入
    R_t = er * N[-1]  # 当年的资源
    Y_t = K[-1] ** alpha * L ** beta * R_t ** gamma  # 生产函数
    Y.append(Y_t)

    # 计算储蓄和资本累积
    S_t = sr * Y_t
    K.append(K[-1] + S_t)

    # 计算资源剩余
    N.append(N[-1] - R_t)

# 绘制时间-收入变化曲线
plt.figure(figsize=(10, 6))
plt.plot(range(years), Y, label="Income (Y) over time", color='b')
plt.title("Income (Y) Change Over Time")
plt.xlabel("Years")
plt.ylabel("Income (Y)")
plt.grid(True)
plt.legend()
plt.show()
