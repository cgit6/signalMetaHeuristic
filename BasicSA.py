# 該函數的輸入包括：
# nodes：表示所有節點的坐標，每行一個節點，列數為節點數量。
# members：表示所有桿件的連接情況，每行一個桿件，包含兩個整數表示連接的兩個節點的編號。
# E：表示所有桿件的材料彈性模量，長度等於桿件數量。
# A：表示所有桿件的截面面積，長度等於桿件數量。
# supports：表示受力支撐點的節點編號。
# loads：表示所有節點的受力情況，每行包含兩個浮點數，分別表示該節點在x軸和y軸方向的受力大小。

# 該函數的輸出包括：
# best：表示找到的最佳解，與輸入的節點坐標相加得到實際節點坐標。
# best_energy：表示找到的最佳解的能量。
# energy_history：表示找到的所有解的能量，用於後續繪圖等用途。

import numpy as np

# 生成初始解
def initialization(pop,ub,lb,dim):
    
    X = np.zeros([pop,dim]) #声明空间
    for i in range(pop):
        for j in range(dim):
            X[i,j]=(ub[j]-lb[j])*np.random.random()+lb[j] #生成[lb,ub]之间的随机数
    
    return X

# 計算適應值
def CaculateFitness(X,fun):

    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness

# 定義冷卻時間表
def cool(t):
    return 0.99*t

#定義移動函數
def move(x):
    dx = np.random.randn(*x.shape)*0.01
    return x + dx

# 定義接受概率函數
def accept_prob(delta_e, t):
    return np.exp(-delta_e/t)

def SA(pop,ub,lb,dim,MaxIter,fun):
    # 定義初始解決方案
    X= initialization(pop,ub,lb,dim)

    # 初始化當前解和能量
    current = X
    # 計算適應度值
    fitness = CaculateFitness(X,fun)

    # 存儲迄今為止找到的最佳解決解與適應值
    best = current
    best_fitness = fitness

    # 存儲適應值紀錄
    fitness_history = [fitness]

    # 迭代階段
    for i in range(MaxIter):
        # Update the temperature
        t = cool(i/MaxIter)

        # 生成新的候選解
        candidate = move(current)
        candidate_fitness = fun(candidate)

        # 計算delta of fitness
        delta_e = candidate_fitness - current_fitness

        # 決定是否接受候選解

        if delta_e < 0 or np.random.rand() < accept_prob(delta_e, t):
            current = candidate
            current_fitness = candidate_fitness

            # 更新目前找到的最佳方案
            if current_fitness < best_fitness:
                best = current
                best_fitness = current_fitness

        # 存儲能源歷史
        fitness_history.append(current_fitness)

    # 返回找到的最佳解決方案
    return best, best_fitness, fitness_history


