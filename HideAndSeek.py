import math
import numpy as np
from random import random
import copy as copy
# 卡方表
chiSquare =[0.0,6.63490, 9.21034, 11.3449, 13.2767, 15.09,
               16.8119, 18.4753, 20.0902, 21.6660, 23.2093,
               24.7250, 26.2170, 27.6883, 29.1413, 30.5779,
               31.9999, 33.4087, 34.8053, 36.1908, 37.5662,
               38.9321, 40.2984, 41.6384, 42.9798, 44.3141,
               45.6417, 46.9630, 48.2782, 49.5879, 50.8922]
def initialization(ub,lb,dim):
    # 声明空间
    X = np.zeros([dim])  
      
    
    for j in range(dim):
        # 生成[lb,ub]之间的随机数
        X[j] = (ub[j]-lb[j])*np.random.random()+lb[j]  

    return X

# 生成隨機向量
def generate_random_vector(dim,d):
    rand_vec = []
    for i in range(dim):
        rand_vec.append(np.random.random())
    return rand_vec
# 計算適應值
def CaculateFitness(X_new,fun):
    
    fitness = 0    
    fitness = fun(X_new)
    return fitness
# #隨機擾動
# def generate_new(X):   
#     while True:
#         X[0] = X[0] + X[0].T * (random() - random())
#         X[1] = X[1] + X[1].T * (random() - random())
#         if (-5 <= X[0] <= 5) & (-5 <= X[1] <= 5):  
#             #重复得到新解，直到产生的新解满足约束条件
#             break                                  
#     return X
# 边界检查函数
'''
dim:为每个个体数据的维度大小
X:为输入数据，维度为[pop,dim]
ub:为个体数据上边界，维度为[dim,1]
lb:为个体数据下边界，维度为[dim,1]
'''
def BorderCheck(X,ub,lb,dim):
    
    for i in range(dim):
        if X[i]>ub[i]:
            X[i] = ub[i]
            
        elif X[i]<lb[i]:
            X[i] = lb[i]
    return X
# 某種算法

#Metropolis法
# 概念 => 比較好一定接受，比較差有機率(p)接受，(1-p)機率拒絕
def Metrospolis( T, f_new,fitness):   
    if f_new <= fitness:
        return f_new
    elif f_new > fitness:
        # 生成p 值
        p = math.exp((fitness - f_new) / T)
        if random() < p:
            return f_new
        else:
            return fitness
# 主要算法
def SA(dim,lb,ub,nrun,MaxIter,fun):
    # dim 變數個數
    if dim<=30:
        chi2 = chiSquare[dim]
    else:
        chi2 = dim + np.sqrt(2.0 * dim) * 2.33
    

    # 计算分母以估计最佳值
    alpha=dim/2
    # 分母
    denom = 1/0.9**alpha -1
    best_f = 1000000
    worst_f = -1000000
    d=np.zeros([dim,1])
    SA_Curve = np.zeros([MaxIter,1])



    # nrun => 算法執行次數
    for i in range(nrun):
        # 开始第i次捉迷藏算法
        print('SA第',i+1,'次運行')
        # 找一個初解
        
        # 生成隨機向量
        # gen_vec=generate_random_vector(dim)
        
        # 生成一組初始解
        X = initialization(ub, lb, dim)
        X_new = np.zeros([dim,1])
        # 初始解
        fitness = CaculateFitness(X, fun)

        # 當前最佳解xopt[i]=初始解x[i]
        # 记录當前最优适应度值
        GbestScore = copy.copy(fitness)
        # print("GbestScore",GbestScore)
        GbestScore_2 = np.zeros([dim,1])
        # 记录當前最优解
        GbestPositon = np.zeros([dim, 1])
        
        
        # 記錄一個新的解
        # 紀錄一個新的適應值
        f_new = np.zeros([dim,1])
        # 初始溫度
        T = 1000000

        # 一種邊界更新的方法
        for j in range(MaxIter):
            # 生成一組隨機向量
            d=generate_random_vector(dim,d)
            r=0
            for k in range(dim):
                r=r+d[k]*d[k]
            r=np.sqrt(r)
            for k in range(dim):
                d[k] = d[k]/r

            min_pos = 1000000
            max_neg = -1000000

            la_ub = np.zeros([dim])
            la_lb = np.zeros([dim])
            #minpos 與 maxneg 帶入
            for k in range(dim):

                # 先計算lambda[i]
                # 上界
                la_ub[k] = (ub[k]-X[k]) / d[k]
                # 下界
                la_lb[k] = (lb[k]-X[k]) / d[k]
                
                # 收斂邊界
                # 上界
                # lambda 的意義是什麼?
                if 0<=la_ub[k]<=min_pos:
                    min_pos = la_ub[k]
                elif 0>la_ub[k]>max_neg:
                    max_neg = la_ub[k]

                # 下界
                if 0<=la_lb[k]<= min_pos:
                    min_pos = la_lb[k]
                elif 0 > la_lb[k] > max_neg:
                    max_neg = la_lb[k]
                
                # return minpos,maxneg
            
            # 生成新解，跟引領蜂有點像
            # 基向量 + r*(r*range+上界)
            for j in range(dim):
                X_new[j] = X[j]+(max_neg + np.random.random() * (min_pos - max_neg))*d[j]
                X_new = BorderCheck(X_new,ub,lb,dim)
            
            # 因為是求極小，若 f_new<fitness 視為較優解
            # metroplis法開始
            # 更新besf_f 與 worst_f
            # f_new 是另一個解算出的適應值
            f_new = CaculateFitness(X_new, fun)
            # Metrospolis法
            fitness = Metrospolis(T,fitness, f_new)
            # print("fitness:",fitness)
            # 將新的解放入bestf 與 worstf 中
            # metroppolis法結束

            # 結果比較
            # 先跟當次迭代的適應值比較 fitness <=> GbestScore
            # 在將當次最佳解跟全部的當前最佳解比較
            
            if fitness < GbestScore:
                GbestScore_2 = GbestScore
                GbestScore = fitness
                # 
                fest = GbestScore+(GbestScore_2-GbestScore)/ denom
                # T 溫度
                T =2*(fest-fitness)/chi2
                # print("溫度:",T)
                # for i in range(dim):
                    # X_new[i] = X[i]
                # print(f"第{j}次:{X_new}")
                SA_Curve[j] = best_f
                print(f"X_new:{X_new}")
                print(f"T:{T}")

                print(f"fitness:{fitness}")
                print(f"GbestScore:{GbestScore}")


            


    return X_new,fitness,SA_Curve