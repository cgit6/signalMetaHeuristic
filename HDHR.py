import numpy as np

# 這個示例中，我們使用 NumPy 库中的 numpy.random.randn 函数來隨機生成方向向量，
# 使用 numpy.linalg.norm 函数計算向量的歐氏長度，使用 numpy.random.randint 
# 函数生成移動步數，並使用循環來實現移動操作。值得注意的是，為了使取樣點均勻地分布在
# 超球體內部，我們需要將初始點標準化，並乘上超球體半徑。最後，我們將所有取樣點存儲在
# 一個矩陣中，並返回該矩陣作為最終結果。

def hdhr(n, d, m, r):
    # n: 取樣點數量，d: 維度，m: 移動步數，r: 超球體半徑
    points = np.zeros((n, d))  # 初始化取樣點矩陣
    p0 = np.random.randn(d)    # 隨機選擇初始點
    points[0, :] = p0 / np.linalg.norm(p0) * r  # 將初始點標準化，並乘上超球體半徑

    for i in range(1, n):
        # 隨機選擇超球面方向
        direction = np.random.randn(d)
        direction /= np.linalg.norm(direction)

        # 計算移動步數
        step = np.random.randint(1, m + 1)

        # 在所選擇的超球面方向上移動
        for j in range(step):
            p = points[i-1, :] + direction * r / m
            if np.linalg.norm(p) <= r:
                points[i, :] = p
            else:
                break

    return points