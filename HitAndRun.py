import numpy as np

def hit_and_run(polytope, num_samples, max_iterations=10000):
    """
    Hit and Run算法，用于在凸多面体内生成随机样本。
    
    Args:
        polytope: numpy数组，形状为(n, m)，其中n是顶点数量，m是维数。
                  每行表示凸多面体的一个顶点。
        num_samples: int，要生成的随机样本数量。
        max_iterations: int，Hit and Run算法的最大迭代次数，默认为10000。
    
    Returns:
        numpy数组，形状为(num_samples, m)，其中每行表示一个随机样本。
    """
    # 初始化
    num_vertices, num_dimensions = polytope.shape
    current_point = polytope[0]
    samples = np.zeros((num_samples, num_dimensions))
    
    # 迭代生成样本
    for i in range(num_samples):
        for j in range(max_iterations):
            # 随机生成方向向量
            direction = np.random.uniform(low=-1, high=1, size=num_dimensions)
            
            # 计算与凸多面体的交点
            t_min = np.inf
            t_max = -np.inf
            for k in range(num_vertices):
                vertex = polytope[k]
                dot_product = np.dot(vertex - current_point, direction)
                t_min = min(t_min, dot_product)
                t_max = max(t_max, dot_product)
            
            # 生成样本点
            t = np.random.uniform(low=t_min, high=t_max)
            new_point = current_point + t * direction
            
            # 判断新点是否在凸多面体内部
            is_inside = True
            for k in range(num_vertices):
                vertex = polytope[k]
                dot_product = np.dot(vertex - new_point, current_point - new_point)
                if dot_product > 0:
                    is_inside = False
                    break
            
            # 如果新点在凸多面体内部，则接受新点，否则重试
            if is_inside:
                current_point = new_point
                samples[i] = new_point
                break
    
    return samples


# 定义凸多面体的顶点
polytope = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

# 生成随机样本
samples = hit_and_run(polytope, 1000)

# 打印样本均值和方差
print("样本均值：", np.mean(samples, axis=0))
print("样本方差：", np.var(samples, axis=0))