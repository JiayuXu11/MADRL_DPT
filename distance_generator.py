import numpy as np

n = 10  # 点的数量
a = 0  # 随机点的坐标范围
b = 10

# 生成随机点
points = np.random.uniform(a, b, size=(n, 2))

# 计算距离矩阵
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])

# 线性缩放距离矩阵到300-2000的范围
d_min = dist_matrix[dist_matrix > 0].min()
d_max = dist_matrix[dist_matrix > 0].max()
scaled_matrix = (dist_matrix - d_min) / (d_max - d_min) * (2000 - 300) + 300

np.fill_diagonal(scaled_matrix, 0)
scaled_matrix = np.round(scaled_matrix).astype(int)

# matrix = np.full((n,n),1000)
# np.fill_diagonal(matrix, 0)
# scaled_matrix = np.round(matrix).astype(int)
# print(repr(matrix))