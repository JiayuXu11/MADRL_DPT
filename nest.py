import numpy as np

def hierarchical_clustering(R, phi):
    #R是距离矩阵，phi是根据距离计算成本的公式，L是计算簇与簇之间距离的公式
    #获取矩阵R的行数n=5
    n = len(R)
    #返回一个从 0 到 n-1 的整数序列，代表当前level的cluster
    S = [{i} for i in range(n)]
    #为矩阵R创建一个副本矩阵R_bar
    R_bar = [[R[i][j] for j in range(n)] for i in range(n)]
    #创建一个单位矩阵E，大小是n*n
    #Set level 0 structure to identity matrix
    E = [[[int(i == j) for j in range(n)] for i in range(n)] for _ in range(n)]
    #根据距离矩阵R创建成本矩阵s，格式为（成本，序号）
    # s = [(phi(R[k][k]), k) for k in range(n)]
    s=[]
    # 集合间距离
    def L(I,J):
        return sum(R_bar[i][j] for i in I for j in J) / (len(I)*len(J))
    for k in range(n):
        s.append((phi(R[k][k]),0,k))

    S_level=[[] for i in range(n)]
    S_level[0]=S
    for level in range(1, n):
        # for cluster in range(n - level + 1):

        #找到变量 S 中任意两个值组成的元组中，使得变量 R_bar 中对应元素最小的元组
        i_star, j_star = min([(S[i], S[j]) for i in range(len(S)) for j in range(len(S)) if i < j], key=lambda x: L(x[0],x[1]))
        #将变量 S 中的元素 i_star 和 j_star 删除，并将不可变集合 frozenset((i_star, j_star)) 添加到集合中。
        S = [s_ for s_ in S if s_ not  in [i_star,j_star]]
        S = S + [i_star | j_star]
        S_level[level]=S
        #I遍历0到4
        R_bar1=[[0 for _ in S] for _ in S]

        for I in range(len(S)):
            I_set = S[I]
            #I=0时，J=1,2,3,4;I=1时，J=2,3,4;I=2时，J=3,4;I=3时，J=4
            for J in range(I + 1, len(S)):
                J_set = S[J]
                # #从变量 S 中找到包含变量 I 的集合，并将其赋值给变量 I_set
                # I_set = S[next((i for i, S_i in enumerate(S) if I in S_i), None)]
                # #从变量 S 中找到包含变量 J 的集合，并将其赋值给变量 J_set
                # J_set = S[next((j for j, S_j in enumerate(S) if J in S_j), None)]
                #将 L(I_set, J_set) 的值分别赋给 E[level][I][J] 和 E[level][J][I]
                R_bar1[I][J] = R_bar1[J][I] = L(I_set,J_set)

            #level=1时，k=5，level=2时，k=4，level=3时，k=3，level=4时，k=2，level=5时，k=1
        for k in range(len(S) ):
            I = S[k]
            E[level][k]=[1 if i in I else 0 for i in range(n)] 
            #计算变量 R_bar 中所有索引对 (i, j) 所对应的元素的 phi 值，并将phi 值作为一个元组传递给元组的第一个元素
            #计算变量 k 和变量 n 的和，并将其作为元组的第二个元素
            s.append((phi(R_bar1[k][k]),level,k))
            # s.append([(phi(R_bar1[i][j]),k+level) for i in I for j in I])
    return S_level

def phi(distance):
    return 10+0.005*distance

#随机生成一个5*5的随机数数组，服从均值为1标准差为0的正太分布
np.random.seed(0)
arr = np.random.rand(5, 5)

# Set the diagonal elements to 0
#将矩阵的对角线设置为0
np.fill_diagonal(arr, 0)

# Multiply all elements by a random number between 1 and 10 to ensure all elements are greater than 0
#将数组中每一个元素乘以一个介于1和10之间的随机整数
arr *= np.random.randint(1, 11)

# The resulting array has diagonal elements of 0 and all other elements greater than 0

# Make the matrix symmetric
#让数组成为一个对称矩阵
arr = arr + arr.T - np.diag(arr.diagonal())
print(arr)


hierarchical_clustering(arr,phi)