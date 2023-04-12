import numpy as np

def hierarchical_clustering(R, phi, L):
    n = len(R)
    S = [{i} for i in range(n)]
    R_bar = [[R[i][j] for j in range(n)] for i in range(n)]
    E = [[int(i == j) for j in range(n)] for i in range(n)]
    s = [(phi(R[k][k]), k) for k in range(n)]
    for level in range(1, n):
        for cluster in range(n - level + 1):
            print([(i, j) for i in S[cluster] for j in S[cluster] if i <= j])
            i_star, j_star = min(((i, j) for i in S[cluster] for j in S[cluster] if i <= j), key=lambda x: R_bar[x[0]][x[1]])
            S[cluster] = S[cluster] - {i_star, j_star} | {frozenset((i_star, j_star))}
    for I in range(n):
        for J in range(I + 1, n):
            I_set = S[next((i for i, S_i in enumerate(S) if I in S_i), None)]
            J_set = S[next((j for j, S_j in enumerate(S) if J in S_j), None)]
            E[level][I][J] = E[level][J][I] = L(I_set, J_set)
    for k in range(n - level + 1):
        I = S[k]
    s.append((phi(R_bar[i][j] for i in I for j in I), k + n))
    return s, E

def phi(distance):
    return 10+0.005*distance

arr = np.random.rand(5, 5)

# Set the diagonal elements to 0
np.fill_diagonal(arr, 0)

# Multiply all elements by a random number between 1 and 10 to ensure all elements are greater than 0
arr *= np.random.randint(1, 11)

# The resulting array has diagonal elements of 0 and all other elements greater than 0

# Make the matrix symmetric
arr = arr + arr.T - np.diag(arr.diagonal())
print(arr)
# 集合间距离
def L(I,J):
    return sum(arr[i][j] for i in I for j in J) / (len(I)*len(J))

hierarchical_clustering(arr,phi,L)