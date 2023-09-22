import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
def sample_spiral():
    points_per_cluster = 500
    bandwidth = 0.1
    data = np.empty((points_per_cluster, 2))
    w = np.arange(1, points_per_cluster + 1).astype(np.float32) / points_per_cluster
    data[:,0] = (4 * w + 1) * np.cos(2*np.pi * w) + np.random.randn(points_per_cluster) * bandwidth
    #print("data1",data.shape,data[:,0])
    data[:,1] = (4 * w + 1) * np.sin(2*np.pi * w) + np.random.randn(points_per_cluster) * bandwidth
    #print("data2", data[:, 1])
    data = np.vstack((data, -data))
    return data

X = sample_spiral()
print(X.shape)
N = X.shape[0]
plt.scatter(X[:, 0],X[:,1], s = 10, c = 'k')
plt.axis('square')

# scaling paramater (we may want to try different values and select the least-distored clusters)
sigma = 0.1
def calculate_w_ij(a, b, sigma=sigma):
    w_ab = np.exp(-(np.linalg.norm(x=a-b, ord=2))**2 / (2 * sigma ** 2))
    return w_ab


# 计算邻接矩阵
def Construct_Matrix_W(data, k=5):
    rows = len(data)  # 取出数据行数
    W = np.zeros((rows, rows))  # 对矩阵进行初始化：初始化W为rows*rows的方阵
    for i in range(rows):  # 遍历行
        for j in range(rows):  # 遍历列
            if (i != j):  # 计算不重复点的距离
                W[i][j] = calculate_w_ij(data[i,:], data[j,:])  # 调用函数计算距离
        t = np.argsort(W[i, :])  # 对W中进行行排序，并提取对应索引
        for x in range(rows - k):  # 对W进行处理
            W[i][t[x]] = 0
    W = (W + W.T) / 2  # 主要是想处理可能存在的复数的虚部，都变为实数
    return W


def Calculate_Matrix_L_sym(W):  # 计算标准化的拉普拉斯矩阵
    degreeMatrix = np.sum(W, axis=1)  # 按照行对W矩阵进行求和
    L = np.diag(degreeMatrix) - W  # 计算对应的对角矩阵减去w
    # 拉普拉斯矩阵标准化，就是选择Ncut切图
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))  # D^(-1/2)
    L_sym = np.dot(np.dot(sqrtDegreeMatrix, L), sqrtDegreeMatrix)  # D^(-1/2) L D^(-1/2)
    return L_sym


def normalization(matrix):  # 归一化
    print("matrix",matrix.shape,matrix)
    #sum = np.sqrt(np.sum(matrix ** 2, axis=1, keepdims=True))  # 求数组的正平方根
    ormed_matrix = normalize(matrix, axis=1, norm='l2')
    #nor_matrix = matrix / sum  # 求平均
    print("ormed_matrix=",ormed_matrix)
    return ormed_matrix


W = Construct_Matrix_W(X)  # 计算邻接矩阵
L_sym = Calculate_Matrix_L_sym(W)  # 依据W计算标准化拉普拉斯矩阵
lam, H = np.linalg.eig(L_sym)  # 特征值分解

t = np.argsort(lam)  # 将lam中的元素进行排序，返回排序后的下标
print("lam=",lam)
print("t=",t)
H = np.c_[H[:, t[0]], H[:, t[1]]]  # 0和1类的两个矩阵按行连接，就是把两矩阵左右相加，要求行数相等。
H = normalization(H)  # 归一化处理

# Kmeans Test
model_k = KMeans(n_clusters=2)  # 新建20簇的Kmeans模型
model_k.fit(X)  # 训练
labels = model_k.labels_  # 得到聚类后的每组数据对应的标签类型
res = np.c_[X, labels]  # 按照行数连接data和labels
# if y is the assignment of the N points to clusters 0 and 1, then plot as
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s = 10, c = labels)

model = KMeans(n_clusters=2)  # 新建20簇的Kmeans模型
model.fit(H)  # 训练
labels = model.labels_  # 得到聚类后的每组数据对应的标签类型

res = np.c_[X, labels]  # 按照行数连接data和labels

print("res",res)
# if y is the assignment of the N points to clusters 0 and 1, then plot as
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s = 10, c = labels)

plt.show()

# YOUR CODE GOES HERE
