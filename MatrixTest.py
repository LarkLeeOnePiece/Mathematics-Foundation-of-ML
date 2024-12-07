import numpy as np

# 定义一个4x4的矩阵
matrix = np.array([[2, 1, -1, 3],
                   [0, -1, 2, -4],
                   [-2, 3, 1, 0],
                   [1, 2, 0, -1]])

# 使用numpy.linalg.inv函数计算逆矩阵
inverse_matrix = np.linalg.inv(matrix)

# 定义三维旋转函数
def rotation_matrix_x(angle_degrees):
    angle_radians = np.radians(angle_degrees)
    c = np.cos(angle_radians)
    s = np.sin(angle_radians)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rotation_matrix_y(angle_degrees):
    angle_radians = np.radians(angle_degrees)
    c = np.cos(angle_radians)
    s = np.sin(angle_radians)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def rotation_matrix_z(angle_degrees):
    angle_radians = np.radians(angle_degrees)
    c = np.cos(angle_radians)
    s = np.sin(angle_radians)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

# 旋转角度（度）
angle_x = -30
angle_y = -45
angle_z = -60

# 计算绕x、y、z轴的旋转矩阵
R_x = rotation_matrix_x(angle_x)
R_y = rotation_matrix_y(angle_y)
R_z = rotation_matrix_z(angle_z)

# 打印旋转矩阵
print("Rotation Matrix around X-axis:")
print(R_x)

print("\nRotation Matrix around Y-axis:")
print(R_y)

print("\nRotation Matrix around Z-axis:")
print(R_z)

print("\After combination:")
print(np.dot(np.dot(R_x, R_y),R_z))# I guess the order is Z * Y *X

import numpy as np

# 定义矩阵
ofGlobalToCamMat = np.array([
    [1, 0, 0, -0],
    [0, 1, 0, -0],
    [0, 0, 1, -4],
    [0, 0, 0, 1]
])

ofCameraToCanonical_Morth = np.array([
    [0.25, 0, 0, -0],
    [0, 0.25, 0, -0],
    [0, 0, 0.333333, 1],
    [0, 0, 0, 1]
])

ofCanonicaltoScreen_Mvp = np.array([
    [512, 0, 0, 511],
    [0, 384, 0, 383],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# 计算乘积
result = np.dot(np.dot(ofCanonicaltoScreen_Mvp, ofCameraToCanonical_Morth), ofGlobalToCamMat)
print("result")
print(result)
vertices1=np.array([5.92969, 4.125, 0,1])
frag1=np.dot(result,vertices1)
print("frag2",frag1)
vertices2=np.array([5.38719,4.125,2.7475,1])
frag2=np.dot(result,vertices2)
print("frag2",frag2)
vertices3=np.array([5.2971,4.49414,2.70917,1])
frag3=np.dot(result,vertices3)
print("frag3",frag3)



import numpy as np

# 定义旋转函数
def rotation_matrix(angle):
    angle_rad = np.deg2rad(angle)
    return np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
# 定义平移矩阵
def translation_matrix(shift):
    return np.array([
        [1, 0, 0],
        [0, 1, shift],
        [0, 0, 1]
    ])
M1=np.dot(translation_matrix(1),rotation_matrix(-45))
M2=np.dot(translation_matrix(1),rotation_matrix(-135))
M3=np.dot(translation_matrix(1),rotation_matrix(-45))
M=np.dot(np.dot(M1,M2),M3)
point=np.array([0,1,1])
print(f"M1={M1}\n,M2={M2}\n,M3={M3},\nM={M}")
print(np.dot(M,point))