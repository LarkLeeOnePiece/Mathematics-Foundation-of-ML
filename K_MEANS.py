# author: Da Li
# University: KAUST
# Usage: some wheels for k-means algorithm
# theory:
#Given x1,x2...xn and z1,z2,...zk(data, representatives)
#repeat
    #update partition : assign xi to Gj, make sure ||xi-zj||^2 minimum
    #update representatives
#until z1,z2,...zk remain unchanged  #convert to J unchanged
import numpy as np
from numpy import linalg as la
def k_means(data:'array',K:"int"=3,maxiters:'int'=30,loss:'f32'=0.00001):
    J_List=[]
    N=data.shape[0]# row of data
    assignment=np.zeros(N)
    initial=np.random.choice(N,K,replace=False)
    reps=data[initial,:]#
    for j in range(K):#update assignment, otherwise some groups number is 0
        assignment[initial[j]]=j
    distance=np.zeros(N)
    Jprev=np.infty
    for iter in range(maxiters):
        for i in range(N):
            ci=np.argmin([la.norm(data[i]-reps[p])for p in range(K)])
            assignment[i]=ci
            distance[i]=la.norm(data[i]-reps[ci])
        for j in range(K):#update reprentatives
            group=[i for i in range(N) if assignment[i]==j]
            SUM=np.sum(data[group],axis=0)
            lenN=len(group)
            #print(lenN)
            reps[j]=SUM/lenN
        J=la.norm(distance)**2/N# **2 means ^2
        if(iter>maxiters) or np.abs(J-Jprev)<(loss)*J:
            break
        Jprev=J
        J_List.append(Jprev)
    print("iter:",iter+1)
    return assignment,reps,J_List
