import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
import time
from sklearn.cluster import KMeans
import math
import sys
from itertools import product
import os
from scipy.linalg import qr
from networkx.generators.community import LFR_benchmark_graph

# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
print(current)
# Getting the parent directory name where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to the sys.path.
sys.path.append(parent)

from load_datasets import karate_club,dolphins,football,polbooks
from utils import eign_computing,SSBM,plot_heatmap,OrderedLabelEncoder,plot_heatmap_triangle,Kempe_2004_KT, construct_DS_matrix

def check_sign(w2_pre,w2):
    eigen_sign = np.sign(w2_pre) == np.sign(w2)
    eigen_sign = eigen_sign.astype(int)
    eigen_sign[eigen_sign==0] = -1
    # print(eigen_sign)
    return eigen_sign

def Noisy_PM_K2(A,W,L,T1,T2,n,w1_t,w2_t):
    # print(T1)
    # print(T2)
    # initialization       
    # Power method
    w1 = w1_t.copy()
    w2 = w2_t.copy()
    for t in range(T1):
        # calculate u_1 (A u_1)^H u'
        w1 = np.matmul(A,w1)   # A*u1
        lambda1_2 = w1*w1
        for round in range(L):
            lambda1_2 = np.matmul(W,lambda1_2)
        w1 = w1 / np.sqrt(n*lambda1_2)
        lambda1 = np.sqrt(n*lambda1_2)
        
    for t in range(T2):
        # compute lambda*u_1*u_1^H u_2
        temp = w1*w2  # element-wise multiplication
        # gossip with DS weight matrix
        for round in range(L):
            temp = np.matmul(W,temp)

        weighted_w1 = lambda1*w1*n*temp
        
        # Do => A u' - lambda_1 u_1 (u_1)^H u' 
        w2_pre = w2.copy()
        w2 = np.matmul(A,w2) - weighted_w1
        eigen_sign = check_sign(w2_pre,w2)
        
        q = w2*w2   # element-wise multiplication
        # gossip with DS weight matrix
        for round in range(L):
            q = np.matmul(W,q) 
        # Normalization
        w2 = w2 / np.sqrt(n*q)
        lambda2 = eigen_sign * np.sqrt(n*q)
 
    lambda_mat = np.hstack((lambda1.reshape(-1,1),lambda2.reshape(-1,1)))

    return w1,w2,lambda_mat

def Compute_next_eigenvector(U,A,W,L,T,w_k,lambda_mat):
    '''
    U: n by k-1 eigenspace
    return n by k eigenspace
    '''
    n = U.shape[0]      # number of nodes
    # initialization
    tic = time.time()
    U_t = U.copy()
    U_weighted = U.copy()
    # Power method
    for t in range(T):
        for index in range(U.shape[1]):
            # compute u_i^H u_k
            U_t[:,index] = U[:,index]*w_k  # element-wise multiplication
            # gossip with DS weight matrix
            for round in range(L):
                U_t[:,index] = np.matmul(W, U_t[:,index])
            U_weighted[:,index] = lambda_mat[:,index]*U[:,index]*n*U_t[:,index]  # element-wise multiplication
        
        # Do : A u' - \sum_i lambda_i u_i (u_i)^H u'
        wk_pre = w_k.copy() 
        w_k = np.matmul(A,w_k) - U_weighted.sum(axis=1)
        eigen_sign = check_sign(wk_pre,w_k)
        
        q = w_k*w_k   # element-wise multiplication
        # gossip with DS weight matrix
        for rou in range(L):
            q = np.matmul(W,q) 
        # Normalization
        w_k = w_k / np.sqrt(n*q)
        lambdak = eigen_sign * np.sqrt(n*q)
      
    V = np.hstack((U,w_k.reshape(-1,1)))
    lambda_new = np.hstack((lambda_mat,lambdak.reshape(-1,1)))
    
    return V,lambda_new


def Myalgo(adj_mat:np.array,T_list:list,L,V_init,U_gt, the_num_eigevectors,epsilon):

    # parameters
    num_nodes = len(adj_mat)
    W = construct_DS_matrix(num_nodes,adj_mat)
        
    u1_initial = V_init[:,0]
    u2_initial = V_init[:,1]
    
    u1,u2,lambda_mat = Noisy_PM_K2(A = adj_mat,W=W,L=L,T1=T_list[0],T2 = T_list[1],n=num_nodes,w1_t=u1_initial,w2_t=u2_initial)
    U = np.hstack((u1.reshape(-1,1),u2.reshape(-1,1)))
    num_clusters = 2
    
    for i in range(2,the_num_eigevectors):
        # print("Computing {} th eigenvecotrs".format(i))
        w_init = V_init[:,i]
        U,lambda_mat = Compute_next_eigenvector(U=U,A=adj_mat,W=W,L=L,T=T_list[i],w_k=w_init,lambda_mat=lambda_mat)
        num_clusters+= 1

        
    error = np.linalg.norm(np.matmul(U,U.T)-np.matmul(U_gt[:,:num_clusters],U_gt[:,:num_clusters].T))
    return error


def compute_communiction_complexity(T_list):
    complexity = 0
    for i in range(len(T_list)):
        complexity += i *T_list[i]
        
    return complexity
        
    

if __name__ == '__main__':
    
    total_num_clusters = 5
    the_num_eigevectors = 5
    seed = total_num_clusters
    np.random.seed(seed)
          
    n1 = 35
    n2 = 30
    n3 = 25
    n4 = 20
    n5 = 15

    print(n1,n2,n3,n4,n5)
    # eigenValues,eigenVectors,adj,gt = SSBM(sizes=[n1,n2,n3,n4,n5],alpha=25,beta=5,num_clusters=total_num_clusters,seed = 0,isplot=False)
    adj = np.load("./numpy_array/SBM/SBMa15b0_5_adj.npy")
    gt = np.load("./numpy_array/SBM/SBMa15b0_5_gt.npy")
    eigenValues,eigenVectors = eign_computing(adj,name = "adj")
    
    eigenValues_sorted = sorted(eigenValues, key=abs,reverse=True)
    # print(eigenValues_sorted)
    sort_index = sorted(range(len(eigenValues)), key=lambda k: np.abs(eigenValues[k]),reverse=True)
    eigenVectors_sorted = eigenVectors[:,sort_index]
    
    num_nodes = adj.shape[0]
    
    W = construct_DS_matrix(num_nodes=num_nodes,adj_mat = adj)
    
    # V_init = np.random.normal(0, 1/num_nodes, size=(num_nodes,the_num_eigevectors))
    # np.save("./numpy_array/SBM/initial_vector"+ str(seed) ,V_init)
    V_init = np.load("./numpy_array/SBM/initial_vector" + str(seed) + ".npy")
    # print(V_init.shape)
    L_set =[50,75,100]
    epsilon_list = [1e-1,1e-2,1e-3]
    
    # Kempe
    for item in L_set:
        # print(item)
        for epsilon in epsilon_list:
            # print(epsilon)
            num_PI, est_error = Kempe_2004_KT(A=adj,L=item,n=num_nodes,V =V_init,W =W ,epsilon = epsilon,U_gt = eigenVectors_sorted[:,:the_num_eigevectors])
            if est_error:
                print("epsilon:",epsilon,"L:",item, "The number of power iterations:", num_PI)
                print("Complexity:", the_num_eigevectors**2*num_PI)
                print("error:", est_error)