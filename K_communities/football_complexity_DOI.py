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

# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
print(current)
# Getting the parent directory name where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to the sys.path.
sys.path.append(parent)

from load_datasets import karate_club,dolphins,football,polbooks,football_localfile
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


def Myalgo(adj_mat:np.array,T_list:list,L,V_init,U_gt, total_num_clusters,epsilon):

    # parameters
    num_nodes = len(adj_mat)
    W = construct_DS_matrix(num_nodes,adj_mat)
        
    u1_initial = V_init[:,0]
    u2_initial = V_init[:,1]
    
    u1,u2,lambda_mat = Noisy_PM_K2(A = adj_mat,W=W,L=L,T1=T_list[0],T2 = T_list[1],n=num_nodes,w1_t=u1_initial,w2_t=u2_initial)
    U = np.hstack((u1.reshape(-1,1),u2.reshape(-1,1)))
    num_clusters = 2

    # if np.linalg.norm(np.matmul(U,U.T)-np.matmul(U_gt[:,:num_clusters],U_gt[:,:num_clusters].T)) > epsilon:
    #     # print(np.linalg.norm(np.matmul(U,U.T)-np.matmul(U_gt[:,:num_clusters],U_gt[:,:num_clusters].T)))
    #     # print(num_clusters)
    #     error = np.linalg.norm(np.matmul(U,U.T)-np.matmul(U_gt[:,:num_clusters],U_gt[:,:num_clusters].T))
    #     return error
    # print(np.linalg.norm(np.matmul(U,U.T)-np.matmul(U_gt[:,:num_clusters],U_gt[:,:num_clusters].T)))
    for i in range(2,total_num_clusters):
        # print("Computing {} th eigenvecotrs".format(i))
        w_init = V_init[:,i]
        U,lambda_mat = Compute_next_eigenvector(U=U,A=adj_mat,W=W,L=L,T=T_list[i],w_k=w_init,lambda_mat=lambda_mat)
        num_clusters+= 1
        # if np.linalg.norm(np.matmul(U,U.T)-np.matmul(U_gt[:,:num_clusters],U_gt[:,:num_clusters].T)) > epsilon:
        #     # print(np.linalg.norm(np.matmul(U,U.T)-np.matmul(U_gt[:,:num_clusters],U_gt[:,:num_clusters].T)))
        #     # print(num_clusters)
        #     error = np.linalg.norm(np.matmul(U,U.T)-np.matmul(U_gt[:,:num_clusters],U_gt[:,:num_clusters].T))
        #     return error
        
    error = np.linalg.norm(np.matmul(U,U.T)-np.matmul(U_gt[:,:num_clusters],U_gt[:,:num_clusters].T))
    return error


def plot_eigenvalues(centralized, average, outlier,total_num_clusters,name,num_nodes,T,L,save_path):
    labels  = [r"$\lambda_1$",r"$\lambda_2$",r"$\lambda_3$",r"$\lambda_4$",r"$\lambda_5$",r"$\lambda_6$",
               r"$\lambda_7$",r"$\lambda_8$",r"$\lambda_9$",r"$\lambda_{10}$",r"$\lambda_{11}$",r"$\lambda_{12}$",r"$\lambda_{13}$"]
    n = range(1,total_num_clusters+1)
    plt.figure()
    plt.plot(n,centralized,"o-",linewidth = 3,markersize=14,markeredgewidth=3,markerfacecolor="none",label="Exact eigenvalues")
    plt.plot(n,average,"s:",linewidth = 3,markersize=14,markeredgewidth=3,markerfacecolor="none",label="Average of DNPM")
    plt.plot(n,outlier,"x-.",linewidth = 3,markersize=14,markeredgewidth=3,markerfacecolor="none",label="Outlier of DNPM")
    
    plt.xlabel(r"Eigenvalues",fontsize = 20,x=0.5,y=-0.125)
    plt.xlim([1,total_num_clusters])
    plt.xticks(n,labels,fontsize=18)
    plt.yticks(fontsize=18)

    plt.ylabel("Value",fontsize = 20,x=-0.5,y=0.45,rotation=45)
    plt.title(r"$(N,T,L)$=({},{},{})".format(num_nodes,T,L),fontsize = 20,x=0.5, y=1.025)
    # plt.title(r'Estimated error of $2^{nd}$ eigenvector',fontsize = 24)
    # ax.xaxis.set_label_coords(0.5, -0.125)
    # ax.yaxis.set_label_coords(-0.125, 0.5)
    plt.grid(alpha=0.45)
    plt.legend(fontsize=18,loc="lower left")
    plt.savefig(save_path,bbox_inches='tight')

def compute_communiction_complexity(T_list):
    complexity = 0
    for i in range(len(T_list)):
        complexity += i *T_list[i]
        
    return complexity
        
    

if __name__ == '__main__':
    
    total_num_clusters = 11
    seed = total_num_clusters*1              
    np.random.seed(seed)
    print("seed:",seed)


    # adj , gt = football(False)
    adj , gt = football_localfile(False)
    # print(gt)
    num_nodes = adj.shape[0]
    print("Number of nodes", num_nodes)
    eigenValues, eigenVectors = eign_computing(adj,name="football")

    eigenValues_sorted = sorted(eigenValues, key=abs,reverse=True)
    # print("Values",eigenValues)
    sort_index = sorted(range(len(eigenValues)), key=lambda k: np.abs(eigenValues[k]),reverse=True)
    # print("index",sort_index)
    eigenVectors_sorted = eigenVectors[:,sort_index]
    # print(eigenVectors_sorted[:,total_num_clusters])
    # assert False
    W = construct_DS_matrix(num_nodes=num_nodes,adj_mat = adj)
    # print(W.shape)
    V_init = np.random.normal(0, np.sqrt(1/num_nodes), size=(num_nodes,total_num_clusters))
    np.save("./numpy_array/football/initial_vector"+ str(seed) ,V_init)
    # print(V_init.shape)
    # assert False
    V_init = np.load("./numpy_array/football/initial_vector" + str(seed) + ".npy")
    

    U_gt = eigenVectors_sorted[:,:total_num_clusters]
    # print(np.linalg.norm(np.matmul(U_gt,U_gt.T)-np.matmul(V_init,V_init.T)))
    for i in range(total_num_clusters):
        print(i," ",np.sum(V_init[:,i]*U_gt[:,i])/np.linalg.norm(V_init[:,i]),np.linalg.norm(V_init[:,i]),np.linalg.norm(U_gt[:,i]))

    L_set =[40,60,80]
    epsilon_list = [1e-1,1e-2,1e-3]
    
    # Kempe
    for item in L_set:
        for epsilon in epsilon_list:
            num_PI, est_error = Kempe_2004_KT(A=adj,L=item,n=num_nodes,V =V_init,W =W ,epsilon = epsilon,U_gt = eigenVectors[:,:total_num_clusters])
            if est_error:
                print("epsilon:",epsilon,"L:",item, "The number of power iterations:", num_PI)
                print("Complexity:", total_num_clusters**2*num_PI)
                print("error:", est_error)
