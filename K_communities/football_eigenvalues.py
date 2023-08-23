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
import os
from scipy.linalg import qr

# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
print(current)
# Getting the parent directory name where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to the sys.path.
sys.path.append(parent)

from load_datasets import karate_club,dolphins,football,polbooks
from utils import eign_computing,SSBM,plot_heatmap,OrderedLabelEncoder,plot_heatmap_triangle,construct_DS_matrix

def check_sign(w2_pre,w2):
    eigen_sign = np.sign(w2_pre) == np.sign(w2)
    eigen_sign = eigen_sign.astype(int)
    eigen_sign[eigen_sign==0] = -1
    # print(eigen_sign)
    return eigen_sign

def Noisy_PM_K2(A,W,L,T,n,w1_t,w2_t):
    # initialization       
    # Power method
    w1 = w1_t.copy()
    w2 = w2_t.copy()
    for t in range(T):
        # calculate u_1 (A u_1)^H u'
        w1 = np.matmul(A,w1)   # A*u1
        lambda1_2 = w1*w1
        for round in range(L):
            lambda1_2 = np.matmul(W,lambda1_2)
        w1 = w1 / np.sqrt(n*lambda1_2)
        lambda1 = np.sqrt(n*lambda1_2)
        
    for t in range(T):
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

def Myalgo(adj_mat:np.array,T:int,L:int,seed:int,gt, the_num_eigenvectors):
    
    # parameters
    num_nodes = len(adj_mat)
    W = construct_DS_matrix(num_nodes,adj_mat)

    u1_initial = np.random.normal(0, 1/num_nodes, size=(num_nodes))
    u2_initial = np.random.normal(0, 1/num_nodes, size=(num_nodes))
    u1,u2,lambda_mat = Noisy_PM_K2(A = adj_mat,W=W,L=L,T=T,n=num_nodes,w1_t=u1_initial,w2_t=u2_initial)

    U = np.hstack((u1.reshape(-1,1),u2.reshape(-1,1)))
    # print(centroids)
    for i in range(3,the_num_eigenvectors+1):
        print("Computing {} th eigenvecotrs".format(i))
        w_init = np.random.normal(0, 1/num_nodes, size=(num_nodes))
        U,lambda_mat = Compute_next_eigenvector(U=U,A=adj_mat,W=W,L=L,T=T,w_k=w_init,lambda_mat=lambda_mat)
        

    return U, lambda_mat

def find_outlier(data:np.array):
    average = data.mean()
    index = np.argmax(np.abs(data-average))
    outlier = data[index]
    
    return average, outlier,index

# def plot_eigenvalues(centralized, average, outlier,total_num_clusters,name,num_nodes,T,L,save_path):
#     labels  = [r"$\lambda_1$",r"$\lambda_2$",r"$\lambda_3$",r"$\lambda_4$",r"$\lambda_5$",r"$\lambda_6$",
#                r"$\lambda_7$",r"$\lambda_8$",r"$\lambda_9$",r"$\lambda_{10}$",r"$\lambda_{11}$",r"$\lambda_{12}$",r"$\lambda_{13}$"]
#     n = range(1,total_num_clusters+1)
#     plt.figure()
#     plt.plot(n,centralized,"o-",linewidth = 3,markersize=14,markeredgewidth=3,markerfacecolor="none",label="Exact eigenvalues")
#     plt.plot(n,average,"s:",linewidth = 3,markersize=14,markeredgewidth=3,markerfacecolor="none",label="Average of DNPM")
#     plt.plot(n,outlier,"x-.",linewidth = 3,markersize=14,markeredgewidth=3,markerfacecolor="none",label="Outlier of DNPM")
    
#     plt.xlabel(r"Eigenvalues",fontsize = 20,x=0.5,y=-0.125)
#     plt.xlim([1,total_num_clusters])
#     plt.xticks(n,labels,fontsize=18)
#     plt.yticks(fontsize=18)

#     plt.ylabel("Value",fontsize = 20,x=-0.5,y=0.45)
#     plt.title(r"$(N,T,L)$=({},{},{})".format(num_nodes,T,L),fontsize = 20,x=0.5, y=1.025)
#     # plt.title(r'Estimated error of $2^{nd}$ eigenvector',fontsize = 24)
#     # ax.xaxis.set_label_coords(0.5, -0.125)
#     # ax.yaxis.set_label_coords(-0.125, 0.5)
#     plt.grid(alpha=0.45)
#     plt.legend(fontsize=18,loc="lower left")
#     plt.savefig(save_path,bbox_inches='tight')
    
def plot_eigenvalues_range(centralized, lower, upper,the_num_eigenvectors,num_nodes,T,L,save_path):
    labels  = [r"$\lambda_1$",r"$\lambda_2$",r"$\lambda_3$",r"$\lambda_4$",r"$\lambda_5$",r"$\lambda_6$",
               r"$\lambda_7$",r"$\lambda_8$",r"$\lambda_9$",r"$\lambda_{10}$",r"$\lambda_{11}$",r"$\lambda_{12}$",r"$\lambda_{13}$"]
    labels = labels[:the_num_eigenvectors]
    n = range(1,the_num_eigenvectors+1)
    plt.figure()
    plt.plot(n,centralized,"o",linewidth = 3,markersize=14,markeredgewidth=3,markerfacecolor="none",label="Exact eigenvalues")
    plt.plot(n,upper,"-",linewidth = 3,markerfacecolor="none",label=r"Largest estimation")
    plt.plot(n,lower,"-",linewidth = 3,markerfacecolor="none",label=r"Smallest estimation")
    
    plt.xlabel(r"Eigenvalues",fontsize = 20,x=0.5,y=-0.125)
    plt.xlim([1,the_num_eigenvectors])
    plt.xticks(n,labels,fontsize=18)
    plt.yticks(fontsize=18)

    plt.ylabel("Value",fontsize = 20,x=-0.5,y=0.45)
    # plt.title(r"$(N,L)$=({},{})".format(num_nodes,L),fontsize = 20,x=0.5, y=1.025)
    # plt.title(r'Estimated error of $2^{nd}$ eigenvector',fontsize = 24)
    # ax.xaxis.set_label_coords(0.5, -0.125)
    # ax.yaxis.set_label_coords(-0.125, 0.5)
    plt.grid(alpha=0.45)
    plt.legend(fontsize=18,loc="lower left")
    plt.savefig(save_path,bbox_inches='tight')

if __name__ == '__main__':
    
    the_num_eigenvectors =13
    np.random.seed(the_num_eigenvectors)
    
    T = 100
    L = 30
    print(T,L)
    adj , gt = football(False)
    eigenValues, eigenVectors = eign_computing(adj,name="football")
    # print(eigenValues[:the_num_eigenvectors])
    
    eigenValues_sorted = sorted(eigenValues, key=abs,reverse=True)
    # print(eigenValues_sorted)
    sort_index = sorted(range(len(eigenValues)), key=lambda k: np.abs(eigenValues[k]),reverse=True)

    eigenVectors_sorted = eigenVectors[:,sort_index]
    
    U,lambda_mat = Myalgo(adj_mat=adj,T=T,L=L,seed=0,gt=gt, the_num_eigenvectors=the_num_eigenvectors)
    print(lambda_mat[21,:])
    print(U.shape)
    print(eigenVectors_sorted[:,:the_num_eigenvectors].shape)
    error = np.linalg.norm(np.matmul(U,U.T) - np.matmul(eigenVectors_sorted[:,:the_num_eigenvectors],eigenVectors_sorted[:,:the_num_eigenvectors].T))
    print(error)
    
    upper = []
    lower = []
    for i in range(the_num_eigenvectors):
        upp = np.max(lambda_mat[:,i])
        low = np.min(lambda_mat[:,i])
        
        lower.append(low)
        upper.append(upp)
        
    lower= np.array(lower)
    upper = np.array(upper)
    save_path = "football_eigenvalues.eps"
    plot_eigenvalues_range(centralized = eigenValues_sorted[:the_num_eigenvectors], lower = lower, upper = upper,
                           the_num_eigenvectors = the_num_eigenvectors,num_nodes = adj.shape[0],T = T,L=L,save_path = save_path)
    

