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
# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
print(current)
# Getting the parent directory name where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to the sys.path.
sys.path.append(parent)

from load_datasets import karate_club,dolphins,football,polbooks
from utils import eign_computing,SSBM,plot_heatmap,OrderedLabelEncoder,plot_heatmap_triangle

def Noisy_PM_K2(A,W,L,T1,T2,n,w1_t,w2_t):
    # initialization
    tic = time.time()        
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
        w2 = np.matmul(A,w2) - weighted_w1
        q = w2*w2   # element-wise multiplication
        # gossip with DS weight matrix
        for rou in range(L):
            q = np.matmul(W,q) 
        # Normalization
        w2 = w2 / np.sqrt(n*q)
        lambda2 = np.sqrt(n*q)
        # print(w1,np.linalg.norm(w1[:15]))
    lambda_mat = np.hstack((lambda1.reshape(-1,1),lambda2.reshape(-1,1)))
    print("Time to calculate first and second largest eigenvector:",time.time()-tic)

    return w1,w2,lambda_mat

def Compute_next_eigenvector(U,A,W,L,T,w_k,lambda_mat,num_clusters):
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
        w_k = np.matmul(A,w_k) - U_weighted.sum(axis=1)
        q = w_k*w_k   # element-wise multiplication
        # gossip with DS weight matrix
        for rou in range(L):
            q = np.matmul(W,q) 
        # Normalization
        w_k = w_k / np.sqrt(n*q)
        lambdak = np.sqrt(n*q)
      
    V = np.hstack((U,w_k.reshape(-1,1)))
    lambda_new = np.hstack((lambda_mat,lambdak.reshape(-1,1)))
    
    return V,lambda_new

def constuct_DS_matrix(num_nodes,adj_mat):
    # construct doubly stochastic weight matrix
    degree = np.sum(adj_mat,axis=1)
    W = np.zeros((num_nodes,num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mat[i,j] == 1:
                # den = 1 + max(degree[i],degree[j])
                den = max(degree[i],degree[j])
                W[i,j] = 1/den
        W[i,i] = 1-np.sum(W[i,:])
    return W

def constuct_normalized_matrix(num_nodes,adj_mat):
    # construct normalized matrix
    degree = np.sum(adj_mat,axis=1)
    W = np.zeros((num_nodes,num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mat[i,j] == 1:
                den = np.sqrt(degree[i])*np.sqrt(degree[j])
                W[i,j] = 1/den
        W[i,i] = 1-np.sum(W[i,:])
    return W

    # local clustering
    # num_clusters = 4
    # n = range(2,2+count+1)
    # plt.figure()
    # plt.plot(n,centralized_result,'bo--',linewidth=2,markersize=12,label="Centralized")
    # plt.plot(n,score_set,'rs-',linewidth=2,markersize=12,label="Decentralized")
    # # plt.yticks(range(0,100,10))
    # plt.xlabel("The number of eigenvector",fontsize=16)
    # plt.ylabel("Small centroid",fontsize=16)
    # plt.title("SBM,K={}".format(num_clusters))
    # # plt.title("Karate Club,K={}".format(num_clusters))
    # plt.grid(alpha=0.4)
    # plt.legend(loc="lower right")
    # savepath = "./dataset/2means_"+str(num_clusters)+"_SBM_de.jpg"
    # # savepath = "./dataset/2means_Karate_de.jpg"
    # plt.savefig(savepath,dpi=300)
    # print(score_set)


def centralized_Kmeans_plus(X,gt,num_clusters,init_centroids):
    # central = KMeans(n_clusters=num_clusters,random_state=0,n_init="auto",init = init_centroids).fit(X)
    central = KMeans(n_clusters=num_clusters,random_state=0,n_init=1,init = "k-means++").fit(X)
    le = OrderedLabelEncoder()
    pseudo_label = le.fit_transform(central.labels_) 
    accuracy = accuracy_score(gt,pseudo_label)
    
    return accuracy
def centralized_Kmeans_fixed(X,gt,num_clusters,init_centroids):
    central = KMeans(n_clusters=num_clusters,random_state=0,n_init=1,init = init_centroids).fit(X)
    # central = KMeans(n_clusters=num_clusters,random_state=0,n_init=1,init = "k-means++").fit(X)
    le = OrderedLabelEncoder()
    pseudo_label = le.fit_transform(central.labels_) 
    accuracy = accuracy_score(gt,pseudo_label)
    
    return accuracy

def termination(mse,mse_old,delta_min):
    # print("mse",mse)
    # print("mse decreasing ratio",(mse_old-mse)/mse_old)
    if np.abs((mse_old-mse)/mse_old) < delta_min : 
        nonstop = False
    else:
        nonstop = True
        mse_old = mse
    
    return nonstop,mse_old

def generate_centroids(data:np.array,num_clusters,dim_d):
    # data: K dim vector, a row of top K eigenspace
    
    sign_data = np.flip(np.sign(data))      # flip to meet the format of binary counting
    sign_data[sign_data==-1] = 0
    
    C_index = 0
    for i in range(dim_d-1):    # ignore the first eigenvector
        C_index += sign_data[i]* 2**(i)

    if C_index <=1:
        C_index  = 0
    else:
        # C_index = round(np.log2(C_index))
        C_index = math.floor(np.log2(C_index))
    # print(C_index)

    centroids = np.zeros((num_clusters,dim_d))
    centroids[C_index,:] = data.copy()
    
    return C_index,centroids
    
def decentralized_Kmeans_sign(num_nodes,W,L,X,num_clusters,dim_data):
    # intialization
    # X : dictionary , with np array
    D = dim_data  # dimension of each data
    mse_old = 1e+7
    delta_min = 1e-2
    
    centroids = np.zeros((num_nodes,num_clusters,dim_data))
    Index_set = []
    for i in range(num_nodes):
        C_index,centroids[i,:,:] = generate_centroids(data = X[i],num_clusters=num_clusters,dim_d = D)
        Index_set.append(C_index)

    # le = OrderedLabelEncoder()
    # pseudo_label = le.fit_transform(Index_set)
    plt.figure()
    plt.plot(Index_set)
    plt.savefig("initial.jpg",dpi=300)
    # assert False
    nonstop = True
    count = 0
    print("Decentralized K-means")
    while  nonstop :
        print(count,nonstop)
        num = np.zeros((num_nodes,num_clusters))
        summation = np.zeros((num_nodes,num_clusters,D))
        square_error = np.zeros((num_nodes,num_clusters))
        w_kplus1 = np.ones(num_nodes)
        
        # cluster assignment
        cluster_dict = dict()
        for i in range(num_nodes):
            mu = centroids[i,:,:]           # K by D
            if X[i].ndim==1:
                data = X[i].reshape(-1,D)
            else:
                data = X[i]
                
            for j in range(data.shape[0]):
                diff = mu -np.repeat(data[j,:].reshape(1,-1),num_clusters,axis=0)
                obj = np.linalg.norm(diff, axis=1)
                #print(obj)  # 1-d
                cluster = np.argmin(obj)
                # print("node",i,cluster)
                cluster_dict[i] = cluster
                num[i,cluster] += 1
                summation[i,cluster,:] += data[j,:]
                square_error[i,cluster] += np.linalg.norm(data[j,:]-mu[cluster,:])**2
        # gossip
        summation = summation.reshape((num_nodes,-1))
        for l in range(L):
            num = np.matmul(W,num)
            summation = np.matmul(W,summation)
            square_error = np.matmul(W,square_error)
            w_kplus1 = np.matmul(W,w_kplus1) 
        summation = summation.reshape((num_nodes,num_clusters,-1))
    
        # codebook update
        for i in range(num_nodes):
            nonzero_index = np.where(num[i,:]!=0)[0]
            zero_index = list(np.where(num[i,:]==0)[0])
            denum = np.repeat(num[i,:].reshape(-1,num_clusters),D,axis=0)
            denum = denum.T
            centroids[i,nonzero_index,:] = summation[i,nonzero_index,:]/denum[nonzero_index,:]
            if list(zero_index) != 0:
                for item in zero_index:
                    print("Node",i,"exists empty clusters.")
                    
                    centroids[i,item,:] = np.random.rand(1,D)
                    
            square_error[i,:] = square_error[i,:]/w_kplus1[i]
            # print(centroids.shape)
        mse = np.mean(square_error[i,:])
        mu = centroids[i,:,:].copy()
        count +=1
        
        # termination conditions
        if np.isnan(mse):
            # print(denum)
            print(centroids.reshape(num_nodes,-1)[i,:])
            assert False
        
        nonstop, mse_old = termination(mse,mse_old,delta_min) 
        
    return centroids.reshape(num_nodes,-1), list(cluster_dict.values())

def decentralized_Kmeans_rand_partition(num_nodes,W,L,X,num_clusters,dim_data,init_centroids):
    # intialization
    # X : dictionary , with np array
    D = dim_data  # dimension of each data
    mse_old = 1e+7
    delta_min = 1e-2
    
    # X = {0:np.array([[3,4,5,6],[0,0,1,0]])}
    # centroids = np.random.rand(num_nodes,num_clusters,D)
    centroids = init_centroids.copy()
    nonstop = True
    count = 0
    print("Decentralized K-means")
    while  nonstop :
        print(count,nonstop)
        num = np.zeros((num_nodes,num_clusters))
        summation = np.zeros((num_nodes,num_clusters,D))
        square_error = np.zeros((num_nodes,num_clusters))
        w_kplus1 = np.ones(num_nodes)
        
        # cluster assignment
        cluster_dict = dict()
        for i in range(num_nodes):
            mu = centroids[i,:,:]           # K by D
            if X[i].ndim==1:
                data = X[i].reshape(-1,D)
            else:
                data = X[i]
                
            for j in range(data.shape[0]):
                diff = mu -np.repeat(data[j,:].reshape(1,-1),num_clusters,axis=0)
                obj = np.linalg.norm(diff, axis=1)
                #print(obj)  # 1-d
                cluster = np.argmin(obj)
                # print("node",i,cluster)
                cluster_dict[i] = cluster
                num[i,cluster] += 1
                summation[i,cluster,:] += data[j,:]
                square_error[i,cluster] += np.linalg.norm(data[j,:]-mu[cluster,:])**2
        # gossip
        summation = summation.reshape((num_nodes,-1))
        for l in range(L):
            num = np.matmul(W,num)
            summation = np.matmul(W,summation)
            square_error = np.matmul(W,square_error)
            w_kplus1 = np.matmul(W,w_kplus1) 
        summation = summation.reshape((num_nodes,num_clusters,-1))

        
        # codebook update
        for i in range(num_nodes):
            nonzero_index = np.where(num[i,:]!=0)[0]
            zero_index = list(np.where(num[i,:]==0)[0])
            denum = np.repeat(num[i,:].reshape(-1,num_clusters),D,axis=0)
            denum = denum.T
            centroids[i,nonzero_index,:] = summation[i,nonzero_index,:]/denum[nonzero_index,:]
            if list(zero_index) != 0:
                for item in zero_index:
                    print("Node",i,"exists empty clusters. rp")
                    # print(centroids[i,item,:])
                    
                    # centroids[i,item,:] = np.random.rand(1,D)
                    # print(centroids[i,item,:])
            square_error[i,:] = square_error[i,:]/w_kplus1[i]
            # print(centroids.shape)
        mse = np.mean(square_error[i,:])
        mu = centroids[i,:,:].copy()
        if count == 0:
            first_centroids = mu.copy()
        count +=1
        
        # termination conditions
        if np.isnan(mse):
            # print(denum)
            print(centroids.reshape(num_nodes,-1)[i,:])
            assert False
        
        nonstop, mse_old = termination(mse,mse_old,delta_min) 
        
    return first_centroids,centroids.reshape(num_nodes,-1), list(cluster_dict.values())

def Myclustering_with_K(adj_mat:np.array,T:int,L:int,seed:int,isself,gt, total_num_clusters):
    
    # fix seed
    np.random.seed(seed)
    # parameters
    num_nodes = len(adj_mat)
    W = constuct_DS_matrix(num_nodes,adj_mat)
    if isself:
        adj_mat = adj_mat + num_nodes*np.eye(num_nodes)
        
    u1_initial = np.random.normal(0, 1/num_nodes, size=(num_nodes))
    u2_initial = np.random.normal(0, 1/num_nodes, size=(num_nodes))
    u1,u2,lambda_mat = Noisy_PM_K2(A = adj_mat,W=W,L=L,T1=T,T2=T,n=num_nodes,w1_t=u1_initial,w2_t=u2_initial)

    U = np.hstack((u1.reshape(-1,1),u2.reshape(-1,1)))
    
    # print(centroids)
    for i in range(3,total_num_clusters+1):
        print("Computing {} th eigenvecotrs".format(i))
        w_init = np.random.normal(0, 1/num_nodes, size=(num_nodes))
        U,lambda_mat = Compute_next_eigenvector(U=U,A=adj_mat,W=W,L=L,T=T,w_k=w_init,lambda_mat=lambda_mat,num_clusters=i)
        

    X = dict()
    X_data = []
    for i in range(num_nodes):
        X[i] = np.matmul(U[i,:],np.diag(lambda_mat[i,:]))
        X_data.append(np.matmul(U[i,:],np.diag(lambda_mat[i,:])))
        
    init_centroids = np.random.rand(num_nodes,total_num_clusters,total_num_clusters)
    
    acc_dec_8,first_centroids = DKmeans_to_acc(num_nodes,W,80,X,num_clusters=total_num_clusters,dim_data=total_num_clusters,init_centroids = init_centroids)
    init_centroids = np.repeat(first_centroids.reshape(1,-1,total_num_clusters),num_nodes,axis=0)
    print(init_centroids.shape)
    acc_dec_12,_ = DKmeans_to_acc(num_nodes,W,16,X,num_clusters=total_num_clusters,dim_data=total_num_clusters,init_centroids = init_centroids)
    acc_dec_16,_ = DKmeans_to_acc(num_nodes,W,16,X,num_clusters=total_num_clusters,dim_data=total_num_clusters,init_centroids = init_centroids)
    acc_dec_20,_ = DKmeans_to_acc(num_nodes,W,16,X,num_clusters=total_num_clusters,dim_data=total_num_clusters,init_centroids = init_centroids)
    
    # centroids,label = decentralized_Kmeans_sign(num_nodes,W,L,X,num_clusters=total_num_clusters,dim_data=total_num_clusters)
    # pseudo_label = le.fit_transform(label) 
    # acc_dec_sign = accuracy_score(gt,pseudo_label)

    X_array = np.array(X_data)
    print(X_array)

    acc_cen_fixed = centralized_Kmeans_fixed(X = X_array, gt = gt,num_clusters = total_num_clusters, init_centroids = first_centroids)   
    
    return acc_cen_fixed, acc_dec_8,acc_dec_12,acc_dec_16,acc_dec_20

def DKmeans_to_acc(num_nodes,W,L,X,num_clusters,dim_data,init_centroids):
    first_centroids,centroids,label = decentralized_Kmeans_rand_partition(num_nodes,W,L,X,num_clusters,dim_data, init_centroids = init_centroids)
    le = OrderedLabelEncoder()
    pseudo_label = le.fit_transform(label) 
    acc_dec = accuracy_score(gt,pseudo_label)
    
    return acc_dec,first_centroids

if __name__ == '__main__':
    
    # Synthetic dataset

    realizations = 10

    result_cen_fixed = []
    result_dec_8 = []      # random partition
    result_dec_12 = []
    result_dec_16 = [] 
    result_dec_20 = [] 
    

    total_num_clusters = 5
    for index in range(realizations):
        n1 = int(np.random.uniform(200,250))
        n2 = int(np.random.uniform(200,250))
        n3 = int(np.random.uniform(200,250))
        
        n4 = int(np.random.uniform(200,250))
        n5 = int(np.random.uniform(200,250))
        n6 = int(np.random.uniform(200,250))
        n7 = int(np.random.uniform(200,250))
        print(n1,n2,n3)
        # n1 = n2 = n3 = n4 = n5 = 20
        if total_num_clusters == 7:
            eigenValues,eigenVectors,adj_mat,gt = SSBM(sizes=[n1,n2,n3,n4,n5,n6,n7],alpha=30,beta=8,num_clusters=total_num_clusters,seed = index+34,isplot=False)
        elif total_num_clusters ==5:
            eigenValues,eigenVectors,adj_mat,gt = SSBM(sizes=[n1,n2,n3,n4,n5],alpha=30,beta=8,num_clusters=total_num_clusters,seed = index+34,isplot=False)
        else:
            eigenValues,eigenVectors,adj_mat,gt = SSBM(sizes=[n1,n2,n3],alpha=30,beta=8,num_clusters=total_num_clusters,seed = index+34,isplot=False)
        
        acc_cen_fixed, acc_dec_8,acc_dec_12,acc_dec_16,acc_dec_20= Myclustering_with_K(adj_mat = adj_mat,T=50,L=50,seed=index,isself=False,gt = gt,
                                                               total_num_clusters = total_num_clusters)
        
        result_cen_fixed.append(acc_cen_fixed)
        result_dec_8.append(acc_dec_8)
        result_dec_12.append(acc_dec_12)
        result_dec_16.append(acc_dec_16)
        result_dec_20.append(acc_dec_20)
        
    print("Accuracy of centralized spectral clustering with random initial:",np.array(result_cen_fixed).mean(),np.array(result_cen_fixed).std())
    print("Accuracy of decentralized spectral clustering L=4",np.array(result_dec_8).mean(),np.array(result_dec_8).std())  
    print("Accuracy of decentralized spectral clustering L=16",np.array(result_dec_12).mean(),np.array(result_dec_12).std())  
    print("Accuracy of decentralized spectral clustering L=64",np.array(result_dec_16).mean(),np.array(result_dec_16).std())  
    print("Accuracy of decentralized spectral clustering L=128",np.array(result_dec_20).mean(),np.array(result_dec_20).std())  
      
    # print(eigenValues)
    # print(eigenVectors[:,0])
    # AA = np.array([[1,2,3],[2,3,4],[3,4,5]])
    # value,vector = eign_computing(AA,name="test")
    # print(value)
    # print(vector)
    # s,v,d = np.linalg.svd(AA)
    # print(v)
    # print(s)
    # print(d)
    
    # eigenValues,eigenVectors,adj_mat,gt = SSBM(sizes=[50,50,50,50],alpha=20,beta=5,num_clusters=4,seed = 0,isplot=False)

    # corner case
    # eigenValues,eigenVectors,adj_mat,gt = SSBM(sizes=[40,50,30,30],alpha=20,beta=5,num_clusters=4,seed = 10,isplot=False)

    # eigenValues,eigenVectors,adj_mat,gt = SSBM(sizes=[50,35,50,50,60,60,45,30],alpha=32,beta=8,num_clusters=8,seed = 10,isplot=False)

    # eigenValues,eigenVectors,adj_mat,gt = SSBM(sizes=[40,50,40,30,41],alpha=20,beta=5,num_clusters=5,seed = 0,isplot=False)
    # eigenValues,eigenVectors,adj_mat,gt = SSBM(sizes=[40,45,40,30,41],alpha=20,beta=5,num_clusters=5,seed = 0,isplot=False)
    
    # Real-world dataset

    # adj_mat, gt = karate_club(isplot=False,isKmeans = False)
    # adj_mat, gt = dolphins(isplot=False)
    # adj_mat, gt = polbooks(isplot=False)


    # num_clusters = 4
    # result_all = np.array(result_all)
    # n = range(2,10,2)
    # print(n)
    # plt.figure()
    # plt.plot(n,result_all[0,:],'bo--',linewidth=2,markersize=12,label="T=3")
    # plt.plot(n,result_all[1,:],'ro--',linewidth=2,markersize=12,label="T=4")
    # plt.plot(n,result_all[2,:],'go--',linewidth=2,markersize=12,label="T=5")
    # plt.plot(n,result_all[3,:],'yo--',linewidth=2,markersize=12,label="T=6")
    # plt.plot(n,result_all[4,:],'mo--',linewidth=2,markersize=12,label="T=7")
    # plt.xlabel("The number of gossip iterations",fontsize=16)
    # plt.ylabel("NMI",fontsize=16)
    # plt.title("SBM,K={}".format(num_clusters))
    # plt.grid(alpha=0.4)
    # plt.legend(loc="lower right")
    # savepath = str(num_clusters)+"_SBM_de.jpg"
    # # savepath = "./dataset/2means_Karate_de.jpg"
    # plt.savefig(savepath,dpi=300)