import networkx as nx
import math
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import eigh
import numpy as np
import numpy.matlib
from sklearn.metrics.pairwise import euclidean_distances,manhattan_distances,cosine_distances,cosine_similarity

import time
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import column_or_1d

np.set_printoptions(suppress=True)

def ordered_encode_python(values, uniques=None, encode=False):
    # only used in _encode below, see docstring there for details
    if uniques is None:
        uniques = list(dict.fromkeys(values))
        uniques = np.array(uniques, dtype=values.dtype)
    if encode:
        table = {val: i for i, val in enumerate(uniques)}
        try:
            encoded = np.array([table[v] for v in values])
        except KeyError as e:
            raise ValueError("y contains previously unseen labels: %s"
                             % str(e))
        return uniques, encoded
    else:
        return uniques

class OrderedLabelEncoder(LabelEncoder):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = ordered_encode_python(y)
    def fit_transform(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_, y = ordered_encode_python(y, encode=True)
        return y 

def plot_heatmap_triangle(A,save_path="eigenspace.jpg"):
    plt.figure()
    # sns.heatmap(A,vmin=-0.02, vmax=0.1)
    # sns.heatmap(A,cmap="gray")
    
    # x_labels = range(1,35)
    # y_labels = range(1,35)
    # if len(A) > 10:
    sns.heatmap(A,cmap="PiYG")
    # else:
        # sns.heatmap(A,cmap="PiYG", annot=True,vmin=vmin, vmax=vmax)
        
    plt.savefig(save_path)

def plot_heatmap(A,n,K,a,b,save_path="adjacency.jpg",vmin=0,vmax=1):
    plt.figure()
    # sns.heatmap(A,vmin=-0.02, vmax=0.1)
    # sns.heatmap(A,cmap="gray")
    
    # x_labels = range(1,35)
    # y_labels = range(1,35)
    if len(A) > 10:
        sns.heatmap(A,cmap="gray")
    else:
        sns.heatmap(A,cmap="PiYG", annot=True,vmin=vmin, vmax=vmax)
        
    if a==0 and b==0:
        plt.title("n={},K={}".format(n,K))
    else:    
        plt.title("n={},K={},a={},b={}".format(n,K,a,b))
    plt.savefig(save_path)
    
def plot_linechart(result,n,K,L,save_path="error.jpg"):
    plt.figure()
    plt.plot(range(1,len(result)+1),result)
    plt.xlabel("T(iterations)",fontsize = 18)
    plt.title("Decentralzied computing error, (n,K,L)=({},{},{}).".format(n,K,L))
    plt.grid(alpha=0.4)
    plt.savefig(save_path)


def plot_sign_2nd_eigenvector(u2,A,a,b,save_path):
    u2 = np.squeeze(u2)
    u2_sign = np.sign(u2)
    print(u2_sign)
    plt.figure()
    n = range(1,len(u2_sign)+1)
    plt.bar(n,u2)
    plt.xlabel("Node index",fontsize = 18)
    plt.xlim([1,len(u2)])
    plt.ylabel("Value",fontsize = 18)
    plt.title("Second largest eigenvector")
    plt.grid(alpha=0.4)
    plt.savefig(save_path)


def show_graph_with_labels(adjacency_matrix, save_path):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    plt.figure()
    nx.draw(gr, node_size=300,with_labels=True)
    plt.savefig(save_path)
 
def eign_computing(adj_matrix,name,isplot=False):
    eigenValues,eigenVectors = eigh(adj_matrix)
    if isplot:
        print("Eigenvalues of {}".format(name),eigenValues[np.abs(eigenValues)>1e-4])
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    return eigenValues, eigenVectors

def max_min(result,sizes):
    
    result_max = np.zeros((len(sizes),len(sizes)))
    result_min = np.zeros((len(sizes),len(sizes)))
    node_ids = []
    temp = 0
    for i in range(len(sizes)):
        temp += sizes[i]
        node_ids.append(temp)
    result_temp = result.copy()
    result_temp[result_temp==0] = 100
     
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            if i>0 and j>0:
                result_max[i,j] = np.max(result[node_ids[i-1]:node_ids[i],node_ids[j-1]:node_ids[j]])
                result_min[i,j] = np.min(result_temp[node_ids[i-1]:node_ids[i],node_ids[j-1]:node_ids[j]])
            elif j>0:
                result_max[i,j] = np.max(result[0:node_ids[i],node_ids[j-1]:node_ids[j]])
                result_min[i,j] = np.min(result_temp[0:node_ids[i],node_ids[j-1]:node_ids[j]])
            elif i>0:
                result_max[i,j] = np.max(result[node_ids[i-1]:node_ids[i],0:node_ids[j]])
                result_min[i,j] = np.min(result_temp[node_ids[i-1]:node_ids[i],0:node_ids[j]])
            else:
                result_max[i,j] = np.max(result[0:node_ids[i],0:node_ids[j]])
                result_min[i,j] = np.min(result_temp[0:node_ids[i],0:node_ids[j]])
    return result_min,result_max
    
def SSBM(sizes,alpha,beta,num_clusters:int,seed = 0,isplot=False):
    # return eigenvavlues, eigenvector adjacency matrix, ground truth

    if len(sizes) != num_clusters:
        print("The length of sizes should match the number of clusters.")
        assert False

    num_nodes = np.sum(sizes)
    prob_in = alpha*math.log(num_nodes)/num_nodes
    prob_out = beta*math.log(num_nodes)/num_nodes 
    probs = np.ones((num_clusters,num_clusters))*prob_out - np.eye(num_clusters)*prob_out + np.eye(num_clusters)*prob_in
    
    g = nx.stochastic_block_model(sizes,probs,seed=seed)
    adj_matrix = nx.to_numpy_array(g)
    if isplot:
        file_name = "./Observation/adjacency" + str(num_clusters) + ".jpg"
        plot_heatmap(adj_matrix,num_nodes,num_clusters,prob_in,prob_out,save_path=file_name)

    eigenValues,eigenVectors = eign_computing(adj_matrix,"SSBM")
    gt = []
    for i in range(len(sizes)):
       gt.extend(i*np.ones(sizes[i])) 

    return eigenValues,eigenVectors,adj_matrix,gt

def SSBM_sparse(sizes,alpha,beta,num_clusters:int,seed = 0,isplot=False):
    # return eigenvavlues, eigenvector adjacency matrix, ground truth

    if len(sizes) != num_clusters:
        print("The length of sizes should match the number of clusters.")
        assert False

    num_nodes = np.sum(sizes)
    prob_in = alpha/num_nodes
    prob_out = beta/num_nodes 
    probs = np.ones((num_clusters,num_clusters))*prob_out - np.eye(num_clusters)*prob_out + np.eye(num_clusters)*prob_in
    
    g = nx.stochastic_block_model(sizes,probs,seed=seed)
    adj_matrix = nx.to_numpy_array(g)
    if isplot:
        # file_name = ".adjacency" + str(num_clusters) + ".jpg"
        # plot_heatmap(adj_matrix,num_nodes,num_clusters,prob_in,prob_out,save_path=file_name)
        options = {"node_size": 20, "linewidths": 0, "width": 0.5,"with_labels": False}
        pos = nx.spring_layout(g, seed=800)  # Seed for reproducible layout
        # pos = nx.spectral_layout(G)
        plt.figure()
        nx.draw(g, pos ,**options)
        # plt.legend()                          # 0509 should make legends based on the color ...
        plt.savefig("SBM.jpg", dpi=300)

    eigenValues,eigenVectors = eign_computing(adj_matrix,"SSBM")
    gt = []
    for i in range(len(sizes)):
       gt.extend(i*np.ones(sizes[i])) 

    return eigenValues,eigenVectors,adj_matrix,gt

def SSBM_K2(sizes,alpha,beta,num_clusters:int,seed = 0):
    # return eigenvavlues, eigenvector adjacency matrix

    num_nodes = np.sum(sizes)
    prob_in = alpha*math.log(num_nodes)/num_nodes
    prob_out = beta*math.log(num_nodes)/num_nodes 
    # print("prob_in:",prob_in)
    # print("prob_out:",prob_out)
    probs = np.ones((num_clusters,num_clusters))*prob_out - np.eye(num_clusters)*prob_out + np.eye(num_clusters)*prob_in

    g = nx.stochastic_block_model(sizes,probs,seed=seed)
    adj_matrix = nx.to_numpy_array(g)

    eigenValues,eigenVectors = eign_computing(adj_matrix,"SSBM")

    return eigenValues,eigenVectors,adj_matrix

def SSBM_K2_noeig(sizes,alpha,beta,num_clusters:int,seed = 0):
    # return eigenvavlues, eigenvector adjacency matrix

    num_nodes = np.sum(sizes)
    prob_in = alpha*math.log(num_nodes)/num_nodes
    prob_out = beta*math.log(num_nodes)/num_nodes 
    probs = np.ones((num_clusters,num_clusters))*prob_out - np.eye(num_clusters)*prob_out + np.eye(num_clusters)*prob_in

    g = nx.stochastic_block_model(sizes,probs,seed=seed)
    is_con = nx.is_connected(g)
    adj_matrix = nx.to_numpy_array(g)

    return adj_matrix,is_con

def Kempe_2004(A,L,T,n,vec1,vec2):
    
    V = np.zeros((n,2))
    V[:,0] = vec1
    V[:,1] = vec2
    # print(V)
    w = np.zeros(n)
    w[0] = 1
    
    # construct B
    row_sums = A.sum(axis=1)
    new_matrix = A / row_sums[:, numpy.newaxis]
    B = new_matrix.T

    for t in range(T):
        V = np.matmul(A,V)
        # print(V.shape)
        K = np.zeros((n,4))
        R = np.zeros((2,2,n))
        # print(K.shape)
        
        for index in range(n):
            K[index,:] = np.matmul(V[index,:].reshape((2,1)),V[index,:].reshape((1,2))).reshape(1,4)
        # print(K)

        # Push-Sum algorithm
        for rou in range(L):
            K = np.matmul(B,K)
            w = np.matmul(B,w)
        
        for index in range(n):
            # print(K[index,:] )
            K[index,:] = K[index,:] / w[index]
            # print(K[index,:] )
            R[:,:,index] = np.linalg.cholesky(K[index,:].reshape(2,2))
            R[:,:,index] = R[:,:,index].T
            V[index,:] = np.matmul(V[index,:],np.linalg.inv(R[:,:,index]))
    
    return V
def Kempe_2004_DS(A,L,T,n,vec1,vec2,W):
    
    V = np.zeros((n,2))
    V[:,0] = vec1
    V[:,1] = vec2
    # print(V)
    w = np.zeros(n)
    w[0] = 1    

    for t in range(T):
        V = np.matmul(A,V)

        # QR decomposition
        K = np.zeros((n,4))
        R = np.zeros((2,2,n))
        
        for index in range(n):
            K[index,:] = np.matmul(V[index,:].reshape((2,1)),V[index,:].reshape((1,2))).reshape(1,4)

        # Gossip algorithm
        for rou in range(L):
            K = np.matmul(W,K)
            w = np.matmul(W,w)
        # print(K)
        # print(w)
        # assert False    
        for index in range(n):
            # print(K[index,:] )
            K[index,:] = K[index,:] / w[index]
            
            # print(K[index,:] )
            R[:,:,index] = np.linalg.cholesky(K[index,:].reshape(2,2))
            R[:,:,index] = R[:,:,index].T
            V[index,:] = np.matmul(V[index,:],np.linalg.inv(R[:,:,index]))
    
    return V
   

def Kempe_2004_K(A,L,T,n,V,W):
    
    # print(V)
    num_cluster = V.shape[1]
    w = np.zeros(n)
    w[0] = 1    

    for t in range(T):
        V = np.matmul(A,V)

        # QR decomposition
        K = np.zeros((n,num_cluster**2))
        R = np.zeros((num_cluster,num_cluster,n))
        
        for index in range(n):
            K[index,:] = np.matmul(V[index,:].reshape((num_cluster,1)),V[index,:].reshape((1,num_cluster))).reshape(1,num_cluster**2)

        # Gossip algorithm
        for rou in range(L):
            K = np.matmul(W,K)
            w = np.matmul(W,w)
        # print(K)
        # print(w)
        # assert False    
        for index in range(n):
            # print(K[index,:] )
            K[index,:] = K[index,:] / w[index]
            
            # print(K[index,:] )
            R[:,:,index] = np.linalg.cholesky(K[index,:].reshape(num_cluster,num_cluster))
            R[:,:,index] = R[:,:,index].T
            V[index,:] = np.matmul(V[index,:],np.linalg.inv(R[:,:,index]))
        print(V[0,:])
    
    return V

def Kempe_2004_KT(A,L,n,V:np.array,W,epsilon,U_gt):
    
    # print(V)
    num_cluster = V.shape[1]
    w = np.zeros(n)
    w[0] = 1    
    t = 0
    while np.linalg.norm(np.matmul(V,V.T)-np.matmul(U_gt,U_gt.T)) > epsilon:
        # print(np.linalg.norm(V-U_gt,'fro'))
        # print(np.linalg.norm(np.matmul(V,V.T)-np.matmul(U_gt,U_gt.T)))
        # print("1",np.linalg.norm(V[:,1]-U_gt[:,1]))
        # print("2",np.linalg.norm(V[:,2]-U_gt[:,2]))
        # print("1minus",np.linalg.norm(V[:,1]+U_gt[:,1]))
        # print("2minus",np.linalg.norm(V[:,2]+U_gt[:,2]))
        # print("all",np.linalg.norm(V-U_gt,'fro'))
        t += 1
        V = np.matmul(A,V)

        # QR decomposition
        K = np.zeros((n,num_cluster**2))
        R = np.zeros((num_cluster,num_cluster,n))
        
        for index in range(n):
            K[index,:] = np.matmul(V[index,:].reshape((num_cluster,1)),V[index,:].reshape((1,num_cluster))).reshape(1,num_cluster**2)

        # Gossip algorithm
        for rou in range(L):
            K = np.matmul(W,K)
            w = np.matmul(W,w)
        # print(K)
        # print(w)
        # assert False    
        for index in range(n):
            # print(K[index,:] )
            K[index,:] = K[index,:] / w[index]
            
            # print(K[index,:] )
            R[:,:,index] = np.linalg.cholesky(K[index,:].reshape(num_cluster,num_cluster))
            R[:,:,index] = R[:,:,index].T
            V[index,:] = np.matmul(V[index,:],np.linalg.inv(R[:,:,index]))
    
        if t >=2000:
            print(np.linalg.norm(np.matmul(V,V.T)-np.matmul(U_gt,U_gt.T)))
            est_error = False
            break
        else:
            est_error = np.linalg.norm(np.matmul(V,V.T)-np.matmul(U_gt,U_gt.T))
    
    return t, est_error

def construct_DS_matrix(num_nodes,adj_mat):
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

def compute_communiction_complexity(T_list):
    complexity = 0
    for i in range(1,len(T_list)+1):
        complexity += i *T_list[i-1]
        
    return complexity