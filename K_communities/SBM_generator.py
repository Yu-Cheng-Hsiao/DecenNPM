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
from utils import eign_computing,SSBM,SSBM_sparse,plot_heatmap,OrderedLabelEncoder,plot_heatmap_triangle,construct_DS_matrix

    
def plot_graph_communities(adj_mat,gt):
    
    G = nx.from_numpy_array(adj_mat) 
    
    initialpos = {}
    for i in range(adj_mat.shape[0]):
        if int(gt[i]) == 0:
            initialpos[i] = (np.random.uniform(30,60),np.random.uniform(5,25))
        elif int(gt[i]) == 1:
            initialpos[i] = (np.random.uniform(5,30),np.random.uniform(-10,-35))
        elif int(gt[i]) == 2:
            initialpos[i] = (np.random.uniform(-20,-60),np.random.uniform(0,20))
        elif int(gt[i]) == 3:
            initialpos[i] = (np.random.uniform(-10,-40),np.random.uniform(-15,-30))
        elif int(gt[i]) == 4:
            initialpos[i] = (np.random.uniform(-15,15),np.random.uniform(25,40))
        else:
            assert False
        
    pos = nx.spring_layout(G, seed=800,pos = initialpos, fixed = range(adj_mat.shape[0]))  # Seed for reproducible layout
    # pos = nx.spectral_layout(G)
    options = {"node_size": 200, "linewidths": 1, "width": 0.1,"with_labels": False, "edge_color":"darkgrey" }
    plt.figure()
    nx.draw(G, pos, node_color=gt, **options)
    ax = plt.gca() # to get the current axis
    ax.collections[0].set_edgecolor("#FF0000")

    plt.savefig("SBM.eps")

if __name__ == '__main__':
    
    total_num_clusters = 5
    
    np.random.seed(total_num_clusters)
    
    n1 = 35
    n2 = 30
    n3 = 25
    n4 = 20
    n5 = 15

    print(n1,n2,n3,n4,n5)
    # eigenValues,eigenVectors,adj,gt = SSBM(sizes=[n1,n2,n3,n4,n5],alpha=25,beta=5,num_clusters=total_num_clusters,seed = 0,isplot=False)
    eigenValues,eigenVectors,adj,gt = SSBM(sizes=[n1,n2,n3,n4,n5],alpha=15,beta=0.5,num_clusters=total_num_clusters,seed = 0,isplot=False)

    np.save("./numpy_array/SBM/SBMa15b0_5_adj",adj)
    np.save("./numpy_array/SBM/SBMa15b0_5_gt",np.array(gt))
    # print(eigenValues)
    plot_graph_communities(adj_mat = adj,gt = gt)

    