import networkx as nx
import math
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import eigh
import numpy as np
import numpy.matlib
from sklearn.metrics.pairwise import euclidean_distances,manhattan_distances,cosine_distances,cosine_similarity
from scipy.spatial.distance import pdist
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from scipy.linalg import sqrtm
import time
import urllib.request
import io
import zipfile
from utils import plot_heatmap,plot_linechart,plot_sign_2nd_eigenvector,show_graph_with_labels,eign_computing
import os 

np.set_printoptions(suppress=True)


def karate_club(isplot=False,isKmeans = False):
    # G = nx.karate_club_graph()
    # pos = nx.kamada_kawai_layout(G)
    # nx.draw(G,pos,with_labels=True)
    # plt.savefig("karate_club.jpg")
    # adj_mat = nx.adjacency_matrix(G)
    # nlap = nx.normalized_laplacian_matrix(G)
    # nlap = nlap.todense()
    # lap = nx.laplacian_matrix(G)
    # lap = lap.todense()
    url = "./dataset/gml/karate.gml"
    G = nx.read_gml(url,label="id")
    
    gt = []
    for node_id, node_attributes in G.nodes(data=True):
        # print(node_id, node_attributes)
        # print(type(node_attributes.values()))
        label = str(node_attributes["value"])
        gt.append(int(label))

    print("karate",gt)
    options = {"node_size": 200, "linewidths": 0, "width": 0.5,"with_labels": False}
    
    pos = nx.spring_layout(G, seed=800)  # Seed for reproducible layout
    # pos = nx.spectral_layout(G)
    if isplot:
        plt.figure()
        nx.draw(G, pos, node_color=gt, **options)
        plt.title("Karate club network")
        # plt.legend()                          # 0509 should make legends based on the color ...
        plt.savefig("./dataset/karate.jpg", dpi=300)

    # construct the adjacency matrix
    adj_mat = nx.adjacency_matrix(G)
    adj_mat = adj_mat.todense()
    adj_mat[adj_mat!=0] = 1

    eigVal,eigVec= eign_computing(adj_mat,"Karate")
    if isplot:
        plot_sign_2nd_eigenvector(u2=eigVec[:,1],A=adj_mat,a=0,b=0,save_path="./dataset/Karate_club_2nd.jpg")
    if isKmeans:
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(eigVec[:,0:2])
        assignment = kmeans.predict(eigVec[:,0:2])
        group1 = np.where(assignment == 0)[0] + 1
        group2 = np.where(assignment == 1)[0] + 1
        print('people in group 1:', group1)
        print('people in group 2:', group2)
    
    return adj_mat, gt
    
def football(isplot):
    url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"
    sock = urllib.request.urlopen(url)  # open URL
    s = io.BytesIO(sock.read())  # read into BytesIO "file"
    sock.close()

    zf = zipfile.ZipFile(s)  # zipfile object
    txt = zf.read("football.txt").decode()  # read info file
    gml = zf.read("football.gml").decode()  # read gml data
    # throw away bogus first line with # from mejn files
    gml = gml.split("\n")[1:]
    G = nx.parse_gml(gml)  # parse gml data

    gt = []
    for node_id, node_attributes in G.nodes(data=True):
        # print(node_id, node_attributes)
        # print(type(node_attributes.values()))
        label = str(node_attributes["value"])
        gt.append(int(label))

    # print("football",gt)
    options = {"node_size": 200, "linewidths": 0, "width": 0.5,"with_labels": False}
    
    pos = nx.spring_layout(G, seed=800)  # Seed for reproducible layout
    # pos = nx.spectral_layout(G)
    if isplot:
        plt.figure()
        nx.draw(G, pos, node_color=gt, **options)
        plt.title("football network")
        # plt.legend()                          # 0509 should make legends based on the color ...
        plt.savefig("./dataset/football.jpg", dpi=300)

    # construct the adjacency matrix
    adj = nx.adjacency_matrix(G)
    adj_football = adj.todense()
    adj_football[adj_football!=0] = 1

    return adj_football, gt

def polbooks(isplot=False):

    path = "./dataset/gml/polbooks.gml"
    G = nx.read_gml(path)
    # print(dir(G))
    # print(G.edges)
    # print degree for each dolphin
    degree = 0
    count = 0
    for n,d in G.degree():
        degree +=d
        count +=1

    node_colors = dict()
    gt = []
    for node_id, node_attributes in G.nodes(data=True):
        # print(node_id, node_attributes)
        # print(type(node_attributes.values()))
        label = node_attributes["value"]
        gt.append(label)
        if label == "n":    #label = {n,c,l}
            node_colors[node_id] = "red"
        elif label=="c":
            node_colors[node_id] = "blue"
        else:
            node_colors[node_id] = "black"
            
    print("polbooks",gt)
    # print("polbooks",node_colors)
    
    options = {"node_size": 100, "linewidths": 0, "width": 0.5,"with_labels": False}
    
    pos = nx.spring_layout(G, seed=38)  # Seed for reproducible layout
    # pos = nx.spectral_layout(G)
    if isplot:
        plt.figure()
        nx.draw(G, pos, node_color=[node_colors[node] for node in G.nodes()], **options)
        plt.title("polbooks network")
        plt.savefig("./dataset/polbooks.jpg", dpi=300)

    # construct the adjacency matrix
    adj = nx.adjacency_matrix(G)
    adj_mat = adj.todense()
    adj_mat[adj_mat!=0] = 1

    return adj_mat,gt    # ground_truth is a dictionary.

def dolphins(isplot=False):
    # url = "http://www-personal.umich.edu/~mejn/netdata/dolphins.zip"
    # sock = urllib.request.urlopen(url)  # open URL
    # s = io.BytesIO(sock.read())  # read into BytesIO "file"
    # sock.close()
    # zf = zipfile.ZipFile(s)  # zipfile object
    # txt = zf.read("dolphins.txt").decode()  # read info file
    # gml = zf.read("dolphins.gml").decode()  # read gml data
    # gml = gml.split("\n")[1:]
    # G = nx.parse_gml(gml)  # parse gml data

    # getting the name of the directory where the this file is present.
    current = os.path.dirname(os.path.realpath(__file__))
    url = current + "/dataset/gml/dolphins.gml"
    
    G = nx.read_gml(url)
    # print(dir(G))
    # print(G.edges)

    # print degree for each dolphin
    degree = 0
    count = 0
    for n,d in G.degree():
        degree +=d
        count +=1

    ground_truth = dict()
    node_colors = dict()
    gt = []
    for node_id, node_attributes in G.nodes(data=True):
        # print(node_id, node_attributes)
        # print(type(node_attributes.values()))
        ground_truth[node_id] = node_attributes.values()
        
        label = str(list(node_attributes.values())).replace("[","").replace("]","")
        
        
        if label == "0":
            node_colors[node_id] = "red"
            gt.append(0)
        else:
            node_colors[node_id] = "blue"
            gt.append(1)
            
    # print("dolphins",ground_truth)
    # print("dolphins",node_colors)
    nx.set_node_attributes(G, values=node_colors, name='color')

    options = {"node_size": 200, "linewidths": 0, "width": 0.5,"with_labels": False}

    pos = nx.spring_layout(G, seed=300)  # Seed for reproducible layout
    # pos = nx.spectral_layout(G)
    if isplot:
        plt.figure()
        nx.draw(G, pos, node_color=[node_colors[node] for node in G.nodes()], **options)
        plt.title("dolphins social network")
        plt.savefig("./dataset/dolphins.jpg", dpi=300)

    # construct the adjacency matrix
    adj = nx.adjacency_matrix(G)
    adj_dolphins = adj.todense()
    adj_dolphins[adj_dolphins!=0] = 1

    return adj_dolphins,gt,G    # ground_truth is a dictionary.

if __name__ == '__main__':
   
    seed = 56
    np.random.seed(seed)
    num_clusters = 5
    adj_mat,gt = dolphins(isplot=False)
    adj_mat,gt = polbooks(isplot=True)
    # eigenValues, eigenVectors= eign_computing(adj_mat,"dolphins")

    # X = eigenVectors[:,0:num_clusters-1]
    # print("The shape of X:", X.shape)
    # result = cosine_distances(X)
    
    adj_mat, gt = football(isplot=False)
    adj_mat, gt = karate_club(isplot=False,isKmeans = True)
