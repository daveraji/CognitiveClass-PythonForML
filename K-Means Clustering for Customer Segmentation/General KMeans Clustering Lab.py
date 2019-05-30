import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
%matplotlib inline 

#create random dataset for the lab
np.random.seed(0)

x,y = make_blobs(n_samples=5000, centers=[[4,4],[-2,-1],[2,-3],[1,1]],cluster_std=0.9)
plt.scatter(x[:,0], x[:,1], marker='.')

#set up the k-means clustering
k_means = KMeans(init = 'k-means++', n_clusters = 4, n_init =12)
k_means.fit(x)
k_means_labels = k_means.labels_
k_means_labels

k_means_centroids = k_means.cluster_centers_
k_means_centroids

#show all this is a graph
fig = plt.figure(figsize=(6,4))
colours = plt.cm.Spectral(np.linspace(0,1,len(set(k_means_labels))))
ax = fig.add_subplot(1,1,1)

#for loop that plots the datapoints and centoids
for k, col in zip(range(len([[4,4],[-2,-1],[2,-3],[1,1]])),colours):
    
    #list of datapoints in clusters
    members = (k_means_labels==k)
    centroid = k_means_centroids[k]
    ax.plot(x[members,0],x[members,1],'w',markerfacecolor=col,marker='.')
    ax.plot(centroid[0],centroid[1],'o',markerfacecolor=col,markeredgecolor='k',markersize=6)
    
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.show()
    
