# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 19:10:30 2017

@author: erank
"""
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance_matrix


def myKmeans(X, K):
    nrows = X.shape[0]
    ncolumns = X.shape[1]
    #first=X[0,:]
    #second=X[1,:]
    #third=X[2,:]
    
    
    
    #find_initial_centroids = np.array([[first],[second],[third]])#
    find_initial_centroids=np.random.choice(2, K, replace=False)
    
    centroids = find_initial_centroids
    old_centroids = np.zeros((K,ncolumns))
    
    labels = np.zeros(nrows)
    while (old_centroids != centroids).any():
        old_centroids = centroids.copy()
        
        
        #distance matrix
        distanceMatrix = distance_matrix(X, centroids, p=2)
        
        for i in np.arange(nrows):
            d = distanceMatrix[i]
            closest_centroid = (np.where(d == np.min(d)))[0][0]
            
            labels[i] = closest_centroid
            
        for k in np.arange(K):
            Xk = X[labels == k]
            centroids[k] = np.apply_along_axis(np.mean, axis=0, arr=Xk)
            
    return(centroids, labels)

def main():
    path = "D:\\ML\\SCLC_study_output_filtered_2.csv"
    dataset = np.genfromtxt(path,delimiter=",")
    dataset1 = dataset[1:,1:]
    #np.random.seed(5)
    #iris = datasets.load_iris()
    #dataset1 = iris.data
    #feature_names = iris.feature_names
    #y = iris.target
    #target_names = iris.target_names
    
    result = myKmeans(dataset1,2)
    centroids = result[0]
    print("Centroids are : ")
    print(centroids)
    cluster_labels = result[1].astype(int)
    print("Labels are: ")
    print(cluster_labels)


    centroids=result[0]
    plt.scatter(dataset1[cluster_labels==0,0], dataset1[cluster_labels==0,1], s=100, c='red', label= 'Cluster 1')
    plt.scatter(dataset1[cluster_labels==1,0], dataset1[cluster_labels==1,1], s=100, c='blue', label= 'Cluster 2')
    plt.title('clusters graph');
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()