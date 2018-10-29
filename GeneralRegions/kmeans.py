# example taken from Pattanayak - Pro Deep Learning...
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

class kmeans(object):

    def __init__(self, _filename):
        t0 = time.clock()
        np.random.seed(0)
        img = cv2.imread(_filename)
        imgray_ori = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #plt.imshow(imgray_ori,cmap='gray')
        ## Save the dimensions of the image
        row,col,depth = img.shape
        ## Collapse the row and column axis for faster matrix operation.
        img_new = np.zeros(shape=(row*col,3))
        glob_ind = 0
        for i in range(row):
            for j in range(col):
                u = np.array([img[i,j,0],img[i,j,1],img[i,j,2]])
                img_new[glob_ind,:] = u
                glob_ind += 1
        # Set the number of clusters
        K = 5
        # Run the K-means for
        num_iter = 20
        for g in range(num_iter):
            # Define cluster for storing the cluster number and out_dist to store the distances from centroid
            clusters = np.zeros((row*col,1))
            out_dist = np.zeros((row*col,K))
            centroids = np.random.randint(0,255,size=(K,3))
            for k in range(K):
                diff = img_new - centroids[k,:]
                diff_dist = np.linalg.norm(diff,axis=1)
                out_dist[:,k] = diff_dist
            # Assign the cluster with minimum distance to a pixel location
            clusters = np.argmin(out_dist,axis=1)
            # Recompute the clusters
            for k1 in np.unique(clusters):
                centroids[k1,:] = np.sum(img_new[clusters == k1,:],axis=0)/np.sum([clusters == k1])
        # Reshape the cluster labels in two-dimensional image form
        clusters = np.reshape(clusters,(row,col))
        out_image = np.zeros(img.shape)
        #Form the 3D image with the labels replaced by their correponding centroid pixel intensities
        for i in range(row):
            for j in range(col):
                out_image[i,j,0] = centroids[clusters[i,j],0]
                out_image[i,j,1] = centroids[clusters[i,j],1]
                out_image[i,j,2] = centroids[clusters[i,j],2]

        out_image = np.array(out_image,dtype="uint8")

        plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='r', zorder=10)

        plt.imshow(out_image)
        print('kmeans time:', time.clock()-t0)
        plt.show()

