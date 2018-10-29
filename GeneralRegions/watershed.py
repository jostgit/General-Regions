# example taken from Pattanayak - Pro Deep Learning...
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed

class mywatershed(object):
    def __init__(self, _filename):
        t0 = time.clock()
        ## Load the coins image
        im = cv2.imread(_filename)
        ## Convert the image to grayscale
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        #plt.imshow(imgray,cmap='gray')
        # Threshold the image to convert it to Binary image based on Otsu's method
        thresh = cv2.threshold(imgray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        ## Detect the contours and display them
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        y = cv2.drawContours(imgray, contours, -1, (0,255,0), 3)
        ## If we see the contour plots in the display of "y"
        ## we see that the coins have a common contour and hence it is not possible to separate them
        plt.imshow(y)
        ## Hence we will proceed with the Watershed algorithm so that each of the coins form its own
        ## cluster and thus it’s possible to have separate contours for each coin.
        ## Relabel the thresholded image to be consisting of only 0 and 1
        ## as the input image to distance_transform_edt should be in this format.
        thresh[thresh == 255] = 5
        thresh[thresh == 0] = 1
        thresh[thresh == 5] = 0
        ## The distance_transform_edt and the peak_local_max functions help building the markers by detecting
        ## points near the center points of the coins. One can skip these steps and create a marker
        ## manually by setting one pixel within each coin with a random number representing its cluster
        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=10,labels=thresh)
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        # Provide the EDT distance matrix and the markers to the watershed algorithm to detect the cluster’s
        # labels for each pixel. For each coin, the pixels corresponding to it will be filled with the cluster number
        labels = watershed(-D, markers, mask=thresh)   
        print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
        # Create the contours for each label(each coin) and append to the plot
        for k in np.unique(labels):
            if k != 0 :
                labels_new = labels.copy()
                labels_new[labels == k] = 255
                labels_new[labels != k] = 0
                labels_new = np.array(labels_new,dtype='uint8')
                im2, contours, hierarchy = cv2.findContours(labels_new,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                z = cv2.drawContours(imgray,contours, -1, (0,255,0), 3)
                plt.imshow(z)
        print('watershed time:', time.clock()-t0)
        plt.show()
