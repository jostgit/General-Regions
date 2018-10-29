# example taken from Pattanayak - Pro Deep Learning...
## Binary thresholding Method Based on Histogram of Pixel Intensities
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

class otsu(object):

    def __init__(self, _filename):
        t0 = time.clock()
        ## Otsu's thresholding Method
        img = cv2.imread(_filename)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        row,col = np.shape(gray)
        hist_dist = 256*[0]
        ## Compute the frequency count of each of the pixels in the image
        for i in range(row):
            for j in range(col):
                hist_dist[gray[i,j]] += 1
        # Normalize the frequencies to produce probabilities   
        hist_dist = [c/float(row*col) for c in hist_dist]
        plt.plot(hist_dist)
        ## Compute the between segment variance
        def var_c1_c2_func(hist_dist,t):
            u1,u2,p1,p2,u = 0,0,0,0,0
            for i in range(t+1):
                u1 += hist_dist[i]*i
                p1 += hist_dist[i]
            for i in range(t+1,256):
                u2 += hist_dist[i]*i
                p2 += hist_dist[i]
            for i in range(256):
                u += hist_dist[i]*i
            var_c1_c2 = p1*(u1 - u)**2 + p2*(u2 - u)**2
            return var_c1_c2
        ## Iteratively run through all the pixel intensities from 0 to 255 and choose the one that
        ## maximizes the variance
        variance_list = []
        for i in range(256):
            var_c1_c2 = var_c1_c2_func(hist_dist,i)
            variance_list.append(var_c1_c2)
        ## Fetch the threshold that maximizes the variance
        t_hat = np.argmax(variance_list)
        ## Compute the segmented image based on the threshold t_hat
        gray_recons = np.zeros((row,col))
        for i in range(row):
            for j in range(col):
                if gray[i,j] <= t_hat :
                    gray_recons[i,j] = 255
                else:
                    gray_recons[i,j] = 0
        plt.imshow(gray_recons)
        print('Otsu time:', time.clock()-t0)
        plt.show()

