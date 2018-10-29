import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os

class preProcess(object):
    def __init__(self, numBins):
        self.NUM_BINS=numBins
        self.CLASS_BLANK=0
        self.CLASS_TEXT=1
        self.CLASS_INVERTED=2
        self.CLASS_IMG=3
        self.CLASS_OTHER=4
        self.classnames = ['blank', 'text', 'inverted', 'image','other']

    def LoadImgAndPrepare(self, _filename):
        _rgb = imread(_filename)
        _gray = rgb2gray(_rgb)   
        _flat = _gray.ravel()
        bin_counts, bin_edges = np.histogram(_flat, bins=self.NUM_BINS)
        _sum = len(_flat)
        _percent = bin_counts / _sum
        return _percent

    def GetImagesAndLabels(self, _directory):
        _dataset = np.ndarray((1,self.NUM_BINS))
        _class = np.ndarray(1,dtype=int)
        _files = np.ndarray(1,dtype=object)
        index = 0
        for file in  os.listdir(_directory):
            if file.endswith('.png'):
                fullfile = os.path.join(_directory, file)
                img = self.LoadImgAndPrepare(fullfile)
                if index >= _dataset.shape[0]: 
                    new_img = np.ndarray((1,self.NUM_BINS))
                    _dataset = np.append(_dataset, new_img, axis=0)
                    new_class = np.ndarray(1,dtype=int)
                    _class = np.append(_class, new_class, axis=0)
                    new_file = np.ndarray(1,dtype=object)
                    _files = np.append(_files, new_file, axis=0)
                _dataset[index] = img
                _files[index] = file
                _class[index] = self.CLASS_OTHER
                for n in range(len(self.classnames)):
                    if  (file.find(self.classnames[n]) != -1):
                        _class[index] = n
                        break
                index += 1
        return _dataset, _class, _files

