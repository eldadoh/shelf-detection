from __future__ import print_function
from logging import exception
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import shutil 
from skimage.transform import resize as skimage_resize 
from skimage import io 
from sklearn.cluster import KMeans , SpectralClustering
from sklearn.mixture import GaussianMixture

def normalize_image(img_path) : 

    """
    [0-255] --> [0 - 1]
    Neural networks process inputs using small weight values, and inputs with large integer values can disrupt or slow down the learning process. 
    As such it is good practice to normalize the pixel values so that each pixel value has a value between 0 and 1.
    """

    try :
        img = io.imread(img_path)
    except:  
        img = img_path 

    img = img.astype('float32')
    
    # normalize to the range 0-1
    img /= 255.0

    return img 

def centerize_image(img_path , normalize_before = False) : 

    """
    centerizing and standartizing are used for training neural networks efficently 
    after you done center/standartize image you can not display it because of negative values
    so reverse your operations to display again in range [0-1] / [0 - 255 ]

    Centering, as the distribution of the pixel values is centered on the value of zero.
    Centering can be performed before or after normalization. 
    
    Centering the pixels before normalizing will mean that the pixel values will be centered close to 0.5 and be in the range 0-1. 
    
    Centering after normalization will mean that the pixels will have positive and negative values [-0.5  , 0.5 ], 
    in which case images will not display correctly (e.g. pixels are expected to have value in the range 0-255 or 0-1). 
    """

    normalize_after = not normalize_before

    try :
        img = io.imread(img_path)
    except:  
        img = img_path 

    img = img.astype('float32')

    if normalize_before:

        img /= 255.0

    mean = img.mean()

    img -= mean 

    if normalize_after:

        img /= 255.0

    return img

def standardize_image(img_path) : 

    try :
        img = io.imread(img_path)
    except:  
        img = img_path 

    img = img.astype('float32')

    mean,std  = img.mean() , img.std()

    img -= mean 

    img /= std 

    return img 

def K_means_sklearn(lines,show = False):

    #prepare data
    X = np.empty(shape=(0,2))
    
    for item in lines: 
        first_point,second_point = item[:]
        X = np.vstack([X,first_point,second_point])
        
    assert X.shape[0] == len(lines) * 2 and X.shape[1] == 2 

    model = KMeans(n_clusters=3)
    model.fit(X)
    yhat = model.predict(X)
    clusters = np.unique(yhat)

    if show: 

        for cluster in clusters:
            row_ix = np.where(yhat == cluster)
            plt.scatter(X[row_ix, 0], X[row_ix, 1])

        plt.show()