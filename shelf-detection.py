import os
import math
import numpy as np 
import pandas as pd
import glob
import skimage 
import matplotlib.pyplot as plt
from skimage import io
from skimage import filters
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line
from skimage import data, exposure, img_as_float
from skimage.transform import probabilistic_hough_line
from sklearn.cluster import KMeans , SpectralClustering
from skimage.feature import canny 
from copy import deepcopy
from ml_utils import standardize_image,K_means_sklearn
from img_utils import show_histogram
from img_utils_skimage import threshold_otsu_skimage

def probabilistic_hough_transform(derivative_img,threshold, line_length,line_gap):

    """
    image : (M, N) ndarray
         Input image with nonzero values representing edges.
    threshold : int, optional
        Threshold
    line_length : int, optional
        Minimum accepted length of detected lines. Increase the parameter to extract longer lines.
    line_gap : int, optional
        Maximum gap between pixels to still form a line. Increase the parameter to merge broken lines more aggressively.
    theta : 1D ndarray, dtype=double, optional
        Angles at which to compute the transform, in radians. If None, use a range from -pi/2 to pi/2.
    """
    
    lines = probabilistic_hough_line(derivative_img, threshold=threshold, line_length=line_length,line_gap=line_gap)     

    return lines 
     
def plot_hough_1(lines,img,derivative_img):

    img_copy = deepcopy(img)
    
    #######
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 15), sharex=True, sharey=True)
    ax = axes.ravel()
    #######
    #plot0#
    ax[0].imshow(img_copy, cmap='gray')
    ax[0].set_title('Input')
    #plot1#
    ax[1].imshow(derivative_img * 0)
    for line in lines:
        p0, p1 = line
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
        # ax[1].scatter((p0[0], p1[0]), (p0[1], p1[1]))
    ax[1].imshow(derivative_img, cmap='gray')
    ax[1].set_xlim((0, derivative_img.shape[1]))
    ax[1].set_ylim((derivative_img.shape[0], 0))
    ax[1].set_title('Edges - Sobel_X')
    #plot2#
    ax[2].imshow(img_copy * 0)
    for line in lines:
        p0, p1 = line
        # ax[2].scatter((p0[0], p1[0]), (p0[1], p1[1]))
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].imshow(img_copy, cmap='gray')
    ax[2].set_xlim((0, img_copy.shape[1]))
    ax[2].set_ylim((img_copy.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')
    #######
    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.show()
    #######

    return lines

def plot_hough_2(lines,img):
 
    num_of_points = len(lines)

    #######       
    # fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True)
    ax = axes.ravel()
    #######
    #plot0#
    ax[0].imshow(img * 0)
    for line in lines:
        p0, p1 = line
        ax[0].plot((p0[0], p1[0]), (p0[1], p1[1]))
        ax[0].scatter((p0[0], p1[0]), (p0[1], p1[1]))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_xlim((0, img.shape[1]))
    ax[0].set_ylim((img.shape[0], 0))
    ax[0].set_title(f'img + points')
    #plot1#
    ax[1].imshow(img * 0)
    for line in lines:
        p0, p1 = line
        ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
        # ax[1].scatter((p0[0], p1[0]), (p0[1], p1[1]))
    ax[1].set_title(f'Only Points \nNumber of points: {num_of_points}')
    #######
    plt.tight_layout()
    plt.show()
    
    return fig 



def post_process_line_segments(img_rgb,lines): 

    epsilon = 1e-4 

    new_lines = []

    for line in lines : 

        p1,p2 = line

        p1_x,p1_y = p1 
        p2_x,p2_y = p2 

        m = ((p2_y - p1_y) + epsilon )  / (p2_x - p1_x)

        # Y - p2_y = m (X - p2_x)
        # Y - p2_y = m (0 - p2_x) # force X=0 and solve for Y 

        X_left = 0
        Y_left = m * (X_left - p2_x) + p2_y
        
        X_right = img_rgb.shape[1]
        Y_right = m * (X_right - p2_x) + p2_y

        Y_left = int(Y_left)
        Y_right = int(Y_right) 

        new_lines.append(((X_left,Y_left),(X_right,Y_right)))
    
    return new_lines

def shelf_detection_hough_transform(img_path,param_dict):
    
    img_rgb = io.imread(img_path)

    img_gray = io.imread(img_path,as_gray=True)

    img_standardized = standardize_image(img_gray)

    img_blurred = filters.gaussian(img_standardized, sigma=1, output=None, mode='nearest', cval=0, multichannel=None, preserve_range=False, truncate=4.0)

    img_th = threshold_otsu_skimage(img_blurred)

    img_sobel_x = filters.sobel_h(img_th)

    lines = probabilistic_hough_transform(derivative_img = img_sobel_x, **param_dict )

    new_lines = post_process_line_segments(img_rgb,lines)

    # plot_hough_1(lines,img_rgb,img_sobel_x)

    fig = plot_hough_2(new_lines,img_rgb)

    # K_means_sklearn(new_lines,n_clusters = 3)

    # get the x,y of the clusters and return them as the shelf coordinates 

    return fig 

    pass

    
def main(): 
    
    IMAGES_FOLDER_PATH = 'Data'
    IMAGE_1 = 'Data/2021-06-06-162728_1.jpg'
    IMAGE_2 = 'Data/2021-06-07-090106_1.jpg'
    IMAGE_3 = 'Data/2021-06-07-090106_2.jpg'

    CURRENT_BEST_PARAM_DICT = {'threshold' : 30,
                               'line_length' : 300,
                               'line_gap' : 10
    }

    # shelf_detection_hough_transform(IMAGE_1 , CURRENT_BEST_PARAM_DICT)

    for img_path in glob.glob(IMAGES_FOLDER_PATH + '/*.jpg'):
        fig = shelf_detection_hough_transform(img_path , CURRENT_BEST_PARAM_DICT) 
        fig.savefig('Data/' + os.path.basename(img_path))

    # PARAM_DICT1 = {'threshold' : 80,
    #               'line_length' : 400,
    #               'line_gap' : 10
    #               }
                  
    # PARAM_DICT2 = {'threshold' : 80,
    #               'line_length' : 350,
    #               'line_gap' : 15
    #               }

    # PARAM_DICT3 = {'threshold' : 80,
    #               'line_length' : 150,
    #               'line_gap' : 3
    #               }
    # PARAM_DICT4 = {'threshold' : 80,
    #               'line_length' : 200,
    #               'line_gap' : 3
    #               }     
    # PARAM_DICT5 = {'threshold' : 80,
    #               'line_length' : 350,
    #               'line_gap' : 3
    #               }                                                                       

    # dict_list = [PARAM_DICT1,PARAM_DICT2,PARAM_DICT3,PARAM_DICT4,PARAM_DICT5]

    
    # for param_dict in dict_list : 

    #     shelf_detection_hough_transform(IMAGE_1 , param_dict)
    #     shelf_detection_hough_transform(IMAGE_2 , param_dict)
    #     shelf_detection_hough_transform(IMAGE_3 , param_dict)
        
    
if __name__ == "__main__" : 

    #python3 -m cProfile shelf-detection.py

    main() 


