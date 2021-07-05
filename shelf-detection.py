import numpy as np 
import pandas as pd
import glob
import skimage 
from opencv_utils import *
from img_utils import *
from img_utils_skimage import *
from img_plots import *

from skimage import io
from skimage import filters
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line
from skimage import data, exposure, img_as_float
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny 
from copy import deepcopy

def run_on_single_image_wrapper(func, img_path): 
    func(img_path)

def run_on_dir_wrapper(func,dir_path,suffix = 'jpg'):
    for item_path in glob.glob(dir_path + '/*.' + 'jpg'):
        func(item_path)

def probabilistic_hough_transform(img,derivative_img,show = True):

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

    # lines = probabilistic_hough_line(derivative_img, threshold=50, line_length=100,line_gap=3)
    
    
    
    lines = probabilistic_hough_line(derivative_img, threshold=50, line_length=100,line_gap=3)
    
    if show : 

        img_copy = img 
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(img_copy, cmap='gray')
        ax[0].set_title('Input')

        ax[1].imshow(derivative_img * 0)
        for line in lines:
            p0, p1 = line
            ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
            ax[1].scatter((p0[0], p1[0]), (p0[1], p1[1]))
        ax[1].imshow(derivative_img, cmap='gray')
        ax[1].set_xlim((0, derivative_img.shape[1]))
        ax[1].set_ylim((derivative_img.shape[0], 0))
        ax[1].set_title('Edges - Sobel_X')

        ax[2].imshow(img_copy * 0)
        for line in lines:
            p0, p1 = line
            ax[2].scatter((p0[0], p1[0]), (p0[1], p1[1]))
            ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
        ax[2].imshow(img_copy, cmap='gray')
        ax[2].set_xlim((0, img_copy.shape[1]))
        ax[2].set_ylim((img_copy.shape[0], 0))
        ax[2].set_title('Probabilistic Hough')

        for a in ax:
            a.set_axis_off()

        plt.tight_layout()
        plt.show()

    return lines

def shelf_detection_hough_transform(img_path):
    
    img = io.imread(img_path)

    img_gray = io.imread(img_path,as_gray=True)

    img_th = threshold_otsu_skimage(img_gray)

    img_sobel_x = filters.sobel_h(img_th)
    
    # img_sobel_x = canny(img_th)

    # show_histogram(img = img_sobel_x,bins_num = 256,display = True)
    # plot_subplot_1x2([img,img_sobel_x],titles_list = ['rgb','img_sobel_x'],cmap_list = ['brg','gray']) 

    lines = probabilistic_hough_transform(img = img,derivative_img = img_sobel_x)

    print()


def main(): 

    IMAGES_FOLDER_PATH = 'Data'
    
    TEST_IMAGE_1_PATH = 'Data/2021-06-06-162728_1.jpg'

    run_on_dir_wrapper(func = shelf_detection_hough_transform,dir_path = IMAGES_FOLDER_PATH,suffix = 'jpg')
    
    # run_on_single_image_wrapper(func = shelf_detection_hough_transform,img_path=TEST_IMAGE_1_PATH)
    
if __name__ == "__main__" : 
    main() 
