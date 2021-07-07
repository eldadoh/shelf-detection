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
from skimage.feature import canny 
from copy import deepcopy

def run_func_on_single_item(func, img_path): 
    func(img_path)

def run_func_on_dir(func,dir_path,suffix = 'jpg'):
    for item_path in glob.glob(dir_path + '/*.' + 'jpg'):
        func(item_path)

def show_histogram(img,bins_num,display = True): 

    hist_vals , x_bins =  np.histogram(img,bins_num)
    hist_vals = np.concatenate([np.array([0]),hist_vals])

    if display : 
        
        # calc_image_range(img)
        print('\nThe non-zero values on the histogram are :\n')
        print(x_bins[hist_vals!=0])
        
    fig = plt.figure()
    
    plt.scatter(x_bins[hist_vals!=0],hist_vals[hist_vals!=0],color = 'red')
    plt.scatter(x_bins[hist_vals==0],hist_vals[hist_vals==0],color = 'blue')
    
    plt.show()

def threshold_otsu_skimage(img):
    
    th = filters.threshold_otsu(img, nbins=256)
    
    img_th = img >= th
    
    return img_th

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

def plot_hough_points(lines,img, show = True):
    
    for line in lines:
        p0, p1 = line
        calc_angle(p0,p1)

    if show : 
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(img * 0)
        for line in lines:
            p0, p1 = line
            ax[0].plot((p0[0], p1[0]), (p0[1], p1[1]))
            ax[0].scatter((p0[0], p1[0]), (p0[1], p1[1]))
        
        ax[0].imshow(img, cmap='gray')
        ax[0].set_xlim((0, img.shape[1]))
        ax[0].set_ylim((img.shape[0], 0))
        ax[0].set_title('img + points ')

        ax[1].imshow(img * 0)
        for line in lines:
            p0, p1 = line
            ax[1].scatter((p0[0], p1[0]), (p0[1], p1[1]))
        ax[1].set_title('Only Points ')
        
        plt.show()


def calc_angle(point1,point2,show = True):

    x1,y1 = point1[:]
    x2,y2 = point2[:]

    dy = np.abs(y2 - y1)
    dx = np.abs(x2 - x1)

    angle_radian = math.atan2(dy, dx)  
    angle_degree = np.rad2deg(angle_radian)

    if show: 
        if (angle_degree > 30) : 
            print(f'{point1} , {point2}')
            print(f'The angle between the two line\'s points is probably outlier : {angle_degree}')
    
    return angle_degree

def get_mask_of_hough_line_points(img , lines):

    h,w = img.shape[:2]
    mask = np.zeros((h,w))
    print()    

def shelf_detection_hough_transform(img_path):
    
    img = io.imread(img_path)

    img_gray = io.imread(img_path,as_gray=True)

    img_th = threshold_otsu_skimage(img_gray)

    img_sobel_x = filters.sobel_h(img_th)

    lines = probabilistic_hough_transform(img = img,derivative_img = img_sobel_x)

    get_mask_of_hough_line_points(img,lines)

    plot_hough_points(lines,img)

    # analyse lines array ... 

    # calc angle between 2 points and elimate outliers 
    # clustering ? k-means ? 

def normalize_image(img_path) : 

    """
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

def centerize_image(img_path) : 

    """[summary]
    Centering, as the distribution of the pixel values is centered on the value of zero.

    Centering can be performed before or after normalization. 
    
    Centering the pixels then normalizing will mean that the pixel values will be centered close to 0.5 and be in the range 0-1. 
    
    Centering after normalization will mean that the pixels will have positive and negative values, in which case images will not display correctly 
    (e.g. pixels are expected to have value in the range 0-255 or 0-1). 
    Centering after normalization might be preferred, although it might be worth testing both approaches

    """


    try :
        img = io.imread(img_path)
    except:  
        img = img_path 

    img = img.astype('float32')

    mean = img.mean()

    img -= mean 

    return img

def standartize_image(img_path) : 

    try :
        img = io.imread(img_path)
    except:  
        img = img_path 

    img = img.astype('float32')

    mean,std  = img.mean() , img.std()

    img -= mean 

    img /= std 

    return img 
    

def main(): 


    IMAGES_FOLDER_PATH = 'Data'
    IMAGE_1 = 'Data/2021-06-06-162728_1.jpg'

    # show_histogram(img = img_sobel_x,bins_num = 256,display = True)
    # run_func_on_single_item(func = shelf_detection_hough_transform,img_path=IMAGE_1)

    # run_func_on_dir(func = shelf_detection_hough_transform,dir_path = IMAGES_FOLDER_PATH,suffix = 'jpg')
    
if __name__ == "__main__" : 

    #python3 -m cProfile shelf-detection.py

    main() 

