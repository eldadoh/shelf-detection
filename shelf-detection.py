import numpy as np, pandas as pd ,torch ,skimage, cv2 ,glob ,os
from opencv_utils import *
from img_utils import *
from img_utils_skimage import *
from img_plots import *

from skimage import io
from skimage import filters
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line
from skimage import data, exposure, img_as_float

def run_on_single_image_wrapper(func, img_path): 
    func(img_path)

def run_on_dir_wrapper(func,dir_path,suffix = 'jpg'):
    for item_path in glob.glob(dir_path + f'/*.{suffix}'):
        func(item_path)


def shelf_detection_hough_transform(img_path):
    
    img = io.imread(img_path)
    img_gray = io.imread(img_path,as_gray=True)

    th = filters.threshold_otsu(img_gray, nbins=256)
    img_th = img_gray >= th
    img_sobel_x = filters.sobel_h(img_th)

    show_histogram(img = img_sobel_x,bins_num = 256,display = True)
    # plot_subplot_1x2([img,img_sobel_x],titles_list = ['rgb','img_sobel_x'],cmap_list = ['brg','gray']) 

    hspace, angles, dists = hough_line(img_sobel_x)
    hspace, angles, dists = hough_line_peaks(hspace, angles, dists)
    print()

def main(): 

    IMAGES_FOLDER_PATH = 'Data'
    TEST_IMAGE_1_PATH = 'Data/2021-06-06-162728_1.jpg'

    run_on_single_image_wrapper(func = shelf_detection_hough_transform,img_path=TEST_IMAGE_1_PATH)
    
    # run_on_dir_wrapper(func = shelf_detection_preprocess,dir_path = IMAGES_FOLDER_PATH,suffix = 'jpg')
    
if __name__ == "__main__" : 
    main() 
