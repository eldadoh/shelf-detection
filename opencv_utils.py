from img_plots import plot_img_opencv
import shutil
import itertools
import os
import glob
import numpy as np
import cv2 
from cv2 import BFMatcher as bf
from matplotlib import pyplot as plt
from skimage.transform import resize as skimage_resize 
from collections import defaultdict


def check_diff_feature_extractors_performence_on_singel_image(img_path,show = False):

    """
        applying the next feature extractors : ['SIFT','ORB','FAST', 'SIMPLE_BLOB_DETECTOR']
        returns key_points_dict[nkeypoints,keypoints , descriptors]
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    key_points_dict = defaultdict(list)

    feature_extractor_list = ['SIFT','ORB','SIMPLE_BLOB_DETECTOR']

    for feature_extractor in feature_extractor_list : 

        if feature_extractor == 'SIFT' :
            """ 
            (nfeatures=..., nOctaveLayers=..., contrastThreshold=..., edgeThreshold=..., sigma=...)
            """ 
            SIFT = cv2.SIFT_create()
            kps, des = SIFT.detectAndCompute(gray, None)
            nKeypoints = len(kps)
            key_points_dict['SIFT'].append([nKeypoints,kps,des])

        elif feature_extractor == 'ORB':

            ORB = cv2.ORB_create()
            kps, des = ORB.detectAndCompute(gray, None)
            nKeypoints = len(kps)
            key_points_dict['ORB'].append([nKeypoints,kps,des])

        elif feature_extractor == 'SIMPLE_BLOB_DETECTOR':

            params = cv2.SimpleBlobDetector_Params()

            params.minThreshold = 10
            params.maxThreshold = 200
            params.filterByArea = True
            params.minArea = 1500
            params.filterByCircularity = True
            params.minCircularity = 0.1
            params.filterByConvexity = True
            params.minConvexity = 0.87
            params.filterByInertia = True
            params.minInertiaRatio = 0.01

            SimpleBlobDetector = cv2.SimpleBlobDetector_create(params)

            kps = SimpleBlobDetector.detect(gray)
            nKeypoints = len(kps)
            des = None
            key_points_dict['SIMPLE_BLOB_DETECTOR'].append([nKeypoints,kps,des])

        else :
            continue

    if show : 
        
        for key in key_points_dict.keys():

            print(f'module: {key} Number of key point extracted : {key_points_dict[key][0][0]}')

    return key_points_dict


def drawKeyPts_single_image(img_path,feature_extractor = 'SIFT' ,col = (0,0,255),th = 5 ,circle_visualization = False,show = False,save = False,output_dir_path = None,return_key_points_count = False):
    """ 
        using SIFT 
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(gray, None)
   
    for key_point in kps:

        x=np.int(key_point.pt[0])
        y=np.int(key_point.pt[1])
        
        size = 0

        if circle_visualization :
            size = np.int(key_point.size)
        
        cv2.circle(img,(x,y),size, col,thickness=th, lineType=8, shift=0) 
   
    if show : 

        plt.imshow(img)    
        plt.plot()

    if save and output_dir_path is not None : 
        save_path = os.path.join(output_dir_path, 'key_points_' + os.path.basename(img_path))
        cv2.imwrite(save_path,img)
    
    if return_key_points_count :
        print(f'detect {len(kps)} keypoints in image : {os.path.basename(img_path)}')
    return img, kps, des 



def Calc_and_Plot_matched_keypoints_between_two_images(img1_path, img2_path , show = False):
    
    """
       defualt feature extractor is SIFT
       return matches, kpA, desA,kpB, desB
    """

    img1_name = os.path.basename(img1_path)
    img2_name = os.path.basename(img2_path)

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    sift = cv2.SIFT_create()
    kpA, desA = sift.detectAndCompute(gray1, None)
    kpB, desB = sift.detectAndCompute(gray2, None)

    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(desB, desB)
    matches = sorted(matches, key=lambda x: x.distance)


    if show :

        matched_image = cv2.drawMatches(img1, kpA, img2, kpB, matches, None, flags=2)
        plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
        plt.show() 

    return matches, kpA, desA,kpB, desB


def Calc_hist_grayscale(img , show = False) : 
    
    """
    Params:
    images : it is the source image of type uint8 or float32. it should be given in square brackets, ie, "[img]".
    channels : it is also given in square brackets. It is the index of channel for which we calculate histogram. For example, if input is grayscale image, its value is [0]. For color image, you can pass [0], [1] or [2] to calculate histogram of blue, green or red channel respectively.
    mask : mask image. To find histogram of full image, it is given as "None". But if you want to find histogram of particular region of image, you have to create a mask image for that and give it as mask. (I will show an example later.)
    histSize : this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].
    ranges : this is our RANGE. Normally, it is [0,256].
    """

    # # img = cv2.imread(img)
    # img = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    img_ = img.astype(np.uint8)
    hist = cv2.calcHist([img_],[0],None,[256],[0,256])
    
    
    if show : 
        plt.hist(img.ravel(),256,[0,256]) 
        plt.show()

    return hist     

def Calc_hist_rgb(img , show) : 
    
    color = ('b','g','r')
    
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

def resize_image_to_multiple_scales(img_path, scale_args ,output_path = None ,show=False, save= False):
    
    """ 
        WARNING : THIS FUNCTION FORCE CONVERSION TO 0-255 , NP.UINT8
        args = list of downsampling ratio numbers 
        ex : args = [2,4,8]
    """

    try : 
        img = cv2.imread(img_path)
    except Exception as e : 
        img = img_path

    h,w = img.shape[:2]

    for arg in scale_args : 
        
        resized_img = skimage_resize(img.copy(), ( h//arg , w // arg ), anti_aliasing=True )
        
        resized_img*=255 #convert from float[0-1] to uint8 [0-255] 
        resized_img = resized_img.astype(np.uint8)
        
        h_,w_ = resized_img.shape[:2]


        if show :    
            plot_img_opencv(resized_img)
        if save : 
            resized_image_name = os.path.basename(img_path)[:-len('.jpg')] + '_factor' + str(arg*arg) + '_size' +f'{h_}' + '_' +f'{w_}' + '.jpg'
            resized_img_output_path  = os.path.join(output_path,resized_image_name)
            cv2.imwrite(resized_img_output_path,resized_img)



