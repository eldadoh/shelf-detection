import numpy as np 
import cv2,skimage
import matplotlib.pyplot as plt
from img_utils import Resize,calc_image_range

plt.rcParams["figure.figsize"] = (16,10)
font = {'family' : 'DejaVu Sans',
        'size'   : 18}
plt.rc('font', **font)

"""
    plot_img_matplotlib(image, title = '',show_colorbar = False)
    plot_subplot_1x2(images_list)
    plot_subplot_2x2(images_list)
    plot_img_opencv(image,resize_flag = True,width=400)
    plot_x_y_matplotlib(x = np.empty([]),y = np.empty([]),title = '')

"""

def plot_img_matplotlib(image, title = '',show_colorbar = False):  

    img = image.copy()
         
    fig = plt.figure()

    plt.title(title)
    
    if len(img.shape) == 2:
        plt.imshow(img,cmap = 'gray')
    else:
        plt.imshow(img[:,:,::-1])

    plt.xticks([])
    plt.yticks([])

    if show_colorbar:
        
        plt.colorbar()

    plt.show()

def plot_subplot_1x2(images_list,titles_list = ["",""],cmap_list =['gray','gray']):
    
    img1,img2 = images_list[:]
    title1,title2 = titles_list[:] 
    cmap1,cmap2 = cmap_list[:]

    fig, axes = plt.subplots(nrows=1, ncols=2)
   
    ax = axes.ravel()

    ax[0].imshow(img1,cmap = cmap1)
    ax[0].set_title(title1)
    ax[0].axis('off')

    ax[1].imshow(img2,cmap = cmap2)
    ax[1].set_title(title2)
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()


def plot_subplot_1x3(images_list,titles_list = ["","",""],cmap_list =['gray','gray','gray']):
    
    img1,img2,img3 = images_list[:]
    title1,title2,title3 = titles_list[:]
    cmap1,cmap2,cmap3 = cmap_list[:]

    fig, axes = plt.subplots(nrows=1, ncols=3)

    ax = axes.ravel()

    ax[0].imshow(img1,cmap = cmap1)
    ax[0].set_title(title1)
    ax[0].axis('off')
    ax[1].imshow(img2,cmap = cmap2)
    ax[1].set_title(title2)
    ax[1].axis('off')
    ax[2].imshow(img3,cmap = cmap3)
    ax[2].set_title(title3)
    ax[2].axis('off')
    plt.tight_layout()
    plt.show()


def plot_subplot_2x2(images_list,titles_list = ["","","",""],cmap_list =['gray','gray','gray','gray']):
    
    img1,img2,img3,img4 = images_list[:]
    title1,title2,title3,title4 = titles_list[:]
    cmap1,cmap2,cmap3,cmap4 = cmap_list[:]

    fig, axes = plt.subplots(nrows=2, ncols=2)

    ax = axes.ravel()

    ax[0].imshow(img1,cmap = cmap1)
    ax[0].set_title(title1)
    ax[0].axis('off')
    ax[1].imshow(img2,cmap = cmap2)
    ax[1].set_title(title2)
    ax[1].axis('off')
    ax[2].imshow(img3,cmap = cmap3)
    ax[2].set_title(title3)
    ax[2].axis('off')
    ax[3].imshow(img4,cmap = cmap4)
    ax[3].set_title(title4)
    ax[3].axis('off')
    plt.tight_layout()
    plt.show()

def plot_img_opencv(image,resize_flag = True,width=400):
    """
    This func force conversion to uint8 dtype
    and by default resize the input img
    """
    img = image.copy()

    if not img.dtype == np.uint8:

        img *= 255  
        img = img.astype(np.uint8)
    
    if resize_flag:

        img = Resize(img,width)
    
    cv2.imshow('_', img)
    k = cv2.waitKey(0)

    if k != ord('s'):

        cv2.destroyAllWindows()

    elif k == ord('s'):

        cv2.destroyAllWindows()

        img_name = input('Enter image name for saving :\n')

        cv2.imwrite(img_name +'.jpg', img)
        
    return img 

def plot_x_y_matplotlib(x = np.empty([]),y = np.empty([]),title = ''):

    fig = plt.figure()
    plt.suptitle(f'{title}')
    
    x = np.empty([])
    y = np.empty([]) 

    plt.subplot(1, 2, 1)
    plt.plot(x, y)

    plt.subplot(1, 2, 2)
    plt.plot(x, y)

    plt.show()
  
def show_histogram(img,bins_num,display = True): 

    hist_vals , x_bins =  np.histogram(img,bins_num)
    hist_vals = np.concatenate([np.array([0]),hist_vals])

    if display : 
        
        calc_image_range(img)
        print('\nThe non-zero values on the histogram are :\n')
        print(x_bins[hist_vals!=0])
        
    fig = plt.figure()
    
    plt.scatter(x_bins[hist_vals!=0],hist_vals[hist_vals!=0],color = 'red')
    plt.scatter(x_bins[hist_vals==0],hist_vals[hist_vals==0],color = 'blue')
    
    plt.show()