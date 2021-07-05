from skimage import io
from skimage import filters
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line
from skimage import color 

def convert_rgb_to_hsv(img):
    return color.convert_colorspace(img, 'RGB', 'HSV')

def threshold_otsu_skimage(img):
    
    th = filters.threshold_otsu(img, nbins=256)
    
    img_th = img >= th
    
    return img_th