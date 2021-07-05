from __future__ import print_function
from logging import exception
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import shutil 
from skimage.transform import resize as skimage_resize 


"""
Utility script for general image processing tasks
"""

def resize_img1_according_to_img2(img1_path,img2_path,output_dir_path = None, save = False):
    """
        There is use of :Normalize_img_by_min_max func

        Input : img1_path , img2_path , output_dir_path
        Output: img1 resized as the shape of img2 
    """

    img1_name = os.path.basename(img1_path)[:-len('.jpg')]
    img2_name = os.path.basename(img2_path)[:-len('.jpg')]
    img1_name_resized = f'{img1_name}_resized_according_to_{img2_name}.jpg'
    
    img1 = cv2.imread(img1_path,0)
    img2 = cv2.imread(img2_path,0)
    
    img1_resized_according_to_img2 = skimage_resize(img1.copy(), ( img2.shape[0], img2.shape[1]), anti_aliasing=True )
    img1_resized_according_to_img2 = Normalize_img_by_min_max(img1_resized_according_to_img2)
    
    if save:
        
        img1_resized_path = os.path.join(output_dir_path , img1_name_resized)
        
        cv2.imwrite(img1_resized_path,img1_resized_according_to_img2)
    
    return img1_resized_according_to_img2, img2 , img1_resized_path ,img2_path

  
def Normalize_img_by_min_max(img):

    """ 
    def normalize8(I):

        mn = I.min()
        mx = I.max()

        mx -= mn

        I = ((I - mn)/mx) * 255
        return I.astype(np.uint8)
    """ 

    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

def calc_image_range(img,display = True):

    min_,max_ = np.min(img), np.max(img)

    if display :
        print(f'\nImage values range is ==> Min: {min_} Max: {max_}\n')

    return min_,max_

def threshold_otsu(img_path,show = False):

    """input : grayscale 1d image """

    try : 
        img = cv2.imread(img_path,0)
    except Exception as e : 
        img = img_path

    if img.dtype != np.uint8 : 
        img *= 255  
        img = img.astype(np.uint8)
        
    _,threshold_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    
    if show : 
        
        plot_img_opencv(threshold_img)

    return threshold_img

def threshold_otsu_from_img(img,show = False):

    """input : grayscale 1d image """

    _,threshold_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    
    if show : 
        
        plot_img_opencv(threshold_img)

    return threshold_img

def Blur(img, ker_size=(3, 3),show = False):
    
    blured = cv2.GaussianBlur(img, ksize=ker_size, sigmaX=0)
    
    if show : 
    
        plot_img_opencv(blured)
    
    return blured

def erode(img,structuring=cv2.MORPH_RECT ,size = (3,3),iter = 1 ,show = False):
    
    elem = cv2.getStructuringElement(structuring, size)
    
    eroded = cv2.erode(img, elem, iterations=iter)
    
    if show : 

        plot_img_opencv(eroded)

    return eroded

def dilate(img,structuring=cv2.MORPH_RECT ,size = (3,3),iter = 1 ,show = False):
    
    elem = cv2.getStructuringElement(structuring, size)
    
    dilated = cv2.dilate(img, elem,iterations=iter)
    
    if show : 

        plot_img_opencv(dilated)

    return dilated

def Resize(image, width=None, height=None, inter=cv2.INTER_AREA):

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
        
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

def Canny(image, sigma=0.33,show = False):

    # compute the median of the single channel pixel intensities
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    canny_edged = cv2.Canny(image, lower, upper)

    if show : 
        
        plot_img_matplotlib(canny_edged)

    return canny_edged

def find_maxima_points_on_corr_map_of_template_matching_above_th (img,template,th) : 
    
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where( result >= th) #return [coord_y_arr , coord_x_arr]
    results = zip(*loc[::-1])     #fliping to [coord_x_arr , coord_y_arr]
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(results) #gets only single changel array
    
    return results,min_val, max_val, min_loc, max_loc
            



def pad_image_for_centering(path , Resize = False , Resize_Size = None ):
    """ 
        centering rgb image , padding with constant value
        but can be adjusted
    """
    img = cv2.imread(path)
    h,w = img.shape[:2]
    pad_val = np.abs((h - w ) // 2)

    if h > w :  
        img_pad = np.pad(img, ((0, 0), (pad_val, pad_val),(0, 0)), 'constant', constant_values=255)
    else: # w > h : 
        img_pad = np.pad(img, ((pad_val, pad_val),(0,0),(0, 0)), 'constant', constant_values=255)

    return img_pad

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes


def label_contour_opencv(image, c, i, color=(0, 255, 0), thickness=2):
    # compute the center of the contour area and draw a circle
    # representing the center
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # draw the contour and label number on the image
    cv2.drawContours(image, [c], -1, color, thickness)
    cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)

    # return the image with the contour number drawn on it
    return image

def Adjust_Lumin_Condition_CLAHE(img):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def nms(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
 #
    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms




def translate(image, x, y):
    # define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # return the translated image
    return shifted


def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def skeletonize(image, size, structuring=cv2.MORPH_RECT):
    # determine the area (i.e. total number of pixels in the image),
    # initialize the output skeletonized image, and construct the
    # morphological structuring element
    area = image.shape[0] * image.shape[1]
    skeleton = np.zeros(image.shape, dtype="uint8")
    elem = cv2.getStructuringElement(structuring, size)

    # keep looping until the erosions remove all pixels from the
    # image
    while True:
        # erode and dilate the image using the structuring element
        eroded = cv2.erode(image, elem)
        #temp = cv2.dilate(eroded, elem)

        # subtract the temporary image from the original, eroded
        # image, then take the bitwise 'or' between the skeleton
        # and the temporary image
        temp = cv2.subtract(image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image = eroded.copy()

        # if there are no more 'white' pixels in the image, then
        # break from the loop
        if area == area - cv2.countNonZero(image):
            break

    # return the skeletonized image
    return skeleton

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")



def blured_circle_region(img, type_='Circle'):

    blurred_img = cv2.GaussianBlur(img, (21, 21), 0)
    shape_ = img.shape
    mask = np.zeros(shape_, dtype=np.uint8)
    mask = cv2.circle(mask, (258, 258), 100, (255, 255, 255), -1)

    out = np.where(mask == np.array([255, 255, 255]), img, blurred_img)
    return out


def Connected_Components(img):

    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    num_labels, labels_im = cv2.connectedComponents(img)

    def imshow_components(labels):
        # Map component labels to hue val
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # cvt to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue == 0] = 0

        plot_img_opencv(labeled_img)

    imshow_components(labels_im)

    return num_labels, labels_im

def create_dir_with_override(dir_path):
    try : 
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    except Exception as e : 
        print(e)
        print('Could not create the desired dir with the corersponding dir path : \n' + f'{dir_path}')

def Find_global_min_and_max_in_single_chanel_array(array,mask = np.empty([])):

    """ 
        Finds the global minimum and maximum in an array , and their location .
        if you need this for multi chanel arrays , use reshape .
    """

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(array, mask)

def find_maxima_points_on_corr_map_of_template_matching_above_th (img,template,th) : 
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where( result >= th)
    results = zip(*loc[::-1])
    return results

def resize_scale_downsample_skimage(image = None):
    
    import matplotlib.pyplot as plt
    from skimage import data, color
    from skimage.transform import rescale, resize, downscale_local_mean

    if not image : 
        image = color.rgb2gray(data.astronaut())
    
    image_rescaled = rescale(image, 0.25, anti_aliasing=False)
    image_resized = resize(image, (image.shape[0] // 4, image.shape[1] // 4),
                        anti_aliasing=True)
    image_downscaled = downscale_local_mean(image, (4, 3))

    fig, axes = plt.subplots(nrows=2, ncols=2)

    ax = axes.ravel()

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(image_rescaled, cmap='gray')
    ax[1].set_title("Rescaled image (aliasing)")

    ax[2].imshow(image_resized, cmap='gray')
    ax[2].set_title("Resized image (no aliasing)")

    ax[3].imshow(image_downscaled, cmap='gray')
    ax[3].set_title("Downscaled image (no aliasing)")

    ax[0].set_xlim(0, 512)
    ax[0].set_ylim(512, 0)
    plt.tight_layout()
    plt.show()