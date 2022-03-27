##################################################
# 2. Augment_cls.py
# Note: code modified from original in
# https://github.com/naokishibuya/car-behavioral-cloning/blob/master/utils.py
# Available for audit in audit_files/naoki from sharepoint link
##################################################
import cv2, os
import numpy as np
import matplotlib.image as mpimg
import conf
import Automold as am
import Helpers as hp

class Augment_cls():
  """
  Augmentation methods
  """
  img_dims = []
  
  def __init__(self, model):
    """
    Set image dimensions for model
    Inputs
    model: string, model name
    """
    self.img_dims = self.get_image_dimensions(model)
    
  def get_image_dimensions(self, model):
    """
    Get the required dimensions for image model, used for resizing and cropping images
    Inputs
    model: string, name of network model
    Output
    int: IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH_NET, IMAGE_HEIGHT_NET, TOP_CROP,
    BOTTOM_CROP
    """
    if (model == conf.ALEXNET):
      return conf.alexnet_img_dims
    elif (model == conf.NVIDIA1):
      return conf.nvidia1_img_dims
    elif (model == conf.NVIDIA2):
      return conf.nvidia2_img_dims
    elif (model == conf.NVIDIA_BASELINE):
      return conf.nvidia_baseline_img_dims
    else:
      # default to nvidia1
      return conf.nvidia1_img_dims
    
  def add_rain(self, image_arr, rt=None, st=0):
        """
        Add rain to image
        Inputs:
            image_arr: numpy array containing image
            rt: string, rain type "heavy" or "torrential"
            st: range to draw a random slant from
        Output
            image_arr: numpy array containing image with rain
        Example
        TODO
        Maybe this should kept is a separate class? We are adding noise in this case.
        """
        # print("Adding rain...")
        if(st != 0):
            # draw a random number for slant
            st = np.random.randint(-1 * st, st)

        if(rt!='light'): # heavy or torrential
            image_arr = am.add_rain_single(image_arr, rain_type=rt, slant=st)
        else:
             # no slant
            image_arr = am.add_rain_single(image_arr)

        return image_arr    
    
  def load_image(self, image_path):
    """
    Load RGB images from a file
    """
    return mpimg.imread(image_path)
  
  def crop(self, image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    # this breaks nvidia_baseline
    # return image[60:-25, :, :] # remove the sky and the car front
    return image[self.img_dims[conf.IMG_TOP_CROP_IDX]:self.img_dims[conf.IMG_BOTTOM_CROP_IDX],:,:] # remove the sky and the car front
  
  def resize(self, image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (self.img_dims[conf.IMG_WIDTH_NET_IDX], self.img_dims[conf.IMG_HEIGHT_NET_IDX]), cv2.INTER_AREA)
  
  def resize_expected(self, image):
    """
    Resize the image to the expected original shape
    """
    return cv2.resize(image, (self.img_dims[conf.IMG_WIDTH_IDX], self.img_dims[conf.IMG_HEIGHT_IDX]), cv2.INTER_AREA)
  
  def rgb2yuv(self, image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
  
  def preprocess(self, image):
    """
    Combine all preprocess functions into one
    Input
        self, instance of the class
        image: numpy array of image
    Output
        image
    Example 
    """
    image = self.crop(image)
    image = self.resize(image)
    image = self.rgb2yuv(image)
    return image
  
  def random_flip(self, image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
      image = cv2.flip(image, 1)
      steering_angle = -steering_angle
    return image, steering_angle
  
  def random_translate(self, image, steering_angle, range_x, range_y):
    """
    Randomly shift the image vertically and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle
  
  def random_shadow(self, image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = self.img_dims[conf.IMG_WIDTH_IDX] * np.random.rand(), 0
    x2, y2 = self.img_dims[conf.IMG_WIDTH_IDX] * np.random.rand(), self.img_dims[conf.IMG_HEIGHT_IDX]
    xm, ym = np.mgrid[0:self.img_dims[conf.IMG_HEIGHT_IDX], 0:self.img_dims[conf.IMG_WIDTH_IDX]]
    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down. So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1
    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)
    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
  
  def random_brightness(self, image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
  
  def augment(self, image, steering_angle, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    # resize according to expected input shape e.g. AlexNet 224x224, Udacity 320x160, Unity 160x120, etc
    # set in conf.py
    #image = cv2.resize(image, (self.img_dims[conf.IMG_WIDTH_IDX], self.img_dims[conf.IMG_HEIGHT_IDX]),
    # cv2.INTER_AREA)
    image = self.resize_expected(image)
    # image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = self.random_flip(image, steering_angle)
    image, steering_angle = self.random_translate(image, steering_angle, range_x, range_y)
    image = self.random_shadow(image)
    image = self.random_brightness(image)
    return image, steering_angle
