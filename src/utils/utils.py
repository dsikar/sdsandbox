#!/usr/bin/env python
# coding: utf-8

# In[136]:


def sort_unity_files(path, mask):
    """
    Create a sorted dictionary from unity (SDSandbox) files e.g. 
    C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\logs_Mon_Jul_13_08_29_01_2020\\0_cam-image_array_.jpg
    Where the key in example above is 0 (first characters before underscore in 0_cam-image_array_.jpg)
    
    --------
    
    Input:
        path, string, path to files
        mask, string, file type
    
    -------
    
    Output:
        Sorted dictionary containing key and file path
    
    -------
            
    Example:
    path = 'C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\logs_Mon_Jul_13_08_29_01_2020\\'
    mask = '*.jpg'
    fdict = sort_unity_files(path, mask)
    for key in sorted(fdict):
        print("key: {}, value:{}".format(key,fdict[key]))
    
    -------
       
    Notes:
    File path format is OS dependant, OrderedDict must by sorted to order files in the right order.
    """
    
    import fnmatch
    import os
    from collections import OrderedDict

    #path = 'C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\logs_Mon_Jul_13_08_29_01_2020\\'
    #mask = '*.jpg'
    filemask = os.path.expanduser(path +  mask)
    path, mask = os.path.split(filemask)

    fdict = OrderedDict()
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, mask):
            fdict[int(filename.split('_')[0])] = os.path.join(root, filename)
    print("Use sorted() function in your for loop to sort the output of this sort_unity_files().")
    return fdict


# In[145]:


def plot_img_hist(img):
    """
    Plot histogram for an rgb array
    
    ====
    
    Input:
        img: numpy array
    Output:
        None
    """
    # from https://discuss.pytorch.org/t/plot-a-histogram-for-multiple-images-full-dataset/67600
    # https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    from PIL import Image
    nb_bins = 256
    count_r = np.zeros(nb_bins)
    count_g = np.zeros(nb_bins)
    count_b = np.zeros(nb_bins)

    #img = Image.open('16_left-2.jpeg')

    # Calculate manual hist
    x = np.array(img)
    x = x.transpose(2, 0, 1)
    hist_r = np.histogram(x[0], bins=nb_bins, range=[0, 255])
    hist_g = np.histogram(x[1], bins=nb_bins, range=[0, 255])
    hist_b = np.histogram(x[2], bins=nb_bins, range=[0, 255])
    count_r = hist_r[0]
    count_g = hist_g[0]
    count_b = hist_b[0]

    # Plot manual
    bins = hist_r[1]
    fig = plt.figure()
    plt.bar(bins[:-1], count_r, color='r', alpha=0.5)
    plt.bar(bins[:-1], count_g, color='g', alpha=0.5)
    plt.bar(bins[:-1], count_b, color='b', alpha=0.5)
    return fig


# In[72]:


path = 'C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\logs_Mon_Jul_13_08_29_01_2020\\'
mask = '*.jpg'
fdict = sort_unity_files(path, mask)
for key in sorted(fdict):
    print("key: {}, value:{}".format(key,fdict[key]))


# In[161]:


# fpath = fdict[key]
def get_sdsandbox_json_steer_angle(fpath):
    """
    Get steering angle stored in json file.
    The argument passed in the path for a file, that was stored with a corresponding json file 
    containing a steering angle, looks something like:
    C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\logs_Mon_Jul_13_08_29_01_2020\\ \
    12893_cam-image_array_.jpg
    The json file with with steering angle, in the same path, will be named record_12893.json
    We open that file and return the steering angle.
    
    =======
    
    Input:
        fpath: string, filepath
    Output:
        st_angle: steering angle
        
    =======
    
    Example:
    fpath = 'C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\logs_Mon_Jul_13_08_29_01_2020\\12893_cam-image_array_.jpg'
    jsa = get_sdsandbox_json_steer_angle(fpath)
    print(jsa)
    """ 
    
    import json
    # split string
    fsplit = fpath.split('\\')
    # get name e.g. 12893_cam-image_array_.jpg
    fname = fsplit[-1]
    # get number e.g. 12893
    fnumber = fsplit[-1].split('_')
    fnumber = fnumber[0]
    # build json file name e.g. record_12893.json
    fname = 'record_' + fnumber + '.json'
    # build file path e.g. 'C:\Users\aczd097\Downloads\dataset\unity\log_sample\logs_Mon_Jul_13_08_29_01_2020\record_12893.json'
    idx = fpath.rindex('\\') + 1
    fname = fpath[0:idx] + fname
    # open and read file
    f = open(fname, "r")
    file = f.read()
    # load json
    fjson = json.loads(file)
    # get and return steering angle attribute
    st_angle = fjson['user/angle']
    return st_angle
    
# fpath = 'C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\logs_Mon_Jul_13_08_29_01_2020\\12893_cam-image_array_.jpg'
# jsa = get_sdsandbox_json_steer_angle(fpath)
# print(jsa)  

import cv2
import numpy as np
import matplotlib.pyplot as plt
ipath = 'C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\logs_Mon_Jul_13_08_29_01_2020\\12893_cam-image_array_.jpg'
img = cv2.imread(ipath) # 120x160x3
# resize so we can write some info onto image
#img = cv2.resize(img, (800, 600), cv2.INTER_AREA)
plt.imshow(img)
plt.rcParams["figure.figsize"] = (6,4)
myfig = plot_img_hist(img)
myfig.savefig("temp_plot.png")
img2 = cv2.imread("temp_plot.png")
img3 = overlay_imgs(img2, img1)
plt.close(myfig)


# # Overlay images

# In[165]:


img2 = cv2.imread("temp_plot.png")
img3 = overlay_imgs(img2, img1)
plt.imshow(img3)


# In[147]:


type(np.asarray(myfig)


# In[168]:


# TODO Finish writing this function, looping through fdict instead of current scheme
# Goods-to-have plot RGB distribution on the video as it goes along
def make_video(fdict, model, preproc=False):
    """
    Make video from image dictionary.
    video.avi is written to disk
    
    -------
    
    Inputs
        fdict: collections.OrderedDict, ordered dictionary of file names
        model: string, model name
        preproc: boolean, show preprocessed image next to original
        
    -------
    
    Output
        none
    
    -------
    
    Example

    path = 'C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\logs_Mon_Jul_13_08_29_01_2020\\'
    mask = '*.jpg'
    fdict = sort_unity_files(path, mask)
    
    model = 'nvidia2'
    make_video(fdict, model, True) # saved as nvidia2.avi
    
    
    """
    
    import os
    import sys
    # append local path so we can make use
    # of locally defined modules
    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)


    import argparse
    import fnmatch
    import json
    import os
    from io import BytesIO
    from PIL import Image
    import base64
    import numpy as np
    import matplotlib.pyplot as plt
    import Augment_cls as Augmentation
    import cv2
    import conf    
    
    # instantiate augmentation class
    ag = Augmentation.Augment_cls(model)
    # video name
    video_name = model + '.avi'
    VIDEO_WIDTH, VIDEO_HEIGHT = 800, 600
    IMAGE_WIDTH, IMAGE_HEIGHT = 800, 600
    if(preproc == True): # wide angle
        VIDEO_WIDTH = IMAGE_WIDTH*2
    video = cv2.VideoWriter(video_name, 0, 11, (VIDEO_WIDTH, VIDEO_HEIGHT)) # assumed 11fps
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # frame count
    fno = 1
    try:
        for key in sorted(fdict):

            image = cv2.imread(fdict[key]) # 120x160x3
            # get histogram
            myfig = plot_img_hist(image)
            myfig.savefig("temp_plot.png")
            image2 = cv2.imread("temp_plot.png")
            # save
            plt.close(myfig) 
            
            #plt.imshow(img)
            #plt.rcParams["figure.figsize"] = (6,4)
            #myfig = plot_img_hist(img)
            #myfig.savefig("temp_plot.png")
            #img2 = cv2.imread("temp_plot.png")
            #img3 = overlay_imgs(img2, img1)
            #plt.close(myfig)            
            
            image_copy = image
            # resize so we can write some info onto image
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
            # add Info to frame
            cv2.putText(image, model, (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # overlay histogram
            image = overlay_imgs(image2, image)
            pst = get_sdsandbox_json_steer_angle(fdict[key])
            pst *= conf.norm_const
            simst = "Frame: {}, Actual steering angle: {:.2f}".format(str(fno), pst)
            cv2.putText(image, simst, (50, 115), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # create a preprocessed copy to compare what simulator generates to what network "sees"
            if (preproc == True):  # wide angle
                image2 = ag.preprocess(image_copy)
                image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
                cv2.putText(image2, 'Network Image', (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # histogram on network image
                myfig = plot_img_hist(image)
                myfig.savefig("temp_plot.png")
                image4 = cv2.imread("temp_plot.png")
                # save
                plt.close(myfig)  
                # overlay
                image2 = overlay_imgs(image4, image2)
                
            # concatenate
            if (preproc == True):  # wide angle
                cimgs = np.concatenate((image, image2), axis=1)
                image = cimgs
            # write to video
            video.write(image);
            # increment frame counter
            fno = fno + 1
            
    except Exception as e:
        print("Exception raise: " + str(e))

    cv2.destroyAllWindows()
    video.release()


#if __name__ == "__main__":
#    import argparse
#    parser = argparse.ArgumentParser(description='Make Video script')
#    parser.add_argument('--filename', type=str, help='tcpflow log')
#    parser.add_argument('--model', type=str, help='model name for video label')
#    args = parser.parse_args()
#    make_video(args.filename, args.model, True)
    # example
    # python MakeVideo.py --filename=/tmp/tcpflow.log --model=20201120184912_sanity.h5


# In[ ]:


# path = 'C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\subset_100\\'
path = 'C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\subset\\'
mask = '*.jpg'
fdict = sort_unity_files(path, mask)
# subset 100 images - should be quicker
model = 'nvidia2'
make_video(fdict, model, True) # saved as nvidia2.avi
#for key in sorted(fdict):
#    print(fdict[key])


# In[107]:


import os
import sys
# find the current path, might not be necessary running as a .py script
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

    
import Augment_cls as Augmentation
ag = Augmentation.Augment_cls('nvidia2')  
import conf

path = 'C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\logs_Mon_Jul_13_08_29_01_2020\\'
mask = '*.jpg'
fdict = sort_unity_files(path, mask)
len(fdict) 


# In[152]:


import cv2
import matplotlib.pyplot as plt
img = cv2.imread(fdict[100]) 
img1 = cv2.resize(img, (800, 600), cv2.INTER_AREA)
# plt.rcParams["figure.figsize"] = (20,3)
plt.imshow(img1)

#jsa = get_sdsandbox_json_steer_angle(fdict[100])
#print(jsa)  


# In[64]:


import Augment_cls as Augmentation
ag = Augmentation.Augment_cls('nvidia2')

path = 'C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\logs_Mon_Jul_13_08_29_01_2020\\'
mask = '*.jpg'
fdict = sort_unity_files(path, mask)
print(type(fdict))
print(type(sorted(fdict)))
#for key in sorted(fdict):
#    print("key: {}, value:{}".format(key,fdict[key]))

len(fdict) 


# In[65]:


import cv2
import matplotlib.pyplot as plt
img = cv2.imread(fdict[0]) 
plt.imshow(img)

import cv2
s_img = cv2.imread("smaller_image.png")
l_img = cv2.imread("larger_image.jpg")
x_offset=y_offset=50
l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img


# In[66]:


img_arr = add_rain(img, 'torrential', 20)
plt.imshow(img_arr)


# In[153]:


def overlay_imgs(s_img, l_img, x_offset=50, y_offset=50):
    """
    
    """
    #import cv2
    #s_img = cv2.imread("smaller_image.png")
    #l_img = cv2.imread("larger_image.jpg")
    # x_offset=y_offset=50
    l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
    return l_img

import numpy as np
# img_arr = np.asarray(img_arr, dtype=np.float32)

# set to same image size expected from acquisition process
img_arr = ag.resize_expected(img_arr)

# check for rain
#if(conf.rt != ''):
#    img_arr = add_rain(img_arr, conf.rt, conf.st)
#    self.img_add_rain = img_arr

# same preprocessing as for training
img_arr = ag.preprocess(img_arr)
plt.imshow(img_arr)       
# img = ag.preprocess(img)


# In[18]:


# TODO, need to bring this from existing module
def add_rain(image_arr, rt=None, st=0):
    """
    Add rain to image
    Inputs:
        image_arr: numpy array containing image
        rt: string, rain type "heavy" or "torrential"
        st: range to draw a random slant from
    Output
        image_arr: numpy array containing image with rain
    """
    import Automold as am
    
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

