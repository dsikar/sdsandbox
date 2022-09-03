#!/usr/bin/env python
# coding: utf-8

# In[40]:


def sort_unity_files(path, mask):
    """
    Create a sorted dictionary from unity (SDSandbox) files e.g. 
    C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\logs_Mon_Jul_13_08_29_01_2020\\0_cam-image_array_.jpg
    Where the key in example above is 0 (first characters before underscore in 0_cam-image_array_.jpg)
    
    Parameters
    ----------
    path : string
        path to files
    mask : string
        file type
    
    Returns
    -------    
        fdict: dictionary
        Sorted dictionary containing key and file path
             
    Example
    -------    
    path = 'C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\logs_Mon_Jul_13_08_29_01_2020\\'
    mask = '*.jpg'
    fdict = sort_unity_files(path, mask)
    for key in sorted(fdict):
        print("key: {}, value:{}".format(key,fdict[key]))
       
    Note
    -------    
    File path format is OS dependant. OrderedDict must by sorted to order files in the right order.
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


# In[41]:


def overlay_imgs(s_img, l_img, x_offset=50, y_offset=50):
    """
    Overlay two numpy array images
    
    Parameters
    ----------
        s_img: numpy array, small image
        l_img: numpy array, large image
        x_offset: left padding from large to small overlaid image
        y_offset: top padding from large to small overlaid image
        
    Returns
    -------
        image_arr: numpy array containing large image with insert 
        of small image inlaid
        
    Example
    --------
    
    
    """
    l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
    return l_img


# In[42]:


def plot_img_hist(img, scheme='rgb'):
    """
    Plot histogram for an rgb array
    
    Parameters
    -------    
        img: numpy array
        scheme: string, 'rgb' (default) or , 'yuv-rgb'
        If scheme is rgb, maximum number of values in a bins is expected to 3 digit, otherwise
        6 digits and y-axys is plotted on log scale.
    
    Returns
    -------    
        fig: matplotlib.pyplot figure
    
    Example
    -------  
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    ipath = 'C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\logs_Mon_Jul_13_08_29_01_2020\\12893_cam-image_array_.jpg'
    img1 = cv2.imread(ipath) # 120x160x3
    plt.rcParams["figure.figsize"] = (6,4)
    myfig = plot_img_hist(img)
    """
    # from https://discuss.pytorch.org/t/plot-a-histogram-for-multiple-images-full-dataset/67600
    # https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    
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


# In[43]:


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
    
    Parameters
    -------   
        fpath: string, filepath
    
    Returns
    -------  
        st_angle: steering angle
    
    Example
    ------- 
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
    


# In[ ]:


def overlay_imgs(s_img, l_img, x_offset=50, y_offset=50):
    """
    Overlay two numpy array images
    
    Parameters
    ----------
        s_img: numpy array, small image
        l_img: numpy array, large image
        x_offset: left padding from large to small overlaid image
        y_offset: top padding from large to small overlaid image
        
    Returns
    -------
        image_arr: numpy array containing large image with insert 
        of small image inlaid
        
    Example
    --------
    
    
    """
    #import cv2
    #s_img = cv2.imread("smaller_image.png")
    #l_img = cv2.imread("larger_image.jpg")
    # x_offset=y_offset=50
    l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
    return l_img


# In[ ]:


# TODO, need to bring this from existing module
def add_rain(image_arr, rt=None, st=0):
    """
    Add rain to image
    
    Parameters
    ----------
        image_arr: numpy array containing image
        rt: string, rain type "heavy" or "torrential"
        st: range to draw a random slant from
        
    Returns
    -------
        image_arr: numpy array containing image with rain
        
    Example
    --------
    
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


# In[52]:


def make_video(fdict, model, preproc=False):
    """
    Make video from image dictionary.
    video.avi is written to disk
    
    Parameters
    -------    
        fdict: collections.OrderedDict, ordered dictionary of file names
        model: string, model name
        preproc: boolean, show preprocessed image next to original
    
    Returns
        none
    
    Example
    -------
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
            myfig = plot_img_hist(image, 'rgb')
            myfig.savefig("temp_plot.png")
            image2 = cv2.imread("temp_plot.png")
            # save
            plt.close(myfig)            
            
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
                myfig = plot_img_hist(image, 'yuv-rgb')
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

  
    # subset 100 images - should be quicker
    #path = 'C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\subset_100\\'
    #mask = '*.jpg'
    #fdict = sort_unity_files(path, mask)
    #model = 'nvidia2'
    #make_video(fdict, model, True) # saved as nvidia2.avi

def plot_img_hist(img, scheme='rgb'):
    """
    Plot histogram for an rgb array

    Parameters
    -------
        img: numpy array
        scheme: string, 'rgb' (default) or , 'yuv-rgb'
        If scheme is rgb, maximum number of values in a bins is expected to 3 digit, otherwise
        6 digits and y-axys is plotted on log scale.

    Returns
    -------
        fig: matplotlib.pyplot figure

    Example
    -------
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    ipath = 'C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\logs_Mon_Jul_13_08_29_01_2020\\12893_cam-image_array_.jpg'
    img1 = cv2.imread(ipath) # 120x160x3
    plt.rcParams["figure.figsize"] = (6,4)
    myfig = plot_img_hist(img)
    """
    # from https://discuss.pytorch.org/t/plot-a-histogram-for-multiple-images-full-dataset/67600
    # https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    nb_bins = 256
    count_r = np.zeros(nb_bins)
    count_g = np.zeros(nb_bins)
    count_b = np.zeros(nb_bins)

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
    # figure()
    #plt.yscale('log')
    Rmean = "{:.2f}".format(np.mean(x[0]))
    plt.bar(bins[:-1], count_r, color='r', alpha=0.5, label="red (mean = " + Rmean + ")")
    Gmean = "{:.2f}".format(np.mean(x[1]))
    plt.bar(bins[:-1], count_g, color='g', alpha=0.45, label="green (mean = " + Gmean + ")")
    Bmean = "{:.2f}".format(np.mean(x[2]))
    plt.bar(bins[:-1], count_b, color='b', alpha=0.4, label="blue (mean = " + Bmean + ")")
    # show labels
    plt.legend(loc='upper right')
    plt.xlabel("Bins")
    plt.xticks(np.arange(0, 255, step=25))
    plt.ylabel("Pixels")
    RGBmean = "{:.2f}".format(np.mean(x))
    plt.title("RGB intensity value distributions (mean = " + RGBmean + ")")
    # add a grid
    plt.grid()
    # make y scale logarithmic
    # plt.yscale('log', nonposy='clip')
    # set y limit, may need to change
    # No plotting max for one off images
    #ymax = 10000
    #plt.ylim(0, ymax)
    plt.savefig("temp_plot.jpg")
    plt.close(fig)
    #return fig

# change rgb values
# https://stackoverflow.com/questions/59320564/how-to-access-and-change-color-channels-using-pil
def changeRGB(img, rv=0, gv=0, bv=0):
  """
  Change RGB values using PIL

  Parameters
  -------
  img: uint8 numpy image array
  rv: integer, value to be added to red channel
  gv: integer, value to be added to green channel
  bv, integer, value to be added to blue channel

  Output
  -------
  myimg: uint8 numpy image array

  Example
  -------
  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg
  img = mpimg.imread('steph.jpeg')
  myimg = changeRGB(img, 60, 0, 0)
  plt.imshow(myimg)
  """
  from PIL import Image
  import numpy as np

  im = Image.fromarray(np.uint8(img))

  # Split into 3 channels
  r, g, b = im.split()

  # Red
  r = r.point(lambda i: i + rv)

  # Green
  g = g.point(lambda i: i + gv)

  # Blue
  b = b.point(lambda i: i + bv)

  # Recombine back to RGB image
  result = Image.merge('RGB', (r, g, b))

  # Convert to uint8 numpy array
  myimg = np.asarray(result)

  return myimg

# see https://peps.python.org/pep-0008/ for naming convention
def get_rgb_avgs(img, scheme='rgb'):
    """
    Return individual channel and overall rgb intensity values
    Parameters
    -------
        img: numpy array
        scheme: string, 'rgb' (default) or , 'yuv-rgb'
        If scheme is rgb, maximum number of values in a bins is expected to 3 digit, otherwise
        6 digits and y-axys is plotted on log scale.
    Returns
    -------
        rgb_avg: float, overall rgb intensity average
        r_avg: float, red channel intensity average
        g_avg: float, green channel intensity average
        b_avg: float, blue channel intensity average
    Example
    -------
    from PIL import Image
    import numpy as np
    import cv2
    from google.colab.patches import cv2_imshow
    imgpath = '48140_cam-image_array_.jpg'
    img = Image.open(imgpath)
    rgb_avg, r_avg, g_avg, b_avg = get_rgb_avgs(img)
    print("RGB avg: {:.2f}, Red avg: {:.2f}, Green avg: {:.2f}, Blue avg: {:.2f}".format(rgb_avg, r_avg, g_avg, b_avg))
    """
    # from https://discuss.pytorch.org/t/plot-a-histogram-for-multiple-images-full-dataset/67600
    # https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    #from PIL import Image
    import numpy as np
    #import matplotlib.pyplot as plt

    #nb_bins = 256
    #count_r = np.zeros(nb_bins)
    #count_g = np.zeros(nb_bins)
    #count_b = np.zeros(nb_bins)

    # Calculate manual hist
    x = np.array(img)
    x = x.transpose(2, 0, 1)
    #Rmean = "{:.2f}".format(np.mean(x[0]))
    #Gmean = "{:.2f}".format(np.mean(x[1]))
    #Bmean = "{:.2f}".format(np.mean(x[2]))
    #RGBmean = "{:.2f}".format(np.mean(x))
    Rmean = np.mean(x[0])
    Gmean = np.mean(x[1])
    Bmean = np.mean(x[2])
    RGBmean = (np.mean(x))   
    return RGBmean, Rmean, Gmean, Bmean

# subset 100 images - should be quicker
#path = 'C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\subset_100\\'
#mask = '*.jpg'
#fdict = sort_unity_files(path, mask)
#model = 'nvidia2'
#make_video(fdict, model, True) # saved as nvidia2.avi   

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Make Video script')
    parser.add_argument('--filepath', type=str, help='tcpflow log')
    parser.add_argument('--model', type=str, help='model name for video label')
    parser.add_argument('--mask', type=str, help='image file suffix')    
    args = parser.parse_args()
    #fdict = sort_unity_files(args.filepath, args.mask)
    #make_video(fdict, model, True) # saved as nvidia2.avi 
    #make_video(args.filepath, args.model, True)
    # example
    # python utils.py --filepath=C:\\Users\\aczd097\\Downloads\\dataset\\unity\\log_sample\\subset_100\\ \
    # --model=nvidia2     --mask=*.jpg
    
  


# In[ ]:




