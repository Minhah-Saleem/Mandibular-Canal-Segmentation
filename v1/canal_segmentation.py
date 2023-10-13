#!/usr/bin/env python
# coding: utf-8

# In[161]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# from tensorflow import keras
import tensorflow as tf
tf.random.set_seed(1234)
# from tensorflow.keras.layers import ConvLSTM2D, Bidirectional, BatchNormalization, Conv3D, Cropping3D, ZeroPadding3D, Activation,Input, Conv3D, ZeroPadding3D, UpSampling3D, Dense, concatenate, Conv3DTranspose, Input
# from tensorflow.keras.layers import MaxPooling3D, UpSampling3D, concatenate, GlobalAveragePooling3D
from tensorflow.keras.layers import Conv3DTranspose as Deconvolution3D
# from tensorflow.keras.models import Model
# from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv3D, ZeroPadding3D, UpSampling3D, Dense, concatenate, Conv3DTranspose, Cropping3D, PReLU
from tensorflow.keras.layers import MaxPooling3D, GlobalAveragePooling3D, AvgPool3D
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input

import numpy as np
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
from sys import getsizeof

import os
import cv2
import tensorflow as tf
import pickle
import glob as glob
from sklearn.utils import shuffle
from scipy import ndimage
from tqdm import tqdm
from sklearn import preprocessing
from math import ceil,floor
import json
import SimpleITK as sitk
import os 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import glob
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import skimage
from scipy import ndimage
#from scipy.ndimage import morphology
from skimage import measure, feature
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imutils
from PIL import Image
import pickle
from multiprocessing import Pool


from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
import cv2
from skimage import morphology

import utils as ut
import teeth_segmentation 


# In[13]:


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) +
                                           smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# In[14]:


import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, 1, filters)

    se = GlobalAveragePooling3D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x

def stem_block(x, n_filter, strides):
    x_init = x

    ## Conv 1
    x = Conv3D(n_filter, (3, 3, 3), padding="same", strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv3D(n_filter, (3, 3, 3), padding="same")(x)

    ## Shortcut
    s  = Conv3D(n_filter, (1, 1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x


def resnet_block(x, n_filter, strides=1):
    x_init = x

    ## Conv 1
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv3D(n_filter, (3, 3, 3), padding="same", strides=strides)(x)
    ## Conv 2
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv3D(n_filter, (3, 3, 3), padding="same", strides=1)(x)

    ## Shortcut
    s  = Conv3D(n_filter, (1, 1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x

def aspp_block(x, num_filters, rate_scale=1):
    x1 = Conv3D(num_filters, (3, 3, 3), dilation_rate=(6 * rate_scale, 6 * rate_scale, 6 * rate_scale), padding="same")(x)
    x1 = BatchNormalization()(x1)

    x2 = Conv3D(num_filters, (3, 3, 3), dilation_rate=(12 * rate_scale, 12 * rate_scale, 12 * rate_scale), padding="same")(x)
    x2 = BatchNormalization()(x2)

    x3 = Conv3D(num_filters, (3, 3, 3), dilation_rate=(18 * rate_scale, 18 * rate_scale, 8 * rate_scale), padding="same")(x)
    x3 = BatchNormalization()(x3)

    x4 = Conv3D(num_filters, (3, 3, 3), padding="same")(x)
    x4 = BatchNormalization()(x4)

    y = Add()([x1, x2, x3, x4])
    y = Conv3D(num_filters, (1, 1, 1), padding="same")(y)
    return y

def attetion_block(g, x):
    """
        g: Output of Parallel Encoder block
        x: Output of Previous Decoder block
    """

    filters = x.shape[-1]

    g_conv = BatchNormalization()(g)
    g_conv = Activation("relu")(g_conv)
    g_conv = Conv3D(filters, (3, 3, 3), padding="same")(g_conv)

    g_pool = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(g_conv)

    x_conv = BatchNormalization()(x)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv3D(filters, (3, 3, 3), padding="same")(x_conv)

    gc_sum = Add()([g_pool, x_conv])

    gc_conv = BatchNormalization()(gc_sum)
    gc_conv = Activation("relu")(gc_conv)
    gc_conv = Conv3D(filters, (3, 3, 3), padding="same")(gc_conv)

    gc_mul = Multiply()([gc_conv, x])
    return gc_mul

class ResUnetPlusPlus:
    def __init__(self, input_size_1,input_size_2, input_size_3):
        self.input_size_1 = input_size_1
        self.input_size_2 = input_size_2
        self.input_size_3 = input_size_3
        

    def build_model(self):
        n_filters = [8, 16, 32, 64, 128]
        inputs = Input((self.input_size_1, self.input_size_2,self.input_size_3, 1))

        c0 = inputs
        c1 = stem_block(c0, n_filters[0], strides=1)

        ## Encoder
        c2 = resnet_block(c1, n_filters[1], strides=2)
        c3 = resnet_block(c2, n_filters[2], strides=2)
        c4 = resnet_block(c3, n_filters[3], strides=2)

        ## Bridge
        b1 = aspp_block(c4, n_filters[4])

        ## Decoder
        d1 = attetion_block(c3, b1)
        d1 = UpSampling3D((2, 2, 2))(d1)
        d1 = Concatenate()([d1, c3])
        d1 = resnet_block(d1, n_filters[3])

        d2 = attetion_block(c2, d1)
        d2 = UpSampling3D((2, 2, 2))(d2)
        d2 = Concatenate()([d2, c2])
        d2 = resnet_block(d2, n_filters[2])

        d3 = attetion_block(c1, d2)
        d3 = UpSampling3D((2, 2, 2))(d3)
        d3 = Concatenate()([d3, c1])
        d3 = resnet_block(d3, n_filters[1])

        ## output
        outputs = aspp_block(d3, n_filters[0])
        outputs = Conv3D(1, (1, 1, 1), padding="same")(outputs)
        outputs = Activation("sigmoid")(outputs)

        ## Model
        model = Model(inputs, outputs)
        return model


# In[15]:


def get_full_scan(folder_path):

    files_List  = glob.glob(folder_path + '/**/*.dcm', recursive = True)
    itkimage = sitk.ReadImage(files_List[0])
    rows = int(itkimage.GetMetaData('0028|0010'))
    cols = int(itkimage.GetMetaData('0028|0011'))
    mn = 1000
    mx = 0
    for file in tqdm(files_List):
        itkimage = sitk.ReadImage(file)
        mn = np.min([mn, int(itkimage.GetMetaData('0020|0013'))])
        mx = np.max([mx, int(itkimage.GetMetaData('0020|0013'))])
    full_scan = np.ndarray(shape=(mx-mn+1,rows,cols), dtype=float, order='F')

    for file in tqdm(files_List):
        img, n = dcm_image(file)
        n = int(n)
        full_scan[n-mn,:,:] = img[0,:,:]

    return full_scan

def dcm_image(filename):
    global Algo_flag
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    ins = float(itkimage.GetMetaData('0020|0013'))
    #print(ins)
    #if ins==1.0:
    #    print("we here")
    #    print(itkimage.GetMetaData('0020|1041'))
    #    if int(itkimage.GetMetaData('0020|1041'))==153:
    #        Algo_flag = False
        
#     numpyImage = get_normalized(numpyImage,wc,wl)
    return numpyImage, ins

def readJsonLeft(file,sz):
    
    mask = np.zeros(sz)
    mask = mask.astype('uint8')

    with open(file) as json_file:
        data = json.load(json_file)   

    left_coords = data['annotation']['tooth']['Left']['coordinate']

    l = int(len(left_coords)/3)
    b = {}
    b[0] = np.zeros((l, 1))
    b[1] = np.zeros((l, 1))
    b[2] = np.zeros((l, 1))
    j = 0
    for x in range(0, len(left_coords), 3):
        b[0][j] = int(left_coords[x])
        b[1][j] = int(left_coords[x+1])
        b[2][j] = int(left_coords[x+2])
        j = j + 1

    for i in range(len(b[0])):
        mask[int(b[2][i]), int(b[1][i]), int(b[0][i])] = 1

    return mask
    
def readJsonRight(file,sz):
    
    mask = np.zeros(sz)
    mask = mask.astype('uint8')

    with open(file) as json_file:
        data = json.load(json_file)   

    left_coords = data['annotation']['tooth']['Right']['coordinate']

    l = int(len(left_coords)/3)
    b = {}
    b[0] = np.zeros((l, 1))
    b[1] = np.zeros((l, 1))
    b[2] = np.zeros((l, 1))
    j = 0
    for x in range(0, len(left_coords), 3):
        b[0][j] = int(left_coords[x])
        b[1][j] = int(left_coords[x+1])
        b[2][j] = int(left_coords[x+2])
        j = j + 1

    for i in range(len(b[0])):
        mask[int(b[2][i]), int(b[1][i]), int(b[0][i])] = 1

    return mask


def get_normalized(scan,mn,mx):
    mn = max(mn,np.amin(scan))
    mx = min(mx,np.amax(scan))
    np.clip(scan, mn, mx, out=scan)
    d = mx - mn
    scan = (scan-mn)/d
    return scan


def zero_mean(scan):
    mean_ = np.mean(scan)
    std = np.std(scan)
    scan  = (scan - mean_) / std
    return scan

def plot_3d(image, threshold=100, alpha=0.5):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    face_color = [0.8, 0.2, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


# In[16]:


def new_data(scan,left,right):
    gscan = copy.copy(scan)
    mask = left+right
    mask = np.where(mask>1,1,mask)
    crop = np.argwhere(mask>0)
    depth1 = np.amin(crop[:,0])
    depth2 = np.amax(crop[:,0])
    row1 = np.amin(crop[:,1])
    row2 = np.amax(crop[:,1])
    col1 = np.amin(crop[:,2])
    col2= np.amax(crop[:,2])
    nscan = scan[depth1-10:depth2+10,row1-10:row2+10,col1-10:col2+10]
    nmask = mask[depth1-10:depth2+10,row1-10:row2+10,col1-10:col2+10]
    nscanleft = nscan[:,:,round(nscan.shape[2]/2):]
    nscanright = nscan[:,:,:round(nscan.shape[2]/2)]
    nmaskleft = nmask[:,:,round(nmask.shape[2]/2):]
    nmaskright = nmask[:,:,:round(nmask.shape[2]/2)]
    return nscanleft,nscanright,nmaskleft,nmaskright,depth1,depth2,row1,row2,col1,col2

def res_scan(nscan, shape):
    D,W,H  = shape[0], shape[2],shape[1]
    desired_depth = D
    desired_width = W
    desired_height = H
    current_depth = nscan.shape[0]
    current_width = nscan.shape[2]
    current_height = nscan.shape[1]
    print('current: ',nscan.shape)
    print('desired: ',D,H,W)
    depth_factor = desired_depth/current_depth
    width_factor = desired_width/current_width
    height_factor = desired_height/current_height
    nscan = ndimage.zoom(nscan, (depth_factor, height_factor, width_factor), order=1)
    return nscan

def res_mask(nmask, shape):
    D,W,H  = shape[0], shape[2], shape[1]
    desired_depth = D
    desired_width = W
    desired_height = H
    current_depth = nmask.shape[0]
    current_width = nmask.shape[2]
    current_height = nmask.shape[1]
    depth_factor = desired_depth/current_depth
    width_factor = desired_width/current_width
    height_factor = desired_height/current_height
    nmask = ndimage.zoom(nmask, (depth_factor, height_factor, width_factor), order=1, mode = 'nearest' )
    return nmask
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=8)
    return x, y


# In[159]:


def local_type_1(scan,teeth_mask):
    
    gscan = copy.copy(scan)
    gscan = get_normalized(gscan, 0, 2500)*255 # 0:3000
    gscan = gscan.astype('uint8')
    front_m = np.amax(gscan,axis=1)
    #plt.figure(figsize =(10,10))
    #plt.imshow(front_m,'gray')
    front = front_m>20
    front_mips = front_m > 120
    front_mips = morphology.remove_small_objects(front_mips, min_size=200)
    #plt.imshow(front_mips,'gray')
    front_mips = ndimage.morphology.binary_fill_holes(front_mips)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)) 
    front_mips = ndimage.morphology.binary_closing(front_mips,ker)#
    area = 0
    startt = front_mips.shape[0]
    contours,_ = cv2.findContours(front_mips.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        _,y,w,h = cv2.boundingRect(cnt)
        if (y>front_mips.shape[0]*1/4) and (y<(front_mips.shape[0]*(4/5))):
            if w*h > area:
                area = w*h
                startt = y
    f_mips = np.zeros((front_mips.shape[0],front_mips.shape[1]),dtype = np.uint8)
    f_mips = cv2.drawContours(f_mips, cnt, -1, 255, 1)
    f_mips = ndimage.morphology.binary_fill_holes(f_mips)

    teeth_mask = np.amax(teeth_mask,axis = 1)
    teeth_mask = teeth_mask>0
    teeth_mask = ndimage.morphology.binary_fill_holes(teeth_mask)
    #ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)) 
    #teeth_mask = ndimage.morphology.binary_closing(teeth_mask,ker)#
    teeth_mask = morphology.remove_small_objects(teeth_mask, min_size=200)
    
    new_c = np.argwhere(teeth_mask>0)
    try:
    	start = np.amin(new_c[:,0])
    except:
	start = 0
    #print('thr teeth start: ',startt)
    #print('seg teeth start: ',start)
    start = max(start,front_mips.shape[0]*1/4)
    start = min(start,startt)
    #print('teeth start: ',start)

    start = round(start)
    s = np.sum(front,axis = 1)
    #plt.plot(s)
    end = np.argmin(s) - 10
    #print(end)
    #plt.figure(figsize =(10,10))
    cropped_scan = gscan[start:end,:,:]
    d_mips = np.amax(cropped_scan,axis=0)
    #plt.imshow(d_mips, 'gray')
    thres = d_mips>70 # 70
    thres = morphology.remove_small_objects(thres, min_size=500)
    thres = ndimage.morphology.binary_fill_holes(thres)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)) 
    thres = ndimage.morphology.binary_closing(thres,ker)
    #plt.imshow(thres,'gray')
    contours,_ = cv2.findContours(thres.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    peri=d_mips.shape[0]
    num=0
    for cnt in contours:
        _,y,_,_ = cv2.boundingRect(cnt)
        #print(y)
        if y<peri:
            peri = y
            index = num
        num = num+1
    #print(index)
    thres = np.zeros((d_mips.shape[0],d_mips.shape[0]),dtype = np.uint8)
    thres = cv2.drawContours(thres, contours[index], -1, 255, 1)
    thres=thres>0
    thres = ndimage.morphology.binary_fill_holes(thres)
    #plt.figure(figsize = (10,10))
    #plt.imshow(thres,'gray')
    array = np.argwhere(thres>0)
    row = array[:,0]
    col = array[:,1]
    row1 = np.amin(row)
    row2 = np.amax(row)
    col1 = round(max(np.amin(col)-30,0))
    col2 = round(min(np.amax(col)+30,thres.shape[1]))
    start = round(max(start-50,0))
    #print(start,end,row1,row2,col1,col2)
    newscan = copy.copy(scan)
    newscan = newscan[start:end,row1:row2,col1:col2]
    half = round(newscan.shape[2]/2)
    newleftscan = newscan[:,:,half:]
    newrightscan = newscan[:,:,:half]
    
    return newleftscan,newrightscan, start,end,row1,row2,col1,col2,half

def local_type_2(scan,teeth_mask):

    gscan = copy.copy(scan)
    gscan = get_normalized(gscan, 0, 4500)*255 # 0:3000
    gscan = gscan.astype('uint8')
    front_m = np.amax(gscan,axis=1)
    front = front_m>20
    #plt.figure(figsize =(10,10))
    #plt.imshow(front_m,'gray')
    
    front_mips = front_m>200 #200 
    front_mips = morphology.remove_small_objects(front_mips, min_size=200)
    #plt.imshow(front_mips,'gray')
    front_mips = ndimage.morphology.binary_fill_holes(front_mips)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)) 
    front_mips = ndimage.morphology.binary_closing(front_mips,ker)#
    #plt.figure(figsize = (10,10))
    #plt.imshow(front_mips,'gray')
    area = 0
    startt = front_mips.shape[0]
    contours,_ = cv2.findContours(front_mips.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        _,y,w,h = cv2.boundingRect(cnt)
        if (y>front_mips.shape[0]*1/4) and (y<(front_mips.shape[0]*(4/5))):
            if w*h > area:
                area = w*h
                startt = y
    f_mips = np.zeros((front_mips.shape[0],front_mips.shape[1]),dtype = np.uint8)
    f_mips = cv2.drawContours(f_mips, cnt, -1, 255, 1)
    f_mips = ndimage.morphology.binary_fill_holes(f_mips)
    
    teeth_mask = np.amax(teeth_mask,axis = 1)
    teeth_mask = teeth_mask>0
    teeth_mask = ndimage.morphology.binary_fill_holes(teeth_mask)
    #ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)) 
    #teeth_mask = ndimage.morphology.binary_closing(teeth_mask,ker)#
    teeth_mask = morphology.remove_small_objects(teeth_mask, min_size=200)
    
    new_c = np.argwhere(teeth_mask>0)
    try:
    	start = np.amin(new_c[:,0])
    except:
	start = 0
    #print('thr teeth start: ',startt)
    #print('seg teeth start: ',start)
    start = max(start,front_mips.shape[0]*1/4)
    start = min(start,startt)
    #print('teeth start: ',start)
    start = round(start)
    s = np.sum(front,axis = 1)
    s = s.flatten()
    ss = savgol_filter(s, 105, 3)
    mini = argrelextrema(ss, np.less)
    #plt.plot(ss)
    mini = np.ndarray.flatten(np.asarray(mini))
    if (not mini.shape[0] ==0) and (not mini[-1] < start+100):
        end = round(mini[-1] -20)
    else:
        end = front.shape[0]
    #print('end: ',end)
    cropped_scan = gscan[start:end,:,:]
    #print(cropped_scan.shape)
    d_mips = np.amax(cropped_scan,axis=0)
    #plt.imshow(d_mips, 'gray')
    thres = d_mips>80 # 70
    thres = morphology.remove_small_objects(thres, min_size=500)
    thres = ndimage.morphology.binary_fill_holes(thres)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)) 
    thres = ndimage.morphology.binary_closing(thres,ker)
    #plt.imshow(thres,'gray')
    contours,_ = cv2.findContours(thres.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    peri=d_mips.shape[0]
    num=0
    for cnt in contours:
        _,y,_,_ = cv2.boundingRect(cnt)
        #print(y)
        if y<peri:
            peri = y
            index = num
        num = num+1
    #print(index)
    thres = np.zeros((d_mips.shape[0],d_mips.shape[0]),dtype = np.uint8)
    thres = cv2.drawContours(thres, contours[index], -1, 255, 1)
    thres=thres>0
    thres = ndimage.morphology.binary_fill_holes(thres)
    #plt.figure(figsize = (10,10))
    #plt.imshow(thres,'gray')
    array = np.argwhere(thres>0)
    row = array[:,0]
    col = array[:,1]
    row1 = int(np.amin(row))
    row2 = int(np.amax(row))
    col1 = round(max(np.amin(col)-30,0))
    col2 = round(min(np.amax(col)+30,thres.shape[1]))
    start = round(max(start-50,0))
    #print(row1,row2,col1,col2)
    newscan = copy.copy(scan)
    newscan = newscan[start:end,row1:row2,col1:col2]
    half = round(newscan.shape[2]/2)
    newleftscan = newscan[:,:,half:]
    newrightscan = newscan[:,:,:half]
    
    return newleftscan,newrightscan, start,end,row1,row2,col1,col2,half

def local_type_3(scan,teeth_mask):
    gscan = copy.copy(scan)
    gscan = get_normalized(gscan, 500, 4050)*255 # 0:3000
    gscan = gscan.astype('uint8')

    front_m = np.amax(gscan,axis=1)
    front = front_m>30
    #plt.figure(figsize =(10,10))
    #plt.imshow(front,'gray')
    
    front_mips = front_m > 170
    front_mips = morphology.remove_small_objects(front_mips, min_size=200)
    #plt.imshow(front_mips,'gray')
    front_mips = ndimage.morphology.binary_fill_holes(front_mips)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)) 
    front_mips = ndimage.morphology.binary_closing(front_mips,ker)#
    #plt.figure(figsize =(10,10))
    #plt.imshow(front_mips,'gray')
    contours,_ = cv2.findContours(front_mips.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = 0
    startt = front_mips.shape[0]
    for cnt in contours:
        _,y,w,h = cv2.boundingRect(cnt)
        if (y>front_mips.shape[0]*1/4) and (y<(front_mips.shape[0]*(4/5))):
            if w*h > area:
                area = w*h
                startt = y
            
    f_mips = np.zeros((front_mips.shape[0],front_mips.shape[1]),dtype = np.uint8)
    f_mips = cv2.drawContours(f_mips, cnt, -1, 255, 1)
    f_mips = ndimage.morphology.binary_fill_holes(f_mips)

    teeth_mask = np.amax(teeth_mask,axis = 1)
    teeth_mask = teeth_mask>0
    teeth_mask = ndimage.morphology.binary_fill_holes(teeth_mask)
    #ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)) 
    #teeth_mask = ndimage.morphology.binary_closing(teeth_mask,ker)#
    teeth_mask = morphology.remove_small_objects(teeth_mask, min_size=200)
    
    new_c = np.argwhere(teeth_mask>0)
    try:
    	start = np.amin(new_c[:,0])
    except:
	start = 0
    #print('thr teeth start: ',startt)
    #print('seg teeth start: ',start)
    start = max(start,front_mips.shape[0]*1/4)
    start = min(start,startt)
    #print('teeth start: ',start)
    start = round(start)
    
    s = np.sum(front,axis = 1)
    #plt.figure(figsize =(10,10))
    #plt.plot(s)
    end = np.argmin(s)
    #print('end: ',end)
    cropped_scan = gscan[start:end,:,:]
    d_mips = np.amax(cropped_scan,axis=0)
    #plt.imshow(d_mips, 'gray')
    thres = d_mips>50 # 70
    thres = morphology.remove_small_objects(thres, min_size=500)
    thres = ndimage.morphology.binary_fill_holes(thres)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)) 
    thres = ndimage.morphology.binary_closing(thres,ker)
    
    contours,_ = cv2.findContours(thres.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    peri=d_mips.shape[0]
    num=0
    for cnt in contours:
        _,y,_,_ = cv2.boundingRect(cnt)
        #print(y)
        if y<peri:
            peri = y
            index = num
        num = num+1
    #print(index)
    thres = np.zeros((d_mips.shape[0],d_mips.shape[0]),dtype = np.uint8)
    thres = cv2.drawContours(thres, contours[index], -1, 255, 1)
    thres=thres>0
    thres = ndimage.morphology.binary_fill_holes(thres)
    #plt.figure(figsize = (10,10))
    #plt.imshow(thres,'gray')
    array = np.argwhere(thres>0)
    row = array[:,0]
    col = array[:,1]
    row1 = int(np.amin(row))
    row2 = int(np.amax(row))
    col1 = round(max(np.amin(col)-30,0))
    col2 = round(min(np.amax(col)+30,thres.shape[1]))
    start = round(max(start-70,0))
    #print(row1,row2,col1,col2)
    newscan = copy.copy(scan)
    newscan = newscan[start:end,row1:row2,col1:col2]
    half = round(newscan.shape[2]/2)
    newleftscan = newscan[:,:,half:]
    newrightscan = newscan[:,:,:half]

    return newleftscan,newrightscan, start,end,row1,row2,col1,col2,half


# In[162]:



def prediction_canal(scan):
    sz = scan.shape
    dsc_resize_left = []
    dsc_resize_right = []
    dsc_org_left = []
    dsc_org_right = []    
    
    model_left = ResUnetPlusPlus(144,176 , 176)
    model_left = model_left.build_model()
    # model = MultiResUnet3D(144, 176, 176,1)
    # model  = Attention_ResUNet ()
    model_path_left = "left_sided_weights_v1.hdf5"
    model_left.load_weights(model_path_left)

    model_right = ResUnetPlusPlus(144,176 , 176)
    model_right = model_right.build_model()
    # model = MultiResUnet3D(144, 176, 176,1)
    # model  = Attention_ResUNet ()
    model_path_right = "right_sided_weights_v1.hdf5"
    model_right.load_weights(model_path_right)

#read scan
    counts,bins,bars = plt.hist(scan[:,:,int(scan.shape[2]/2)].flatten())
    plt.close()
    #print(counts)
    #print(bins)
    nscan = copy.copy(scan)
    sscan = copy.copy(scan)
    val = np.argmax(counts)
    if bins[-1]>3200:
        a  = "Type 2"
        #print(a)
        teeth_mask = teeth_segmentation.prediction_teeth(sscan,500,4050)
        nscan_left,nscan_right, start,end,row1,row2,col1,col2,half = local_type_2(nscan,teeth_mask)
        nscan_left = get_normalized(nscan_left, -1000, 5000)
        nscan_right = get_normalized(nscan_right, -1000, 5000)
    else:
        for i in range(len(counts)-1,-1,-1):
            if counts[i]>400:
                if bins[i+1]>2000:
                    a= "Type 3"
                    #print(a)
                    teeth_mask = teeth_segmentation.prediction_teeth(sscan,500,4050)
                    nscan_left,nscan_right, start,end,row1,row2,col1,col2,half = local_type_3(nscan,teeth_mask)
                    nscan_left = get_normalized(nscan_left, -250,1500)
                    nscan_right = get_normalized(nscan_right, -250,1500)
                else:
                    a = "Type 1"
                    #print(a)
                    teeth_mask = teeth_segmentation.prediction_teeth(sscan,250,1500)
                    nscan_left,nscan_right, start,end,row1,row2,col1,col2,half = local_type_1(nscan,teeth_mask)
                    nscan_left = get_normalized(nscan_left, -250,1500)
                    nscan_right = get_normalized(nscan_right, -250,1500)
#                         f_mips = np.amax(nscan,axis = 1)
                break
     
    #saving shapes
        
    scan_shape_left =nscan_left.shape
    scan_shape_right = nscan_right.shape
    #desired shape
    shape = (144, 176, 176)
    #resize and normalize for left
    nscan_left = zero_mean(nscan_left)
    nscan_left_resize = res_scan(nscan_left, shape)

    #resize and normalize for right
    nscan_right = zero_mean(nscan_right)
    nscan_right_resize = res_scan(nscan_right, shape)
    #model loading

    ## predict model for left only
    scan_left = np.expand_dims(nscan_left_resize, axis=0)
    scan_left = np.expand_dims(scan_left, axis=-1)
    pred_mask_left = model_left.predict(scan_left)
    pred_mask_left = np.where(pred_mask_left>0.01, 1, 0)
    pred_mask_left = np.squeeze(pred_mask_left)
    
    ##predict for right canal only
    scan_right = np.expand_dims(nscan_right_resize, axis=0)
    scan_right = np.expand_dims(scan_right, axis=-1)
    pred_mask_right = model_right.predict(scan_right)
    pred_mask_right = np.where(pred_mask_right>0.01, 1, 0)
    pred_mask_right = np.squeeze(pred_mask_right)  
    
    #original size  for left
    pred_mask_left_org_size = res_mask(pred_mask_left, scan_shape_left )
    pred_mask_right_org_size = res_mask(pred_mask_right, scan_shape_right )
    
#combine left and right masks
    mask_combine_resize = np.zeros(sz)
    mask_combined = np.dstack((pred_mask_right_org_size,pred_mask_left_org_size))
    mask_combine_resize[start:end, row1:row2,  col1:col2] = mask_combined
    print("Canal mask generated")
    
    return mask_combine_resize





# In[ ]:




