#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

model_path_region = "teeth_weights.hdf5"
model_path_teeth = "close_teeth.hdf5"

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# from tensorflow import keras
import tensorflow as tf
tf.random.set_seed(1234)

from tensorflow.keras.layers import Conv3DTranspose as Deconvolution3D


from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv3D, ZeroPadding3D, UpSampling3D, Dense, concatenate, Conv3DTranspose, Cropping3D, PReLU
from tensorflow.keras.layers import MaxPooling3D, GlobalAveragePooling3D, AvgPool3D
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input
from tensorflow.keras.models import load_model


import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
tf.compat.v1.disable_eager_execution()
from sys import getsizeof

import cv2
import pickle
from sklearn.utils import shuffle
from scipy import ndimage
from sklearn import preprocessing

import json
import SimpleITK as sitk
from glob import glob 
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io import savemat
from skimage import morphology

from skimage import measure, feature
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imutils
from PIL import Image
import copy

# In[2]:

import tensorflow.keras.backend as K
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg +
                                  (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)

def dice_coefficient(predicted, target):
    smooth = 1
    product = np.multiply(predicted, target)
    intersection = np.sum(product)
    coefficient = (2 * intersection + smooth) / (np.sum(predicted) + np.sum(target) + smooth)
    return coefficient




def resize_mask(nmask, h, w, d):
    current_depth = 128
    current_width = 128
    current_height = 128
    desired_depth = d
    desired_width = w
    desired_height = h
    depth_factor = desired_depth/current_depth
    width_factor = desired_width/current_width
    height_factor = desired_height/current_height
    nmask = ndimage.zoom(nmask, (depth_factor, height_factor, width_factor), order=0, mode= 'nearest')
    return nmask


def get_normalized(scan,mn,mx):
    np.clip(scan, mn, mx, out=scan)
    d = mx - mn
    scan = (scan-mn)/d
    return scan

def dynamic_windowing(scan):
    counts,bins,bars = plt.hist(scan.flatten())
    plt.close()
    if bins[-1]>3200:
        a  = 2
#         print("Type 2")
        scan = get_normalized(scan, -1000, 4500)
    else:
        counts = sorted(counts)
#         c1 = counts[-1]/counts[-2]

        if counts[0]<7000:
        
#             print('Type 1')
            a=1
            scan = get_normalized(scan,-250,1800)
        else:
            a=3
#             print('Type 3')
            scan = get_normalized(scan,-250,3000)
    return scan

def res_scan(nscan):
    desired_depth = 128
    desired_width = 128
    desired_height = 128
    current_depth = nscan.shape[0]
    current_width = nscan.shape[2]
    current_height = nscan.shape[1]
    depth_factor = desired_depth/current_depth
    width_factor = desired_width/current_width
    height_factor = desired_height/current_height
    nscan = ndimage.zoom(nscan, (depth_factor, height_factor, width_factor), order=1)
    minimum = np.amin(nscan)
    maximum = np.amax(nscan)
    d = maximum - minimum
    nscan = (nscan-minimum)/d
    return nscan

def res_mask(nmask):
    desired_depth = 208
    desired_width = 240
    desired_height = 240
    current_depth = nmask.shape[0]
    current_width = nmask.shape[2]
    current_height = nmask.shape[1]
    depth_factor = desired_depth/current_depth
    width_factor = desired_width/current_width
    height_factor = desired_height/current_height
    nmask = ndimage.zoom(nmask, (depth_factor, height_factor, width_factor), order=1, mode = 'nearest')
    return nmask

def get_points(im):
    points = np.argwhere(im>0)
    d1 = int(max(np.amin(points[:,0]) -20,0))
    d2 = int(min(np.amax(points[:,0]) +20,im.shape[0]))
    r1 = int(max(np.amin(points[:,1]) -20,0))
    r2 = int(min(np.amax(points[:,1]) +20,im.shape[1]))
    c1 = int(max(np.amin(points[:,2]) -20,0))
    c2 = int(min(np.amax(points[:,2]) +20,im.shape[2]))
    if d1<0:
        d1=0
    elif d2>im.shape[0]:
        d2=im.shape[0]
    elif r1<0:
        r1=0
    elif r2>im.shape[1]:
        r2=im.shape[1]
    elif c1<0:
        c1=0
    elif c2>im.shape[2]:
        c2=im.shape[2]
    return d1, d2, r1, r2, c1, c2

# In[136]:

def load_the_model():
    global model_path_region
    dependencies = {
        'dice_coef': dice_coef,
        'dice_coef_loss': dice_coef_loss,
        'tversky': tversky
    }
    model = load_model(model_path_region, custom_objects=dependencies, compile=False)
    lr = 1e-4
    optimizer = Nadam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=[dice_coef_loss], metrics=[dice_coef, tversky])
    return model

def load_region_model():
    global model_path_teeth
    dependencies = {
        'dice_coef': dice_coef,
        'dice_coef_loss': dice_coef_loss,
        'tversky': tversky}
    model = load_model(model_path_teeth, custom_objects=dependencies, compile=False)
    lr = 1e-4
    optimizer = Nadam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=[dice_coef_loss], metrics=[dice_coef, tversky])
    return model


def resize_region_mask(nmask, h, w, d):
    current_depth = 208
    current_width = 240
    current_height = 240
    desired_depth = d
    desired_width = w
    desired_height = h
    depth_factor = desired_depth/current_depth
    width_factor = desired_width/current_width
    height_factor = desired_height/current_height
    nmask = ndimage.zoom(nmask, (depth_factor, height_factor, width_factor), order=0, mode= 'nearest')
    return nmask

def res_region_scan(nscan):
    desired_depth = 208
    desired_width = 240
    desired_height = 240
    current_depth = nscan.shape[0]
    current_width = nscan.shape[2]
    current_height = nscan.shape[1]
    depth_factor = desired_depth/current_depth
    width_factor = desired_width/current_width
    height_factor = desired_height/current_height
    nscan = ndimage.zoom(nscan, (depth_factor, height_factor, width_factor), order=1)
    minimum = np.amin(nscan)
    maximum = np.amax(nscan)
    d = maximum - minimum
    nscan = (nscan-minimum)/d
    return nscan

def get_region_prediction(scan):
    full_scan = copy.copy(scan)
    model2 = load_the_model()
    d= full_scan.shape[0]
    w= full_scan.shape[2]
    h= full_scan.shape[1]
    full_scan=dynamic_windowing(full_scan)
    full_scan= res_scan(full_scan)
    pp = np.empty(shape = (1, 128, 128, 128,1), dtype=np.float32)
    pp[0,:,:,:,0]=full_scan
    pred=model2.predict(pp)
    pred=pred[0,:,:,:,0] 
    pred=resize_mask(pred, h, w, d)
    pred[pred>=.002]=1
    pred[pred<.002]=0
    #pred=pred[0,:,:,:,0] 
    pred = ndimage.binary_fill_holes(pred, structure=np.ones((3,3,3))).astype(np.float32)
    #pred=resize_mask(pred, h, w, d)
    K.clear_session()
    return pred

def get_prediction(scan):
    s = copy.copy(scan)
    p=get_region_prediction(s)
    model = load_region_model()
    kernel=np.ones((3,3,3))
    morph=ndimage.morphology.binary_closing(p, structure= kernel, iterations=5)
    cleaned = morphology.remove_small_objects(morph, min_size=250, connectivity=2)
    po=get_points(cleaned)
    x=scan[po[0]:po[1], po[2]:po[3], po[4]:po[5]]
    d= x.shape[0]
    w= x.shape[2]
    h= x.shape[1]
    x=dynamic_windowing(x)
    x=res_region_scan(x)
    pred = np.empty(shape = (1, 208, 240, 240,1), dtype=np.float32)
    pred[0,:,:,:,0]=x
    pred=model.predict(pred)
    pred=pred[0,:,:,:,0] 
    pred=resize_region_mask(pred, h, w, d)
    pred[pred>=.002]=1
    pred[pred<.002]=0
    pred = pred>0
    pred = morphology.remove_small_objects(pred, min_size=250, connectivity=2)
    end_mask=np.zeros([scan.shape[0],scan.shape[1],scan.shape[2]])
    end_mask[po[0]:po[1], po[2]:po[3], po[4]:po[5]] = pred
    K.clear_session()
    return end_mask

