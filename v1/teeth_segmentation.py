#!/usr/bin/env python
# coding: utf-8

# In[18]:



import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import json
import numpy as np
import matplotlib.pyplot as plt
import json
import SimpleITK as sitk
import os
from glob import glob 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import glob
from tqdm import tqdm
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy import ndimage
from scipy.ndimage import morphology
from skimage import measure, feature
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imutils
from PIL import Image
import pickle
from multiprocessing import Pool
import json
import SimpleITK as sitk
import os
from glob import glob 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import glob
from tqdm import tqdm
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy import ndimage
from scipy.ndimage import morphology
from skimage import measure, feature
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imutils
from PIL import Image
import pickle
from multiprocessing import Pool


def zero_mean(scan):
    mean_ = np.mean(scan)
    std = np.std(scan)
    scan  = (scan - mean_) / std
    return scan

# In[17]:


def res_mask(nmask,shape):
    D,W,H  = shape[0], shape[2],shape[1]
    desired_depth = D
    desired_width = W
    desired_height = H
    current_depth = nmask.shape[0]
    current_width = nmask.shape[2]
    current_height = nmask.shape[1]
    depth_factor = desired_depth/current_depth
    width_factor = desired_width/current_width
    height_factor = desired_height/current_height
    nmask = ndimage.zoom(nmask, (depth_factor, height_factor, width_factor), order=1)
    return nmask


# In[19]:




def get_normalized(scan,mn,mx):
    np.clip(scan, mn, mx, out=scan)
    d = mx - mn
    scan = (scan-mn)/d
    return scan

def res_scan(nscan,shape):
    D,W,H  = shape[0], shape[2],shape[1]
    desired_depth = D
    desired_width = W
    desired_height = H
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


# In[20]:


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) +
                                           smooth)


# In[21]:


import os
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

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
        n_filters = [4, 8, 16, 32, 64]
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


# In[24]:


def prediction_teeth(scan,w1,w2):
    
    model = ResUnetPlusPlus(192,192,192)
    model = model.build_model()
    model_path = r"teeth_segmentation_weights_v1.hdf5"
    model.load_weights(model_path)
    
    scan_shape = scan.shape
    print(w1,w2)
    nscan = get_normalized(scan,w1,w2)
    shape = (192, 192, 192)
    nscan_re = res_scan(nscan, shape)
#         nscan_re = resize_data(scan, shape)
    nscan_re = nscan_re.astype(np.float32)
    nscan_re = zero_mean(nscan_re)
    # read mask 

    # add dimension for model prediction
    nscan_re = np.expand_dims(nscan_re, axis=0)
    nscan_re = np.expand_dims(nscan_re, axis=-1)
    
    pre_mask_reo = model.predict(nscan_re)

    #remove dimensions
    pre_mask_reo = np.squeeze(pre_mask_reo) 
    pre_mask_re = np.where(pre_mask_reo>0.01, 1, 0) #0.01

    #pre_mask_re = pre_mask_re.astype(np.float32)

    # resize to original size
    pre_mask_org_size = res_mask( pre_mask_re ,scan_shape )

    
    print("Teeth Mask generated")
    
    return  pre_mask_org_size

