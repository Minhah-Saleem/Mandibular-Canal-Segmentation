local_path = "local_canal.h5"
left_path = "left_cana2.h5"
right_path = "right_canal.h5"

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from numpy.random import seed
seed(42)
import tensorflow as tf
tf.random.set_seed(42)
import os
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import load
import glob
import pickle
from scipy import ndimage
import json
import SimpleITK as sitk
import glob 
from tqdm import tqdm

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage

import gc

def find_gap(img,mode):
    if (mode=='axial'):
        for i in range (img.shape[0]):
            if img[i,:,:].any() != False:
                x=i
                break
        for i in range (img.shape[0]-1,0,-1):
            if img[i,:,:].any() == True:
                y=i
                break
        img_new=img[x:y,:,:]
        mip=np.amax(img_new,axis=1)
        gap=[]
        for i in range(mip.shape[0]):
            count=0
            for j in range (mip.shape[1]):
                if mip[i,j]==False:
                    count=count+1
            if(count==img.shape[2]):
                gap.append(i+x)
        
        if gap:    
            return gap[0],gap[-1]
        else: 
            return None, None
    if (mode=='coronal'):
        for i in range (img.shape[1]):
            if img[:,i,:].any() != False:
                x=i
                break
        for i in range (img.shape[1]-1,0,-1):
            if img[:,i,:].any() == True:
                y=i
                break   
        img_new=img[:,x:y,:]
        mip=np.amax(img_new,axis=2)
        gap=[]
        for i in range(mip.shape[1]):
            count=0
            for j in range (mip.shape[0]):
                if mip[j,i]==False:
                    count=count+1
            if(count==mip.shape[0]):
                gap.append(i+x)
        
        if gap:    
            return gap[0],gap[-1]
        else: 
            return None, None
        

    return f_msk
def get_full_msk(msk1,msk2,stp,mode):
    d1 = ndimage.distance_transform_edt(msk1) - ndimage.distance_transform_edt(~msk1);   # dt is the distance transform
    d2 = ndimage.distance_transform_edt(msk2) - ndimage.distance_transform_edt(~msk2);   # ~ is the logical negation
    a = round(2/stp,2)    
    j = 1
    if (mode=='axial'):
        f_msk=np.ones((stp,msk1.shape[0], msk1.shape[1] ),dtype=bool)
        for i in np.arange(a,a*(stp-1),a):
            msk = ((2-i)*d1+i*d2) >0
            f_msk[j,:,:] = msk
            j = j+1
        f_msk[j,:,:] = msk2
        f_msk[0,:,:] = msk1
        return f_msk
    if (mode=='coronal'):
        f_msk=np.ones((msk1.shape[0], stp, msk1.shape[1] ),dtype=bool)
        for i in np.arange(a,a*(stp-1),a):
            msk = ((2-i)*d1+i*d2) >0
            f_msk[:,j,:] = msk
            j = j+1
        f_msk[:,j,:] = msk2
        f_msk[:,0,:] = msk1
        return f_msk
        

    
def get_omask(clean,mode):
    index1,index2=find_gap(clean,mode)    
    if index1 and index2: 
        x=index2-index1
        if x==0 or x>5:
            return clean,0
        else:
            #print(mode)
            m1=index1-2
            m2=index2+2
            stp=m2-m1+1
            final=clean
            if (mode=='axial'):
                msk1= clean[m1,:,:]
                msk2= clean[m2,:,:]  
            if (mode=='coronal'):
                msk1= clean[:,m1,:]
                msk2= clean[:,m2,:]   
            kernel = np.ones((4,4,4), np.uint8)  
            omask=get_full_msk(msk1,msk2,stp,mode)
            omask=morphology.dilation(omask, kernel)
            if (mode=='axial'):
                final[index1:m2,:,:]=omask[index1-m1:-1,:,:]
            if (mode=='coronal'):
                final[:,index1:m2,:]=omask[:,index1-m1:-1,:]
                
                
            return final,1
    else:
        return clean,0
    

    
def post_processing(mask,mode,pmode):
    a=mask.shape[0]
    b=mask.shape[1]
    c=mask.shape[2]
    if pmode == 'full':
        half_c=int(c/2)
        mask=mask>0
        clean= morphology.remove_small_objects(mask, min_size=15,connectivity=1)
        l_clean= clean[:,:,:half_c]
        r_clean= clean[:,:,half_c:]
        l_omask,l_check=get_omask(l_clean,mode)
        r_omask,r_check=get_omask(r_clean,mode)


        if l_check==0 and r_check==0:
            return clean, "clean" 
        else:
            f_omask=np.ones((a,b,c),dtype=bool)
            f_omask[:,:,:half_c]=l_omask
            f_omask[:,:,half_c:]=r_omask
            return f_omask, "filled"
    else:
        mask=mask>0
        clean= morphology.remove_small_objects(mask, min_size=200,connectivity=1)
        omask,check = get_omask(clean,mode)
        if check==0:
            return clean, "clean" 

        else:
            return omask,"filled"

    

def postprocessing_MCS(segmask,pmode): #the function takes the segemented mask as input
    mode='axial'
    omask,string=post_processing(segmask,mode,pmode)

    while string == "filled":
        omask,string =post_processing(omask,mode,pmode)
    
    mode='coronal'
    fmask,string=post_processing(omask,mode,pmode)
    while string == "filled":
        fmask,string =post_processing(fmask,mode,pmode)
        
    fmask = np.where(fmask==True,1,0)  
    return fmask
 
#the function postprocessing_MCS(segmask) is the function which has to be called and it returns the post processed mask

def get_normalized(scan,mn,mx):
    mn = max(mn,np.amin(scan))
    mx = min(mx,np.amax(scan))
    np.clip(scan, mn, mx, out=scan)
    d = mx - mn
    scan = (scan-mn)/d
    scan = scan.astype(np.float64)
    return scan
def zero_mean(scan):
    mean_ = np.mean(scan)
    std = np.std(scan)
    scan  = (scan - mean_) / std
    return scan
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
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    ins = float(itkimage.GetMetaData('0020|0013'))
    return numpyImage, ins

def res_scan(nscan, shape):
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

def dynamic_windowing_local(scan):
    counts,bins,bars = plt.hist(scan[:,:,int(scan.shape[2]/2)].flatten())
    plt.close()
    
    val = np.argmax(counts)
    if bins[-1]>3200:
        a  = "Type 2"
        scan = get_normalized(scan, -1000, 5000)
    else:
        for i in range(len(counts)-1,-1,-1):
            if counts[i]>400:
                if bins[i+1]>2000:
                    a= "Type 3"
                    scan = get_normalized(scan, -250,1500)
                else:
                    a = "Type 1"
                    scan = get_normalized(scan, -250,1500)
                break
    print(a)
    return scan


def dynamic_windowing(scan):
    counts,bins,bars = plt.hist(scan.flatten())
    plt.close()
    if bins[-1]>3200:
        a  = "Type 2"
        scan = get_normalized(scan, -1000, 5000)
    else:
        counts = sorted(counts)
        c1 = counts[-1]/counts[-2]
        if c1>3:
            a  = "Type 3"
            scan = get_normalized(scan,-250,1500)
        else:
            a  = "Type 1"
            scan = get_normalized(scan,-250,scan.max())
    return scan



#import tensorflow as tf
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

#class ResUnetPlusPlus:
#     def __init__(self, input_size_1,input_size_2, input_size_3):
#         self.input_size_1 = input_size_1
#         self.input_size_2 = input_size_2
#         self.input_size_3 = input_size_3
#         ourmodel = self.build_model()
#         return ourmodel

def ResUnetPlusPlus(input_size_1,input_size_2, input_size_3):
    n_filters = [8, 16, 32, 64, 128]
    inputs = Input((input_size_1, input_size_2,input_size_3, 1))

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


import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K

activation = 'sigmoid'
lpatch_size_x = 208#128#256  #176#208 #224 #128 #144
lpatch_size_y = 208#128#256  #176#208 #272 #128 #176
lpatch_size_z = 208#128#256  #176#208 #208 #128 #176
n_classes = 1
channels=1
LR = 0.0001
optim = tf.keras.optimizers.Adam(LR)
#optim = tf.keras.optimizers.RMSprop(lr=0.001)#, rho=0.9, epsilon=1e-08, decay=0.0)


#limage_size = (lpatch_size_x,lpatch_size_y,lpatch_size_z)


patch_size_x =  176  #208#128#256  #176#208 #224 #128 #144
patch_size_y =  208  #208#128#256  #176#208 #272 #128 #176
patch_size_z =  160  #208#128#256  #176#208 #208 #128 #176
image_size = (patch_size_x,patch_size_y,patch_size_z)
#optim = tf.keras.optimizers.RMSprop(lr=0.001)#, rho=0.9, epsilon=1e-08, decay=0.0)

import copy

def preprocess(origscan):
    pmode = 'full'
    model_local = ResUnetPlusPlus(lpatch_size_x, lpatch_size_y, lpatch_size_z)
    model_local.load_weights(local_path)

    scan_size = origscan.shape
    scan = copy.copy(origscan)
    scan = res_scan(scan,(lpatch_size_x,lpatch_size_y,lpatch_size_z))
    scan = dynamic_windowing_local(scan)
    scan = zero_mean(scan)
    scan = scan.astype(np.float64)
    mn = np.amin(scan)
    mx = np.amax(scan)
    d = mx - mn
    scan = (scan-mn)/d
    scan = np.expand_dims(scan, axis=3)
    scan = np.expand_dims(scan, axis=0)
    msk = model_local.predict(scan)
    gc.collect()
    msk = msk[0,:,:,:,0]
    msk = msk>0.5
    msk = postprocessing_MCS(msk,pmode)
    msk = res_mask(msk,scan_size)
    points = np.argwhere(msk>0)
    d1 = int(max(np.amin(points[:,0]) -20,0))
    d2 = int(min(np.amax(points[:,0]) +20,msk.shape[0]))
    r1 = int(max(np.amin(points[:,1]) -20,0))
    r2 = int(min(np.amax(points[:,1]) +20,msk.shape[1]))
    c1 = np.amin(points[:,2]) 
    c2 = np.amax(points[:,2])
    if c1>=msk.shape[2]/2:
        c1 = msk.shape[2]-c2
    if c2<=msk.shape[2]/2:
        c2 = msk.shape[2] - c1
    c1 = int(max(c1 -20,0))
    c2 = int(min(c2+20,msk.shape[2]))
    localscan = origscan[d1:d2,r1:r2,c1:c2]
    return localscan,d1,d2,r1,r2,c1,c2

def prediction_canal(scan):
    origscan = copy.copy(scan)
    keras.backend.clear_session()
    localscan,d1,d2,r1,r2,c1,c2 = preprocess(origscan)
    div = int(localscan.shape[2]/2)
    leftscan = localscan[:,:,div:]
    rightscan = localscan[:,:,:div]
    leftdim = leftscan.shape
    rightdim = rightscan.shape
    
    leftscan = res_scan(leftscan,image_size)
    leftscan = dynamic_windowing(leftscan)
    leftscan = zero_mean(leftscan)
    leftscan = leftscan.astype(np.float64)
    leftscan = np.expand_dims(leftscan, axis=3)
    leftscan = np.expand_dims(leftscan, axis=0)
    
    rightscan = res_scan(rightscan,image_size)
    rightscan = dynamic_windowing(rightscan)
    rightscan = zero_mean(rightscan)
    rightscan = rightscan.astype(np.float64)
    rightscan = np.expand_dims(rightscan, axis=3)
    rightscan = np.expand_dims(rightscan, axis=0)
    
    keras.backend.clear_session()
    model_left = ResUnetPlusPlus(patch_size_x, patch_size_y, patch_size_z)
    model_left.load_weights(left_path)
    leftmsk = model_left.predict(leftscan)
    
    keras.backend.clear_session()
    model_right = ResUnetPlusPlus(patch_size_x, patch_size_y, patch_size_z)
    model_right.load_weights(right_path)
    rightmsk = model_right.predict(rightscan)
    
    keras.backend.clear_session()
    
    leftmsk = leftmsk[0,:,:,:,0]
    rightmsk = rightmsk[0,:,:,:,0]
    leftmsk= leftmsk>0.25
    rightmsk = rightmsk>0.25
    pmode = 'half'
    leftmsk = postprocessing_MCS(leftmsk,pmode)
    rightmsk = postprocessing_MCS(rightmsk,pmode)
    leftmsk = res_mask(leftmsk,leftdim)
    rightmsk = res_mask(rightmsk,rightdim)
    localmask = np.zeros_like(localscan)
    
    localmask[:,:,div:] = leftmsk
    localmask[:,:,:div] = rightmsk
    
    mask = np.zeros_like(scan)
    mask[d1:d2,r1:r2,c1:c2] = localmask
    
    return mask