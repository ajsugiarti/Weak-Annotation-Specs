# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:39:37 2024

@author: annisa.sugiarti
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import medicaltorch.metrics as mt_metrics
import utils
import torchvision.transforms.functional as Ftv

def enhance_clahe(img,teta=2.0):
    """
    Enhance only the luminance channel of the image
    """
    clahe = cv2.createCLAHE(clipLimit=teta, tileGridSize=(8,8))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    cl1 = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return cl1,lab

def kmeans_cluster(img,k,threshold=False,inverse=False):
    """
    If threshold true, res2 will be mask.
    Only works for 1 channel images (2D).
    """

    if img.ndim == 3:
        # print('3 dimensi')
        x,y,c = img.shape
        vector = img.reshape((-1,c))
    elif img.ndim == 2:
        # print('2 dimensi')
        vector = img.flatten()
    
    # print(vector.shape)
    vector = np.float32(vector)
    
    max_iter = 10
    epsilon = 0.01
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    
    attempts = 10
    flags = cv2.KMEANS_PP_CENTERS
    
    init_labels = None
    
    compactness,labels,centers = cv2.kmeans(vector, k, init_labels, criteria, attempts, flags)
    # print(centers.shape)
    
    centers=np.uint8(centers)
    
    res = centers[labels.flatten()]
       
    res2 = res.reshape((img.shape))
    
    if threshold:
        if img.ndim != 2:
            raise Exception("Thresholding does not work for multi-channel images.")
        
        res3 = res2.copy()
        if inverse:
            min_center = centers.min()
            res2 = (res3 <= min_center)*1
        else:
            max_center = centers.max()
            res2 = (res3 >= max_center)*1
    
    return res2
    
   
def tiles(img,tilesize):
    """
    img is in the shape of (HxWxC) or (HxW)
    """
    shape = img.shape
        
    if np.mod(shape[0],tilesize) !=0 or np.mod(shape[1],tilesize) != 0:
        raise ValueError("img shape must be divisible by tilesize")
        
    if img.ndim == 3:
        tiled = img.reshape(shape[0]//tilesize, tilesize, shape[1]//tilesize, tilesize, shape[2])
    elif img.ndim == 2:
        tiled = img.reshape(shape[0]//tilesize, tilesize, shape[1]//tilesize, tilesize)
        
    tiled = tiled.swapaxes(1,2)
    
    return tiled

def plot_tiles(tiled,vmax=255):
    
    M,N = tiled.shape[:2]
    fig,ax = plt.subplots(M,N)
    for i in range(M):
        for j in range(N):
            
            ax[i,j].imshow(tiled[i,j,...],vmin=0,vmax=vmax,cmap='gray')
            ax[i,j].axis('off')

def merge_tiles(tiled,shape=(248,248)):
    if tiled.ndim - 2 != len(shape):
        raise ValueError("The dimension of the tile image does not match the new shape")
    
    swap_tile = tiled.swapaxes(1,2)
    back_img = np.reshape(swap_tile.ravel(), shape)
    
    return back_img

def local_kmeans(img,mask,k,tilesize=31,k_thresh_min=3,k_thresh_max=None,inverse=False):
    """
    k_thresh is the threshold to decide if a tile will be segmented again using kmeans_cluster
    This decision must be improved.
    """
    if not k_thresh_max:
        k_thresh_max = tilesize**2
    
    tile_im = tiles(img.copy(),tilesize)
    tile_mask = tiles(mask.copy(),tilesize)
    new_mask = np.zeros_like(tile_mask)
    
    M,N = tile_im.shape[:2]
    
    for i in range(M):
        for j in range(N):
            tilesum = np.sum(tile_mask[i,j,...])
            if tilesum > k_thresh_min:
                if tilesum < 0.2*tilesize**2:
                    tile_mask[i,j,...]=kmeans_cluster(tile_im[i,j,...], k,threshold=True,inverse=inverse)                  
    
    # plot_tiles(tile_im)
    # plot_tiles(tile_mask,vmax=1)
    newmask = merge_tiles(tile_mask,mask.shape)
    
    return newmask

def kmeans_mask(img,k_global=2,k_local=2,k_local_min=2,k_local_max=None,tilesize=32,blue_channel=False,dilation=1):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
    # img = enhance_clahe(img)[0]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    value_mask = kmeans_cluster(hsv[:,:,2],k_global,threshold=True)
    sat_mask = kmeans_cluster(hsv[:,:,1],k_global,threshold=True,inverse=True)
    blue_mask = kmeans_cluster(img[:,:,2],k_global,threshold=True)
    
    mask = value_mask*sat_mask*blue_mask
    plt.figure()
    plt.imshow(mask)
    
    if k_local is not None:
        vnewmask = local_kmeans(hsv[:,:,2], mask, k_local, tilesize=tilesize,
                                k_thresh_min=k_local_min,k_thresh_max=k_local_max)
        snewmask = local_kmeans(hsv[:,:,1], mask, k_local,tilesize=tilesize,
                                k_thresh_min=k_local_min,k_thresh_max=k_local_max,inverse=True)
        
        newmask = vnewmask*snewmask
    else:
        newmask = mask
    
    if blue_channel:
        bnewmask = local_kmeans(img[:,:,2], mask, k_local, tilesize=tilesize,
                                k_thresh_min=k_local_min,k_thresh_max=k_local_max)
        newmask*=bnewmask
    
    if dilation is not None:
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        newmask = cv2.dilate((newmask*255).astype(np.uint8), element, iterations = dilation)
    
    return newmask

def plot_compare(img,kmask,mask=None,name=None):
    if mask is not None:
        if mask.dtype != 'uint8':
            mask*=255
            mask = mask.astype(np.uint8)
    if kmask.dtype != 'uint8':
        kmask*=255
        kmask = kmask.astype(np.uint8)
        
    if mask is not None:
        ims = [img,mask,kmask]
        titles = ['image','original_mask','mask_using_kmeans']
    else:
        ims = [img,kmask]
        titles = ['image','mask_using_kmeans']
        
    fig,ax = plt.subplots(1,len(ims),figsize=(12,5))
    for i in range(len(ims)):
        ax[i].imshow(ims[i])
        ax[i].set_title(titles[i])
        
    return fig

    

def main():
    plt.close('all')
    filepath = 'EndoSRR-master/EndoRR_dataset/test'
    impath = '%s/image'%filepath
    maskpath = '%s/mask'%filepath
    tilesize = 32
    k_local_max = 0.2*tilesize**2
    
    kwargs = {
    'tilesize' : tilesize,
    'k_global' : 9,
    'k_local' : 5,
    'k_local_min' : 1,
    'k_local_max' : k_local_max,
    'blue_channel': False,
    'dilation':1}
    
    imfiles = [os.path.join(impath,f) for f in os.listdir(impath)]
    maskfiles = [os.path.join(maskpath,f) for f in os.listdir(maskpath)]
    
    savepath = '%s/kmeans_mask'%filepath
    if not os.path.isdir(savepath):
        os.mkdir(savepath)   
    
    #medical torch dice
    dicescores = 0.0
    
    num = len(imfiles)
    for i in range(num):
        print('Working with image %d'%i)
        img = cv2.imread(imfiles[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(maskfiles[i],0)
        mask = mask.astype(float)
        
        kmask = kmeans_mask(img, **kwargs)
        
        # print('grayscale = %.2f to %.2f'%(kmask.min(),kmask.max()))
        dicescores += mt_metrics.dice_score(kmask/255,mask/255)
        
        # prepare for metrics from endossr
        kmask_fl = kmask
        mask_fl = mask
        
        kmask_u8 = kmask_fl.astype(np.uint8)
        newname = os.path.join(savepath,os.path.splitext(os.path.split(imfiles[i])[-1])[0] + '.png')
        cv2.imwrite(newname,kmask_u8)
        
            
    dicescores_avg = dicescores / num
    print(kwargs)
    print('dicescores = %f'%dicescores_avg)

    kwargs['dicescores'] = dicescores_avg

    jsonfile = '%s/log.json'%savepath
    with open(jsonfile, "w") as outfile: 
        json.dump(kwargs, outfile, indent = 2)
    print(kwargs)    


    
if __name__=='__main__':
    main()

        
    
    

