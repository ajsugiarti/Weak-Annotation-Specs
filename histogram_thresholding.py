# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:03:30 2024

@author: annisa.sugiarti

Combine histogram-based thresholding using wavelet dneoising with thresholding from Arnold

"""

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
from scipy.ndimage import label, generate_binary_structure, find_objects
from skimage.restoration import denoise_wavelet
# from skimage.color import label2rgb
import medicaltorch.metrics as mt_metrics
import sod_metric

def dilate(img,ksize=2,dilation_iter=1):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize,ksize))
    newimg = cv2.dilate(img, element, iterations = dilation_iter) 
    return newimg




def find_thresh(img,levels=2):
    if img.ndim != 3:
        img = np.expand_dims(img, -1)
    
    z = img.shape[-1]
    thresholds =[]
    for i in range(z):
        c = img[...,i]
        # var = 10*(c.max()-c.min())/100
        chist,bins = np.histogram(c.flatten(),bins=256, range=(0,255))
        w = denoise_wavelet(chist,method='VisuShrink',mode='soft',wavelet_levels=levels,wavelet='sym4',rescale_sigma=True)
        
        dw = np.gradient(w)
        w2 = np.ones(w.shape)
        w2[dw <= 0] = 0
        dw2 = np.gradient(w2)
        w3 = np.ones(w2.shape)
        w3[dw2 <= 0] = 0
        
        w3reserve = w3[::-1]
        arg_last_max = len(w3reserve)-w3reserve.argmax()-1
        # arg_last_max -= var
        thresholds.append(arg_last_max)
        
    return thresholds

   

def specs_mask_arnold(rgb,wavelet_levels=2,module2=False,dilation_iter=None,median_kernel=15,lower_thres_ratio=0.85,contrast_ratio=1.0):
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    r,g,b = cv2.split(rgb)

    G95 = np.percentile(grey,95)
    ratioG = np.percentile(g,95)/G95
    ratioB = np.percentile(b,95)/G95

    
    T1 = find_thresh(grey,wavelet_levels)[0]
    T1g = T1*ratioG
    T1b = T1*ratioB
    
    mask1 = np.zeros(grey.shape)
    mask1[g>T1g]=1
    mask1[b>T1b]=1
    mask1[grey>T1]=1
    
    
    if module2:
        
        T1*=lower_thres_ratio
        print(T1)
        T1g = T1*ratioG
        T1b = T1*ratioB
        
        temp_mask = np.zeros(grey.shape)
        temp_mask[g>T1g]=1
        temp_mask[b>T1b]=1
        temp_mask[grey>T1]=1
        
        temp_mask = (temp_mask*255).astype(np.uint8)
        
        dilmask2 = dilate(temp_mask,ksize=2)
        dilmask4 = dilate(temp_mask,ksize=4)
        
        xormask = np.logical_xor(dilmask2,dilmask4)
        
        # # fill each hole
        s = generate_binary_structure(2,2)
        labeled_array, num_features = label(dilmask4, structure=s)
        regions = find_objects(labeled_array)
        
        filtered_rgb = rgb.copy()
        
        for i,region in enumerate(regions):
            xormask2 = xormask[region]
        #     region = np.zeros(temp_mask.shape)
        #     region[labeled_array==i+1] = 1
            
        #     dilmask2 = dilate(region,ksize=2)
        #     dilmask4 = dilate(region,ksize=4)
        
        #     xormask = np.logical_xor(dilmask2,dilmask4)
            
            for j in range(3):
                c = filtered_rgb[...,j]
                colour = xormask2*c[region]
                colour = (np.true_divide(colour.sum(),(colour!=0).sum())).astype(np.uint8) #mean of non zero values
                c[labeled_array==i+1] = colour  #changing value of c also changing filtered_rgb
                
        filtered_rgb = cv2.medianBlur(filtered_rgb, median_kernel) 
        # plt.figure(); plt.imshow(filtered_rgb)
               
        
        ### For presentation only
        view_line = False
        if view_line:
            rgb_line = rgb[:,1166,0].astype(np.float32)
            filtered_line = filtered_rgb[:,1166,0].astype(np.float32)
            x = np.arange(0,rgb_line.shape[0],dtype=np.float32)
            plt.figure(figsize=(4,3))
            plt.plot(x,rgb_line)
            plt.plot(x,filtered_line)
            plt.xlabel('Pixel rows')
            plt.ylabel('Grayscale')
            plt.legend(['Original Image','Median filtered'],loc='lower left')
            plt.tight_layout()
            
        
        # to obtain the contrast compensated intensity ratio 
        mean = np.mean(rgb,axis=(0,1))
        std = np.std(rgb, axis=(0,1))
        contrast = ((mean+std)/mean)**(-1)
        # contrast = 1
        I_ratio = np.max(contrast*np.float32(rgb)/(np.float32(filtered_rgb)+1e-6),axis=-1)
        # plt.figure(); plt.imshow(I_ratio)
        
        mask2 = (I_ratio>contrast_ratio)*1
        mask = (np.logical_or(mask1,mask2)*255).astype(np.uint8)
        # mask = (mask2*255).astype(np.uint8)
    else:
        mask = (mask1*255).astype(np.uint8)
    
    # mask = np.stack((mask,)*3, axis = -1)
        
    if dilation_iter is not None:
        mask = dilate(mask,dilation_iter=dilation_iter)
    # print((T1,G95,ratioG,ratioB))    
    return mask




  
def main():
    plt.close('all')
    filepath = 'EndoSRR-master/EndoRR_dataset/test'
    impath = '%s/image'%filepath

    
    experiments = {
        1:{"experiment_name":"Saint-Pierre + Arnold module 1",
           "mask_fn":specs_mask_arnold,
           "kwargs":  {
                       'wavelet_levels' : 2,
                       'module2' : False,
                       'dilation_iter' : None,
                       'median_kernel' : 13,
                       'lower_thres_ratio' : 0.28,
                       'contrast_ratio': 2.95},
            },
        }
    
    
    imfiles = [os.path.join(impath,f) for f in os.listdir(impath)]
    filenames = [os.path.splitext(f)[0] for f in os.listdir(impath)]
    
    savepath = '%s/masks/SPierre_Arnold1_mask_dilation_1x'%filepath
    if not os.path.isdir(savepath):
        os.mkdir(savepath)   
    
    results = defaultdict()
        
    for j,exp in experiments.items():
        print(exp['experiment_name'])
        mask_fn = exp['mask_fn']
        kwargs = exp['kwargs']
        
        num_start = 14
        num_end = 17#num_start+1#len(imfiles)
        for i in range(num_start,num_end):
            img = cv2.imread(imfiles[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            wmask = mask_fn(img, **kwargs)
                       
            wmask_u8 = wmask.astype(np.uint8)
            name = os.path.splitext(os.path.split(imfiles[i])[-1])[0]
            newname = os.path.join(savepath,name + '.png')
            cv2.imwrite(newname,wmask_u8)


    
        result = {
            'experiment_name':exp['experiment_name'],
            'kwargs':kwargs,}
        
        results[j]=result
    
    jsonfile = '%s/log_used.json'%savepath
    jsonfile = '%s/log_%s.json'%(savepath,filenames[i])
    with open(jsonfile, "w") as outfile: 
        json.dump(results, outfile, indent = 2)

if __name__ == '__main__':
    main()     