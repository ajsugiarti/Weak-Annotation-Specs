# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:03:30 2024

@author: annisa.sugiarti

Combine histogram-based thresholding using wavelet dneoising with thresholding from Arnold

"""

import os
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

def makefloat(img):
    imgnew = img.astype(float)
    imgnew/=255
    return imgnew

def makeuint8(img):
    img*=255
    imgnew = img.astype(np.uint8)
    return imgnew


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
                c[labeled_array==i+1] = colour
                
        filtered_rgb = cv2.medianBlur(filtered_rgb, median_kernel)
        # plt.figure(); plt.imshow(filtered_rgb)
        
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
    data = 'train'
    filepath = '../EndoSRR-master/EndoRR_dataset/%s'%data
    impath = '%s/image'%filepath
    maskpath = '%s/mask'%filepath
    # kmaskpath = '%s/kmeans_mask_with_dilation_1'%filepath  -->to plot together with kmeans mask
    
    experiments = {
        1:{"experiment_name":"Saint-Pierre + Arnold module 1",
           "mask_fn":specs_mask_arnold,
           "kwargs":  {
                       'wavelet_levels' : 2,
                       'module2' : False,
                       'dilation_iter' : 2,
                       'median_kernel' : 13,
                       'lower_thres_ratio' : 0.85,
                       'contrast_ratio': 2},
            },
        4:{"experiment_name":"Saint-Pierre + Arnold module 1 & 2",
           "mask_fn":specs_mask_arnold,
           "kwargs":  {
                       'wavelet_levels' : 2,
                       'module2' : True,
                       'dilation_iter' : 2,
                       'median_kernel' : 5,
                       'lower_thres_ratio' : 0.85,
                       'contrast_ratio': 2},
            },
        }
    
    
    imfiles = [os.path.join(impath,f) for f in os.listdir(impath)]
    maskfiles = [os.path.join(maskpath,f) for f in os.listdir(maskpath)]
    # kmaskfiles = [os.path.join(kmaskpath,f) for f in os.listdir(kmaskpath)]
    

 
    
    results = defaultdict()
    
    
    for j,exp in experiments.items():
        print(exp['experiment_name'])
        
        savepath = os.path.join(filepath,exp['experiment_name'])
        if not os.path.isdir(savepath):
            os.mkdir(savepath)  
            
        mask_fn = exp['mask_fn']
        kwargs = exp['kwargs']
        
        #medical torch dice
        dicescores = 0.0
        
        # weighted fmeasure
        metric_WFM = sod_metric.WeightedFmeasure()
        
        #medical torch iou
        iou = 0.0
        
        num = len(imfiles)
        for i in range(num):
            img = cv2.imread(imfiles[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(maskfiles[i],0)
            mask = mask.astype(float)
            

            wmask = mask_fn(img, **kwargs)
            
            # print('grayscale = %.2f to %.2f'%(kmask.min(),kmask.max()))
            dicescores += mt_metrics.dice_score(wmask/255,mask/255)
            iou += mt_metrics.intersection_over_union(wmask/255, mask/255)
            
            # prepare for metrics from endossr
            metric_WFM.step(pred=wmask, gt=mask)
            
            wmask_u8 = wmask.astype(np.uint8)
            name = os.path.splitext(os.path.split(imfiles[i])[-1])[0]
            newname = os.path.join(savepath,name + '.png')
            cv2.imwrite(newname,wmask_u8)
            
                
        dicescores_avg = dicescores / num
        iou_avg = iou / num
        wfm = metric_WFM.get_results()["wfm"]
        print(kwargs)
        print('dicescores = %f'%dicescores_avg)
        print('Weighted Fscore = %f\n'%wfm)

    
        result = {
            'experiment_name':exp['experiment_name'],
            'kwargs':kwargs,
            'dicescores':dicescores_avg,
            'intersection_of_union':iou_avg,
            'weighted_f':wfm}
        
        results[j]=result
    
    results['Source_code']=os.path.join(os.getcwd(),'wavelet_analysis.py')
    jsonfile = '%s/all_annot_methods_scores_%s_data_FINAL.json'%(savepath,data)
    with open(jsonfile, "w") as outfile: 
        json.dump(results, outfile, indent = 2)
  

if __name__ == '__main__':
    main()     