# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:22:58 2024

@author: annisa.sugiarti

This file was created to inspect the difference of interpolation method and data type of mask
towards the metrics and the performance.
Calculate m

"""

import os
import json
import numpy as np
import cv2
import pandas as pd
import torch
import torchvision.transforms.functional as Ftv
import albumentations as A
from torch.utils.data import DataLoader
import pandas as pd

from collections import defaultdict

import medicaltorch.metrics as mt_metrics

import models as da_models

from main_unet import inputPath, CustomDataset,threshold_predictions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_unet():
    drop_rate = 0.5
    bn_momentum = 0.1
    model = da_models.Unet(drop_rate=drop_rate,
                           bn_momentum=bn_momentum)

    return model.to(device)

def inference(modelfile,model='unet',data='test',datatype=np.float32,
              thr=None,imsave=False,img_interpolation=1,mask_interpolation=0):
    print(model)
    if model == 'unet':
      model = create_unet()
      img_size=(512,640)
      hh,ww = img_size
    
    model.to(device)
    model.load_state_dict(torch.load(modelfile, weights_only=True, map_location=device))
    model.eval()
    
    metric_fn = metric_fns = [mt_metrics.accuracy_score,mt_metrics.dice_score, mt_metrics.jaccard_score, mt_metrics.intersection_over_union]
    path = os.path.split(modelfile)[0]
    print(os.path.split(path)[-1])
    savepath = '%s/%s_inference'%(path,data)
    if not os.path.isdir(savepath):
      os.mkdir(savepath)
    result_dict = defaultdict(float)
    
    # resize = A.Compose([A.Resize(img_size[0],img_size[1],interpolation=3)])
    
    
    impath = "../EndoSRR-master/EndoRR_dataset/test/image"
    gtpath = "../EndoSRR-master/EndoRR_dataset/test/mask"
    imfiles = inputPath(impath)
    gtfiles = inputPath(gtpath)
    num_samples = len(imfiles)
    
    for imfile,gtfile in zip(imfiles,gtfiles):
    
      filename = os.path.splitext(os.path.split(imfile)[-1])[0]
      assert filename == os.path.splitext(os.path.split(gtfile)[-1])[0]
    
      img = cv2.imread(imfile)
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      gt = cv2.imread(gtfile,0)
      mask_size = gt.shape
      hhh, www = mask_size
    
      # sample = resize(image=img)
      # imgsmall = sample['image']

      imgsmall = cv2.resize(img, (ww,hh),interpolation=img_interpolation)
      imgtensor = Ftv.to_tensor(Ftv.to_pil_image(imgsmall))
      imgtensor = imgtensor.unsqueeze(0)
      imgcuda = imgtensor.to(device)
    
      with torch.no_grad():
          newmaskcuda = model(imgcuda)
      newmask = newmaskcuda.cpu().numpy()
      if thr is not None:
          newmask = threshold_predictions(newmask,thr)
      newmask*= 255
      newmask = newmask.astype(np.uint8)
      newmask = newmask.squeeze()
      
      # resizeback = A.Compose([A.Resize(mask_size[0],mask_size[1],interpolation=0)]) #nearest neighbor for mask
      # sample2 = resizeback(image=newmask)
      # newmask = sample2['image']
      
      newmask = cv2.resize(newmask, (www,hhh), interpolation=mask_interpolation)

      
      if imsave:
          cv2.imwrite('%s/%s.png'%(savepath,filename),newmask)
    
      for metric_fn in metric_fns:
        # res = metric_fn((newmask/255).astype(np.float32), (gt/255).astype(np.float32))
        # res = metric_fn((newmask/255).astype(np.uint8), (gt/255).astype(np.uint8))
        res = metric_fn((newmask/255).astype(datatype), (gt/255).astype(datatype))
        dict_key = '{}'.format(metric_fn.__name__)
        result_dict[dict_key] += res
    
    for key, val in result_dict.items():
      val_avg = val / num_samples
      result_dict[key] = val_avg
      print('%s: %f'%(key,val_avg))
      
    return result_dict

def inference_keepsize(modelfile,model='unet',data='test',datatype=np.float32,thr=None,imsave=False,img_interpolation=0,mask_interpolation=0):
    print(model)
    if model == 'unet':
      model = create_unet()
      img_size=(512,640)
    
    model.to(device)
    model.load_state_dict(torch.load(modelfile, weights_only=True, map_location=device))
    model.eval()
    
    metric_fns = [mt_metrics.accuracy_score,mt_metrics.dice_score, mt_metrics.jaccard_score, mt_metrics.intersection_over_union]
    path = os.path.split(modelfile)[0]
    print(os.path.split(path)[-1])
    savepath = '%s/%s_inference'%(path,data)
    if not os.path.isdir(savepath):
      os.mkdir(savepath)
    result_dict = defaultdict(float)
    
    if imsave:
        mask_size=(1024,1280)
        resizeback = A.Compose([A.Resize(mask_size[0],mask_size[1],interpolation=0)])
    
    
    impath = "../EndoSRR-master/EndoRR_dataset/%s/image"%data
    gtpath = "../EndoSRR-master/EndoRR_dataset/%s/mask"%data
    imfiles = inputPath(impath)
    gtfiles = inputPath(gtpath)
    num_samples = len(imfiles)
    
    for imfile,gtfile in zip(imfiles,gtfiles):
    
      filename = os.path.splitext(os.path.split(imfile)[-1])[0]
      assert filename == os.path.splitext(os.path.split(gtfile)[-1])[0]
    
      img = cv2.imread(imfile)
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      gt = cv2.imread(gtfile,0)
    
      # sample = resize(image=img,mask=gt)
      # imgsmall = sample['image']
      # gtsmall = sample['mask']
      
      hh,ww = img_size
      imgsmall = cv2.resize(img, (ww,hh),interpolation=img_interpolation)
      gtsmall = cv2.resize(gt, (ww,hh),interpolation=mask_interpolation)
      
      imgtensor = Ftv.to_tensor(Ftv.to_pil_image(imgsmall))
      imgtensor = imgtensor.unsqueeze(0)
      imgcuda = imgtensor.to(device)
    
      with torch.no_grad():
          newmaskcuda = model(imgcuda)
      newmask = newmaskcuda.cpu().numpy()
      if thr is not None:
          newmask = threshold_predictions(newmask,thr)
      newmask*= 255
      newmask = newmask.astype(np.uint8)
      newmask = newmask.squeeze()
      
      if imsave:
          sample2 = resizeback(image=newmask)
          newmasksave = sample2['image']
          cv2.imwrite('%s/%s.png'%(savepath,filename),newmasksave)
        
      for metric_fn in metric_fns:
        res = metric_fn((newmask/255).astype(datatype), (gtsmall/255).astype(datatype))
        dict_key = '{}'.format(metric_fn.__name__)
        result_dict[dict_key] += res
    
    for key, val in result_dict.items():
      val_avg = val / num_samples
      result_dict[key] = val_avg
      print('%s: %f'%(key,val_avg))
      
    # if imsave:
    #   with open("%s/log.json"%savepath,"w") as outfile:
    #     json.dump(result_dict,outfile,indent=2)
      
    return result_dict

def inference_loader(modelfile,model='unet',data='test',thr=None):
    print(model)
    if model == 'unet':
      model = create_unet()
      img_size=(512,640)
    
    model.to(device)
    model.load_state_dict(torch.load(modelfile, weights_only=True, map_location=device))
    model.eval()
    
    metric_fn = metric_fns = [mt_metrics.accuracy_score,mt_metrics.dice_score, mt_metrics.jaccard_score, mt_metrics.intersection_over_union]
    path = os.path.split(modelfile)[0]
    print(os.path.split(path)[-1])
    savepath = '%s/%s_inference'%(path,data)
    if not os.path.isdir(savepath):
      os.mkdir(savepath)
    result_dict = defaultdict(float)
    
   
    
    impath = "../EndoSRR-master/EndoRR_dataset/test/image"
    gtpath = "../EndoSRR-master/EndoRR_dataset/test/mask"
    imfiles = inputPath(impath)
    gtfiles = inputPath(gtpath)
    num_samples = len(imfiles)
    
    target_test = CustomDataset(imfiles,gtfiles,
                                img_size = img_size,
                                augmentation = None)
    
    test_loader = DataLoader(target_test, batch_size=1,
                                                 shuffle=False, drop_last=False,
                                                )
    
    for i, batch in enumerate(test_loader):
        input_data, gt_data = batch["input"], batch["gt"]

        input_data_gpu = input_data.to(device)
        gt_data_gpu = gt_data.to(device)

        with torch.no_grad():
            model_out1 = model(input_data_gpu)

        gt_masks = gt_data_gpu.cpu().numpy().astype(np.uint8)
        gt_masks = gt_masks.squeeze(axis=1)

        preds1 = model_out1.cpu().numpy()
        if thr is not None:
            preds1 = threshold_predictions(preds1,thr=thr)
        preds1 = preds1.astype(np.uint8)
        preds1 = preds1.squeeze(axis=1)
        
        for metric_fn in metric_fns:
            for prediction, ground_truth in zip(preds1, gt_masks):
                res = metric_fn(prediction, ground_truth)
                dict_key = '{}'.format(metric_fn.__name__)
                result_dict[dict_key] += res

    for key, val in result_dict.items():
      val_avg = val / num_samples
      result_dict[key] = val_avg
      print('%s: %f'%(key,val_avg))
    
    # with open("%s/log.json"%savepath,"w") as outfile:
    #   json.dump(result_dict,outfile,indent=2)
    return result_dict
    
   

def main():
    """
    Using hyperparameter from last discussion.
    datatype = uint8
    threshold = 0.9
    mask interpolation = 0 --> as per default used by albumentation package
    Using inference_keepsize to calculate the metrics before backresize the mask to original size.

    """
    
    modelfiles = [
        'EndoSRR_mask_resized_512x640/model200.pt',
        'SPierre_mask_640x512_200epochs/model200.pt',
        'kmeans_mask_resized_512x640/model200.pt']
    for model in modelfiles[2:]:
        print(model)
        result = inference_keepsize(model,datatype=np.uint8,imsave=False,data='test',
                                    thr=0.9,img_interpolation=1,mask_interpolation=0)
        print(result)
    
if __name__ == '__main__':
    main()