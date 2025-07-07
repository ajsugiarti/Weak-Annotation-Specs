# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:32:56 2024

@author: annisa.sugiarti
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

import albumentations as A

import torch
import torchvision.transforms.functional as Ftv
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import models as da_models

CUDA = True
if not torch.cuda.is_available() and CUDA == True:
    CUDA = False
    
print('using CUDA: ', CUDA)

file1 = '7_5_Right_Image.png'
file2 = '4_2_Left_Image.png'
file3 = '1_1_Left_Image.png'

# files = [file1, file2, file3]

imfile = "../EndoSRR-master/EndoRR_dataset/test/image"
gtfile = "../EndoSRR-master/EndoRR_dataset/test/mask"
samfile ="../EndoSRR-master/EndoRR_dataset/test/SAM_mask"
kmeans = "../EndoSRR-master/EndoRR_dataset/test/kmeans_mask_with_dilation_1"
newsamfile = "../EndoSRR-master/EndoRR_dataset/test/retrain_SAM_kmeans_mask"


modelfile = "kmeans_mask_resized_512x640/model200.pt"

img_size=(1024,1024)
resize = A.Compose([A.Resize(img_size[0],img_size[1],interpolation=3)])
print('Image size: ',img_size)

drop_rate = 0.5 #ctx["drop_rate"]
bn_momentum = 0.1 #ctx["bn_momentum"]
model = da_models.Unet(drop_rate=drop_rate,
                       bn_momentum=bn_momentum)

if CUDA:
    model.cuda()
    model.load_state_dict(torch.load(modelfile))
else:
    model.load_state_dict(torch.load(modelfile,map_location=torch.device('cpu')))

class CustomDataset(Dataset):
    # """
    # img_paths: directory of images
    # mask_paths: directory of mask
    # """
    def __init__(self, img_paths, mask_paths,
                 img_size=None,
                 augmentation = None):
                
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.ids = np.arange(len(self.img_paths))
        self.aug = augmentation
        self.img_size = img_size
        
        if self.img_size is not None:
            self.resize = A.Compose([A.Resize(img_size[0],img_size[1],interpolation=3)])
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx],0)
        
        if self.img_size is not None:
            sample = self.resize(image=image,mask=mask)
            image = sample['image']
            mask = sample['mask']
        
        if self.aug is not None:
            sample = self.aug(image=image,mask=mask)
            image = sample['image']
            mask = sample['mask']


        image = Ftv.to_tensor(Ftv.to_pil_image(image))
        mask = Ftv.to_tensor(Ftv.to_pil_image(mask))

        mydict = {
            'input': image,
            'gt': mask
        }
        return mydict

def inputPath(dirPath):
    img_path = sorted(
                        [
                            os.path.join(dirPath, fname)
                            for fname in os.listdir(dirPath)
                            if os.path.splitext(os.path.split(fname)[-1])[-1] == '.png'
                        ]
                        )
    return img_path

# files = os.listdir(imfile)
files = inputPath(imfile)
times = []

for file in files:
    # img = cv2.imread('%s/%s'%(imfile,file))
    img = cv2.imread(file,cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    sample = resize(image=img) 
    imgsmall = sample['image']
    imgtensor = Ftv.to_tensor(Ftv.to_pil_image(imgsmall))
    imgtensor = imgtensor.unsqueeze(0)
    if CUDA:
        imgcuda = imgtensor.cuda()

    start = time.time()
    with torch.no_grad():
        if CUDA:
            newmaskcuda = model(imgcuda)
            newmask = newmaskcuda.cpu().numpy()*255
        else:
            newmask = model(imgtensor)
            newmask = newmask.numpy()*255
            
        newmask = newmask.astype(np.uint8)
        newmask = newmask.squeeze()
    
    end = time.time()
    elapsed = end-start
    times.append(elapsed)
    
        
    # gt = cv2.imread('%s/%s'%(gtfile,file),0)
    # sam = cv2.imread('%s/%s'%(samfile,file),0)
    # newsam = cv2.imread('%s/%s'%(newsamfile,file),0)
    # km = cv2.imread('%s/%s'%(kmeans,file),0)
    
    # fig,ax = plt.subplots(2,3,figsize=(12,8))
    # ax = ax.ravel()
    # ax[0].imshow(img)
    # ax[1].imshow(gt)
    # ax[2].imshow(km)
    # ax[3].imshow(sam)
    # ax[4].imshow(newsam)
    # ax[5].imshow(newmask)
    
    # ax[0].set_title('Input image')
    # ax[1].set_title('Provided Mask')
    # ax[2].set_title('Kmeans Mask')
    # ax[3].set_title('Provided SAM model')
    # ax[4].set_title('Retrained SAM model')
    # ax[5].set_title('UNet')
                    
    
    # savedir = r"\\FS1\Docs4\annisa.sugiarti\My Documents\Codes\UNet_specular_reflection\resized_im_512x640\inference\train"
    # cv2.imwrite('%s/%s'%(savedir,file),newmask)
    
times = np.array(times)
time_mean = np.mean(times)
time_std = np.std(times)
print('Average time: {}'.format(time_mean))
print('Standard deviation time: {}'.format(time_std))
