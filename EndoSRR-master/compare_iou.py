# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 12:59:23 2024

@author: annisa.sugiarti
"""

import os
import json
import cv2
from collections import defaultdict


import medicaltorch.metrics as mt_metrics


def inputPath(dirPath):
    img_path = sorted(
                        [
                            os.path.join(dirPath, fname)
                            for fname in os.listdir(dirPath)
                            if os.path.splitext(os.path.split(fname)[-1])[-1] == '.png'
                        ]
                        )
    return img_path

file1 = '7_5_Right_Image.png'
file2 = '4_2_Left_Image.png'

files = [file1, file2]

data = 'test'

imfile = "EndoRR_dataset/%s/image"%data
gtfile = "EndoRR_dataset/%s/mask"%data
samfile ="EndoRR_dataset/%s/SAM_mask"%data
kmeans = "EndoRR_dataset/%s/kmeans_mask_with_dilation_1"%data
newsamfile = "EndoRR_dataset/%s/retrain_SAM_kmeans_mask"%data
spierre = "EndoRR_dataset/%s/SPierre_Arnold1_mask"%data
newsamfile2 = "EndoRR_dataset/%s/retrain_SAM_SPierre_Arnold1_mask"%data

inferences = {
    0:{'name':'provided_SAM',
       'model':'sam',
       'maskfile':samfile},
    1:{'name':'SAM_kmeans',
       'model':'sam',
       'maskfile':newsamfile},
    2:{'name':'SAM_histogram-thresholding',
       'model':'sam',
       'maskfile':newsamfile2},
    3:{'name':'kmeans_clustering',
      'maskfile':kmeans},
    4:{'name':'Histogram-based_thresholding',
       'maskfile':spierre}
    
    }

code = 0

metric_fns = [mt_metrics.accuracy_score,mt_metrics.dice_score, mt_metrics.jaccard_score, mt_metrics.intersection_over_union]
gtlist = inputPath(gtfile)
num_samples = len(gtlist)


for i,test in inferences.items():

    result_dict = defaultdict(float)
    
    masklist = inputPath(test['maskfile'])
    assert len(masklist) == num_samples
    for i in range(num_samples):
        gtfile = gtlist[i]
        maskfile = masklist[i]
        gtname = os.path.splitext(os.path.split(gtfile)[-1])[0]
        maskname = os.path.splitext(os.path.split(maskfile)[-1])[0]
        assert gtname == maskname
        
        gt = cv2.imread(gtfile,0).astype(float)
        mask = cv2.imread(maskfile,0).astype(float)
        
        gt/=255
        mask/=255
        
        for metric_fn in metric_fns:
            res = metric_fn(mask, gt)
            dict_key = '{}'.format(metric_fn.__name__)
            result_dict[dict_key] += res
    
    for key, val in result_dict.items():
        result_dict[key] = val / num_samples

    test['results'] = result_dict
    
with open("all_metrics_%s_data_FINAL.json"%data,"w") as outfile:
    json.dump(inferences,outfile,indent=2)