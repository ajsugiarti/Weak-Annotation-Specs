# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:59:37 2024

@author: annisa.sugiarti
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import cv2
# from torch.nn import functional as F
import torchvision.transforms.functional as Ftv
import albumentations as A
from torch.utils.data import Dataset
import sys
import json
import time
from collections.abc import Mapping, Sequence
from collections import defaultdict
import torch
from torch.utils.data import DataLoader


# import medicaltorch.filters as mt_filters
import medicaltorch.losses as mt_losses
import medicaltorch.metrics as mt_metrics
# import medicaltorch.datasets as mt_datasets
# import medicaltorch.transforms as mt_transforms

# import torchvision as tv
import torchvision.utils as vutils

from tqdm import tqdm

from tensorboardX import SummaryWriter

import models as da_models
import sod_metric

"""
__numpy_type_map and mt_collate are taken from docs of medicaltorch.datasets
To avoid error due to unsupported torch._six which imports int_classes and string_classes,
lines containing these instances are ignored.
Sequence and Mapping now imported from collections.abc instead of collections in python 3.10
"""
__numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

def memory_stats():
    print('\nAllocated and reserved memory:')
    print(torch.cuda.memory_allocated()/1024**2)
    print(torch.cuda.memory_reserved()/1024**2)
    
    
def mt_collate(batch):
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        stacked = torch.stack(batch, 0)
        return stacked
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return __numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    # elif isinstance(batch[0], int_classes):
    #     return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    # elif isinstance(batch[0], string_classes):
    #     return batch
    elif isinstance(batch[0], Mapping):
        return {key: mt_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [mt_collate(samples) for samples in transposed]

    return batch


def inputPath(dirPath):
    img_path = sorted(
                        [
                            os.path.join(dirPath, fname)
                            for fname in os.listdir(dirPath)
                            if os.path.splitext(os.path.split(fname)[-1])[-1] == '.jpg'
                            or os.path.splitext(os.path.split(fname)[-1])[-1] == '.png'
                        ]
                        )
    return img_path


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



def decay_poly_lr(current_epoch, num_epochs, initial_lr):
    initial_lrate = initial_lr
    factor = 1.0 - (current_epoch / num_epochs)
    lrate = initial_lrate * np.power(factor, 0.9)
    return lrate


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def cosine_lr(current_epoch, num_epochs, initial_lr):
    return initial_lr * cosine_rampdown(current_epoch, num_epochs)


def decay_constant_lr(current_epoch, num_epochs, initial_lr):
    return initial_lr


def get_current_consistency_weight(weight, epoch, rampup):
    """Consistency ramp-up from https://arxiv.org/abs/1610.02242"""
    return weight * sigmoid_rampup(epoch, rampup)


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def adjust_learning_rate(optimizer, epoch, step_in_epoch,
                         total_steps_in_epoch, initial_lr, rampup_begin):
    lr = initial_lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = linear_rampup(epoch, 15) * (initial_lr - rampup_begin) + rampup_begin

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def weightedfscore(y_pred, y_true):
    """
    Adapted from utils.py in EndoSRR: https://github.com/Tobyzai/EndoSRR/tree/master
    """
    metric_WFM = sod_metric.WeightedFmeasure()
    metric_WFM.step(pred=y_pred*255, gt=y_true*255)
    wfm = metric_WFM.get_results()["wfm"]

    return wfm*100  


def threshold_predictions(predictions, thr=0.9):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds


def create_model(ctx):
    drop_rate = ctx["drop_rate"]
    bn_momentum = ctx["bn_momentum"]
    model = da_models.Unet(drop_rate=drop_rate,
                           bn_momentum=bn_momentum)

    return model.cuda()


def validation(model1, loader,
               metric_fns, epoch, ctx, prefix, writer=None):
    val_source_loss = 0.0

    num_samples = 0
    num_steps = 0

    result_source_dict = defaultdict(float)


    for i, batch in enumerate(loader):
        input_data, gt_data = batch["input"], batch["gt"]

        input_data_gpu = input_data.cuda()
        gt_data_gpu = gt_data.cuda()

        with torch.no_grad():
            model_out1 = model1(input_data_gpu)
            val_source_class_loss = mt_losses.dice_loss(model_out1, gt_data_gpu)
            val_source_loss += val_source_class_loss.item()


        gt_masks = gt_data_gpu.cpu().numpy().astype(np.uint8)
        gt_masks = gt_masks.squeeze(axis=1)

        preds1 = model_out1.cpu().numpy()
        preds1 = threshold_predictions(preds1)
        preds1 = preds1.astype(np.uint8)
        preds1 = preds1.squeeze(axis=1)
        
        for metric_fn in metric_fns:
            for prediction, ground_truth in zip(preds1, gt_masks):
                res = metric_fn(prediction, ground_truth)
                dict_key = 'val_source_{}'.format(metric_fn.__name__)
                result_source_dict[dict_key] += res

        num_samples += len(preds1)
        num_steps += 1

    val_source_loss_avg = val_source_loss / num_steps

    for key, val in result_source_dict.items():
        result_source_dict[key] = val / num_samples
        
    if writer is not None:
        writer.add_scalars(prefix + '_metrics', result_source_dict, epoch)
    
        writer.add_scalars(prefix + '_losses', {
                           prefix + '_source_loss': val_source_loss_avg,
                       },
                       epoch)
        return val_source_loss_avg
    else:
        result_source_dict['val_losses'] = val_source_loss_avg
        im = input_data.permute(0,2,3,1).numpy()
        return result_source_dict,im,gt_masks,preds1
    
def cmd_train(ctx):
    global_step = 0
    num_workers = ctx["num_workers"]
    num_epochs = ctx["num_epochs"]
    initial_lr = ctx["initial_lr"]
    weight_decay = ctx["weight_decay"]
    savepath = ctx["experiment_name"]
    useLRonPlateau = ctx["LRonPlateau"]
    if ctx['img_size_y'] is not None:
        IMG_SIZE = (ctx['img_size_y'],ctx['img_size_x'])
    else:
        IMG_SIZE = None
    print('Resized image size is : '+str(IMG_SIZE))
    
    if not os.path.isdir(savepath):
        os.mkdir(savepath) 


    transform = A.Compose([
            A.Flip(p=0.5),
            A.ShiftScaleRotate(p=0.5, border_mode = 2),
            A.GridDistortion(p=0.5),
              ])
    
        
    Xs_train_dir = ctx['train']['image']
    Ys_train_dir = ctx['train']['mask']
    Xt_valid_dir = ctx['val']['image']
    Yt_valid_dir = ctx['val']['mask']
    
    if Xt_valid_dir is None:
        im_paths = inputPath(Xs_train_dir)
        mask_paths = inputPath(Ys_train_dir)
        fold = int(np.ceil(len(im_paths)/5))
        
        Xs_train_paths = im_paths[:fold*4]
        Ys_train_paths = mask_paths[:fold*4]
        Xt_valid_paths = im_paths[fold*4:]
        Yt_valid_paths = mask_paths[fold*4:]
        
    else:
        Xs_train_paths = inputPath(Xs_train_dir)
        Ys_train_paths = inputPath(Ys_train_dir)
        Xt_valid_paths = inputPath(Xt_valid_dir)
        Yt_valid_paths = inputPath(Yt_valid_dir)
    
    print('Number of training data: '+str(len(Xs_train_paths)))
    print('Number of validation data: '+str(len(Xt_valid_paths)))
    
    # Sample Xs and Ys from this
    source_train = CustomDataset(Xs_train_paths,Ys_train_paths,
                                 img_size = IMG_SIZE,
                                 augmentation = transform)   
    
    # test data for final evaluation
    target_test = CustomDataset(Xt_valid_paths,Yt_valid_paths,
                                img_size = IMG_SIZE,
                                augmentation = None)


    #create data loaders
    source_train_loader = DataLoader(source_train, batch_size=ctx["batch_size"],
                                     shuffle=True, drop_last=False,
                                     num_workers=num_workers
                                    )
                                    
                                    #  collate_fn=mt_datasets.mt_collate,
                                    #  pin_memory=True)

    test_loader = DataLoader(target_test, batch_size=ctx["batch_size"],
                                                 shuffle=False, drop_last=False,
                                                )   

    model = create_model(ctx)
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=initial_lr,
                                 weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    writer = SummaryWriter(log_dir="{}/logs".format(savepath))
    
    # Training loop
    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        start_time = time.time()

        # LR
        lr = scheduler.get_last_lr()[0]
        writer.add_scalar('learning_rate', lr, epoch)
        tqdm.write("Learning Rate: {:.6f}".format(lr))    

        # Train mode
        model.train()
        
        class_loss_total = 0.0
        num_steps = 0
        
        for i, train_batch in enumerate(source_train_loader):
            # Keys: 'input', 'gt', 'input_metadata', 'gt_metadata'

            # Supervised component --------------------------------------------
            train_input, train_gt = train_batch["input"], train_batch["gt"]
            train_input = train_input.cuda()
            train_gt = train_gt.cuda()
            preds_supervised = model(train_input)
            class_loss = mt_losses.dice_loss(preds_supervised, train_gt)
            
            optimizer.zero_grad()
            class_loss.backward()

            optimizer.step()

            class_loss_total += class_loss.item()

            num_steps += 1
            global_step += 1

        npy_supervised_preds = preds_supervised.detach().cpu().numpy()
        writer.add_histogram("Supervised_Preds_Hist", npy_supervised_preds, epoch)
        
        class_loss_avg = class_loss_total / num_steps
        
        tqdm.write("Steps p/ Epoch: {}".format(num_steps))
        tqdm.write("Class Loss: {:.6f}".format(class_loss_avg))

       # Write sample images
        if ctx["write_images"] and epoch % ctx["write_images_interval"] == 0:
            try:
                plot_img = vutils.make_grid(preds_supervised,
                                            normalize=True, scale_each=True)
                writer.add_image('Train_Source_Prediction', plot_img, epoch)
            
                plot_img = vutils.make_grid(train_input,
                                            normalize=True, scale_each=True)
                writer.add_image('Train_Source_Input', plot_img, epoch)
            
                plot_img = vutils.make_grid(train_gt,
                                            normalize=True, scale_each=True)
                writer.add_image('Train_Source_Ground_Truth', plot_img, epoch)
            except:
                 tqdm.write("*** Error writing images ***")
             
            torch.save(model.state_dict(), '%s/model%d.pt'%(savepath,epoch))
    
        writer.add_scalars('losses', {'class_loss': class_loss_avg}, epoch)
                                      
                           
                                                               
        # Evaluation mode
        model.eval()
        
        metric_fns = [mt_metrics.dice_score, mt_metrics.jaccard_score, mt_metrics.intersection_over_union, weightedfscore]
        
        val_loss = validation(model, test_loader, metric_fns,
                              epoch, ctx, 'val_', writer)    
        
        if useLRonPlateau:
            scheduler.step(val_loss)
            
        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))
    
   
    writer.close()
    
    del global_step, num_steps, source_train
    del source_train_loader 
    del model, optimizer, writer
    del class_loss_total
    del train_input, train_gt, preds_supervised, class_loss
    del class_loss_avg
    
    torch.cuda.empty_cache()
  


        
def run_main_unet(json_ctx=None):
    if not json_ctx and len(sys.argv) <= 1:
        print("\ndomainadapt [config filename].json\n")
        return

    # elif json_ctx:
    #     ctx = json_ctx
    else:
        try:
            with open(json_ctx) as fhandle:
                ctx = json.load(fhandle)
        except FileNotFoundError:
            print("\nFile {} not found !\n".format(sys.argv[1]))
            return

    command = ctx["command"]
    # os.environ["CUDA_VISIBLE_DEVICES"]=str(ctx["gpu"])
    torch.cuda.set_device(int(ctx["gpu"]))
    print(ctx['experiment_name'])
    print(ctx['train']['mask'])

    if command == 'train':
        cmd_train(ctx)
        torch.cuda.empty_cache()
        memory_stats()
        

if __name__ == '__main__':
    json_ctx = "configs.json"
    run_main_unet(json_ctx=json_ctx)
