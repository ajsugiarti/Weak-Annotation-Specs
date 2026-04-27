# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 15:31:46 2026

@author: annisa.sugiarti
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import json
import time
import torch
from torch.utils.data import DataLoader
from sklearn.utils import shuffle

# import medicaltorch.filters as mt_filters
import medicaltorch.losses as mt_losses
import medicaltorch.metrics as mt_metrics
# import medicaltorch.datasets as mt_datasets
# import medicaltorch.transforms as mt_transforms

# import torchvision as tv
import torchvision.utils as vutils

from tensorboardX import SummaryWriter
from tqdm import tqdm

from main_resunetpp import CustomDataset, validation, memory_stats
from resunetplusplus_pytorch import build_resunetplusplus as resunetpp



def createDataset(path,excludes: list =[],mask_type='mix'):
    subdirs = [subdir for subdir in os.listdir(path)]
    for exclude in excludes:
        subdirs.remove(exclude)
    
    images = []
    masks = []
    for subdir in subdirs:        
        images += [f for f in glob.glob(os.path.join(path,subdir,'frame','*.png'))]
        masks += [f for f in glob.glob(os.path.join(path,subdir,'mask_{}'.format(mask_type),'*.png'))]
    
    return images,masks
    
def cmd_finetune(ctx):
    global_step = 0
    num_workers = ctx["num_workers"]
    num_epochs = ctx["num_epochs"]
    initial_lr = ctx["initial_lr"]
    weight_decay = ctx["weight_decay"]
    useLRonPlateau = ctx["LRonPlateau"]
    modelfile = ctx["model"]
    
    if ctx['img_size_y'] is not None:
        IMG_SIZE = (ctx['img_size_y'],ctx['img_size_x'])
    else:
        IMG_SIZE = None
    print('Resized image size is : '+str(IMG_SIZE))
    
    
    # Ntrain = ctx['num_finetune']
    
    #Define which folder to exclude 
    #Uncomment all muscle folders if finetuning with lung only.
    # 
    excludes = [
        # 'muscle_123636', 
        # 'muscle_124705', 
        # 'muscle_125041', 
                ]
    
    X_paths,Y_paths = createDataset(ctx['train'],excludes,ctx["mask"])
    
    Ntrain = int(0.8*len(X_paths))
    print('train data: ',Ntrain)
    print('validation data: ',len(X_paths)-Ntrain)
    savepath = 'finetuned_model/{}/{}'.format(ctx["mask"],ctx["experiment_name"])
    
    if not os.path.isdir(savepath):
        os.mkdir(savepath) 

    transform = A.Compose([
            A.ShiftScaleRotate(p=0.5, border_mode = 2),
            A.GridDistortion(p=0.5),
              ])
    
    
    Xs_paths,Ys_paths = shuffle(X_paths,Y_paths,random_state=0)
    
    
    Xs_train_paths = Xs_paths[:Ntrain]
    Ys_train_paths = Ys_paths[:Ntrain]
    Xt_valid_paths = Xs_paths[Ntrain:]
    Yt_valid_paths = Ys_paths[Ntrain:]
    
    # Sample Xs and Ys from this
    source_train = CustomDataset(Xs_train_paths,Ys_train_paths,
                                 img_size = IMG_SIZE,
                                 augmentation = transform)   
    
    # test data for final evaluation
    val_data = CustomDataset(Xt_valid_paths,Yt_valid_paths,
                                img_size = IMG_SIZE,
                                augmentation = None)


    #create data loaders
    source_train_loader = DataLoader(source_train, batch_size=ctx["batch_size"],
                                     shuffle=True, drop_last=False,
                                     num_workers=num_workers
                                    )
                                    
                                    #  collate_fn=mt_datasets.mt_collate,
                                    #  pin_memory=True)

    val_loader = DataLoader(val_data, batch_size=ctx["batch_size"],
                                                 shuffle=False, drop_last=False,
                                                )   

    model = resunetpp()
    model.load_state_dict(torch.load(modelfile, weights_only=True, map_location='cuda:0'))
    model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=initial_lr,
                                 weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    writer = SummaryWriter(log_dir="{}/logs".format(savepath))
    
    # Training loop
    best_dice = 0.0
    best_epoch = 0
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
                writer.add_image('Train_Prediction', plot_img, epoch)
            
                plot_img = vutils.make_grid(train_input,
                                            normalize=True, scale_each=True)
                writer.add_image('Train_Input', plot_img, epoch)
            
                plot_img = vutils.make_grid(train_gt,
                                            normalize=True, scale_each=True)
                writer.add_image('Train_Ground_Truth', plot_img, epoch)
            except:
                 tqdm.write("*** Error writing images ***")
                        
    
        writer.add_scalars('losses', {'class_loss': class_loss_avg}, epoch)
                                      
                           
                                                               
        # Evaluation mode
        model.eval()
        
        metric_fns = [mt_metrics.accuracy_score,mt_metrics.dice_score, mt_metrics.jaccard_score]#, mt_metrics.intersection_over_union]
        
        val_loss,dice = validation(model, val_loader, metric_fns,
                              epoch, ctx, 'val_', writer) 
        # return_val = validation(model, val_loader, metric_fns,
        #                       epoch, ctx, 'val_', writer)  
        # print(type(return_val))
        # print(return_val)
        # if useLRonPlateau:
        #     scheduler.step(val_loss)
        
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), '%s/best_model.pt'%(savepath))
            best_epoch = epoch
            
        if epoch == 70:
            torch.save(model.state_dict(), '%s/70_epoch_model.pt'%(savepath))
            
        
        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))
    
    print('best_epoch is:',best_epoch)
    torch.save(model.state_dict(), '%s/last_epoch_model.pt'%(savepath))
    writer.close()
    
    del global_step, num_steps, source_train
    del source_train_loader 
    del model, optimizer, writer
    del class_loss_total
    del train_input, train_gt, preds_supervised, class_loss
    del class_loss_avg
    
    torch.cuda.empty_cache()


def run(json_ctx=None):
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
    # print(ctx['train']['mask'])
    # print(ctx['test']['mask'])

    if command == 'finetune':
        cmd_finetune(ctx)
        torch.cuda.empty_cache()
        memory_stats()

        

if __name__ == '__main__':
    json_ctx = "configs_finetune2.json"
    run(json_ctx=json_ctx)