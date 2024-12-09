'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_pretrain_MAFA import ALBEF_Base
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from models.GRIT_utils import GRIT,chunks, mini_batch_level_shuffle

import utils
from dataset.handle_data import create_dataset, create_sampler,create_fixed_sampler, create_loader
import scipy.io as sio
from scheduler import create_scheduler
from optim import create_optimizer
import os
import math
from models.model_retrieval import ALBEF

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, filter_model,args):
    # train
    model.train()  
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size 

    # GRIT init (queue init)
    num_steps=len(data_loader)
    grit= GRIT (config,device,num_steps)

    d_idx_arr=[]
    for e in data_loader.sampler:
        d_idx_arr.append(e)
    d_idx_chunk_arr = list(chunks(d_idx_arr, config['batch_size']))


    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()

        image = image.to(device,non_blocking=True)
        d_idx = torch.tensor(d_idx_chunk_arr[i]).to(device)
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device)

        # model pre-training loss
        loss_mlm, loss_ita, loss_itm,image_feat_store,text_feat_store = model(image, text_input, filter_model, args.low_threshold, args.high_threshold ,epoch)
        loss = loss_mlm + loss_ita + loss_itm
        loss.backward()
        optimizer.step()

        # GRIT 1 phases 
        grit.collecting(image_feat_store,text_feat_store,d_idx) 
        # GRIT 2-3 phases
        grit.grit_second_third_phase(i, model.module.temp,num_steps)

        # logger
        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
       
    
    dist.barrier()
    # Mini-batch level shuffle (Phase 4) 
    if not args.mini_batch_shuffle_across_gpu:
            G_index_set = mini_batch_level_shuffle(grit.G_index_set,config['batch_size'])
    else:
            G_index_set=grit.G_index_set
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} ,G_index_set   
 
   
def main(args, config, filter_config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed 
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['train_epochs']
    warmup_steps = config['schedular']['warmup_epochs']    

    #### Dataset #### 
    print("Creating dataset")
    datasets = [create_dataset('pretrain', config)]
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()         

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    
    #### Model #### 
    print("Creating model")
    model = ALBEF_Base(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
    model = model.to(device)   


    ## Con-D: filter_model ###
    filter_model = ALBEF(config=filter_config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    if args.filter_checkpoint:    
        checkpoint = torch.load(args.filter_checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],filter_model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],filter_model.visual_encoder)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
        
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')         
                state_dict[encoder_key] = state_dict[key] 
                del state_dict[key]                
        msg = filter_model.load_state_dict(state_dict,strict=False)  
        print('load filter checkpoint from %s'%args.filter_checkpoint)
    filter_model.eval()
    filter_model = filter_model.to(device)
    for param in filter_model.parameters():
        param.requires_grad = False
    ########
        
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  

    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                       
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']+1         
        else:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped       
        
        model.load_state_dict(state_dict)    
        print('load checkpoint from %s'%args.checkpoint)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    print("Start training")
    start_time = time.time()
   
    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps) 
                    
        if (epoch==0 or args.resume)  and not args.index_warmstart:
            if args.distributed:	
                samplers = create_sampler(datasets, [True], num_tasks, global_rank)	
            else:	
                samplers = [None]            
            data_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]
            if args.distributed:
                data_loader.sampler.set_epoch(epoch)        

        else:
            index_file= sio.loadmat(args.output_dir+'/total_indices.mat')
            previous_index_set= index_file['indices'][0]

            # Data loader
            if args.distributed:	
                samplers = create_fixed_sampler(num_tasks, global_rank,previous_index_set)	
            else:	
                samplers = [None]
            data_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]        
        
        # Training
        train_stats, next_index_set = train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config, filter_model, args) 
        
        # Output index save
        next_index_set = torch.tensor(next_index_set,dtype=torch.long).to(device)
        total_next_index_set = concat_all_gather(next_index_set)
        total_next_index_set = total_next_index_set.detach().cpu().numpy()     
        
        # Mini-batch level shuffle across gpu (Phase 4) 
        if args.mini_batch_shuffle_across_gpu:
            total_next_index_set= mini_batch_level_shuffle(list(total_next_index_set),config['batch_size'])
        
        sio.savemat(args.output_dir+'/total_indices.mat',{'indices':total_next_index_set})
        dist.barrier()        
        
        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        }                     
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        dist.barrier()  
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
                                                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--filter_config', default='')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--filter_checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--low_threshold', default=0.5, type=float, help='low threshold of ITM')    
    parser.add_argument('--high_threshold', default=0.8, type=float, help='high threshold of ITM')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--index_warmstart', default=False, type=bool)
    parser.add_argument('--mini_batch_shuffle_across_gpu', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    filter_config = yaml.load(open(args.filter_config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    yaml.dump(filter_config, open(os.path.join(args.output_dir, 'filter_config.yaml'), 'w'))    
    
    main(args, config, filter_config)
