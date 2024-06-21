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
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset.handle_data import create_dataset, create_sampler,create_fixed_sampler, create_loader
import scipy.io as sio
import os
from models.model_retrieval import ALBEF


@torch.no_grad()
def get_example(data_loader, tokenizer, epoch, device, filter_model):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    wrt_txt_examples = []
    wrt_img_examples = []
    threshold = 0.8

    for i, (image, text, imgstr) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        new_text = []
        for each_text in text:
            each_text = each_text.split('[SEPERATE]')
            new_text.append(random.sample(each_text, 1)[0])
        
        image = image.to(device,non_blocking=True)
        text_input = tokenizer(new_text, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device)

        image_embeds = filter_model.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        image_feat = F.normalize(filter_model.vision_proj(image_embeds[:,0,:]),dim=-1)  
        text_output = filter_model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(filter_model.text_proj(text_embeds[:,0,:]),dim=-1)       

        sim_i2t = image_feat @ text_feat.t() / filter_model.temp
        sim_t2i = sim_i2t.t()

        weights_i2t = F.softmax(sim_i2t, dim=1)
        weights_t2i = F.softmax(sim_t2i, dim=1)

        weights_i2t.fill_diagonal_(0)
        weights_t2i.fill_diagonal_(0)

        itm_output_pos = filter_model.text_encoder(encoder_embeds = text_embeds, 
                                    attention_mask = text_input.attention_mask,
                                    encoder_hidden_states = image_embeds,
                                    encoder_attention_mask = image_atts,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        itm_pos_score_logit = filter_model.itm_head(itm_output_pos.last_hidden_state[:,0,:])
        itm_pos_score = F.softmax(itm_pos_score_logit,dim=1)[:,1]

        top_ks = torch.topk(weights_t2i, k=2, dim=1)
        neg_idxs = top_ks.indices[:, 0]
        sec_hard_idxs = top_ks.indices[:, 1]
        image_neg = image[neg_idxs]
        image_sec_neg = image[sec_hard_idxs]
        imgstr_neg = np.array(imgstr)[neg_idxs.cpu()]
        imgstr_sec_neg = np.array(imgstr)[sec_hard_idxs.cpu()]
        itm_output_neg_wrttxt = filter_model.text_encoder(encoder_embeds = text_embeds, 
                                    attention_mask = text_input.attention_mask,
                                    encoder_hidden_states = image_embeds[neg_idxs],
                                    encoder_attention_mask = image_atts[neg_idxs],                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        itm_neg_wrttxt_score_logit = filter_model.itm_head(itm_output_neg_wrttxt.last_hidden_state[:,0,:])
        itm_neg_wrttxt_score = F.softmax(itm_neg_wrttxt_score_logit, dim=1)[:,1]


        top_ks = torch.topk(weights_i2t, k=2, dim=1)
        neg_idxs = top_ks.indices[:, 0]
        sec_hard_idxs = top_ks.indices[:, 1]
        text_neg = np.array(new_text)[neg_idxs.cpu()]
        text_sec_neg = np.array(new_text)[sec_hard_idxs.cpu()]

        top_ks = torch.topk(weights_i2t, k=1, dim=1)
        neg_idxs = top_ks.indices.squeeze()
        itm_output_neg_wrtimg = filter_model.text_encoder(encoder_embeds = text_embeds[neg_idxs], 
                                    attention_mask = text_input.attention_mask[neg_idxs],
                                    encoder_hidden_states = image_embeds,
                                    encoder_attention_mask = image_atts,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        itm_neg_wrtimg_score_logit = filter_model.itm_head(itm_output_neg_wrtimg.last_hidden_state[:,0,:])
        itm_neg_wrtimg_score = F.softmax(itm_neg_wrtimg_score_logit, dim=1)[:,1]

    
        for b, score in enumerate(itm_neg_wrtimg_score):
            if score > threshold:
                # Save examples for visualization
                wrt_img_examples.append((imgstr[b], new_text[b], text_neg[b], score))

        for b, score in enumerate(itm_neg_wrttxt_score):
            if score > threshold:
                # Save examples for visualization
                wrt_txt_examples.append((new_text[b], imgstr[b], imgstr_neg[b], score))
        
        if i==2:
            break
       
    
    dist.barrier()
    return wrt_txt_examples, wrt_img_examples
 
   
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
    datasets = [create_dataset('pretrain_caplist_imgstr', config)]
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()         

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    
    #### Model #### 
    print("Creating model")
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
        
    
    print("Start Getting ECM examples")
    start_time = time.time()
   
    for epoch in range(start_epoch, max_epoch):
                    
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
        wrt_txt_examples, wrt_img_examples = get_example(data_loader, tokenizer, epoch, device, filter_model)
        with open(args.output_dir+f'/wrt_txt_examples_{utils.get_rank()}.json', 'w') as f:
            json.dump(wrt_txt_examples, f)
        with open(args.output_dir+f'/wrt_img_examples_{utils.get_rank()}.json', 'w') as f:
            json.dump(wrt_img_examples, f)



        dist.barrier()  
        break
                
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
    parser.add_argument('--filter_checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
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
