from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

class ALBEF_Base(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,    
                 temp = 0.07,
                 init_deit = True
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
            
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)      

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.itm_head = nn.Linear(text_width, 2)     
        


    def forward(self, image, text, filter_model, low_threshold, high_threshold ,epoch): 
        
        ### Calculate uni-modal embeddings from visual encoder and text encoder
        with torch.no_grad():
            bs=image.size(0)
            self.temp.clamp_(0.001,0.5)
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  
        text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 
        
        ### Calculate i2t, t2i similarities (ITC loss will be calculated later)
        with torch.no_grad():
            image_feat_sg = image_feat.t().clone()
            text_feat_sg = text_feat.t().clone()   
        
        sim_i2t = image_feat @ text_feat_sg / self.temp 
        sim_t2i = text_feat @ image_feat_sg / self.temp 
        with torch.no_grad():
                sim_targets = torch.zeros(sim_i2t.size()).to(image.device)
                sim_targets.fill_diagonal_(1)
                if epoch>0:
                    all_zeros = torch.zeros(sim_i2t.size()).to(image.device)
                    sim_targets = 0.5*sim_targets + 0.5* F.softmax(all_zeros,dim=1)
                
                image_feat_store=image_feat.clone().detach()
                text_feat_store=text_feat.clone().detach()
                
                sim_i2t_sg= sim_i2t.clone().detach()
                sim_t2i_sg= sim_t2i.clone().detach()

        with torch.no_grad():
            bs = image.size(0)
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

            sim_i2t_targets= torch.zeros(sim_i2t.size()).to(image.device)
            sim_t2i_targets= torch.zeros(sim_t2i.size()).to(image.device)
        ###
        

        ##================= ITM and ECM process ========================##                
        # forward the positve image-text pair for ITM
        output_pos = self.text_encoder.bert(encoder_embeds = text_embeds,
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,
                                        return_dict = True,
                                        mode = 'fusion',
                                       )

        ###  ECM process: Identification phase of ECM for T2I 
        ### select a hardest and a second hardest negative image for each text
        top_ks = torch.topk(weights_t2i, k=2, dim=1)
        neg_idxs_t2i = top_ks.indices[:, 0]
        sec_hard_idxs = top_ks.indices[:, 1]

        ### ECM for T2I
        filter_image_embeds = filter_model.visual_encoder(image) 
        filter_image_atts = torch.ones(filter_image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        filter_text_output = filter_model.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        filter_text_embeds = filter_text_output.last_hidden_state
        itm_output_neg_wrttxt = filter_model.text_encoder(encoder_embeds = filter_text_embeds, 
                                    attention_mask = text.attention_mask,
                                    encoder_hidden_states = filter_image_embeds[neg_idxs_t2i],
                                    encoder_attention_mask = filter_image_atts[neg_idxs_t2i],                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        itm_output_neg_wrttxt = filter_model.itm_head(itm_output_neg_wrttxt.last_hidden_state[:,0,:])
        itm_neg_wrttxt_score = F.softmax(itm_output_neg_wrttxt, dim=1)[:,1]
        
        ###  ITM for T2I 
        image_embeds_neg = image_embeds[neg_idxs_t2i][itm_neg_wrttxt_score < low_threshold]
        remained_text_embeds = text_embeds[itm_neg_wrttxt_score < low_threshold]
        remained_text_atts = text.attention_mask[itm_neg_wrttxt_score < low_threshold]

        image_embeds_pos = image_embeds[neg_idxs_t2i][itm_neg_wrttxt_score > high_threshold]
        additional_text_embeds = text_embeds[itm_neg_wrttxt_score > high_threshold]
        additional_text_atts = text.attention_mask[itm_neg_wrttxt_score > high_threshold]

        t2i_fp_t = torch.arange(bs).to(image.device)[itm_neg_wrttxt_score > high_threshold]
        t2i_fp_i= neg_idxs_t2i[itm_neg_wrttxt_score > high_threshold]

        image_embeds_sechard = image_embeds[sec_hard_idxs][(itm_neg_wrttxt_score > low_threshold)*(itm_neg_wrttxt_score < high_threshold)]
        additional_text_embeds_sec = text_embeds[(itm_neg_wrttxt_score > low_threshold)*(itm_neg_wrttxt_score < high_threshold)]
        additional_text_atts_sec = text.attention_mask[(itm_neg_wrttxt_score > low_threshold)*(itm_neg_wrttxt_score < high_threshold)]
        ###

        ###  ECM process: Identification phase of ECM for I2T 
        ### select a hardest and a second hardest negative text for each image
        top_ks = torch.topk(weights_i2t, k=2, dim=1)
        neg_idxs_i2t = top_ks.indices[:, 0]
        sec_hard_idxs = top_ks.indices[:, 1]
        
        ### Identification phase of ECM for I2T 
        itm_output_neg_wrtimg = filter_model.text_encoder(encoder_embeds = filter_text_embeds[neg_idxs_i2t], 
                                    attention_mask = text.attention_mask[neg_idxs_i2t],
                                    encoder_hidden_states = filter_image_embeds,
                                    encoder_attention_mask = filter_image_atts,                             
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        itm_output_neg_wrtimg = filter_model.itm_head(itm_output_neg_wrtimg.last_hidden_state[:,0,:])
        itm_neg_wrtimg_score = F.softmax(itm_output_neg_wrtimg, dim=1)[:,1]
        text_embeds_neg = text_embeds[neg_idxs_i2t][itm_neg_wrtimg_score < low_threshold]
        text_atts_neg = text.attention_mask[neg_idxs_i2t][itm_neg_wrtimg_score < low_threshold]
        remained_image_embeds = image_embeds[itm_neg_wrtimg_score < low_threshold]

        text_embeds_pos = text_embeds[neg_idxs_i2t][itm_neg_wrtimg_score > high_threshold]
        text_atts_pos = text.attention_mask[neg_idxs_i2t][itm_neg_wrtimg_score > high_threshold]
        additional_image_embeds = image_embeds[itm_neg_wrtimg_score > high_threshold]

        i2t_fp_i = torch.arange(bs).to(image.device)[itm_neg_wrtimg_score > high_threshold]
        i2t_fp_t= neg_idxs_i2t[itm_neg_wrtimg_score > high_threshold]

        text_embeds_sechard = text_embeds[sec_hard_idxs][(itm_neg_wrtimg_score > low_threshold)*(itm_neg_wrtimg_score < high_threshold)]
        text_atts_sechard = text.attention_mask[sec_hard_idxs][(itm_neg_wrtimg_score > low_threshold)*(itm_neg_wrtimg_score < high_threshold)]
        additional_image_embeds_sec = image_embeds[(itm_neg_wrtimg_score > low_threshold)*(itm_neg_wrtimg_score < high_threshold)]
        ###
        

        text_embeds_all = torch.cat([remained_text_embeds, text_embeds_neg, additional_text_embeds, text_embeds_pos, additional_text_embeds_sec, text_embeds_sechard],dim=0)
        text_atts_all = torch.cat([remained_text_atts, text_atts_neg, additional_text_atts, text_atts_pos, additional_text_atts_sec, text_atts_sechard],dim=0)
        image_embeds_all = torch.cat([image_embeds_neg,remained_image_embeds, image_embeds_pos, additional_image_embeds, image_embeds_sechard, additional_image_embeds_sec],dim=0)
        image_atts_all = torch.cat([image_atts[:image_embeds_neg.size(0)],image_atts[:remained_image_embeds.size(0)], image_atts[:image_embeds_pos.size(0)], image_atts[:additional_image_embeds.size(0)], image_atts[:image_embeds_sechard.size(0)], image_atts[:additional_image_embeds_sec.size(0)]],dim=0)

        output_neg_pos = self.text_encoder.bert(encoder_embeds = text_embeds_all,
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = image_embeds_all,
                                        encoder_attention_mask = image_atts_all,
                                        return_dict = True,
                                        mode = 'fusion',
                                       )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg_pos.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(remained_text_embeds.size(0)+text_embeds_neg.size(0), dtype=torch.long), torch.ones(additional_text_embeds.size(0)+text_embeds_pos.size(0), dtype=torch.long), torch.zeros(additional_text_embeds_sec.size(0)+text_embeds_sechard.size(0), dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)


        ##================= MLM with ECM========================##                
        add_pos_input_ids = torch.cat([text.input_ids.clone(), text.input_ids[neg_idxs_i2t][itm_neg_wrtimg_score > high_threshold],text.input_ids[itm_neg_wrttxt_score > high_threshold] ], dim=0)
        add_pos_labels = add_pos_input_ids.clone()
        add_pos_atts = torch.cat([text.attention_mask, text.attention_mask[neg_idxs_i2t][itm_neg_wrtimg_score > high_threshold], text.attention_mask[itm_neg_wrttxt_score > high_threshold] ],dim=0)
          
        probability_matrix = torch.full(add_pos_labels.shape, self.mlm_probability)
        add_pos_input_ids, add_pos_labels = self.mask(add_pos_input_ids, self.text_encoder.config.vocab_size, image.device, targets=add_pos_labels,
                                      probability_matrix = probability_matrix)
        
        add_pos_images = torch.cat([image_embeds, image_embeds[itm_neg_wrtimg_score > high_threshold], image_embeds[neg_idxs_t2i][itm_neg_wrttxt_score > high_threshold]],dim=0)
        add_image_atts = torch.ones(add_pos_images.size()[:-1],dtype=torch.long).to(image.device)
        
        mlm_output = self.text_encoder(add_pos_input_ids, 
                                       attention_mask = add_pos_atts, 
                                       encoder_hidden_states = add_pos_images,  
                                       encoder_attention_mask =  add_image_atts,     
                                       return_dict = True,
                                       labels = add_pos_labels, 
                                      )                           
        loss_mlm = mlm_output.loss       

        ##================= ITC with ECM========================##                
        with torch.no_grad():
             sim_i2t_targets = torch.zeros(sim_i2t.size()).to(image.device)
             sim_t2i_targets = torch.zeros(sim_t2i.size()).to(image.device)
             sim_i2t_targets.fill_diagonal_(1)
             sim_t2i_targets.fill_diagonal_(1)

             all_zeros = torch.zeros(sim_i2t.size()).to(image.device)

             bs_indices = torch.arange(bs).to(image.device)

             valid_t2i = (t2i_fp_t == bs_indices.unsqueeze(1)).any(dim=1)
             sim_t2i_targets[bs_indices[valid_t2i], t2i_fp_i] += 1

             valid_i2t = (i2t_fp_i == bs_indices.unsqueeze(1)).any(dim=1)
             sim_i2t_targets[bs_indices[valid_i2t], i2t_fp_t] += 1

             sim_t2i_targets = sim_t2i_targets / sim_t2i_targets.sum(1,keepdim=True)
             sim_i2t_targets = sim_i2t_targets / sim_i2t_targets.sum(1,keepdim=True)

             if epoch>0:
                sim_t2i_targets = 0.5*sim_t2i_targets + 0.5* F.softmax(all_zeros,dim=1)
                sim_i2t_targets = 0.5*sim_i2t_targets + 0.5* F.softmax(all_zeros,dim=1)

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean()

        loss_ita = (loss_i2t+loss_t2i)/2

        return loss_mlm, loss_ita, loss_itm, image_feat_store, text_feat_store
        
        
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

