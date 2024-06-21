#!/bin/sh

python3 -m torch.distributed.launch --nproc_per_node=4 --use_env Get_ECM_example.py --config ./configs/Pretrain_4Mclean_16blockimagenet_imagenet_for_grit_search5.yaml --output_dir output/ECM_example --resume True --index_warmstart True --filter_config ./configs/Retrieval_coco_filter.yaml  --filter_checkpoint output/Retrieval_coco/4M_grit_itc_fix_search5/checkpoint_best.pth