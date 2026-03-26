#!/bin/bash
# --arch: CLIP:GeoRSCLIP / CLIP:ViT-L/14 / CLIP:RN50
# --decoder_type: conv-20, linear
# --use_noise_view: None, light, cbam, group
# --use_noise_guided_amplification
# --use_aspp
# --use_conprn
# --use_area_loss
# --use_simdet 

CUDA_VISIBLE_DEVICES=0 python train.py \
    --arch=CLIP:GeoRSCLIP \
    --batch_size=2 \
    --num_threads=0 \
    --name=GeoRSCLIP_try_1000_fixbackbone_NAA \
    --train_dataset=DOTA_PS_ALL \
    --data_root_path=/irip/fanziyu_bishe/data/DOTA-PS \
    --train_path=train_data-ps-1000.txt \
    --valid_path=test_data-ps-all.txt \
    --decoder_type=conv-20 \
    --feature_layer=layer20 \
    --use_noise_view=group \
    --use_noise_guided_amplification \
    --use_aspp  \
    --use_conprn \
    --use_simdet \
    --use_area_loss \
    --fix_backbone \
    --fully_supervised 
    # --data_root_path=/nas/datasets/DOTA-PS \
    # --train_path=train_data-ps-all.txt \
    # --valid_path=test_data-ps-all.txt \

