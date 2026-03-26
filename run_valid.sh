#!/bin/bash

# --fully_supervised \
# --data_aug=all
# checkpoints/GeoRSCLIP_crossdomain_SOTA_obj2_simdet/model_epoch_best_10.pth
# checkpoints/GeoRSCLIP_crossdomain_SOTA_splice_simdet/model_epoch_best_19.pth

CUDA_VISIBLE_DEVICES=8 python validate.py \
        --arch=CLIP:GeoRSCLIP \
	    --ckpt=checkpoints/GeoRSCLIP_DOTA_PS_deepfake_all/model_epoch_best.pth \
	    --result_folder=checkpoints/GeoRSCLIP_DOTA_PS_deepfake_all/result \
	    --output_save_path=checkpoints/GeoRSCLIP_DOTA_PS_deepfake_all/result/preds \
        --decoder_type=conv-20 \
        --feature_layer=layer23 \
        --use_aspp \
        --use_simdet \
        --fully_supervised \
