#!/bin/bash

#python train.py --dataroot ./datasets/mxifAll2he --name mxifAll2he_no_flip_inoutput_no_preprocess_with_dropout --model cycle_gan --lambda_identity 0 --input_nc 27 --no_flip --batch_size 4 --norm batch --load_size 256 --crop_size 256
#python train.py --dataroot ./datasets/mxifAll2he --name mxifAll2he_no_flip_input_no_preprocess_only --model cycle_gan --lambda_identity 0 --input_nc 27 --no_flip --batch_size 4 --norm batch
python train.py --dataroot ./datasets/mxifAll2he --name mxifAll2he_fix_seed_random_flip_scale_crop --model cycle_gan --lambda_identity 0 --input_nc 27 --no_flip --batch_size 4 --norm batch

