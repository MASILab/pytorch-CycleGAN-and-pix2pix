#!/bin/bash

#python train.py --dataroot ./datasets/mxifAll2he --name mxifAll2he_no_flip_inoutput_no_preprocess_with_dropout --model cycle_gan --lambda_identity 0 --input_nc 27 --no_flip --batch_size 4 --norm batch --load_size 256 --crop_size 256
#python train.py --dataroot ./datasets/mxifAll2he --name mxifAll2he_no_flip_input_no_preprocess_only --model cycle_gan --lambda_identity 0 --input_nc 27 --no_flip --batch_size 4 --norm batch
#python train.py --dataroot ./datasets/mxifAll2he --name mxifAll2he_fix_seed_no_flip_random_scale_crop --model cycle_gan --lambda_identity 0 --input_nc 27 --no_flip --batch_size 4 --norm batch
#python train.py --dataroot ./datasets/mxifAll2he --name mxifAll2he_fix_seed_random_flip_scale_crop --model cycle_gan --lambda_identity 0 --input_nc 27 --batch_size 4 --norm batch
#python train.py --dataroot ./datasets/mxifAll2he --name mxifAll2he_fix_seed_random_flip_scale_crop_no_dropout --model cycle_gan --lambda_identity 0 --input_nc 27 --batch_size 4 --norm batch --no_dropout
#python train.py --dataroot ./datasets/mxifAll2he --name mxifAll2he_fix_seed_random_flip_scale_crop_no_dropout_HE_aug --model cycle_gan --lambda_identity 0 --input_nc 27 --batch_size 4 --norm batch --no_dropout
#python train.py --dataroot ./datasets/mxifAll2he --name mxifAll2he_fix_seed_random_flip_scale_crop_dropout_HE_aug --model cycle_gan --lambda_identity 0 --input_nc 27 --batch_size 4 --norm batch
#python train.py --dataroot ./datasets/af2he --name af2he_fix_seed_random_flip_scale_crop_dropout_HE_aug --model cycle_gan --lambda_identity 0 --input_nc 4 --batch_size 4 --norm batch
#python train.py --dataroot ./datasets/af2he --name af2he_DAPI_WITH_AF_fix_seed_random_flip_scale_crop_dropout_HE_aug --model cycle_gan --lambda_identity 0 --input_nc 1 --batch_size 4 --norm batch
# python train.py --dataroot ./datasets/af2he --name af2he_DAPI_AFRemoved_WITH_AF_fix_seed_random_flip_scale_crop_dropout_HE_aug --model cycle_gan --lambda_identity 0 --input_nc 4 --batch_size 4 --norm batch
# python train.py --dataroot ./datasets/af99Tohe --name af2he_DAPI_AFRemoved_WITH_AF99MinMaxNorm_fix_seed_random_flip_scale_crop_dropout_HE_aug --model cycle_gan --lambda_identity 0 --input_nc 4 --batch_size 4 --norm batch
# python test.py --dataroot ./datasets/af99Tohe --name af2he_DAPI_AFRemoved_WITH_AF99MinMaxNorm_fix_seed_random_flip_scale_crop_dropout_HE_aug --model cycle_gan --input_nc 2 --norm batch
python train.py --dataroot ./datasets/afMinMaxToHE --name af2he_DAPI_AFRemoved_WITH_AFMinMaxNorm_fix_seed_random_flip_scale_crop_dropout_HE_aug --model cycle_gan --lambda_identity 0 --input_nc 2 --batch_size 4 --norm batch
python test.py --dataroot ./datasets/afMinMaxToHE --name af2he_DAPI_AFRemoved_WITH_AFMinMaxNorm_fix_seed_random_flip_scale_crop_dropout_HE_aug --model cycle_gan --input_nc 2 --norm batch
