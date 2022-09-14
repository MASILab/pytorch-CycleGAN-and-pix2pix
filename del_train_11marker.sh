#!/bin/bash

python train.py --dataroot ./datasets/mxifAll2he --name mxifStruct2he --model cycle_gan --lambda_identity 0 --input_nc 11 --batch_size 4 --norm batch
