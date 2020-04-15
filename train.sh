#!/bin/bash
#
#SBATCH --job-name=resnet-unet-segmentation
#SBATCH --output=unet_train_out.txt

srun --gres=gpu:1 python src/train.py config/segmentation_poles.json
