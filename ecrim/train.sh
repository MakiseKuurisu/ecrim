#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py --train --dev --test --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 --learning_rate 3e-5 --epochs 10