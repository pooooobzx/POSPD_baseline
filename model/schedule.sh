#!/bin/bash
#SBATCH --mem=20000
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=def-lilimou
#EXPORT NLTK_DATA=/home/xzhang23/nltk_data
python3 train.py
