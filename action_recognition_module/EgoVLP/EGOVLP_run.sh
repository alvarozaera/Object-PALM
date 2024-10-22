#!/bin/bash
#SBATCH --job-name=egovlp_train_feats

#SBATCH --gpus=rtx_3090:1
#SBATCH --time=48:00:00
#SBATCH --tmp=16G
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=egovlp_val_feats.txt

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source /cluster/project/cvg/students/azaera/egovlp_venv/bin/activate

python run/test_lta.py --config configs/eval/lta.json --save_feats /cluster/scratch/azaera/ego4d/v2/ego_vlp_feats/ --split val