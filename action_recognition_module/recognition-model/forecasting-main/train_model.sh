#!/bin/bash
#SBATCH --job-name=train_ar_model

#SBATCH --gpus=1 
#SBATCH --gres=gpumem:20g
#SBATCH --time=120:00:00
#SBATCH --tmp=16G
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=out_train.txt

module load stack/2024-06 gcc/12.2.0 python/3.9.18
source /cluster/project/cvg/students/azaera/ar_venv/bin/activate

bash tools/action_recognition/ego4d_train_ar.sh /cluster/scratch/azaera/models/palm_ar_egovlp