#!/bin/bash
#SBATCH --job-name=grounding-dino_ego4d

#SBATCH --gpus=1 
#SBATCH --gres=gpumem:20g
#SBATCH --time=120:00:00
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=32G
#SBATCH --output=out_hands.txt

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source /cluster/home/azaera/grounding_dino_venv/bin/activate

python grounding_dino.py