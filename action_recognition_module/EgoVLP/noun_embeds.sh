#!/bin/bash
#SBATCH --job-name=egovlp_nouns

#SBATCH --time=1:00:00
#SBATCH --gpus=rtx_3090:1
#SBATCH --tmp=16G
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=egovlp_new-nouns_descriptions_cls.txt

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source /cluster/project/cvg/students/azaera/egovlp_venv/bin/activate

python get_egovlp_noun_embeddings.py

