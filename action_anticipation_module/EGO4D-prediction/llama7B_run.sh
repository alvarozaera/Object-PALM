#!/bin/bash
#SBATCH --job-name=llama7B_run

#SBATCH --gpus=a100_80gb:1
#SBATCH --time=120:00:00
#SBATCH --tmp=16G
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=llama7B_run.txt

module load gcc/8.2.0 python_gpu/3.11.2
source /cluster/work/cvl/xiwang1/sankim/vclip/bin/activate

python main_run.py --split test_unannotated \
 --caption_file /cluster/work/cvl/xiwang1/sankim/data/captions/caption_blip2_intention_ncap4_test_unannotated_v2.json \
 --text_generation_model llama7B_converted --nexample 4 --ntot_example 32 \
 --prompt_design maxmargin --log_file llama7B_run