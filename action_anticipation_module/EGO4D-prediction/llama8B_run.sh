#!/bin/bash
#SBATCH --job-name=llama8B_run

#SBATCH --gpus=1 
#SBATCH --gres=gpumem:20g
#SBATCH --time=120:00:00
#SBATCH --tmp=16G
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=llama8B_intention_summary_2.txt

module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source /cluster/home/azaera/groundedsam_venv/bin/activate

python main_run.py --split val \
 --text_generation_model Llama-3-8b --nexample 4 --ntot_example 32 \
 --prompt_design maxmargin --log_file llama8B_run_intention_summary \
 --caption_file ../../captioning_module/EILEV/samples/captions/summary_caption_video_blip_intention_ncap4_val.json
 #--caption_file ../../captioning_module/EILEV/samples/captions/caption_video_blip_intention_ncap4_val.json
 