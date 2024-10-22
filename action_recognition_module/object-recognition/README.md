# Object-PALM: Exploring VLMs for Object-Centric Action Recognition and Forecasting
## Object Recognition Module

The object recognition module aims to detect which objects are present in the scene while an action is being performed. It consists of two main components:

### Hand Region Extraction

First, [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) is used to extract regions near the hands of the camera-wearer. Since the goal is to detect potential objects to which the action is being performed, we assume that they should consistently appear close to the camera-wearer's hands. Use [grounding_dino.py](hand-region-detection/grounding_dino/grounding_dino.py) or [run_ego4d_grounding_dino_hands.sh](hand-region-detection/grounding_dino/run_ego4d_grounding_dino_hands.sh) to obtain the hand regions as json files. Remember to adjust the Ego4d path inside the python script. Besides the software stack present in Euler, the environment used had the libraries specified in [requirements_grounding_dino_venv.txt](hand-region-detection/grounding_dino/requirements_grounding_dino_venv.txt).

### RAM++ for Object Recognition
Then, [RAM++](https://github.com/xinyu1205/recognize-anything) evaluates those hand regions and extract scores for all the objects present in the Ego4d taxonomy. The RAM++ and Ego4d taxonomies are different. Thus, we had to do a mapping where several RAM++ nouns can belong to the same Ego4D category. This mapping can be found in [ram_nouns_mapping.json](ram_nouns_mapping.json) and [ram_nouns_mapping_ids.json](ram_nouns_mapping_ids.json). However, [some Ego4d nouns](ram_nouns_not_in_tax.txt) do not have a direct representation in RAM++. To perform RAM++ openset inference we extract several descriptions of these nouns using [generate_tag_des_llm_llama.py](recognize-anything/generate_tag_des_llm_llama.py). These descriptions are stored [here](recognize-anything/datasets/ego4d_extra_78_nouns/ego4d_extra_78_nouns_llm_tag_descriptions.json).

Finally, run this module to obtain scores for all Ego4d classes per action segment using [inference_ram_plus_ego4d_hands_regions.py](recognize-anything/inference_ram_plus_ego4d_hands_regions.py) (or the associated SLURM scripts [run_ego4d_ram_plus_train.sh](recognize-anything/run_ego4d_ram_plus_train.sh) and [run_ego4d_ram_plus_val.sh](recognize-anything/run_ego4d_ram_plus_val.sh)). The libraries installed in the environment used for this task are in [requirements_groundedsam_venv.txt](requirements_groundedsam_venv.txt). Remember to adjust the necessary paths given as an argument when running the script. In addition, to run this code is necessary to download the weights of the RAM++ model (`ram_plus_swin_large_14m.pth`) from the [official repository](https://github.com/xinyu1205/recognize-anything).

### Postprocessing

The outputs of this module require some postprocessing to adapt the scores to the egocentric setting. We perform that without training based on the quantiles of the distribution of the scores found in the dataset. Some variants are considered: computing new thresholds using the Q-0.95 quantile and applying a piecewise linear transformation using several quantiles (with and without a global threshold of 0.91). The relevant code for this purpose can be found in [ram_analysis](ram_analysis). The script [compute_logit_distr_noun_train_hands_mean.py](ram_analysis/compute_logit_distr_noun_train_hands_mean.py) saves the quantiles found for all clases in [non-gt_quantiles_train_ram_plus_trunc-mean-3_logits_hand_boxes_mapped_ego4d_complete.pt](ram_analysis/non-gt_quantiles_train_ram_plus_trunc-mean-3_logits_hand_boxes_mapped_ego4d_complete.pt). These quantiles are used to compute a rescaled version for all the action segments using the script [compute_quantile_rescaling_and_check_acc_length_hand_detections_truncated_mean_top_15.py](ram_analysis/compute_quantile_rescaling_and_check_acc_length_hand_detections_truncated_mean_top_15.py). The rest of python files are used to obtain some relevant statistics regarding the recognition performance (uploaded as .txt files). Remember to adapt the path variables inside the scripts before using them. Finally, that directory also contains the two variants of thresholds used for recognition.