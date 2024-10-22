# Object-PALM: Exploring VLMs for Object-Centric Action Recognition and Forecasting
This repository extends the codebase from  [&lt;PALM: Predicting Actions through Language Models>](https://github.com/kim-sanghwan/PALM/tree/main) for the [Ego4d](https://github.com/facebookresearch/Ego4d) long-term action anticipation task. The original code can also be found in the branch named `PALM-original-code`.

Similar to PALM, our codebase contains three main modules: action_recognition_module, image_captioning_module, and action_anticipation_module.

1. The [action_recognition_module](action_recognition_module/) is used to perform action recognition to obtain a (verb,noun) pair for each action segment. This module is the one with the most changes compared to PALM due to the inclusion of an [object_recognition_module](action_recognition_module/object-recognition) in order to develop an object-centric architecture. 
2. The [captioning_module](image_captioning_module/) is used to obtain a text description per action segment. Conceptually it is similar to PALM, but the codebase has been changed.
3. The information gathered using the two previous modules is combined to form a prompt to predict future actions using LLM in [action_anticipation_module](action_anticipation_module/EGO4D-prediction). This module remains very similar to the original one. In this work, we propose a new summarization step of the captions obtained in the [captioning_module](image_captioning_module/). However, to facilitate the usage of the code, that step is performed in the captioning module itself.

To be able to run the experiments of this repository, first you need to install the LTA v2 benchmark of the Ego4d dataset (annotations, clips and lta_models) following the instructions given by the official Ego4d [website](https://ego4d-data.org/) and [repository](https://github.com/EGO4D/forecasting/blob/main/LONG_TERM_ANTICIPATION.md). 

Note that this project was run using ETH Euler cluster. Throughout this repository, there are several SLURM scripts used to run the code. For more information about the specific libraries and software used, refer to the [Euler wiki](https://scicomp.ethz.ch/wiki/Python_on_Euler_(Ubuntu)).
