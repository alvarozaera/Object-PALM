# Object-PALM
## Captioning Module

The captioning module uses the VideoBLIP module presented in [EILEV](https://github.com/yukw777/EILEV/tree/main) to generate captions that describe the action segments of the Ego4d dataset.

To obtain those descriptions run
```bash
cd EILEV/samples

python video_blip_generate_action_narration_ego4d.py --device cuda --split train
```
Update the necessary configuration variables when running the script to specify the ego4d path and the dataset split to process. You can also use the SLURM script [run_ego4d_videoblip.sh](EILEV/samples/run_ego4d_videoblip.sh) after adapting the environment. The libraries installed in the environment used (eilev_venv) can be found in [requirements_eilev_venv.txt](requirements_eilev_venv.txt).

The output captions are saved in [captions](EILEV/samples/captions/) as a .txt file. Use the script [create_json_with_output.py](EILEV/samples/create_json_with_output.py) with the path name of the .txt file to convert it into a .json file.

As part of the action anticipation module, we propose an extra summarization step that combines the caption of the past actions into a single one. Use `python create_summary.py` to generate these captions. To do so, adapt the necessary split and path variables inside the script. It generates a .txt file in the same [captions](EILEV/samples/captions/) folder. Use the script [create_json_with_output.py](EILEV/samples/create_json_with_output.py) again to obtain the .json file that is going to be used in the action anticipation module. In the case of the summary, the action segment id present in the .txt and .json files correspond to the first action segment of the past context. 

For the summary you can use also the script [run_summary.sh](EILEV/samples/run_summary.sh) adapting the environment. The environment used had [these](../action_recognition_module/object-recognition/requirements_groundedsam_venv.txt) libraries installed.