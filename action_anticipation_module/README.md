# Object-PALM: Exploring VLMs for Object-Centric Action Recognition and Forecasting
## Action Anticipation Module

The action anticipation module uses the .json files extracted from the [captioning_module](../captioning_module/) and the [action_recognition_module](../action_recognition_module/) to generate the LTA prediction using an LLM. To run the code use the script `llama8b_run.sh` that uses `main_run.py`. The outputs are saved in [llm-log](EGO4D-prediction/llm-log/). We also provide `main_run_evaluate.py` for the evaluation of already generated outputs.

This module uses Llama3-8b for the generation. Update [this function](EGO4D-prediction/utils/prompt_utils.py#L115) with the location of the model after downloading it (or to add the definition of another model if required).

The path of the Ego4d dataset needs to be updated inside the python scripts. Also, we add to the scripts the variable `use_narration_summary`, that must be set True to properly apply the new addition to this module (the previous summarization step to combine all the past captions into a single one). If set to False, it applies the original version that does not include the summarization step.

