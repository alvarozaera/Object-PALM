# PALM
Official implementation of &lt;PALM: Predicting Actions through Language Models> at ECCV 2024

Our codebase consists of three part as explained in the paper: action_recognition_module, image_captioning_module, and action_anticipation_module.
To run the code, Follow below instructions.

1. Please refer to [action_recognition_module](action_recognition_module/) to generate recognized action files in json (ex. outputs_train.json, outputs_val.json, and outputs_test.json).
2. Then, generate captions based on [image_captioning_module](image_captioning_module/EGO4D-caption/) (ex. caption_blip2_ncap4_train_v2.json).
3. Finally, predict future actions using LLM in [action_anticipation_module](action_anticipation_module/EGO4D-prediction).
