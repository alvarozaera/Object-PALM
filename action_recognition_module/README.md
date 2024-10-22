To generate the recognized actions, please follow below instructions

1. Generate and save video features at [EgoVLP](EgoVLP/).
   - Refer to official github page (https://github.com/showlab/EgoVLP/tree/main) to download pretrained model (EgoVLP_PT_BEST: egovlp.pth) and set environment.
   - You should also download Ego4D video clips and annotations following official Ego4D github for long-term action anticipation https://github.com/EGO4D/forecasting/blob/main/LONG_TERM_ANTICIPATION.md and change the corresponding directory in this codebase.
   - Save train data features
   ```python run/test_lta.py --gpu 1 --config configs/eval/lta.json --save_feats /your/save/directory/ --split train```.

2. Train our action recognition model at [forecasting-main](forecasting-main/).
   - We use checkpoint model ```pretrained_models/long_term_anticipation/lta_slowfast_trf.ckpt``` from [Ego4D_LTA](https://github.com/EGO4D/forecasting/blob/main/LONG_TERM_ANTICIPATION.md) to initialize the transformer network.
   - Train the model ```bash tools/long_term_anticipation/evaluate_forecasting.sh output/directory```.
   - Evaluate the model ```bash tools/long_term_anticipation/evaluate_test.sh output/directory```.
