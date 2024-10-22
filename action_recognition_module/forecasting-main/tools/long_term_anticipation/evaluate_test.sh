function run(){
    
  NAME=$1
  CONFIG=$2
  shift 2;

  CUDA_VISIBLE_DEVICES=1 python -m scripts.run_lta \
    --job_name $NAME \
    --working_directory ${WORK_DIR} \
    --cfg $CONFIG \
    DATA.PATH_TO_DATA_DIR ${EGO4D_ANNOTS} \
    DATA.PATH_PREFIX ${EGO4D_VIDEOS} \
    CHECKPOINT_LOAD_MODEL_HEAD True \
    TRAIN.ENABLE False \
    TEST.EVAL_VAL False \
    DATA.CHECKPOINT_MODULE_FILE_PATH "" \
    FORECASTING.AGGREGATOR "" \
    FORECASTING.DECODER "" \
    CHECKPOINT_FILE_PATH "" \
    FORECASTING.NUM_SEQUENCES_TO_PREDICT 5 \
    $@
}

#-----------------------------------------------------------------------------------------------#

WORK_DIR=$1
mkdir -p ${WORK_DIR}

EGO4D_ANNOTS=$PWD/data/long_term_anticipation/annotations/
EGO4D_VIDEOS=$PWD/data/long_term_anticipation/clips/
#CLUSTER_ARGS="--on_cluster NUM_GPUS 8"

# SlowFast-Transformer
LOAD=/your/trained/model.ckpt
run slowfast_trf \
    configs/Ego4dLTA/MULTISLOWFAST_8x8_R101_v1.yaml \
    FORECASTING.AGGREGATOR TransformerAggregator \
    FORECASTING.DECODER MultiHeadDecoder \
    FORECASTING.NUM_INPUT_CLIPS 4 \
    FORECASTING.NUM_ACTIONS_TO_PREDICT 4 \
    MODEL.TRANSFORMER_ENCODER_HEADS 16 \
    MODEL.TRANSFORMER_ENCODER_LAYERS 12 \
    CHECKPOINT_FILE_PATH $LOAD

#-----------------------------------------------------------------------------------------------#
#                                    OTHER OPTIONS                                              #
#-----------------------------------------------------------------------------------------------#

# # SlowFast-Concat
# BACKBONE_WTS=$PWD/pretrained_models/long_term_anticipation/ego4d_slowfast8x8.ckpt
# run slowfast_concat \
#     configs/Ego4dLTA/MULTISLOWFAST_8x8_R101.yaml \
#     FORECASTING.AGGREGATOR ConcatAggregator \
#     FORECASTING.DECODER MultiHeadDecoder \
#     DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS}

# # MViT-Concat
# BACKBONE_WTS=$PWD/pretrained_models/long_term_anticipation/ego4d_mvit16x4.ckpt
# run mvit_concat \
#     configs/Ego4dLTA/MULTIMVIT_16x4.yaml \
#     FORECASTING.AGGREGATOR ConcatAggregator \
#     FORECASTING.DECODER MultiHeadDecoder \
#     DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS}

# # Debug locally using a smaller batch size / fewer GPUs
# CLUSTER_ARGS="NUM_GPUS 2 TRAIN.BATCH_SIZE 8 TEST.BATCH_SIZE 32"

