function run(){
    
  NAME=$1
  CONFIG=$2
  shift 2;

  python -m scripts.run_ar_module \
    --job_name $NAME \
    --working_directory ${WORK_DIR} \
    --cfg $CONFIG \
    ${CLUSTER_ARGS} \
    DATA.PATH_TO_DATA_DIR ${EGO4D_ANNOTS} \
    DATA.PATH_PREFIX ${EGO4D_VIDEOS} \
    CHECKPOINT_LOAD_MODEL_HEAD False \
    DATA.CHECKPOINT_MODULE_FILE_PATH "" \
    $@
}

#-----------------------------------------------------------------------------------------------#

WORK_DIR=$1
mkdir -p ${WORK_DIR}

EGO4D_ANNOTS=/cluster/project/cvg/data/Ego4d/v2/annotations
EGO4D_VIDEOS=/cluster/project/cvg/data/Ego4d/v2/clips

# SlowFast-Transformer (Path to lta_slowfast_trf_v2.ckpt from ego4d v2 LTA models)
BACKBONE_WTS=/cluster/scratch/azaera/ego4d/v2/lta_models/lta_slowfast_trf_v2.ckpt
run slowfast_trf \
    configs/Ego4dLTA/MULTISLOWFAST_8x8_R101.yaml \
    FORECASTING.NUM_INPUT_CLIPS 4 \
    FORECASTING.NUM_ACTIONS_TO_PREDICT 4 \
    DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS} \
    MODEL.TRANSFORMER_HEADS 16 \
    MODEL.TRANSFORMER_LAYERS 9 \
    SOLVER.MAX_EPOCH 200


