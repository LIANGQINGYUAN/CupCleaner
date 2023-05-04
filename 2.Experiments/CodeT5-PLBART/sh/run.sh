gpu=$1
subtask=$2

GPUID=$gpu
BATCH_SIZE=10
MODEL_TAG=codet5_base
SUB_TASK=$subtask

python run_exp.py \
    --model_tag $MODEL_TAG \
    --task comment_update \
    --sub_task $SUB_TASK \
    --gpu $GPUID \
    --bs $BATCH_SIZE