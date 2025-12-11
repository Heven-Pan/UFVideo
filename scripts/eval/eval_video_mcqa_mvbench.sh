EVAL_DATA_DIR=
OUTPUT_DIR=
CKPT=
CKPT_NAME=

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

ARG_WORLD_SIZE=${3:-1}
ARG_NPROC_PER_NODE=${4:-8}

ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=${6:-0}

if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    eval/inference_video_mcqa_mvbench.py \
    --model-path ${CKPT} \
    --video-folder ${EVAL_DATA_DIR}/mvbench/video \
    --question-file ${EVAL_DATA_DIR}/mvbench/json \
    --answer-file ${OUTPUT_DIR}/mvbench/answers/${CKPT_NAME} \
    --num-chunks 8 \


python3 eval/eval_video_mcqa_mvbench.py \
    --pred_path ${OUTPUT_DIR}/mvbench/answers/${CKPT_NAME}.json \
