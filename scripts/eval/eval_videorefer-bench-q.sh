CKPT=
MODEL_NAME=
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
    eval/inference_videorefer_q_bench.py \
    --video-folder VideoRefer-Bench \
    --question-file VideoRefer-Bench/VideoRefer-Bench-Q.json \
    --model-path ${CKPT} \
    --output-file output/videorefer-bench-q/${MODEL_NAME} \
    --num-chunks $ARG_NPROC_PER_NODE \



python3 eval/eval_videorefer_bench_q.py --pred-path output/videorefer-bench-q/${MODEL_NAME}.json