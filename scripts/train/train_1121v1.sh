#!/bin/bash
# 激活 conda 环境
source miniconda3/bin/activate VideoRefer

# 检查 ffprobe 版本以确保路径正确
ffprobe -version

# # Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=${3:-0}

# Multiple conditions
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


# Training Arguments
GLOBAL_BATCH_SIZE=384
LOCAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]

# Log Arguments
export TRANSFORMERS_OFFLINE=1
RUN_NAME=videorefer

# Image QA
IMAGE_DIR=data/llava_v1_5_mix665k_onlyMM_filtered.json
# Video QA
VIDEO_QA_DIR=data/llava-st_qa.json
VIDEO_QA2_DIR=data/LLaVA-Video-178K/LLava-Video-178K_QA.json
# Video referring + multi frames + single/multi objects
VIDEO4_DIR=data/VideoRefer-700K/videorefer-detailed-caption-125k_new.json
VIDEO5_DIR=data/VideoRefer-700K/videorefer-qa-75k_new_filter.json
# Video referring + single frames + single objects
VIDEO6_DIR=data/VideoRefer-700K/videorefer-short-caption-500k_new.json  
VIDEO7_DIR=data/VideoRefer-700K/videorefer-detailed-caption-125k_new_id.json  

# Temporal video grounding
TVG_DIR=data/ST-Grounding/stage2/vtime_1015.json
# Dense video caption
DVC1_DIR=data/ST-Grounding/stage2/momentor_1015.json
DVC2_DIR=data/ST-Grounding/stage2/intervid_G_1015.json


# Temporal grounded conversation
TGC1_DIR=data/ST-Grounding/stage3/activitynet_rtl_1017.json
TGV2_DIR=data/ST-Grounding/stage3/grounded_qa_1017.json
TGV3_DIR=data/ST-Grounding/stage3/momentor_1017.json
# Temporal video grounding
TVG1_DIR=data/ST-Grounding/stage3/didemo_1017.json
TVG2_DIR=data/ST-Grounding/stage3/hirest_1017.json
TVG3_DIR=data/ST-Grounding/stage3/queryd_1017.json
TVG4_DIR=data/ST-Grounding/stage3/vtg_it_1017.json
# Dense video caption
DVC21_DIR=data/ST-Grounding/stage3/coin_1017.json
DVC22_DIR=data/ST-Grounding/stage3/vitt_1017.json
DVC23_DIR=data/ST-Grounding/stage3/youcook_dvc_1017.json

ANET_v1_expand_DIR=data/DisTime-Data/anet_grounding_37k_TIME_STAMP_internvl2_v1_expand.json
ANET_v1_DIR=data/DisTime-Data/anet_grounding_37k_TIME_STAMP_internvl2_v1.json
ET_DIR=data/DisTime-Data/et_instruct_136k_TIME_STAMP_inetrnvl2_filter.json
INTERVID_DIR=data/DisTime-Data/internvid_86k_distime.json

# Seg
SEG1_DIR=data/seg_data/refyoutube_train.json
SEG2_DIR=data/seg_data/mevis_train.json
SEG3_DIR=data/seg_data/refdavis_train.json
SEG4_DIR=data/seg_data/ReVOS_train.json
SEG5_DIR=data/seg_data/ref-sav_train.json
SEG6_DIR=data/seg_data/refcoco_train_processed.json
SEG7_DIR=data/seg_data/refcoco_plus_train_processed.json
SEG8_DIR=data/seg_data/refcocog_train_processed.json

UNIBENCH1_DIR=data/UniBench/task1_obj3.json
UNIBENCH2_DIR=data/UniBench/task2_obj3.json
UNIBENCH3_DIR=data/UniBench/task3.json

# QA + Refer + Temp (1/3)
SAMPLED_DATA=data/VideoRefer-700K/three_tasks_sampled_data.json
# QA 1/2
SAMPLED_QA_DATA=data/QA_tasks_sampled_data.json
# Temp 3/5
SAMPLED_TEMP_DATA=data/ST-Grounding/Temp_tasks_sampled_data.json



DATA_PATHS=("$VIDEO_QA_DIR" "$VIDEO_QA2_DIR" "$VIDEO4_DIR" "$VIDEO5_DIR" "$VIDEO5_DIR" "$VIDEO6_DIR" "$VIDEO7_DIR" "$TVG_DIR" "$DVC1_DIR" "$DVC2_DIR" "$TGC1_DIR" "$TGV2_DIR" "$TGV3_DIR" "$TVG1_DIR" "$TVG2_DIR" "$TVG3_DIR" "$TVG4_DIR" "$DVC21_DIR" "$DVC22_DIR" "$DVC23_DIR" "$ANET_v1_expand_DIR" "$ANET_v1_DIR" "$ET_DIR" "$INTERVID_DIR" "$UNIBENCH1_DIR" "$UNIBENCH2_DIR" "$UNIBENCH2_DIR" "$UNIBENCH2_DIR" "$UNIBENCH2_DIR" "$UNIBENCH2_DIR" "$UNIBENCH3_DIR" "$UNIBENCH3_DIR" "$UNIBENCH3_DIR" "$UNIBENCH3_DIR" "$UNIBENCH3_DIR" "$SEG1_DIR" "$SEG2_DIR" "$SEG3_DIR" "$SEG4_DIR" "$SEG1_DIR" "$SEG2_DIR" "$SEG3_DIR" "$SEG4_DIR" "$SEG1_DIR" "$SEG2_DIR" "$SEG3_DIR" "$SEG4_DIR" "$SEG1_DIR" "$SEG2_DIR" "$SEG3_DIR" "$SEG4_DIR" "$SEG1_DIR" "$SEG2_DIR" "$SEG3_DIR" "$SEG4_DIR" "$SEG3_DIR" "$SEG3_DIR" "$SEG3_DIR" "$SEG3_DIR" "$SEG3_DIR" "$SEG1_DIR" "$SEG1_DIR" "$SEG5_DIR" "$SEG6_DIR" "$SEG7_DIR" "$SEG8_DIR")



miniconda3/envs/VideoRefer/bin/torchrun \
    --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE  \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    code/UFVideo/videorefer/train.py \
    --deepspeed code/UFVideo/scripts/zero2.json \
    --model_type videorefer_qwen2 \
    --model_path   \
    --vision_tower data/ckpts/VideoRefer/ckpts/siglip-so400m-patch14-384 \
    --sam_pretrained data/ckpts/VideoRefer/sam2-hiera-large/sam2_hiera_large.pt \
    --train_mask_decoder True \
    --mm_projector_type stc_connector_v35 \
    --data_path ${DATA_PATHS[@]} \
    --image_aspect_ratio square \
    --mm_vision_select_layer -2 \
    --mm_region_encoder_type pooling \
    --num_frames 32 \
    --num_frames_sam 4 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --lora_enable False \
    --freeze_backbone False \
    --output_dir data/output/UFVideo_ckpt \
    --num_train_epochs 2 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 4 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --report_to tensorboard \
    --run_name $RUN_NAME \

