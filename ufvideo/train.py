# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

import re
import copy
import json
import random
import pathlib
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from PIL import Image
from decord import VideoReader, cpu
# torch-related packages
# NOTE: torch must be imported before transformers. Otherwise, `Segmentation fault (core dumped)` will occur.
import torch
from torch.utils.data import Dataset

import transformers
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import sys
sys.path.append('code/UFVideo')
from ufvideo.model import *
from ufvideo.constants import *
from ufvideo.mm_utils import DirectResize, tokenizer_multimodal_token, process_video, process_image, annToMask, sam_preprocess
from ufvideo.videorefer_trainer import (VideoReferTrainer,
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, 
    find_all_linear_names, safe_save_model_for_hf_trainer
)
import numpy as np

# NOTE: fast tokenizer warning issue: https://github.com/huggingface/transformers/issues/5486   
os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ModelArguments:
    # LLM Arguments
    model_type: Optional[str] = field(default="videorefer_qwen2", metadata={"help": "Model type selected in the list: " + ", ".join(VLLMs.keys())})
    model_path: Optional[str] = field(default="")
    version: Optional[str] = field(default="v1", metadata={"help": "Version of the conversation template."})
    freeze_backbone: bool = field(default=False, metadata={"help": "Whether to freeze the LLM backbone."})
    # Connector Arguments
    mm_projector_type: Optional[str] = field(default='stc_connector_v35')
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    # Region encoder Arguments
    mm_region_encoder_type: Optional[str] = field(default='pooling')
    tune_region_encoder: bool = field(default=False)
    pretrain_region_encoder: Optional[str] = field(default=None)
    # Vision tower Arguments
    vision_tower: Optional[str] = field(default='')
    mm_vision_select_layer: Optional[int] = field(default=-2)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # SAM Arguments
    sam_pretrained: Optional[str] = field(default='')
    train_mask_decoder: bool = field(default=True)
    sam_out_dim: int = field(default=256)
    

@dataclass
class DataArguments:
    # Path Arguments
    # data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_path: List[str] = field(default_factory=list, metadata={"help": "List of paths to the training data."})
    # image_folder: Optional[str] = field(default=None)
    # video_folder: Optional[str] = field(default=None)
    data_folder: Optional[str] = field(default='')
    # Loading Arguments
    is_multimodal: bool = False
    lazy_preprocess: bool = False
    num_frames: Optional[int] = field(default=32)
    num_mask_frames: Optional[int] = field(default=2) # 32
    region_token_num: Optional[int] = field(default=8) # 32
    # Preprocess Arguments
    image_aspect_ratio: str = 'square'
    # SAM parameters
    num_frames_sam: Optional[int] = field(default=4)
    image_size_sam: Optional[int] = field(default=1024)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mm_projector_lr: Optional[float] = None
    freeze_mm_mlp_adapter: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    output_dir: str = field(default='')
    ce_loss_weight: float = field(default=1.0)
    dice_loss_weight: float = field(default=0.5)
    bce_loss_weight: float = field(default=2.0)
    # Training Data Arguments 
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    modal_token: str = None,
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    conversations = []
    input_ids = []
    targets = []
    for source in sources:
        # 1. apply chat template for input conversation
        assert len(source) == 2
        assert modal_token in source[0]['value']
        message = [
            {'role': 'user', 'content': modal_token},
            {'role': 'assistant', 'content': source[1]['value']}
        ]
        conversation = " ".join([sentence['value'] for sentence in source])

        input_id = tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt')
        target = copy.deepcopy(input_id)
        target[input_id == MODAL_INDEX_MAP[modal_token]] = IGNORE_INDEX

        input_ids.append(input_id)
        targets.append(target)

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    modal_token: str = None,
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    # Apply prompt templates
    conversations = []
    input_ids = []
    targets = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != "user":
            # Skip the first one if it is not from human
            source = source[1:]

        message = [{'role': roles[sentence['from']], 'content': sentence['value']} for sentence in source]
        conversation = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        input_ids.append(tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt'))
        targets.append(copy.deepcopy(input_ids[-1]))

        assert len(source) % 2 == 0, f"Invalid conversation length {len(source)}."

        cur = 0
        message = []
        for idx, sentence in enumerate(source):
            if idx % 2 == 1:
                tmp_message = [
                    {'role': roles[source[idx-1]['from']], 'content': source[idx-1]['value']}, 
                    {'role': roles[sentence['from']], 'content': sentence['value']}
                ]

                instruction = tokenizer.apply_chat_template(message + tmp_message[:1], tokenize=False, add_generation_prompt=True)
                conversation = tokenizer.apply_chat_template(message + tmp_message, tokenize=False, add_generation_prompt=False)

                instruction_len = len(tokenizer_multimodal_token(instruction, tokenizer, modal_token, return_tensors='pt'))
                conversation_len = len(tokenizer_multimodal_token(conversation, tokenizer, modal_token, return_tensors='pt'))

                targets[-1][cur:instruction_len] = IGNORE_INDEX

                cur = conversation_len
                message += tmp_message

    return dict(input_ids=input_ids, labels=targets)


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
    modal_token: str = None,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    assert modal_token in MODAL_INDEX_MAP, f"Unsupported modal token {modal_token}."

    for source in sources:
        for sentence in source:
            if modal_token in sentence['value']:
                sentence['value'] = sentence['value'].replace(modal_token, '').strip()
                sentence['value'] = modal_token + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            replace_token = modal_token
            # TODO: fix this for multimedia, e.g., <video>, <audio>, etc.
            sentence["value"] = sentence["value"].replace(modal_token, replace_token)

    return sources


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        # list_data_dict = json.load(open(data_path, "r"))
        list_data_dict = []

        for path in data_path:
            with open(path, "r") as file:
                data = json.load(file)
                list_data_dict.extend(data)
        
        rank0_print("data dict length: ", len(list_data_dict))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.question_prompt = QUESTION_LIST
        self.answer_prompt = ANSWER_LIST
        self.transform = DirectResize(data_args.image_size_sam)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 576 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
    
    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax  # y1, y2, x1, x2
    
    def get_dense_indices(self):
        sequence = np.arange(self.data_args.num_frames)
        random_numbers = np.random.choice(sequence, size=self.data_args.num_frames_sam, replace=False)

        return sorted(random_numbers.tolist())

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        image_processor = self.data_args.image_processor
        video_processor = self.data_args.video_processor

        num_frames = NUM_FRAMES if self.data_args.num_frames is None else self.data_args.num_frames
 
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        ann_indices = []
        frame_nums = 1
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            # print('image_file: ', image_file)
            try:
                image, height, width, _ = process_image(image_file, image_processor, self.data_args.image_aspect_ratio)
                image = image[0]
            except Exception as e:
                traceback.print_exc()
                backup_idx = random.randint(0, len(self.list_data_dict)-1)
                print(f"Encounted error when reading image {image_file}, use {backup_idx}-th example instead!!!")
                return self.__getitem__(backup_idx)
            # place <image> tag to question head.
            modal_token = "<image>"
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args, modal_token)
            
        elif 'video' in sources[0]:
            # print(sources[0]['video'])
            video_file = self.list_data_dict[i]['video']
            # print('video_file: ', video_file)
            # video_file = os.path.join(self.data_args.data_folder, video_file)

            all_frames = set()
            
            if 'seg' not in sources[0]:
                try: 
                    if 'annotation' in sources[0]:
                        if 'new_id' in sources[0]:
                            ann = random.choice(sources[0]['annotation'])
                            # 提取所有键
                            keys = list(ann.keys())
                            # selected_key = None
                            k = random.choice(keys)
                            all_frames.add(k)
                            ann_indices.append([0])  # 因为 selected_key 是唯一的，索引为0

                        else:
                            # 截断
                            for ann in sources[0]['annotation']:
                                all_frames.update(list(ann.keys()))
                            all_frames = list(all_frames)
                            frame_nums = len(all_frames)
                            for ann in sources[0]['annotation']:
                                frame_list = list(ann.keys())
                                indices = []
                                for frame in frame_list:
                                    indices.append(all_frames.index(frame))
                                ann_indices.append(indices)
                    else: 
                        all_frames.add(0)  
                        ann_indices.append([0])

                    all_frames = [int(f) for f in all_frames]
                    # 检查帧数是否超过150
                    if len(all_frames) > 150:
                        raise ValueError(f"Frame count {len(all_frames)} exceeds maximum limit of 150")
                    video, frame, height, width, _ = process_video(video_file, video_processor, aspect_ratio=self.data_args.image_aspect_ratio, num_frames=num_frames, frame_idx=all_frames) #frame [1,3,336,336]
                except Exception as e:
                    traceback.print_exc()
                    backup_idx = random.randint(0, len(self.list_data_dict)-1)
                    print(f"Encounted error when reading video {video_file}, use {backup_idx}-th example instead!!!")
                    return self.__getitem__(backup_idx)

                # place <video> tag to question head.
                modal_token = "<video>"
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args, modal_token)

                # NOTE：对于不是分割任务，可以使用下面这种方法，随机取image_sam, mask是0
                dense_indices = self.get_dense_indices()

                # pre-process for SAM
                image_sam_list = []
                for idx, image_s in enumerate(video):
                    if idx in dense_indices:
                        image_sam = self.transform.apply_image(image_s)
                        image_sam_list.append(image_sam)
                
                image_sam_list_proc = []
                for image in image_sam_list:
                    image_sam = sam_preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
                    image_sam_list_proc.append(image_sam)
                image_sam_tsr = torch.stack(image_sam_list_proc, dim=0)
                        
                masks_sam = torch.rand(0, *(height, width))
                sam_label = torch.ones((height, width)) * IGNORE_INDEX
                masks_sam = torch.cat([masks_sam] * self.data_args.num_frames_sam, dim=0)
                assert image_sam_tsr.shape[0] == self.data_args.num_frames_sam
                assert masks_sam.shape[0] == self.data_args.num_frames_sam or masks_sam.shape[0] == 0

            # 如果是分割任务
            elif 'seg' in sources[0]:
                if 'unibench' in sources[0]:
                    if sources[0]['unibench'] == 'task1':
                        try:
                            # 获取所有可用帧并排序（确保帧序号有序）
                            all_available_frames = sorted(sources[0]['frame_list'])
                            total_frames = len(all_available_frames)

                            # 计算前1/4帧的范围（至少包含1帧，最多到总帧数的1/4）
                            quarter = max(1, total_frames // 4)  # 前1/4区域至少保留1帧
                            candidate_first_frames = all_available_frames[:quarter]

                            # 筛选出后面至少有4帧的候选第一帧
                            valid_first_frames = []
                            for idx, frame in enumerate(candidate_first_frames):
                                # 计算该帧在总列表中的位置，确保后面有足够帧
                                frame_pos = all_available_frames.index(frame)
                                if frame_pos + 4 < total_frames:
                                    valid_first_frames.append(frame)

                            # 选择第一帧（如果筛选后为空，说明前1/4区域不满足，从能满足的区域选最前的帧）
                            if not valid_first_frames:
                                # 从总帧中找最后一个满足"后面有4帧"条件的位置
                                # max_first_pos = total_frames - 5  # 保证后面有4帧（索引+4 < 总长度）
                                first_frame = all_available_frames[0]
                            else:
                                first_frame = random.choice(valid_first_frames)

                            # 记录第一帧
                            all_frames.add(first_frame)
                            # frame_idx = sources[0]['frame_list'].index(first_frame)
                            # ann_indices.append([frame_idx])

                            # 获取第一帧之后的所有帧，从中随机选4帧并排序
                            first_pos = all_available_frames.index(first_frame)
                            # ann_indices.append([first_pos])
                            
                            remaining_frames = all_available_frames[first_pos + 1:]  # 只取第一帧后面的帧
                            random_numbers = sorted(random.sample(remaining_frames, k=4))  # 无需replace=False（sample默认不重复）

                            # # 添加随机帧并转换为整数列表
                            # TODO:确认是否是由于index错了
                            all_frames.update(random_numbers)
                            all_frames = [int(f) for f in all_frames]
                            ann_indices.append([0])
                            ann_indices = ann_indices * len(self.list_data_dict[i]['annotation'])
                            video, frame, height, width, frames_list = process_video(video_file, video_processor, aspect_ratio=self.data_args.image_aspect_ratio, num_frames=num_frames, frame_idx=all_frames) #frame [1,3,336,336]
                        except Exception as e:
                            traceback.print_exc()
                            backup_idx = random.randint(0, len(self.list_data_dict)-1)
                            print(f"Encounted error when reading video {video_file}, use {backup_idx}-th example instead!!!")
                            return self.__getitem__(backup_idx)

                            
                        sam_video_data = frames_list[1:]
                        frame = frame[0].unsqueeze(0)
                        # pre-process for SAM
                        image_sam_list = []
                        for image_s in sam_video_data:
                            image_sam = self.transform.apply_image(np.array(image_s))
                            image_sam_list.append(image_sam)

                        image_sam_list_proc = []
                        for image_sam in image_sam_list:
                            image_sam = sam_preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
                            image_sam_list_proc.append(image_sam)
                        image_sam_tsr = torch.stack(image_sam_list_proc, dim=0)
                        
                        sam_masks = []
                        # 对于多个物体的话暂时按顺序放
                        for n in range(len(sources[0]['annotation'])):
                            for j in random_numbers:
                                sam_mask = annToMask(sources[0]['annotation'][n][str(j)]['segmentation'])
                                sam_mask = torch.from_numpy(sam_mask)
                                sam_masks.append(sam_mask)
                        masks_sam = torch.stack(sam_masks, dim=0)
                        sam_label = torch.ones(masks_sam.shape[1], masks_sam.shape[2]) * IGNORE_INDEX

                        # place <video> tag to question head.
                        modal_token = "<video>"
                        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args, modal_token)
                    # UniBench-task2和task3处理
                    elif sources[0]['unibench'] == 'task2' or sources[0]['unibench'] == 'task3':
                        try:
                            frames_lists = sources[0]['frame_idx']
                            random_numbers = sorted(random.sample(frames_lists, k=4))

                            ann_indices.append([0])

                            all_frames = [int(f) for f in random_numbers]
                            video, frame, height, width, frames_list = process_video(video_file, video_processor, aspect_ratio=self.data_args.image_aspect_ratio, num_frames=num_frames, frame_idx=all_frames) #frame [1,3,336,336]
                        except Exception as e:
                            traceback.print_exc()
                            backup_idx = random.randint(0, len(self.list_data_dict)-1)
                            print(f"Encounted error when reading video {video_file}, use {backup_idx}-th example instead!!!")
                            return self.__getitem__(backup_idx)

                        frame = frame[0].unsqueeze(0)
                        sam_video_data = frames_list
                        # pre-process for SAM
                        image_sam_list = []
                        for image_s in sam_video_data:
                            image_sam = self.transform.apply_image(np.array(image_s))
                            image_sam_list.append(image_sam)

                        image_sam_list_proc = []
                        for image_sam in image_sam_list:
                            image_sam = sam_preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
                            image_sam_list_proc.append(image_sam)
                        image_sam_tsr = torch.stack(image_sam_list_proc, dim=0)
                        
                        sam_masks = []
                        # 对于多个物体的话暂时按顺序放
                        for n in range(len(sources[0]['annotation'])):
                            for j in random_numbers:
                                sam_mask = annToMask(sources[0]['annotation'][n][str(j)]['segmentation'])
                                sam_mask = torch.from_numpy(sam_mask)
                                sam_masks.append(sam_mask)
                        masks_sam = torch.stack(sam_masks, dim=0)
                        sam_label = torch.ones(masks_sam.shape[1], masks_sam.shape[2]) * IGNORE_INDEX

                        # place <video> tag to question head.
                        modal_token = "<video>"
                        sources = preprocess_multimodal(copy.deepcopy([e["conversations"][0] for e in sources]), self.data_args, modal_token)
                
                else:
                    try:
                        all_frames.add(0)
                        ann_indices.append([0])

                        all_frames = [int(f) for f in all_frames]
                        all_frames = sources[0]['no_none_frame_idx']
                        video, frame, height, width, _ = process_video(video_file, video_processor, aspect_ratio=self.data_args.image_aspect_ratio, num_frames=num_frames, frame_idx=all_frames) #frame [1,3,336,336]
                        frame = frame[0].unsqueeze(0)
                    except Exception as e:
                        traceback.print_exc()
                        backup_idx = random.randint(0, len(self.list_data_dict)-1)
                        print(f"Encounted error when reading video {video_file}, use {backup_idx}-th example instead!!!")
                        return self.__getitem__(backup_idx)

                    sequence = sources[0]['no_none_frame_idx']
                    # print('sequence: ', sequence)
                    # print('video path: ', video_file)
                    try:
                        random_numbers = sorted(np.random.choice(sequence, size=self.data_args.num_frames_sam, replace=False))
                    except:
                        print('sequence: ', sequence)
                        print('video path: ', video_file)
                        
                    sam_video_files = [video_file[x] for x in random_numbers]
                    sam_video_data = [Image.open(f) for f in sam_video_files]
                    # pre-process for SAM
                    image_sam_list = []
                    for image_s in sam_video_data:
                        image_sam = self.transform.apply_image(np.array(image_s))
                        image_sam_list.append(image_sam)

                    image_sam_list_proc = []
                    for image_sam in image_sam_list:
                        image_sam = sam_preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
                        image_sam_list_proc.append(image_sam)
                    image_sam_tsr = torch.stack(image_sam_list_proc, dim=0)
                    
                    sam_masks = []
                    for j in random_numbers:
                        sam_mask = annToMask(sources[0]['segmentations'][j])
                        sam_mask = torch.from_numpy(sam_mask)
                        sam_masks.append(sam_mask)
                    masks_sam = torch.stack(sam_masks, dim=0)
                    sam_label = torch.ones(masks_sam.shape[1], masks_sam.shape[2]) * IGNORE_INDEX

                    assert image_sam_tsr.shape[0] == masks_sam.shape[0] == self.data_args.num_frames_sam

                    # place <video> tag to question head.
                    question_template = random.choice(self.question_prompt)
                    question = question_template.format(class_name=sources[0]['conversations'][0])
                    answer = random.choice(self.answer_prompt)
                    message = [[{'from': 'human', 'value': f'<video>\n{question}'}, {'from': 'gpt', 'value': answer}]]
                    modal_token = "<video>"
                    sources = preprocess_multimodal(message, self.data_args, modal_token)
                

        else:
            modal_token = None
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # print(sources)
        masks = []
        if 'seg' in self.list_data_dict[i] and 'unibench' in self.list_data_dict[i]:
            if self.list_data_dict[i]['unibench'] == 'task1':
                for ann in self.list_data_dict[i]['annotation']:
                    mask = annToMask(ann[str(first_frame)]['segmentation'])
                    masks.append(mask)
                masks = np.array(masks)
            else:
                masks = np.zeros((1, 336, 336))
        else:
            if 'annotation' in self.list_data_dict[i]:
                if 'height' in self.list_data_dict[i]:
                    h = self.list_data_dict[i]['height']
                    w = self.list_data_dict[i]['width']
                else:
                    h = None
                    w = None
                if 'image' in self.list_data_dict[i]:
                    mask = annToMask(self.list_data_dict[i]['annotation'])
                    masks.append(mask)
                    ann_indices = [[0]]
                
                else:
                    for anns in self.list_data_dict[i]['annotation']:
                        for ann_idx in anns.keys():
                            if anns[ann_idx]['segmentation'] is None:
                                mask = np.zeros((height, width))
                            else:
                                mask = annToMask(anns[ann_idx]['segmentation'], h, w)
                            masks.append(mask)
                        
                masks = np.array(masks)      
            else:
                masks = np.zeros((1, 336, 336))
            
    
        if self.data_args.is_pretraining:
            data_dict = preprocess_plain(sources, self.tokenizer, modal_token=modal_token)
        else:
            data_dict = preprocess(sources, self.tokenizer, modal_token=modal_token)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['frame'] = image.unsqueeze(0)
            data_dict['image_sam'] = image_sam_tsr
            data_dict['masks_list'] = masks_sam
            data_dict['label_list'] = sam_label
            data_dict['video_file'] = image_file
        elif 'video' in self.list_data_dict[i]:
            data_dict['video'] = video
            data_dict['frame'] = frame 
            data_dict['image_sam'] = image_sam_tsr
            data_dict['masks_list'] = masks_sam
            data_dict['label_list'] = sam_label
            data_dict['video_file'] = video_file
        elif self.data_args.is_multimodal:
            data_dict['image'] = torch.zeros(3, self.data_args.image_size, self.data_args.image_size)
            data_dict['frame'] = torch.zeros(1, 3, self.data_args.image_size, self.data_args.image_size)

        data_dict['frame_nums'] = frame_nums

        data_dict['masks'] = torch.Tensor(masks)
        if len(ann_indices)==0:
            ann_indices = [[0]]
        data_dict['ann_indices'] = ann_indices
        

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, masks, frame, ann_indices, frame_nums, image_sam, masks_list, label_list, video_file = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "masks", "frame", "ann_indices", "frame_nums", "image_sam", "masks_list", "label_list", "video_file"))
        
        # TODO:加上offset，mask_list, image_sam, label_list, 确定shape
        cur_frame_num = 0
        for i, num in enumerate(frame_nums):
            ann_indices[i] = [[x + cur_frame_num for x in sublist] for sublist in ann_indices[i]] 
            cur_frame_num += int(num)
        
        offset_list = [x for x in range(len(instances)+1)]


        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            masks=masks,
            frame=frame,
            ann_indices=ann_indices,
            frame_nums=frame_nums,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            offset=offset_list,
            images_sam=torch.stack(image_sam, dim=0),
            masks_list=masks_list,
            label_list=label_list,
            video_file=video_file
        )
        
        batch['images'] = []
        for instance in instances:
            for modal_token in MODAL_INDEX_MAP.keys():
                modal_token = modal_token.lower()
                # MODAL_TOKEN shape like: <image>, <video>, ...
                modal_name = re.findall(f'[<](.*)[>]', modal_token)
                assert len(modal_name) == 1
                modal_name = modal_name[0]
                if modal_name in instance:
                    batch['images'].append((instance[modal_name], modal_name))

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank
    set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            # device_map={"": training_args.device},
            # BUG: High version transformers report error: 
            # ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time
            # load_in_4bit=training_args.bits == 4,
            # load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type, # {'fp4', 'nf4'}
                bnb_4bit_quant_storage=compute_dtype,
            )
        ))

    config = VLLMConfigs[model_args.model_type].from_pretrained(model_args.model_path, trust_remote_code=True)
    config._attn_implementation = attn_implementation
    config.train_mask_decoder = model_args.train_mask_decoder

    # 加载模型
    if model_args.vision_tower is not None:
        model = VLLMs[model_args.model_type].from_pretrained(
            model_args.model_path,
            config=config,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            do_sample=True,
            **bnb_model_from_pretrained_args
        )
        if 'mixtral' in model_args.model_type:
            # import deepspeed
            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_path,
            config=config,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            do_sample=True,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    # 冻结backbone
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    # 训练精度设置
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # 梯度检查
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # 设置lora微调参数
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # 加载tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    # 加载视觉模块
    if model_args.vision_tower is not None:
        # initialize vision encoder + multi-modal projector
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_size = vision_tower.image_size

        data_args.image_processor = vision_tower.image_processor
        data_args.video_processor = vision_tower.video_processor if hasattr(vision_tower, "video_processor") else vision_tower.image_processor

        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        # 配置模型的多模态MLP适配器调优参数
        # 将tune_mm_mlp_adapter参数同步设置到model.config和training_args中
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.tune_region_encoder = training_args.tune_region_encoder = model_args.tune_region_encoder
    
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        
        if model_args.tune_region_encoder:
            model.requires_grad_(False)
            for p in model.get_model().region_encoder.parameters():
                p.requires_grad = True
                
        if model_args.tune_mm_mlp_adapter:
            data_args.is_pretraining = True
        else:
            data_args.is_pretraining = False

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.ce_loss_weight = training_args.ce_loss_weight
        model.config.dice_loss_weight = training_args.dice_loss_weight
        model.config.bce_loss_weight = training_args.bce_loss_weight
        model.config.num_frames = NUM_FRAMES if data_args.num_frames is None else data_args.num_frames
        rank0_print('tokenizer length: ', len(tokenizer))
        model.initialize_MM_tokenizer(tokenizer=tokenizer)
        model.config.seg_token_id = tokenizer.convert_tokens_to_ids('[SEG]')
    
    model_args.hidden_size = config.hidden_size
    if model_args.sam_pretrained is not None:
        model.get_model().initialize_sam_modules(model_args)
    
    rank0_print('new tokenizer length: ', len(tokenizer))

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    temp_name = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            temp_name.append(name)
    rank0_print(f"Trainable parameters: {temp_name}")
    rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
    rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")
    rank0_print(f"Traning args: ", training_args)
    rank0_print(f"Model args: ", model_args)
    rank0_print(f"Data args: ", data_args)

    # print("Current model:", model)
    # print("Model config:", model.config)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # print("Current data_module:", data_module)
    # select a Trainer
    trainer = VideoReferTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    print("Saving the final model")
    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train("flash_attention_2")
