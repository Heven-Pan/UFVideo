import math
import os
import argparse
import json
import warnings
from tqdm import tqdm
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import torch
import sys
sys.path.append('code/UFVideo')
from ufvideo import model_init, mm_infer
from pycocotools import mask as maskUtils
import numpy as np
from ufvideo.mm_utils import process_video
from functools import partial
from matplotlib import pyplot as plt
from PIL import Image
from ufvideo.utils import disable_torch_init        
import pycocotools.mask as maskUtils
from torch.utils.data import Dataset, DataLoader
import cv2

def annToMask(mask_ann, h=None, w=None):
    if isinstance(mask_ann, list):
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def sam_preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    # h, w = x.shape[-2:]
    # padh = img_size - h
    # padw = img_size - w
    # x = F.pad(x, (0, padw, 0, padh))
    return x

def temporal_iou(A, B):
    max0 = max((A[0]), (B[0]))
    min0 = min((A[0]), (B[0]))
    max1 = max((A[1]), (B[1]))
    min1 = min((A[1]), (B[1]))
    _iou = max(min1 - max0, 0) / (max1 - min0)
    return max(0, _iou)

from torchvision.transforms.functional import resize, to_pil_image
class DirectResize:
    def __init__(self, target_length: int):
        self.target_length = target_length

    def apply_image(self, image: np.ndarray):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))


class VideoRefer_Bench_D(Dataset):
    def __init__(self, video_folder, data_list, processor, mode):
        self.video_folder = video_folder
        self.data_list = data_list
        self.processor = processor
        self.mode = mode
        self.transform = DirectResize(1024)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        video_path = line['video']


        question = line['conversations'][0][0]['value']
        caption = line['conversations'][0][1]['value']
        # question = question + "When is <event> occur in the video? Only give the start and end timestamp."
        # question = 'There is 1 objects in the video: object_1: [<region>]. Please give a detailed description of what is the object_1 doing in the video. And please generate the mask in every frames?'
        video_name = line['video']
        annotations = line['annotation']
        
        # if self.mode=='single':
        #     frame_idx = str(line['frame_idx'])
        #     annotations_single = []
        #     for ann in annotations:
        #         annotations_single.append({frame_idx: ann[frame_idx]})
        #     annotations = annotations_single
        
        all_frames = line['frame_idx']

        video_tensor, frame_data, height, width, frames_list = process_video(video_path, processor=self.processor, aspect_ratio='square', frame_idx=all_frames)

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


        return {
            'video_name': line['video'],
            'video': video_tensor,
            'question': question,
            'images_sam': image_sam_tsr,
            'height': height,
            'width': width, 
            'total_frame': len(line['frame_idx']),
            'id': line['id'],
            'caption': caption
        }

def collate_fn(batch):
    vin = [x['video_name'] for x in batch]
    vid = [x['video'] for x in batch]
    qs = [x['question'] for x in batch]
    img_sam = [x['images_sam'] for x in batch]
    h = [x['height'] for x in batch]
    w = [x['width'] for x in batch]
    tf = [x['total_frame'] for x in batch]
    id = [x['id'] for x in batch]
    cap = [x['caption'] for x in batch]
    return vin, vid, qs, img_sam, h, w, tf, id, cap

def build_videorefer_bench_d_eval(args, processor, local_rank):
    # convert parquet to json
    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, local_rank)
    dataset = VideoRefer_Bench_D(args.video_folder, questions, processor, args.mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return dataloader

import re
def replace_and_normalize(input_str, return_token=False):
    pattern = re.compile(r'(<WIDTH-(\d+)>|<HEIGHT-(\d+)>|<TEMP-(\d+)>)')

    def normalize(match):
        if match.group(2):
            value = int(match.group(2))
        elif match.group(3):
            value = int(match.group(3))
        elif match.group(4):
            value = int(match.group(4))

        normalized_value = value / 99.0

        if return_token:
            return '{:d},'.format(value)
        return '{:.5f},'.format(normalized_value)

    # 使用 re.sub 进行替换，调用 normalize 函数进行处理
    result_str = re.sub(pattern, normalize, input_str)

    return result_str.replace(",]", "]").replace(",}", "}")

def parse_span_from_text(s):
    pattern = r"{\s*(\d+(?:\.\d+)?)\,\s*(\d+(?:\.\d+)?)\s*}"
    # pattern = r"{\s*<time_start>\s*(\d+(?:\.\d+)?)\s*<time_end>\s*<time_start>\s*(\d+(?:\.\d+)?)\s*<time_end>\s*}"
    match = re.search(pattern, s)
    if match:
        start_time = float(match.group(1))
        end_time = float(match.group(2))
        return [start_time, end_time]
    else:
        print("No match found.")
        return [0, 0]

def compare_lists(list1, list2):
    """
    比较两个list的0位和1位差值，选择差值更小的位置
    如果0位差值更小，返回两个list的0位最大值
    如果1位差值更小，返回两个list的1位最小值
    """
    # 计算0位和1位的差值
    diff_0 = abs(list1[0] - list2[0])
    diff_1 = abs(list1[1] - list2[1])
    
    # # 比较差值大小
    # if diff_0 < diff_1:
    #     result = max(list1[0], list2[0])
    #     res2 = min(list1[1], list2[1])
    #     return 0, result, [result, res2]
    # elif diff_1 < diff_0:
    #     res1 = max(list1[0], list2[0])
    #     result = min(list1[1], list2[1])
    #     return 1, result, [res1, result]
    # else:
    return max(list1[0], list2[0]), [max(list1[0], list2[0]), min(list1[1], list2[1])]




import datetime
import torch.distributed as dist
import traceback

def run_inference(args):
    dist.init_process_group(backend="gloo", timeout=datetime.timedelta(minutes=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()
    # local_rank = 0
    # global_rank = 0
    disable_torch_init()

    model, processor, tokenizer = model_init(args.model_path, device_map={"": f"cuda:{local_rank}"})

    for m in model.modules():
        m.tokenizer = tokenizer
    
    # for m in model.modules():
    #     m.tokenizer = tokenizer

    model = model.to(device='cuda', dtype=torch.float16)

    answer_file = os.path.expanduser(f"{args.output_file}_rank{global_rank}.json")
    time_answer_file = os.path.expanduser(f"{args.time_output_file}_rank{global_rank}.json")
    print(f"Rank {global_rank} writing to {answer_file}")
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    os.makedirs(os.path.dirname(time_answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")
    time_ans_file = open(time_answer_file, "w")
    val_loader = build_videorefer_bench_d_eval(args, processor, local_rank)
    
    final_data = []
    for i, (video_names, video, questions, img_sam, h, w, tf, d, cap) in enumerate(tqdm(val_loader, desc=f"Rank {global_rank}", position=local_rank)):
        video_name = video_names[0]
        video_tensor = video[0]
        question = questions[0]
        images_sam = img_sam[0].unsqueeze(0)
        height = h[0]
        width = w[0]
        masks_sam = torch.rand(0, *(height, width))
        sam_label = torch.ones((height, width)) * (-200)
        masks_sam = torch.cat([masks_sam] * 4, dim=0)
        total_frames = tf[0]
        id = d[0]
        caption = cap[0]
        
        output, pred = mm_infer(
            video_tensor,
            question,
            model=model,
            tokenizer=tokenizer,
            choice=2,
            images_sam=images_sam,
            masks_list=masks_sam,
            offset=[0,1],
            label_list=sam_label,
            seg=False
        )
        # print(output)
        gt_temporal = replace_and_normalize(caption)
        out_temporal = replace_and_normalize(output)
        gt_temporal = parse_span_from_text(gt_temporal)
        out_temporal = parse_span_from_text(out_temporal)
        
        record = {
            'video': video_name,
            'caption': caption.split('.', 1)[1],
            'pred': output.split('.', 1)[1].split('The segmentation mask')[0],
        }
        ans_file.write(json.dumps(record) + "\n")
        time_ans_file.write(json.dumps({"pred": out_temporal, "gt": gt_temporal}) + '\n')

        # sample_iou = temporal_iou(out_temporal, gt_temporal)
        # s = 0
        pred_masks = pred['pred_masks']
        
        # print(f'saving the mask to {ans_file} ...')

        for i, pred_mask_vid in enumerate(pred_masks):
            if pred_mask_vid.shape[0] == 0:
                continue

            assert total_frames == pred_mask_vid.shape[0]

            mask_file = os.path.join(args.mask_output_file, str(id), str(i))
            os.makedirs(mask_file, exist_ok=True)

            for frame_idx in range(total_frames):
                pred_mask = pred_mask_vid.detach().cpu().numpy()[frame_idx]
                pred_mask = pred_mask > 0

                save_path = "{}/{}.png".format(mask_file, frame_idx)
                binary_mask = np.where(pred_mask > 0, 1, 0)
                cv2.imwrite(save_path, binary_mask * 255)
        
    ans_file.close()
    time_ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', default='')
    parser.add_argument('--video-folder', help='Directory containing video files.', default='')
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', default='')
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', default='')
    parser.add_argument('--time_output_file', help='Directory to save the model results JSON.', default='')
    parser.add_argument('--mask_output_file', help='Directory to save the model results JSON.', default='')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mode", type=str, default='single')
    args = parser.parse_args()

    run_inference(args)
