import os
import re
import math
import json
import argparse
import warnings
import traceback
import random

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('code/UFVideo')
from ufvideo import model_init, mm_infer
from ufvideo.utils import disable_torch_init
from functools import partial
from ufvideo.mm_utils import process_video
from ufvideo.constants import *

# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


# tvg_prompts = [
#     ("Give you a textual query: '<event>'. When does the described content "
#      "occur in the video? Please return the start and end timestamps.")
# ]
tvg_prompts = [
    ("When is <event> occur in the video? Only give the start and end timestamp.")
]

def format_1d_box(text, ):
    pattern = r"{\s*(\d+(?:\.\d+)?)\,\s*(\d+(?:\.\d+)?)\s*}"
    match = re.search(pattern, text)
    if match:
        start_time = float(match.group(1))
        end_time = float(match.group(2))
        return start_time, end_time
    else:
        # print("No match found.")
        return None

def format_2d_box(text):
    pattern = (r"\[\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),"
               r"\s*(\d+(?:\.\d+)?)\s*\]")
    match = re.search(pattern, text)
    if match:
        a = float(match.group(1))
        b = float(match.group(2))
        c = float(match.group(3))
        d = float(match.group(4))
        return [a, b, c, d]
    else:
        # print("No match found.")
        return None
    

def bbox_post_refine(bbox, height, width):
    if height >= width:
        x1, y1, x2, y2 = (i * height for i in bbox)
        pad = (height - width) // 2
        x1 -= pad
        x2 -= pad
    else:
        x1, y1, x2, y2 = (i * width for i in bbox)
        pad = (width - height) // 2
        y1 -= pad
        y2 -= pad
    res = [x1 / width, y1 / height, x2 / width, y2 / height]
    return res

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def format_box_in_text(text: str, pad_proc=False, **kwargs):
    box = format_2d_box(text)
    if box is None:
        return text
    in_out = kwargs["in_out"]
    pattern = (r"\[\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),"
               r"\s*(\d+(?:\.\d+)?)\s*\]")

    def replace_inside_braces(match):
        a = float(match.group(1))
        b = float(match.group(2))
        c = float(match.group(3))
        d = float(match.group(4))
        box = [a, b, c, d]
        if pad_proc:
            box = bbox_post_refine(box, kwargs["h"], kwargs["w"])

        def clip(x):
            return x if x >= 0 and x <= 1 else (0 if x < 0 else 1)

        box = [round(clip(x), 3) for x in box]
        return (f" [<WIDTH-{in_out}{box[0]}><HEIGHT-{in_out}{box[1]}>"
                f"<WIDTH-{in_out}{box[2]}><HEIGHT-{in_out}{box[3]}>]")

    res = re.sub(pattern, replace_inside_braces, text)
    return res


def format_span_in_text(text: str, in_out: str):
    span = format_1d_box(text)
    if span is None:
        return text
    pattern = r"{\s*(\d+(?:\.\d+)?)\,\s*(\d+(?:\.\d+)?)\s*}"

    def replace_inside_braces(match):
        s = float(match.group(1))
        e = float(match.group(2))
        if s >= e:
            print(f"start {s} >= end {e}")
            return "{<error>}"
        return (f" {'{'}<TEMP-{in_out}{round(s,3)}>"
                f"<TEMP-{in_out}{round(e,3)}>{'}'} ")

    res = re.sub(pattern, replace_inside_braces, text)
    return res


def format_float_in_text(text: str, in_out: str):
    pattern = r"Starts in (\d+(?:\.\d+)?)"

    def replace_inside_braces(match):
        s = float(match.group(1))
        return f" Starts in <TEMP-{in_out}{round(s,3)}> "

    res = re.sub(pattern, replace_inside_braces, text)
    return res


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

def seperate_token_number(text, token):
    pattern = rf'<{token}(.*?)>'
    matches = re.finditer(pattern, text)
    lis = [(i.group(1), i.start()) for i in matches]
    lis = sorted(lis, key=lambda x: x[-1])
    values = []
    for k, _ in lis:
        text = re.sub(re.escape(f"<{token}{k}>"), f"<{token}>", text, count=1)
        values.append(eval(k))
    return text, values

def get_variables(conversations):
    # Extract numerical information from tokens
    variables_dict = {
        "TEMP-INPUT": [],
        "TEMP-OUTPUT": [],
        "HEIGHT-INPUT": [],
        "HEIGHT-OUTPUT": [],
        "WIDTH-INPUT": [],
        "WIDTH-OUTPUT": []
    }
    for con in conversations:
        text = con["value"]
        if text is None:
            continue
        for key in variables_dict.keys():
            text, lis = seperate_token_number(text, key)
            variables_dict[key].extend(lis)
        con["value"] = text
    variables = {
        "temporal_input_locations": variables_dict["TEMP-INPUT"],
        "temporal_output_locations": variables_dict["TEMP-OUTPUT"],
        "spatial_height_input_locations": variables_dict["HEIGHT-INPUT"],
        "spatial_height_output_locations": variables_dict["HEIGHT-OUTPUT"],
        "spatial_width_input_locations": variables_dict["WIDTH-INPUT"],
        "spatial_width_output_locations": variables_dict["WIDTH-OUTPUT"]
    }
    return conversations, variables


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

import traceback


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

class CharadesSTGDataset(Dataset):

    def __init__(self, data_list, processor):
        self.data_list = data_list
        self.data_folder = ''
        self.processor = processor
        self.transform = DirectResize(1024)

    def __len__(self):
        return len(self.data_list)

    def __getitem(self, idx):
        data = self.data_list[idx]

        img_id = data['image_id']
        caption = data['caption'].strip(".")
        caption = caption.strip(" ").lower()
        video_path = os.path.join(self.data_folder, img_id)
        # video_path = img_id
        torch_imgs, _, height, width, frames_list = self.processor(video_path, frame_idx=[0])

        prompt = DEFAULT_VIDEO_TOKEN + "\n" + random.choice(
            tvg_prompts).replace("<event>", caption)
        # prompt = caption
        gt = data["timestamp"]


        sam_video_data = frames_list
        # pre-process for SAM
        image_sam_list = []
        for image_s in sam_video_data:
            image_sam = self.transform.apply_image(np.array(image_s))
            image_sam_list.append(image_sam)

        image_sam_list_proc = []
        for idx in range(4):
            image_sam = image_sam_list[0]
            image_sam = sam_preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
            image_sam_list_proc.append(image_sam)
        image_sam_tsr = torch.stack(image_sam_list_proc, dim=0)

        return {
            'video': torch_imgs, 
            'video_path': video_path,
            'instruct': prompt,
            'gt': gt,
            'images_sam': image_sam_tsr,
            'height': height,
            'width': width
        }

    def __getitem__(self, idx):
        try:
            return self.__getitem(idx)
        except Exception:
            traceback.print_exc()
            return self.__getitem__(idx + 1)

def build_charades_eval(args, processor, local_rank):
    data_list = []
    json_file = args.question_file
    vis_folder = args.video_folder
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    for data in json_data:
        data_list.append({
            'image_id': data['image_id'],
            'caption': data['caption'],
            'timestamp': data['timestamp'],
        })
    data_list = get_chunk(data_list, args.num_chunks, local_rank)
    dataset = CharadesSTGDataset(data_list, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return dataloader


def mvbench_dump(vid, instruct, letters, options, output):
    
    output = output.replace('answer', '')
    output = output.replace('Answer', '')
    pred_answer = re.findall(f'[\(,\ ]*[{letters[0]}-{letters[-1]}][\),\ ]*', output)
    try:
        find_flag = False
        if len(pred_answer) == 0:
            for idx, opt in enumerate(options):
                # Arabic numerals -> English words
                if opt.lower() in output.lower():
                    pred_idx = idx
                    find_flag = True
                    break
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
            find_flag = True

        assert find_flag, 'The video \"{}\" instruct: \n\"{}\"\n output: \n\"{}\"\n is not in the expected format'.format(vid, instruct, output)
    except:
        traceback.print_exc()
        pred_idx = 2
    
    return pred_idx


import re

def extract_times(text):
    # 用正则表达式来匹配<time_start>和<time_end>之间的数字
    pattern = r'<time_start>(.*?)<time_end>'
    matches = re.findall(pattern, text)
    # 将匹配的结果转换为整数
    times = [int(match) for match in matches]
    return times



import datetime
import torch.distributed as dist
import traceback

def run_inference(args):
    dist.init_process_group(backend="gloo", timeout=datetime.timedelta(minutes=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()
    disable_torch_init()
    # local_rank = 0
    # global_rank = 0

    model, processor, tokenizer = model_init(args.model_path, device_map={"": f"cuda:{local_rank}"}, lora=args.lora_enable)

    
    for m in model.modules():
        m.tokenizer = tokenizer
    
    answer_file = os.path.expanduser(f"{args.answer_file}_rank{global_rank}.json")
    print(f"Rank {global_rank} writing to {answer_file}")
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")
    
    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES
    processor = partial(process_video, processor=processor, aspect_ratio='square', num_frames=num_frames)

    val_loader = build_charades_eval(args, processor, local_rank)
    print(len(val_loader))

    # NOTE: only support batch size 1 for now
    for i, line in enumerate(tqdm(val_loader, desc=f"Rank {global_rank}", position=local_rank)):
        vid = line['video_path'][0]
        video_tensor = line['video'][0]
        instruct  = line['instruct'][0]
        gt = [line['gt'][0].item(), line['gt'][1].item()]
        images_sam = line['images_sam']
        height = line['height'][0].item()
        width = line['width'][0].item()
        masks_sam = torch.rand(0, *(height, width))
        sam_label = torch.ones((height, width)) * (-200)
        masks_sam = torch.cat([masks_sam] * 4, dim=0)
        
        output, _ = mm_infer(
            video_tensor,
            instruct,
            model=model,
            tokenizer=tokenizer,
            modal='video',
            do_sample=False,
            choice=2,
            images_sam=images_sam,
            masks_list=masks_sam,
            offset=[0,1],
            label_list=sam_label,
            seg=False
        )
        
        # pred_idx = mvbench_dump(vid, instruct, letters, options, output)
        # print('output: ', output, 'gt: ', gt)
        pred = replace_and_normalize(output)
        # print('output: ', pred, 'gt: ', gt)

        ans_file.write(json.dumps({"vid": vid, "pred": pred, "gt": gt}) + '\n')

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', default='')
    parser.add_argument('--video-folder', help='Directory containing video files.', default='')
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', default='')
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', default='')
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lora-enable", type=bool, default=False)
    args = parser.parse_args()

    run_inference(args)
