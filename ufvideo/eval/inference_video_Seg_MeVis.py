import os
import re
import math
import json
import argparse
import warnings
import traceback

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
import cv2

import sys
sys.path.append('code/UFVideo')
from ufvideo import model_init, mm_infer
from ufvideo.utils import disable_torch_init
from functools import partial
from ufvideo.mm_utils import process_video
from ufvideo.constants import *
import random

# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


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


from pycocotools import mask as maskUtils

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

class MVBenchDataset(Dataset):

    def __init__(self, data_list, processor):
        self.data_list = data_list
        self.processor = processor
        self.transform = DirectResize(1024)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        video_paths = self.data_list[idx]['file_names']
        torch_imgs, _ ,_ ,_, _ = self.processor(video_paths)
        total_frames = self.data_list[idx]['length']

        question = self.data_list[idx]['expressions'][0]
        prompt_template = "Please segment the {class_name} in this image."
        instruct = prompt_template.format(class_name=question.lower())
        answer = "Sure, [SEG]."

        message = [[{'from': 'user', 'value': f'<video>\n{instruct}'}, {'from': 'gpt', 'value': answer}]]

        # sequence = np.arange(self.data_list[idx]['length'])
        # print('sequence: ', sequence)
        # print('video path: ', video_file)
        # try:
        # random_numbers = sorted(np.random.choice(sequence, size=4, replace=False))
        # except:
        #     print('sequence: ', sequence)
        #     print('video path: ', video_file)
        nums = []
        for x in range(self.data_list[idx]['length']):
            if self.data_list[idx]['segmentations'][x] is not None:
                nums.append(x)
            
        sam_video_files = [video_paths[j] for j in nums]
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
        for j in nums:
            sam_mask = annToMask(self.data_list[idx]['segmentations'][j])
            sam_mask = torch.from_numpy(sam_mask)
            sam_masks.append(sam_mask)
        masks_sam = torch.stack(sam_masks, dim=0)
        sam_label = torch.ones(masks_sam.shape[1], masks_sam.shape[2]) * IGNORE_INDEX

        # assert image_sam_tsr.shape[0] == masks_sam.shape[0] == 4
        

        
        return {
            'video': torch_imgs, 
            'video_path': video_paths,
            'instruct': message,
            'answer': answer,
            'images_sam': image_sam_tsr,
            'offset': [0, 1],
            'masks_list': masks_sam,
            'label_list': sam_label,
            'total_frames':len(nums),
            'id': self.data_list[idx]['id']
        }



def build_mvbench_eval(args, processor, local_rank):
    data_list = []
    
    json_file = os.path.join(args.question_file)
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    for data in json_data:
        data_list.append({
            'file_names': data['video'],
            'expressions': data['conversations'],
            'length': data['length'],
            'segmentations': data['segmentations'],
            'id': data['id']
        })
    data_list = get_chunk(data_list, args.num_chunks, local_rank)
    dataset = MVBenchDataset(data_list, processor)
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
    # model_args = {
    #     "train_mask_decoder": False,
    #     "seg_token_idx": args.seg_token_idx,
    # }
    model, processor, tokenizer = model_init(args.model_path, args=args, device_map={"": f"cuda:{local_rank}"})
    
    for m in model.modules():
        m.tokenizer = tokenizer
    
    # answer_file = os.path.expanduser(f"{args.answer_file}_rank{global_rank}.json")
    # print(f"Rank {global_rank} writing to {answer_file}")
    os.makedirs(args.answer_file, exist_ok=True)
    # ans_file = open(answer_file, "w")
    
    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES
    processor = partial(process_video, processor=processor, aspect_ratio=None, num_frames=num_frames)

    val_loader = build_mvbench_eval(args, processor, local_rank)

    # NOTE: only support batch size 1 for now
    for i, line in enumerate(tqdm(val_loader, desc=f"Rank {global_rank}", position=local_rank)):
        vid = line['video_path'][0]
        video_tensor = line['video'][0]
        instruct  = line['instruct'][0]
        label = line['answer'][0]
        images_sam = line['images_sam']
        offset = line['offset']
        masks_list = line['masks_list']
        label_list = line['label_list']
        total_frames = line['total_frames']
        id = line['id'][0].item()

        path_parts = [part for part in vid[0].split("/") if part]

        output = mm_infer(
            video_tensor,
            instruct,
            model=model,
            tokenizer=tokenizer,
            modal='video',
            do_sample=False,
            choice=3,
            images_sam=images_sam,
            offset=offset,
            masks_list=masks_list,
            label_list=label_list,
            seg=True
        )

        # print(output)
        pred_masks = output['pred_masks']
        ans_file = os.path.join(args.answer_file, str(id))
        os.makedirs(ans_file, exist_ok=True)
        # print(f'saving the mask to {ans_file} ...')
        for i, pred_mask_vid in enumerate(pred_masks):
            if pred_mask_vid.shape[0] == 0:
                continue

            assert total_frames == pred_mask_vid.shape[0]

            for frame_idx in range(total_frames):
                pred_mask = pred_mask_vid.detach().cpu().numpy()[frame_idx]
                pred_mask = pred_mask > 0

                save_path = "{}/{}.png".format(ans_file, frame_idx)
                binary_mask = np.where(pred_mask > 0, 1, 0)
                cv2.imwrite(save_path, binary_mask * 255)

        # pred_idx = mvbench_dump(vid, instruct, letters, options, output)

        # ans_file.write(json.dumps({"vid": vid, "pred": output, "gt": answer_idx}) + '\n')

    # ans_file.close()


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
    parser.add_argument("--sam_pretrained", type=str, default='')
    parser.add_argument("--train_mask_decoder", type=bool, default=False)
    parser.add_argument("--hidden_size", type=int, default=3584)
    parser.add_argument("--sam_out_dim", type=int, default=256)
    args = parser.parse_args()

    run_inference(args)
