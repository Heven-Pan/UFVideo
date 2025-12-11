import json
import argparse
from tabulate import tabulate
import glob
import tqdm
from moviepy.editor import VideoFileClip
import re

def temporal_iou(A, B):
    max0 = max((A[0]), (B[0]))
    min0 = min((A[0]), (B[0]))
    max1 = max((A[1]), (B[1]))
    min1 = min((A[1]), (B[1]))
    _iou = max(min1 - max0, 0) / (max1 - min0)
    return max(0, _iou)



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
    
import cv2

def total_time(vid):
    cap = cv2.VideoCapture(vid)
    # 获取帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取总帧数
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # 计算视频时长（秒）
    duration = total_frames / fps
    # 释放视频捕获对象
    cap.release()
    return duration

def main():
    args = parse_args()
    data_paths = glob.glob(args.pred_path.replace('.json', '_*.json'))
    res = []
    for data_path in data_paths:
        for line in open(data_path):
            res.append(json.loads(line))
    
    lis = res
    total_iou = 0
    r_3 = 0
    r_5 = 0
    r_7 = 0
    for i in tqdm.tqdm(range(len(lis))):
        vid = lis[i]["vid"]
        pred = lis[i]["pred"]
        gt_box = lis[i]["gt"]
        ttime = total_time(vid)
        gt_box = [gt_box[0]*ttime, gt_box[1]*ttime]
        pred_box = parse_span_from_text(pred)
        pred_box = [pred_box[0]*ttime, pred_box[1]*ttime]
        if pred_box == [0, 0]:
            continue
        _iou = temporal_iou(pred_box, gt_box)
        total_iou += _iou
        if _iou > 0.3:
            r_3 += 1
        if _iou > 0.5:
            r_5 += 1
        if _iou > 0.7:
            r_7 += 1

    miou = total_iou / len(lis)
    r_3 /= (len(lis) / 100)
    r_5 /= (len(lis) / 100)
    r_7 /= (len(lis) / 100)
    print("miou: ", str(miou))
    print("R@1(0.3): ", str(r_3))
    print("R@1(0.5): ", str(r_5))
    print("R@1(0.7): ", str(r_7))
    return miou, r_3, r_5, r_7

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate video captioning.")
    parser.add_argument("--pred_path", default='', help="The path to file containing prediction.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
