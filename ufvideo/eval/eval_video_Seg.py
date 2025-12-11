import os
import time
import argparse
import cv2
import json
import numpy as np
from pycocotools import mask as maskUtils
import math

import multiprocessing as mp
from functools import partial

from PIL import Image
import glob

def rle_decode(rle, shape):
    """将RLE编码解码为二值mask"""
    mask = maskUtils.decode(rle)
    return mask

def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap

def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels

    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res

# 新增：计算J&F指标（IoU和F-measure的平均值）
def calculate_jf(iou, f_score):
    """计算J&F指标（IoU和边界F-measure的算术平均值）"""
    return (iou + f_score) / 2.0

def process_sample(data, pred_mask_root, gt_mask_root):
    """处理单个样本，返回该样本的所有帧的IoU、Boundary F-score和J&F
    
    只计算GT和预测mask都存在的帧
    
    Args:
        data: 样本数据字典
        pred_mask_root: 预测mask根目录
        gt_mask_root: GT mask根目录
        
    Returns:
        tuple: (sample_id, frame_iou, frame_boundary, frame_jf, valid_count, total_count)
    """
    sample_id = data['id']
    pred_frame_dir = os.path.join(pred_mask_root, str(sample_id))
    
    if not os.path.exists(pred_frame_dir):
        print(f"样本{sample_id}: 预测目录不存在")
        return (sample_id, [], [], [], 0, 0)  # 空结果表示跳过
    
    # 按帧号排序预测mask
    pred_mask_paths = sorted(
        glob.glob(os.path.join(pred_frame_dir, '*.png')),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    
    if not pred_mask_paths:
        print(f"样本{sample_id}: 没有预测mask文件")
        return (sample_id, [], [], [], 0, 0)
    
    frame_iou = []
    frame_boundary = []
    frame_jf = []
    valid_count = 0  # 成功计算的帧数
    total_count = 0  # 尝试处理的总帧数
    
    for frame_idx, pred_path in enumerate(pred_mask_paths):
        total_count += 1
        
        # 检查GT文件名是否存在
        if frame_idx >= len(data['file_names']):
            print(f"样本{sample_id}帧{frame_idx}: 超出file_names范围")
            continue
        
        gt_filename = data['file_names'][frame_idx]
        if gt_filename is None:
            print(f"样本{sample_id}帧{frame_idx}: GT文件名为None，跳过")
            continue
        
        gt_mask_path = os.path.join(gt_mask_root, gt_filename)
        gt_mask_path = gt_mask_path.replace('.jpg', '.png')
        
        # 检查GT文件是否存在
        if not os.path.exists(gt_mask_path):
            # print(f"样本{sample_id}帧{frame_idx}: GT文件不存在 {gt_mask_path}，跳过")
            continue
        
        # 读取预测mask
        try:
            pred_mask = Image.open(pred_path).convert('L')
            pred_mask = np.array(pred_mask) == 255  # 二值化
        except Exception as e:
            print(f"样本{sample_id}帧{frame_idx}: 预测mask读取失败 {e}，跳过")
            continue
        
        # 读取GT mask
        try:
            gt_mask = Image.open(gt_mask_path).convert('L')
            gt_mask = np.array(gt_mask) > 0  # 二值化
        except Exception as e:
            print(f"样本{sample_id}帧{frame_idx}: GT mask读取失败 {e}，跳过")
            continue
        
        # 检查形状是否匹配
        if gt_mask.shape != pred_mask.shape:
            print(f"样本{sample_id}帧{frame_idx}: 形状不匹配 GT{gt_mask.shape} vs Pred{pred_mask.shape}，跳过")
            continue
        
        # 计算指标
        try:
            iou = db_eval_iou(gt_mask, pred_mask)
            f_score = db_eval_boundary(gt_mask, pred_mask)
            jf = calculate_jf(iou, f_score)
            
            frame_iou.append(iou)
            frame_boundary.append(f_score)
            frame_jf.append(jf)
            valid_count += 1
            
        except Exception as e:
            print(f"样本{sample_id}帧{frame_idx}: 指标计算失败 {e}，跳过")
            continue
    
    # 输出统计信息
    if valid_count > 0:
        print(f"样本{sample_id}: 成功计算 {valid_count}/{total_count} 帧")
    else:
        print(f"样本{sample_id}: 没有有效帧可计算 (0/{total_count})")
    
    return (sample_id, frame_iou, frame_boundary, frame_jf, valid_count, total_count)


# 多进程处理函数
def parallel_process(data_list, pred_mask_root, gt_mask_root, num_workers=None):
    """
    并行处理样本列表
    num_workers: 进程数，默认使用CPU核心数
    """
    # 设置进程数（默认使用所有可用核心）
    num_workers = num_workers or mp.cpu_count()
    print(f"使用{num_workers}个进程并行处理...")
    
    # 创建进程池，使用partial固定pred_mask_root参数
    with mp.Pool(processes=num_workers) as pool:
        # 映射样本列表到进程池
        results = pool.map(
            partial(process_sample, pred_mask_root=pred_mask_root, gt_mask_root=gt_mask_root),
            data_list
        )
    
    # 汇总所有结果
    all_iou = []
    all_boundary_f = []
    all_jf = []
    for sample_id, ious, f_scores, jfs, valid_count, total_count in results:
        if valid_count > 0:
            if ious:  # 只处理有效结果
                all_iou.extend(ious)
                all_boundary_f.extend(f_scores)
                all_jf.extend(jfs)
                # print(f"样本{sample_id} - 平均IoU: {np.mean(ious):.4f}, 平均边界F: {np.mean(f_scores):.4f}, 平均J&F: {np.mean(jfs):.4f}")
    
    return all_iou, all_boundary_f, all_jf

def main(json_path, pred_mask_path, gt_mask_root, num_workers=None):
    # 读取JSON数据
    with open(json_path, 'r') as f:
        data_list = json.load(f)
    
    # 并行处理
    all_iou, all_boundary_f, all_jf = parallel_process(data_list, pred_mask_path, gt_mask_root, num_workers)
    
    # 输出整体结果
    if all_iou:
        print("\n===== 整体评估结果 =====")
        print(f"所有帧平均IoU: {np.mean(all_iou):.4f} (±{np.std(all_iou):.4f})")
        print(f"所有帧平均边界F-measure: {np.mean(all_boundary_f):.4f} (±{np.std(all_boundary_f):.4f})")
        print(f"所有帧平均J&F: {np.mean(all_jf):.4f} (±{np.std(all_jf):.4f})")  # 打印整体J&F
        print(f"总帧数: {len(all_iou)}")
    else:
        print("没有有效帧被处理")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mevis_gt_path", type=str, default="")
    parser.add_argument("--mevis_gt_mask_root", type=str, default="")
    parser.add_argument("--mevis_pred_path", type=str, default="")
    args = parser.parse_args()
    
    main(args.mevis_gt_path, args.mevis_pred_path, args.mevis_gt_mask_root, num_workers=128)