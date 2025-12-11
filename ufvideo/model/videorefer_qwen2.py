# Adopted from: https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import sys

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .videorefer_arch import VideoReferMetaModel, VideoReferMetaForCausalLM
from .sam2 import SAM2


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def postprocess_masks(
    masks: torch.Tensor,
    input_size: Tuple[int, ...],
    original_size: Tuple[int, ...],
):
    masks = F.interpolate(
        masks.float(),
        (1024, 1024),
        mode="bilinear",
        align_corners=False,
    )

    masks = masks[..., : input_size[0], : input_size[1]] #(768, 1024)
    masks = F.interpolate(
        masks, original_size, mode="bilinear", align_corners=False
    ) #(480, 640)
    return masks
    

class VideoReferQwen2Config(Qwen2Config):
    model_type = "videorefer_qwen2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "videorefer_qwen2"


class VideoReferQwen2Model(VideoReferMetaModel, Qwen2Model):
    config_class = VideoReferQwen2Config

    def __init__(self, config: VideoReferQwen2Config):
        super(VideoReferQwen2Model, self).__init__(config)


class VideoReferQwen2ForCausalLM(Qwen2ForCausalLM, VideoReferMetaForCausalLM):
    config_class = VideoReferQwen2Config

    def __init__(self, config, **kwargs):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = VideoReferQwen2Model(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        masks = None,
        frame = None,
        ann_indices = None,
        frame_nums = None,
        return_dict: Optional[bool] = None,
        images_sam: Optional[torch.FloatTensor] = None,
        offset: Optional[torch.LongTensor] = None,
        masks_list: Optional[List[torch.FloatTensor]] = None,
        label_list: Optional[List[torch.Tensor]] = None,
        inference: bool = False,
        video_file = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
         
        batch_size, num_frames_sam = images_sam.shape[:2]
        # print("batch_size: ", batch_size)
        num_frames_sam = 4
        # print('images_sam: ', images_sam.shape)
        device = images_sam.device
        # assert batch_size == len(offset) - 1

        images_reshape = rearrange(images_sam, 'b t c h w -> (b t) c h w')
        # print('images_reshape: ', images_reshape.shape)

        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                masks,
                frame,
                ann_indices,
                frame_nums,
                video_file,
            )

        if inference == True:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )

            out_hidden_states = output.hidden_states
            output_logits = output.logits
            output_ce_loss = output.loss
            output_ce_loss = output_ce_loss * self.get_model().config.ce_loss_weight  

            # print('labels:', labels)
            # print('seg token id: ', self.get_model().config.seg_token_id)
            seg_token_mask = (labels == self.get_model().config.seg_token_id)  
            # print('seg_token_mask:', torch.where(seg_token_mask))
            seg_token_mask = torch.cat([seg_token_mask[:, 1:], torch.zeros_like(seg_token_mask[:, 0].unsqueeze(1))], dim=1)
            # print('seg_token_mask:', torch.where(seg_token_mask))

            hidden_states = []

            assert len(self.get_model().text_hidden_fcs) == 1
            hidden_states.append(self.get_model().text_hidden_fcs[0](out_hidden_states[-1]))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            # print('last_hidden_state: ', last_hidden_state.shape)
            # print('seg_token_mask: ', seg_token_mask.shape)
            pred_embeddings = last_hidden_state[seg_token_mask]
            seg_token_counts = seg_token_mask.int().sum(dim=-1)

            seg_token_offset = seg_token_counts.cumsum(dim=-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1, device=device, dtype=torch.long), seg_token_offset], dim=0
            )
            seg_token_offset = seg_token_offset[offset]
            # print('seg_token_offset: ', seg_token_offset)
            # print('offset: ', offset)

            obj_num_list = []
            language_embeddings = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                # print('start_i: ', start_i, 'end_i: ', end_i)
                if start_i == end_i:
                    language_embeddings = language_embeddings + [torch.zeros((1, 256), device=device, dtype=images_sam.dtype)] * num_frames_sam
                else:
                    language_embeddings = language_embeddings + [pred_embeddings[start_i: end_i]] * num_frames_sam
                obj_num_list.append(end_i.item()-start_i.item())
            language_embeddings = torch.cat(language_embeddings, dim=0).unsqueeze(1)

            # if inference:
            #     language_embeddings = language_embeddings.reshape(batch_size, num_frames_sam, 1, 256)
            #     pred_masks = []
            #     for i in range(batch_size):
            #         language_embeddings = language_embeddings[i]
            #         sam_states = self.get_model().mask_encoder.get_sam2_embeddings(images_sam.squeeze(0))
            #         masks = self.get_model().mask_encoder.language_embd_inference(sam_states, language_embeddings)

            #         h, w = label_list[i].shape # TODO: label_list记得加上
            #         masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
            #         masks = masks[:, 0]
            #         masks = masks.sigmoid() > 0.5

            #         pred_masks.append(masks)
            
            #     return {
            #         "pred_masks": pred_masks,
            #         "gt_masks": masks_list     
            #     }
            

            has_seg_token = seg_token_counts.bool()
            # print('has_seg_token:', has_seg_token)
            g_pixel_values = images_reshape # (B*T, 3, 1024, 1024)

            sam_states = self.get_model().mask_encoder.get_sam2_embeddings_train(g_pixel_values, obj_num_list=obj_num_list, BS=batch_size, T=num_frames_sam)
            high_res_masks = self.get_model().mask_encoder.inject_language_embd_train(sam_states, language_embeddings, nf_nobj=None)
            # print('high_re_masks: ', high_res_masks.shape)
            # TODO：动态obj_nums
            # obj_num_list = []
            cumulative_len = 0
            high_res_masks_new = []
            for bs in range(batch_size):
                if obj_num_list[bs] == 0:
                    current_len = 1 * num_frames_sam  # 每个样本取 4 个元素
                else:
                    current_len = obj_num_list[bs] * num_frames_sam
                # 截取当前样本的子张量，并删除冗余的通道维度（[1, 1024, 1024] → [1024, 1024]）
                current_mask = high_res_masks[cumulative_len : cumulative_len + current_len].squeeze(1)
                # print('current_maks: ', current_mask.shape, 'current2: ', current_mask.squeeze(1).shape)
                high_res_masks_new.append(current_mask)
                cumulative_len += current_len
            
            

            valid_pred_masks, valid_gt_masks = [], [] 
            for i in range(batch_size):
                pred_mask_cur_vid = F.interpolate(high_res_masks_new[i].unsqueeze(1), size=label_list[i].shape, mode='bilinear', align_corners=False)[:, 0]
                gt_mask_cur_vid = masks_list[i]
                
                valid_pred_masks.append(pred_mask_cur_vid)
                valid_gt_masks.append(gt_mask_cur_vid)


            mask_bce_loss = 0
            mask_dice_loss = 0
            num_masks = 0
            for batch_idx in range(batch_size):
                gt_mask = valid_gt_masks[batch_idx]
                pred_mask = valid_pred_masks[batch_idx]

                if not has_seg_token[batch_idx]:
                    pred_mask = pred_mask[0: 0]


                assert (
                    gt_mask.shape[0] == pred_mask.shape[0]
                ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                    gt_mask.shape, pred_mask.shape
                )
                # print('pred_mask:', pred_mask.dtype)
                # print('gt_mask:', gt_mask.dtype)
                mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask, gt_mask.to(pred_mask.dtype), num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0] # maybe zero
                )
                mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0] # maybe zero
                )
                num_masks += gt_mask.shape[0] # maybe zero

            mask_bce_loss = self.config.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
            mask_dice_loss = self.config.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
            mask_loss = mask_bce_loss + mask_dice_loss

            loss = output_ce_loss + mask_loss
            # if mask_loss != 0:
            #     print('mask_bce_loss: ', mask_bce_loss.item(), 'mask_dice_loss: ', mask_dice_loss.item(), 'output_ce_loss: ', output_ce_loss.item())

            torch.cuda.empty_cache()
            # print('='*80)
            return {
                "loss": loss,
                "ce_loss": output_ce_loss,
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss,
                "mask_loss": mask_loss,
            }




    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        masks = None,
        frame = None,
        ann_indices = None,
        frame_nums = None,
        images_sam: Optional[torch.FloatTensor] = None,
        offset: Optional[torch.LongTensor] = None,
        masks_list: Optional[List[torch.FloatTensor]] = None,
        label_list: Optional[List[torch.Tensor]] = None,
        inference: bool = True,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        with torch.no_grad():
            batch_size, num_frames_sam = images_sam.shape[:2]
            device = images_sam.device
            assert batch_size == len(offset) - 1

            # images_reshape = rearrange(images_sam, 'b t c h w -> (b t) c h w')
            seg_token_mask = (inputs == self.get_model().config.seg_token_id)  
            seg_token_mask = torch.cat([seg_token_mask[:, 1:], torch.zeros_like(seg_token_mask[:, 0].unsqueeze(1))], dim=1)
            if images is not None:
                (
                    input_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    _,
                    mark_mm_token_index
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    labels=None,
                    images=images,
                    masks=masks,
                    frame=frame,
                    ann_indices=ann_indices,
                    frame_nums=frame_nums,
                )
            if torch.where(seg_token_mask)[0].numel() == 0:
                # return  super().forward(
                #     input_ids=input_ids,
                #     attention_mask=attention_mask,
                #     past_key_values=past_key_values,
                #     inputs_embeds=inputs_embeds,
                #     output_hidden_states=True,
                #     return_dict=True
                # )
                output = super().generate(
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    images_sam=images_sam,
                    offset=offset,
                    masks_list=masks_list,
                    label_list=label_list,
                    inference=inference,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    **kwargs
                )

                output_token = output.sequences
                output_seg_token_masks = (output_token == self.get_model().config.seg_token_id)  
                output_seg_token_masks = torch.cat([output_seg_token_masks[:, 1:], torch.zeros_like(output_seg_token_masks[:, 0].unsqueeze(1))], dim=1)
                
                output_seg_token_hidden_states = []
                for o_idx, token_mask in enumerate(output_seg_token_masks[0]):
                    if token_mask == True:
                        output_seg_token_hidden_states.append(output.hidden_states[o_idx][-1])
                if output_seg_token_hidden_states != []:
                    output_seg_token_hidden_states = torch.cat(output_seg_token_hidden_states, dim=1) # [1, n, D]
                    output_pred_embeddings = self.get_model().text_hidden_fcs[0](output_seg_token_hidden_states) # [1, n, 256]
                    output_pred_embeddings = output_pred_embeddings.squeeze(0)  # [n, 256]

                    pred_masks = []
                    for p_embed in output_pred_embeddings:
                        language_embeddings = p_embed.unsqueeze(0)
                        sam_states = self.get_model().mask_encoder.get_sam2_embeddings(images_sam.squeeze(0))
                        masks = self.get_model().mask_encoder.language_embd_inference(sam_states, [language_embeddings] * images_sam.shape[1])

                        h, w = label_list.shape  
                        masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
                        masks = masks[:, 0]
                        masks = masks.sigmoid() > 0.5

                        pred_masks.append(masks)
                else:
                    pred_masks = []
                
                return {
                    'output': output.sequences,
                    "pred_masks": pred_masks,  
                }

            else:
                output = super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=True,
                )

                mm_feat_index, mm_input = mark_mm_token_index[0][0], mark_mm_token_index[0][1]

                # 1. 创建前size1长度的全False张量
                front_part = torch.zeros((1, mm_feat_index), dtype=torch.bool, device=seg_token_mask.device)
                back_part = seg_token_mask[:, -mm_input:]
                expanded_mask = torch.cat([front_part, back_part], dim=1)
                seg_token_mask = expanded_mask


                out_hidden_states = output.hidden_states

                hidden_states = []
                assert len(self.get_model().text_hidden_fcs) == 1
                hidden_states.append(self.get_model().text_hidden_fcs[0](out_hidden_states[-1])) # [[1, token_size, 256]]

                last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1) # [[1, token_size, 256]]
                pred_embeddings = last_hidden_state[seg_token_mask] # [n, 256]
                seg_token_counts = seg_token_mask.int().sum(dim=-1)

                seg_token_offset = seg_token_counts.cumsum(dim=-1)
                seg_token_offset = torch.cat(
                    [torch.zeros(1, device=seg_token_offset.device, dtype=torch.long), seg_token_offset], dim=0
                )
                # seg_token_offset = seg_token_offset[offset]

                pred_embeddings_ = []
                for i in range(len(seg_token_offset) - 1):
                    start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                    pred_embeddings_.append(pred_embeddings[start_i:end_i])
                pred_embeddings = pred_embeddings_

                pred_masks = []
                for i in range(len(pred_embeddings)):
                    language_embeddings = pred_embeddings[i]
                    sam_states = self.get_model().mask_encoder.get_sam2_embeddings(images_sam.squeeze(0))
                    masks = self.get_model().mask_encoder.language_embd_inference(sam_states, [language_embeddings] * images_sam.shape[1])

                    h, w = label_list[i].shape  
                    masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
                    masks = masks[:, 0]
                    masks = masks.sigmoid() > 0.5

                    pred_masks.append(masks)
            
                return {
                    'output': output,
                    "pred_masks": pred_masks,
                    "gt_masks": masks_list     
                }


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("videorefer_qwen2", VideoReferQwen2Config)
AutoModelForCausalLM.register(VideoReferQwen2Config, VideoReferQwen2ForCausalLM)
