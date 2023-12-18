import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from timm.models.helpers import named_apply, adapt_input_conv
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput

from typing import Any, List, Optional, Tuple, Union

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

from models.adapter import Adapter, AttnAdapter

class ChineseCLIPVisionAttentionWrapper(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, self_attn):
        super().__init__()
        self.self_attn = self_attn

    def get_intermediate(self, hidden_states):
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.self_attn.q_proj(hidden_states) * self.self_attn.scale
        key_states = self.self_attn._shape(self.self_attn.k_proj(hidden_states), -1, bsz)
        value_states = self.self_attn._shape(self.self_attn.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.self_attn.num_heads, -1, self.self_attn.head_dim)
        query_states = self.self_attn._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.self_attn.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.self_attn.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(attn_weights, p=self.self_attn.dropout,
                                           training=self.self_attn.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.self_attn.num_heads, tgt_len, self.self_attn.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.self_attn.num_heads, tgt_len, self.self_attn.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.self_attn.num_heads, tgt_len,
                                       self.self_attn.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output_x = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.self_attn.out_proj(attn_output_x)

        query_states = query_states.view(bsz, self.self_attn.num_heads, tgt_len, self.self_attn.head_dim).transpose(1, 2).contiguous().view(bsz * tgt_len, embed_dim)
        key_states = key_states.view(bsz, self.self_attn.num_heads, tgt_len, self.self_attn.head_dim).transpose(1, 2).contiguous().view(bsz * tgt_len, embed_dim)
        value_states = value_states.view(bsz, self.self_attn.num_heads, tgt_len, self.self_attn.head_dim).transpose(1, 2).contiguous().view(bsz * tgt_len, embed_dim)

        return query_states, key_states, value_states, attn_output_x, attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        prompts=None,
        attn_feat=False
    ):
        """Input shape: Batch x Time x Channel"""
        short_cut_flag = False

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.self_attn.q_proj(hidden_states) * self.self_attn.scale
        return_attn_feat = None
        if prompts is not None:
            if isinstance(prompts, AttnAdapter):
                key_states = self.self_attn.k_proj(hidden_states)
                value_states = self.self_attn.v_proj(hidden_states)

                # q_res, k_res, v_res = prompts(query_states, key_states, value_states)
                q_res, k_res, v_res = prompts(hidden_states, hidden_states, hidden_states)
                if short_cut_flag:
                    query_states = q_res
                    key_states = k_res
                    value_states = v_res
                else:
                    query_states = query_states + q_res
                    key_states = key_states + k_res
                    value_states = value_states + v_res
                key_states = self.self_attn._shape(key_states, -1, bsz)
                value_states = self.self_attn._shape(value_states, -1, bsz)
            else:
                Ek, Ev = prompts
                key_input = torch.cat(
                    [Ek.expand(hidden_states.shape[0], -1, -1).to(hidden_states.device), hidden_states],
                    dim=1)
                value_input = torch.cat(
                    [Ev.expand(hidden_states.shape[0], -1, -1).to(hidden_states.device), hidden_states],
                    dim=1)
                key_states = self.self_attn._shape(self.self_attn.k_proj(key_input), -1, bsz)
                value_states = self.self_attn._shape(self.self_attn.v_proj(value_input), -1, bsz)
        else:
            key_states = self.self_attn._shape(self.self_attn.k_proj(hidden_states), -1, bsz)
            value_states = self.self_attn._shape(self.self_attn.v_proj(hidden_states), -1, bsz)
        if attn_feat:
            return_attn_feat = [query_states, key_states, value_states]

        proj_shape = (bsz * self.self_attn.num_heads, -1, self.self_attn.head_dim)
        query_states = self.self_attn._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.self_attn.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.self_attn.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.self_attn.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.self_attn.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.self_attn.dropout, training=self.self_attn.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.self_attn.num_heads, tgt_len, self.self_attn.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.self_attn.num_heads, tgt_len, self.self_attn.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.self_attn.num_heads, tgt_len, self.self_attn.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        if prompts is not None:
            if isinstance(prompts, AttnAdapter):
                if short_cut_flag:
                    attn_output = prompts(None, None, None, proj=attn_output)
                else:
                    attn_output = self.self_attn.out_proj(attn_output) + prompts(None, None, None, proj=attn_output)
            else:
                attn_output = self.self_attn.out_proj(attn_output)
        else:
            attn_output = self.self_attn.out_proj(attn_output)

        if attn_feat:
            return [attn_output, return_attn_feat], attn_weights_reshaped

        return attn_output, attn_weights_reshaped

class ChineseCLIPVisionLayerWrapper(nn.Module):
    def __init__(self, encoder_layer):
        super().__init__()
        self.encoder_layer = encoder_layer
        self.encoder_layer.self_attn = ChineseCLIPVisionAttentionWrapper(self.encoder_layer.self_attn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        prompts=None,
        prompt_start_layer=False,
        prompt_end_layer=False,
        attn_feat=False
    ):

        adapter = None
        attn_prompts = None
        if prompts is not None:
            if isinstance(prompts, list) or isinstance(prompts, AttnAdapter):
                attn_prompts = prompts
            elif isinstance(prompts, Adapter):
                adapter = prompts
            else:
                if prompt_start_layer:
                    hidden_states = torch.cat([prompts.expand(hidden_states.shape[0], -1, -1).to(hidden_states.device),
                                               hidden_states], dim=1)
                else:
                    hidden_states = torch.cat([prompts.expand(hidden_states.shape[0], -1, -1).to(hidden_states.device),
                                           hidden_states[:, prompts.shape[1]:, :]], dim=1)

        residual = hidden_states

        hidden_states = self.encoder_layer.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.encoder_layer.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
            prompts=attn_prompts,
            attn_feat=attn_feat
        )
        return_attn_feat = None
        if attn_feat:
            return_attn_feat = hidden_states[1]
            hidden_states = hidden_states[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.encoder_layer.layer_norm2(hidden_states)
        hidden_states = self.encoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if adapter is not None:
            hidden_states = hidden_states + adapter(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if prompts is not None and prompt_end_layer and attn_prompts is None and adapter is None:
            hidden_states = outputs[0]
            hidden_states = hidden_states[:, prompts.shape[1]:, :]
            if output_attentions:
                attn_weights = outputs[1]
                if attn_weights is not None:
                    attn_weights = attn_weights[:, :, prompts.shape[1]:, prompts.shape[1]:]
                    adj_factor = 1 - attn_weights[:, :, :prompts.shape[1], :prompts.shape[1]].sum(dim=-1, keepdim=True)
                    attn_weights /= adj_factor
                outputs = (hidden_states, attn_weights, )
            else:
                outputs = (hidden_states,)
        if attn_feat:
            return outputs, return_attn_feat
        return outputs

class ChineseCLIPVisionEncoderWrapper(nn.Module):

    def __init__(self, clip_model, prompt_start_layer=0, prompt_end_layer=-1):
        super().__init__()
        self.config = clip_model.vision_model.encoder.config
        self.encoder = clip_model.vision_model.encoder
        for idx, encoder_layer in enumerate(self.encoder.layers):
            self.encoder.layers[idx] = ChineseCLIPVisionLayerWrapper(encoder_layer)
        self.prompt_start_layer = prompt_start_layer
        self.prompt_end_layer = prompt_end_layer

    def forward(
        self,
        inputs_embeds,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        prompts=None,
        attn_feat=False
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        return_attn_feats = []
        for idx, encoder_layer in enumerate(self.encoder.layers):
            layer_prompt = None
            is_prompt_start_layer = False
            is_prompt_end_layer = False
            if prompts is not None:
                is_prompt_start_layer = (idx == self.prompt_start_layer)
                is_prompt_end_layer = (idx == self.prompt_end_layer)
                if isinstance(prompts, list):
                    if idx >= self.prompt_start_layer and (self.prompt_end_layer == -1 or idx <= self.prompt_end_layer):
                        layer_prompt = prompts[idx - self.prompt_start_layer]
                else:
                    if idx == self.prompt_start_layer:
                        layer_prompt = prompts
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.encoder.gradient_checkpointing and self.encoder.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    output_attentions=output_attentions,
                    prompts=layer_prompt,
                    prompt_start_layer=is_prompt_start_layer,
                    prompt_end_layer=is_prompt_end_layer,
                    attn_feat=attn_feat
                )
            if attn_feat:
                return_attn_feats.append(layer_outputs[1])
                layer_outputs = layer_outputs[0]

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        if attn_feat:
            return BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=encoder_states,
                attentions=all_attentions
            ), return_attn_feats
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class ChineseCLIPVitWrapper(nn.Module):
    def __init__(self, org_clip_model, pretrained_proj = False, prompt_start_layer=0, prompt_end_layer=-1):
        super().__init__()
        clip_model = copy.deepcopy(org_clip_model)
        self.clip_model = clip_model
        self.prompt_start_layer = prompt_start_layer
        self.prompt_end_layer = prompt_end_layer
        self.clip_model.vision_model.encoder = ChineseCLIPVisionEncoderWrapper(clip_model,
                                                                               prompt_start_layer=prompt_start_layer,
                                                                               prompt_end_layer=prompt_end_layer)
        self.vision_model = self.clip_model.vision_model
        self.config = self.clip_model.vision_model.config
        self.pretrained_proj = pretrained_proj

    @property
    def device(self):
        return self.clip_model.device

    def get_attn_weights(self, proj=True):
        attn_weight_list = []
        for encoder_layer in self.vision_model.encoder.encoder.layers:
            attn = encoder_layer.encoder_layer.self_attn.self_attn
            if proj:
                attn_weight_list.append([attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj])
            else:
                attn_weight_list.append([attn.q_proj, attn.k_proj, attn.v_proj])
        return attn_weight_list

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        register_blk=-1,
        prompts=None,
        attn_feat=False):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.vision_model.embeddings(pixel_values)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)

        encoder_outputs = self.vision_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            prompts=prompts,
            attn_feat=attn_feat
        )
        return_attn_feats = None
        if attn_feat:
            return_attn_feats = encoder_outputs[1]
            encoder_outputs = encoder_outputs[0]

        last_hidden_state = encoder_outputs[0]
        if prompts is not None and self.prompt_end_layer == -1:
            if isinstance(prompts, list):
                if not isinstance(prompts[0], list) and not isinstance(prompts[0], Adapter) and not isinstance(prompts[0], AttnAdapter):
                    last_hidden_state[:, 0, :] = last_hidden_state[:, prompts[0].shape[1], :]
            else:
                last_hidden_state[:, 0, :] = last_hidden_state[:, prompts.shape[1], :]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.vision_model.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        vision_outputs = BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

        if self.pretrained_proj or (prompts is not None):
            pooled_output = vision_outputs[1]  # pooled_output
            image_features = self.clip_model.visual_projection(pooled_output)

            if output_hidden_states:
                return image_features.unsqueeze(1), vision_outputs.hidden_states
            if attn_feat:
                return image_features.unsqueeze(1), return_attn_feats

            return image_features.unsqueeze(1)


        return vision_outputs[0]


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, register_hook=False, prompts=None):
        B, N, C = x.shape
        if prompts is not None:
            if isinstance(prompts, AttnAdapter):
                qkv_raw = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
                q_raw, k_raw, v_raw = qkv_raw[0], qkv_raw[1], qkv_raw[2]
                # q_res, k_res, v_res = prompts(q_raw, k_raw, v_raw)
                q_res, k_res, v_res = prompts(x, x, x)
                q_raw = q_raw + q_res
                k_raw = k_raw + k_res
                v_raw = v_raw + v_res
                # q_raw = q_res
                # k_raw = k_res
                # v_raw = v_res

                q = q_raw.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                k = k_raw.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                v = v_raw.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            else:
                Ek, Ev = prompts
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                          C // self.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                k = torch.cat([Ek.expand(k.shape[0], -1, -1).to(k.device),k], dim=1)
                v = torch.cat([Ev.expand(v.shape[0], -1, -1).to(v.device),v], dim=1)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3,
                                                                                            1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grad_checkpointing=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)

    def forward(self, x, register_hook=False, prompts=None,
        prompt_start_layer=False,
        prompt_end_layer=False):

        adapter = None
        attn_prompts = None
        if prompts is not None:
            if isinstance(prompts, list) or isinstance(prompts, AttnAdapter):
                attn_prompts = prompts
            elif isinstance(prompts, Adapter):
                adapter = prompts
            else:
                if prompt_start_layer:
                    x = torch.cat([prompts.expand(x.shape[0], -1, -1).to(x.device), x], dim=1)
                else:
                    x = torch.cat([prompts.expand(x.shape[0], -1, -1).to(x.device), x[:, prompts.shape[1]:, :]], dim=1)

        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook,
                                         prompts=attn_prompts))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if adapter is not None:
            x = x + adapter(x)

        return x

    
class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 use_grad_checkpointing=False, ckpt_layer=0,
                 prompt_start_layer=0, prompt_end_layer=-1):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.prompt_start_layer = prompt_start_layer
        self.prompt_end_layer = prompt_end_layer
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_grad_checkpointing=(use_grad_checkpointing and i>=depth-ckpt_layer)
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, register_blk=-1, prompts=None, attention_mask=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
  
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        for i,blk in enumerate(self.blocks):
            layer_prompt = None
            is_prompt_start_layer = False
            is_prompt_end_layer = False

            if prompts is not None:
                is_prompt_start_layer = (i == self.prompt_start_layer)
                is_prompt_end_layer = (i == self.prompt_end_layer)
                if isinstance(prompts, list):
                    if i >= self.prompt_start_layer and (self.prompt_end_layer == -1 or i <= self.prompt_end_layer):
                        layer_prompt = prompts[i - self.prompt_start_layer]
                else:
                    if i == self.prompt_start_layer:
                        layer_prompt = prompts

            x = blk(x, register_blk==i, prompts=layer_prompt,
                    prompt_start_layer=is_prompt_start_layer,
                    prompt_end_layer=is_prompt_end_layer)
        x = self.norm(x)

        if prompts is not None and self.prompt_end_layer == -1:
            if isinstance(prompts, list):
                if not isinstance(prompts[0], list) and not isinstance(prompts[0], Adapter) and not isinstance(prompts[0], AttnAdapter):
                    x[:, 0, :] = x[:, prompts[0].shape[1], :]
            else:
                x[:, 0, :] = x[:, prompts.shape[1], :]
        
        return x

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)
        

@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))

            
def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint