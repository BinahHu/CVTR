"""PyTorch BERT model. """
import copy
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, device, dtype, nn
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from typing import Union, List
from models.adapter import Adapter, AttnAdapter

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    BaseModelOutputWithPooling,
    BaseModelOutput
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.clip.modeling_clip import _prepare_4d_attention_mask, _create_4d_causal_attention_mask

import transformers
transformers.logging.set_verbosity_error()

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]

class ChineseCLIPBertTokenizerWrapper(nn.Module):
    def __init__(self, clip_processor):
        super().__init__()
        self.clip_processor = clip_processor
        self.translation = None

    def forward(self, text, **kwargs):
        kwargs["text"] = text
        return self.clip_processor(**kwargs)

    def __len__(self):
        return len(self.clip_processor.tokenizer)

    @property
    def pad_token_id(self):
        return self.clip_processor.tokenizer.pad_token_id

    @property
    def cls_token_id(self):
        return self.clip_processor.tokenizer.cls_token_id

    @property
    def mask_token_id(self):
        return self.clip_processor.tokenizer.mask_token_id

    @property
    def eos_token_id(self):
        return self.clip_processor.tokenizer.eos_token_id
class ChineseCLIPTextSelfAttentionWrapper(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        prompts = None,
        attn_feat=False,
        short_cut_flag=False
    ):
        mixed_query_layer = self.attention.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.attention.transpose_for_scores(self.attention.key(encoder_hidden_states))
            value_layer = self.attention.transpose_for_scores(self.attention.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.attention.transpose_for_scores(self.attention.key(hidden_states))
            value_layer = self.attention.transpose_for_scores(self.attention.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            if prompts is not None:
                if isinstance(prompts, AttnAdapter):
                    key_layer = self.attention.key(hidden_states)
                    value_layer = self.attention.value(hidden_states)
                    # q_res, k_res, v_res = prompts(mixed_query_layer, key_layer, value_layer)
                    q_res, k_res, v_res = prompts(hidden_states, hidden_states, hidden_states)
                    if short_cut_flag:
                        mixed_query_layer = q_res
                        key_layer = k_res
                        value_layer = v_res
                    else:
                        mixed_query_layer = mixed_query_layer + q_res
                        key_layer = key_layer + k_res
                        value_layer = value_layer + v_res

                    key_layer = self.attention.transpose_for_scores(key_layer)
                    value_layer = self.attention.transpose_for_scores(value_layer)
                else:
                    Ek, Ev = prompts
                    key_input = torch.cat(
                        [Ek.expand(hidden_states.shape[0], -1, -1).to(hidden_states.device), hidden_states],
                        dim=1)
                    value_input = torch.cat(
                        [Ev.expand(hidden_states.shape[0], -1, -1).to(hidden_states.device), hidden_states],
                        dim=1)
                    key_layer = self.attention.transpose_for_scores(self.attention.key(key_input))
                    value_layer = self.attention.transpose_for_scores(self.attention.value(value_input))
            else:
                key_layer = self.attention.transpose_for_scores(self.attention.key(hidden_states))
                value_layer = self.attention.transpose_for_scores(self.attention.value(hidden_states))

        query_layer = self.attention.transpose_for_scores(mixed_query_layer)
        return_attn_feat = None
        if attn_feat:
            return_attn_feat = [query_layer, key_layer, value_layer]

        use_cache = past_key_value is not None
        if self.attention.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.attention.position_embedding_type == "relative_key" or self.attention.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.attention.distance_embedding(distance + self.attention.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.attention.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.attention.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in ChineseCLIPTextModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.attention.is_decoder:
            outputs = outputs + (past_key_value,)
        if attn_feat:
            return outputs, return_attn_feat
        return outputs
class ChineseCLIPTextAttentionWrapper(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.attention.self = ChineseCLIPTextSelfAttentionWrapper(self.attention.self)

    def get_intermediate(self, hidden_states, attention_mask):
        self_outputs = self.attention.self(
            hidden_states,
            attention_mask
        )
        proj_x = self_outputs[0]
        proj_y = self.attention.output.dense(self_outputs[0])
        return proj_x, proj_y
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        prompts=None,
        attn_feat=False
    ):
        short_cut_flag = False
        self_outputs = self.attention.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            prompts,
            attn_feat=attn_feat,
            short_cut_flag=short_cut_flag
        )
        return_attn_feat = None
        if attn_feat:
            return_attn_feat = self_outputs[1]
            self_outputs = self_outputs[0]
        if prompts is not None:
            if isinstance(prompts, AttnAdapter):
                if short_cut_flag:
                    attention_output = self.attention.output.LayerNorm(hidden_states + prompts(None, None, None, proj=self_outputs[0]))
                else:
                    attention_output = self.attention.output.dense(self_outputs[0])
                    attention_output = self.attention.output.dropout(attention_output)
                    attention_output = attention_output + prompts(None, None, None, proj=self_outputs[0])
                    attention_output = self.attention.output.LayerNorm(attention_output + hidden_states)
            else:
                attention_output = self.attention.output(self_outputs[0], hidden_states)
        else:
            attention_output = self.attention.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        if attn_feat:
            return outputs, return_attn_feat
        return outputs
class ChineseCLIPTextLayerWrapper(nn.Module):
    def __init__(self, layer_module):
        super().__init__()
        self.layer_module = layer_module
        self.layer_module.attention = ChineseCLIPTextAttentionWrapper(self.layer_module.attention)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        prompts=None,
        prompt_start_layer=False,
        prompt_end_layer=False,
        attn_feat=False
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        attention_prompts = None
        adapter = None
        if prompts is not None:
            if isinstance(prompts, list) or isinstance(prompts, AttnAdapter):
                attention_prompts = prompts
            elif isinstance(prompts, Adapter):
                adapter = prompts
            else:
                if prompt_start_layer:
                    hidden_states = torch.cat([prompts.expand(hidden_states.shape[0], -1, -1).to(hidden_states.device),
                                               hidden_states], dim=1)
                else:
                    hidden_states = torch.cat([prompts.expand(hidden_states.shape[0], -1, -1).to(hidden_states.device),
                                           hidden_states[:, prompts.shape[1]:, :]], dim=1)

        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.layer_module.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            prompts=attention_prompts,
            attn_feat=attn_feat
        )
        return_attn_feat = None
        if attn_feat:
            return_attn_feat = self_attention_outputs[1]
            self_attention_outputs = self_attention_outputs[0]
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.layer_module.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.layer_module.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.layer_module.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.layer_module.feed_forward_chunk, self.layer_module.chunk_size_feed_forward, self.layer_module.seq_len_dim, attention_output
        )

        if adapter is not None:
            layer_output = layer_output + adapter(layer_output)

        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.layer_module.is_decoder:
            outputs = outputs + (present_key_value,)

        if attn_feat:
            return outputs, return_attn_feat

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.layer_module.intermediate(attention_output)
        layer_output = self.layer_module.output(intermediate_output, attention_output)
        return layer_output
class ChineseCLIPTextEncoderWrapper(nn.Module):
    def __init__(self, clip_model, prompt_start_layer=0, prompt_end_layer=-1):
        super().__init__()
        self.prompt_start_layer = prompt_start_layer
        self.prompt_end_layer = prompt_end_layer
        self.clip_text_encoder = clip_model.text_model.encoder
        for i, layer_module in enumerate(self.clip_text_encoder.layer):
            self.clip_text_encoder.layer[i] = ChineseCLIPTextLayerWrapper(layer_module)
        self.config = clip_model.text_model.encoder.config
        self.config.__setattr__('fusion_layer', 6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        mode='multi_modal',
        prompts=None,
        attn_feat=False
    ):
        if isinstance(attention_mask, list):
            attention_mask, attention_mask_with_prompts = attention_mask
        else:
            attention_mask_with_prompts = None
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.clip_text_encoder.gradient_checkpointing and self.clip_text_encoder.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        if mode == 'text':
            start_layer = 0
            output_layer = self.config.fusion_layer
        elif mode == 'fusion':
            start_layer = self.config.fusion_layer
            output_layer = self.config.num_hidden_layers
        elif mode == 'multi_modal':
            start_layer = 0
            output_layer = self.config.num_hidden_layers
        else:
            raise NotImplementedError

        return_attn_feat = []
        for i in range(start_layer, output_layer):
            layer_prompt = None
            is_prompt_start_layer = False
            is_prompt_end_layer = False
            actual_attention_mask = attention_mask
            if prompts is not None:
                is_prompt_start_layer = (i == self.prompt_start_layer)
                is_prompt_end_layer = (i == self.prompt_end_layer)
                if isinstance(prompts, list):
                    if i >= self.prompt_start_layer and (self.prompt_end_layer == -1 or i <= self.prompt_end_layer):
                        layer_prompt = prompts[i - self.prompt_start_layer]
                        if not isinstance(prompts[0], Adapter) and not isinstance(prompts[0], AttnAdapter):
                            actual_attention_mask = attention_mask_with_prompts
                else:
                    if i == self.prompt_start_layer:
                        layer_prompt = prompts
                    actual_attention_mask = attention_mask_with_prompts
            layer_module = self.clip_text_encoder.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.clip_text_encoder.gradient_checkpointing and self.clip_text_encoder.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    actual_attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    prompts=layer_prompt,
                    prompt_start_layer=is_prompt_start_layer,
                    prompt_end_layer=is_prompt_end_layer,
                    attn_feat=attn_feat
                )
            if attn_feat:
                return_attn_feat.append(layer_outputs[1])
                layer_outputs = layer_outputs[0]
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        if attn_feat:
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            ), return_attn_feat

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
class ChineseCLIPBertWrapper(nn.Module):
    def __init__(self, clip_model, pretrained_proj = False, prompt_start_layer=0, prompt_end_layer=-1):
        super().__init__()
        self.text_model = clip_model.text_model
        self.prompt_start_layer = prompt_start_layer
        self.prompt_end_layer = prompt_end_layer
        self.text_model.encoder = ChineseCLIPTextEncoderWrapper(clip_model, prompt_start_layer=prompt_start_layer,
                                                                               prompt_end_layer=prompt_end_layer)
        self.last_projection = clip_model.text_projection
        self.config = clip_model.text_model.config
        self.pretrained_proj = pretrained_proj

    def get_attn_weights(self, proj=True):
        attn_weight_list = []
        for layer_module in self.text_model.encoder.clip_text_encoder.layer:
            attn = layer_module.layer_module.attention
            self_attn = attn.attention.self.attention
            out = attn.attention.output
            if proj:
                attn_weight_list.append([self_attn.query, self_attn.key, self_attn.value, out.dense])
            else:
                attn_weight_list.append([self_attn.query, self_attn.key, self_attn.value])
        return attn_weight_list

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            is_decoder=False,
            mode='multi_modal',
            prompts=None,
            attn_feat=False
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif encoder_embeds is not None:
            input_shape = encoder_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = encoder_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or encoder_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.text_model.get_extended_attention_mask(attention_mask, input_shape)

        if prompts is not None:
            if isinstance(prompts, list):
                if isinstance(prompts[0], list):
                    prompt_length = prompts[0][0].shape[1]
                elif isinstance(prompts[0], Adapter) or isinstance(prompts[0], AttnAdapter):
                    prompt_length = 0
                else:
                    prompt_length = prompts[0].shape[1]
            else:
                prompt_length = prompts.shape[1]
            if prompt_length > 0:
                attention_mask_with_prompts = torch.cat([torch.ones((batch_size, prompt_length), device=device),
                                                         attention_mask], dim=1)
                input_shape_with_prompts = torch.Size([batch_size, input_shape[1] + prompt_length])
                extended_attention_mask_with_prompts = self.text_model.get_extended_attention_mask(attention_mask_with_prompts,
                                                                                                    input_shape_with_prompts)
                extended_attention_mask = [extended_attention_mask, extended_attention_mask_with_prompts]

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [self.text_model.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.text_model.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.text_model.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.text_model.get_head_mask(head_mask, self.config.num_hidden_layers)

        if encoder_embeds is None:
            embedding_output = self.text_model.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length
            )
        else:
            embedding_output = encoder_embeds
        encoder_outputs = self.text_model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mode=mode,
            prompts=prompts,
            attn_feat=attn_feat
        )
        return_attn_feat = None
        if attn_feat:
            return_attn_feat = encoder_outputs[1]
            encoder_outputs = encoder_outputs[0]
        sequence_output = encoder_outputs[0]
        if prompts is not None and self.prompt_end_layer == -1:
            if isinstance(prompts, list):
                if not isinstance(prompts[0], list) and not isinstance(prompts[0], Adapter) and not isinstance(prompts[0], AttnAdapter):
                    sequence_output[:, 0, :] = sequence_output[:, prompts[0].shape[1], :]
            else:
                sequence_output[:, 0, :] = sequence_output[:, prompts.shape[1], :]
        pooled_output = self.text_model.pooler(sequence_output) if self.text_model.pooler is not None else None

        if not return_dict:
            res = (sequence_output, pooled_output) + encoder_outputs[1:]
        else:
            res = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

        if self.pretrained_proj or (prompts is not None):
            res.last_hidden_state = self.last_projection(res.last_hidden_state[:, 0, :]).unsqueeze(1)
        if attn_feat:
            return res, return_attn_feat
        return res
class ChineseCLIPBertMaskedLMWrapper(nn.Module):
    def __init__(self, org_clip_model, pretrained_proj=False, chinese_clip=True,
                 prompt_start_layer=0, prompt_end_layer=-1):
        super().__init__()
        clip_model = copy.deepcopy(org_clip_model)
        self.clip_model = clip_model
        if chinese_clip:
            self.bert = ChineseCLIPBertWrapper(clip_model, pretrained_proj=pretrained_proj,
                                               prompt_start_layer=prompt_start_layer,
                                               prompt_end_layer=prompt_end_layer)
        else:
            self.bert = CLIPBertWrapper(clip_model, pretrained_proj=pretrained_proj,
                                        prompt_start_layer=prompt_start_layer,
                                        prompt_end_layer=prompt_end_layer)
        self.config = clip_model.config.text_config
        self.cls = BertOnlyMLMHead(self.config)

    @property
    def device(self):
        return self.clip_model.device

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
            inputs_embeds=None, encoder_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
            labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, is_decoder=False,
            mode='multi_modal', soft_labels=None, alpha=0, return_logits=False, prompts=None):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_embeds=encoder_embeds,
            encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict,
            is_decoder=is_decoder, mode=mode, prompts=prompts)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if soft_labels is not None:
            loss_distill = -torch.sum(F.log_softmax(prediction_scores, dim=1) * soft_labels, dim=-1)
            loss_distill = loss_distill[labels != -100].mean()
            masked_lm_loss = (1 - alpha) * masked_lm_loss + alpha * loss_distill

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            # hidden_states=outputs.hidden_states,
            hidden_states=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )
class CLIPAttentionWrapper(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, attention):
        super().__init__()
        self.attention = attention
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        prompts=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.attention.q_proj(hidden_states) * self.attention.scale
        if prompts is not None:
            Ek, Ev = prompts
            key_input = torch.cat(
                [Ek.expand(hidden_states.shape[0], -1, -1).to(hidden_states.device), hidden_states],
                dim=1)
            value_input = torch.cat(
                [Ev.expand(hidden_states.shape[0], -1, -1).to(hidden_states.device), hidden_states],
                dim=1)
            key_states = self.attention._shape(self.attention.k_proj(key_input), -1, bsz)
            value_states = self.attention._shape(self.attention.v_proj(value_input), -1, bsz)
        else:
            key_states = self.attention._shape(self.attention.k_proj(hidden_states), -1, bsz)
            value_states = self.attention._shape(self.attention.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.attention.num_heads, -1, self.attention.head_dim)
        query_states = self.attention._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.attention.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.attention.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.attention.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.attention.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.attention.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.attention.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.attention.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.attention.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.attention.dropout, training=self.attention.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.attention.num_heads, tgt_len, self.attention.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.attention.num_heads, tgt_len, self.attention.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.attention.num_heads, tgt_len, self.attention.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.attention.out_proj(attn_output)

        return attn_output, attn_weights_reshaped
class CLIPEncoderLayerWrapper(nn.Module):
    def __init__(self, layer_module):
        super().__init__()
        self.layer_module = layer_module
        self.layer_module.self_attn = CLIPAttentionWrapper(self.layer_module.self_attn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        prompts=None,
        prompt_start_layer=False,
        prompt_end_layer=False
    ) -> Tuple[torch.FloatTensor]:

        attention_prompts = None
        if prompts is not None:
            if isinstance(prompts, list):
                attention_prompts = prompts
            else:
                if prompt_start_layer:
                    hidden_states = torch.cat(
                        [prompts.expand(hidden_states.shape[0], -1, -1).to(hidden_states.device),
                         hidden_states], dim=1)
                else:
                    hidden_states = torch.cat(
                        [prompts.expand(hidden_states.shape[0], -1, -1).to(hidden_states.device),
                         hidden_states[:, prompts.shape[1]:, :]], dim=1)

        residual = hidden_states
        hidden_states = self.layer_module.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.layer_module.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            prompts=attention_prompts
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_module.layer_norm2(hidden_states)
        hidden_states = self.layer_module.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
class CLIPTextEncoderWrapper(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, clip_model, prompt_start_layer=0, prompt_end_layer=-1):
        super().__init__()

        self.prompt_start_layer = prompt_start_layer
        self.prompt_end_layer = prompt_end_layer
        self.clip_text_encoder = clip_model.text_model.encoder
        for i, layer_module in enumerate(self.clip_text_encoder.layers):
            self.clip_text_encoder.layers[i] = CLIPEncoderLayerWrapper(layer_module)
        self.config = clip_model.text_model.encoder.config
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask= None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        prompts=None
    ) -> Union[Tuple, BaseModelOutput]:
        if isinstance(attention_mask, list):
            attention_mask, attention_mask_with_prompts = attention_mask
        else:
            attention_mask_with_prompts = None

        if isinstance(causal_attention_mask, list):
            causal_attention_mask, causal_attention_mask_with_prompts = causal_attention_mask
        else:
            causal_attention_mask_with_prompts = None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.clip_text_encoder.layers):
            layer_prompt = None
            is_prompt_start_layer = False
            is_prompt_end_layer = False
            actual_attention_mask = attention_mask
            actual_causal_attention_mask = causal_attention_mask
            if prompts is not None:
                is_prompt_start_layer = (idx == self.prompt_start_layer)
                is_prompt_end_layer = (idx == self.prompt_end_layer)
                if isinstance(prompts, list):
                    if idx >= self.prompt_start_layer and (
                            self.prompt_end_layer == -1 or idx <= self.prompt_end_layer):
                        layer_prompt = prompts[idx - self.prompt_start_layer]
                        actual_attention_mask = attention_mask_with_prompts
                        actual_causal_attention_mask = causal_attention_mask_with_prompts
                else:
                    if idx == self.prompt_start_layer:
                        layer_prompt = prompts
                    actual_attention_mask = attention_mask_with_prompts
                    actual_causal_attention_mask = causal_attention_mask_with_prompts

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    actual_attention_mask,
                    actual_causal_attention_mask,
                    output_attentions=output_attentions,
                    prompts=layer_prompt,
                    prompt_start_layer=is_prompt_start_layer,
                    prompt_end_layer=is_prompt_end_layer
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class CLIPBertWrapper(nn.Module):
    def __init__(self, clip_model, pretrained_proj = False, prompt_start_layer=0, prompt_end_layer=-1):
        super().__init__()
        self.text_model = clip_model.text_model
        self.prompt_start_layer = prompt_start_layer
        self.prompt_end_layer = prompt_end_layer
        self.text_model.encoder = CLIPTextEncoderWrapper(clip_model,
                                                        prompt_start_layer=prompt_start_layer,
                                                        prompt_end_layer=prompt_end_layer)
        self.last_projection = clip_model.text_projection
        self.config = clip_model.text_model.config
        self.pretrained_proj = pretrained_proj

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        encoder_embeds=None,
        mode='multi_modal',
        prompts=None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None  and encoder_embeds is None:
            raise ValueError("You have to specify input_ids")

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = encoder_embeds.size()[:-1]
        input_ids = input_ids.view(-1, input_shape[-1])

        if encoder_embeds is None:
            hidden_states = self.text_model.embeddings(
                input_ids=input_ids,
                position_ids=position_ids)
        else:
            hidden_states = encoder_embeds

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        if prompts is not None:
            if isinstance(prompts, list):
                if isinstance(prompts[0], list):
                    prompt_length = prompts[0][0].shape[1]
                else:
                    prompt_length = prompts[0].shape[1]
            else:
                prompt_length = prompts.shape[1]
            bsz, _, tgt_seq_len, src_seq_len = attention_mask.shape
            attention_mask_with_prompts = torch.cat([torch.ones((bsz, 1, tgt_seq_len,
                                                                 src_seq_len + prompt_length),
                                                                device=attention_mask.device),
                                                                attention_mask], dim=1)
            attention_mask_with_prompts = torch.cat([torch.ones((bsz, 1, tgt_seq_len + prompt_length,
                                                                 src_seq_len + prompt_length),
                                                                device=attention_mask.device),
                                                                attention_mask_with_prompts], dim=1)
            causal_attention_mask_attention_mask_with_prompts = torch.cat([torch.ones((bsz, 1, tgt_seq_len,
                                                                 src_seq_len + prompt_length),
                                                                device=attention_mask.device),
                                                     causal_attention_mask], dim=1)
            causal_attention_mask_attention_mask_with_prompts = torch.cat(
                [torch.ones((bsz, 1, tgt_seq_len + prompt_length,
                             src_seq_len + prompt_length),
                            device=attention_mask.device),
                 causal_attention_mask_attention_mask_with_prompts], dim=1)

            attention_mask = [attention_mask, attention_mask_with_prompts]
            causal_attention_mask = [causal_attention_mask, causal_attention_mask_attention_mask_with_prompts]

        encoder_outputs = self.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            prompts=prompts
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        prompt_shift = 0
        if prompts is not None and self.prompt_end_layer == -1:
            if isinstance(prompts, list):
                if not isinstance(prompts[0], list):
                    prompt_shift = prompts[0].shape[1]
            else:
                prompt_shift = prompts.shape[1]
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            prompt_shift + input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]

        if not return_dict:
            res = (last_hidden_state, pooled_output) + encoder_outputs[1:]
        else:
            res = BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

        if self.pretrained_proj or (prompts is not None):
            res.last_hidden_state = self.last_projection(res.pooler_output).unsqueeze(1)
        return res

def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args = e.args+ (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        
        self.config = config

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)  

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings+ position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.save_attention = False   
            
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        prompts = None
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None


        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            if prompts is not None:
                if isinstance(prompts, AttnAdapter):
                    key_layer = self.key(hidden_states)
                    value_layer = self.value(hidden_states)
                    # q_res, k_res, v_res = prompts(mixed_query_layer, key_layer, value_layer)
                    q_res, k_res, v_res = prompts(hidden_states, hidden_states, hidden_states)
                    mixed_query_layer = mixed_query_layer + q_res
                    key_layer = key_layer + k_res
                    value_layer = value_layer + v_res
                    # mixed_query_layer = q_res
                    # key_layer = k_res
                    # value_layer = v_res
                    key_layer = self.transpose_for_scores(key_layer)
                    value_layer = self.transpose_for_scores(value_layer)
                else:
                    Ek, Ev = prompts
                    key_input = torch.cat(
                        [Ek.expand(hidden_states.shape[0], -1, -1).to(hidden_states.device),
                         hidden_states],
                        dim=1)
                    value_input = torch.cat(
                        [Ev.expand(hidden_states.shape[0], -1, -1).to(hidden_states.device),
                         hidden_states],
                        dim=1)
                    key_layer = self.transpose_for_scores(self.key(key_input))
                    value_layer = self.transpose_for_scores(
                        self.value(value_input))
            else:
                key_layer = self.transpose_for_scores(self.key(hidden_states))
                value_layer = self.transpose_for_scores(
                    self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)         

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        prompts=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            prompts
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)

        self.has_cross_attention = (layer_num >= config.fusion_layer) #选择后6层
        if self.has_cross_attention:           
            self.layer_num = layer_num                
            self.crossattention = BertAttention(config, is_cross_attention=True)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        prompts=None,
        prompt_start_layer=False,
        prompt_end_layer=False
    ):
        attention_prompts = None
        adapter = None
        if prompts is not None:
            if isinstance(prompts, list) or isinstance(prompts, AttnAdapter):
                attention_prompts = prompts
            elif isinstance(prompts, Adapter):
                adapter = prompts
            else:
                if prompt_start_layer:
                    hidden_states = torch.cat(
                        [prompts.expand(hidden_states.shape[0], -1, -1).to(hidden_states.device),
                         hidden_states], dim=1)
                else:
                    hidden_states = torch.cat(
                        [prompts.expand(hidden_states.shape[0], -1, -1).to(hidden_states.device),
                         hidden_states[:, prompts.shape[1]:, :]], dim=1)

        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            prompts=attention_prompts
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        if self.has_cross_attention:
            assert encoder_hidden_states is not None, "encoder_hidden_states must be given for cross-attention layers"
            
            if type(encoder_hidden_states) == list:
                cross_attention_outputs = self.crossattention(
                    attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states[(self.layer_num-self.config.fusion_layer)%len(encoder_hidden_states)],
                    encoder_attention_mask[(self.layer_num-self.config.fusion_layer)%len(encoder_hidden_states)],
                    output_attentions=output_attentions,
                )    
                attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[1:-1]
         
            else:
                cross_attention_outputs = self.crossattention(
                    attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights                               
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config, prompt_start_layer=0, prompt_end_layer=-1):
        super().__init__()
        self.config = config
        self.prompt_start_layer = prompt_start_layer
        self.prompt_end_layer = prompt_end_layer
        self.layer = nn.ModuleList([BertLayer(config,i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        mode='multi_modal',
        prompts=None
    ):
        if isinstance(attention_mask, list):
            attention_mask, attention_mask_with_prompts = attention_mask
        else:
            attention_mask_with_prompts = None
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        
                
        if mode=='text': 
            start_layer = 0
            output_layer = self.config.fusion_layer
            
        elif mode=='fusion':
            start_layer = self.config.fusion_layer
            output_layer = self.config.num_hidden_layers
            
        elif mode=='multi_modal':
            start_layer = 0
            output_layer = self.config.num_hidden_layers
        else:
            raise NotImplementedError
        
        for i in range(start_layer, output_layer):
            layer_prompt = None
            is_prompt_start_layer = False
            is_prompt_end_layer = False
            actual_attention_mask = attention_mask
            if prompts is not None:
                is_prompt_start_layer = (i == self.prompt_start_layer)
                is_prompt_end_layer = (i == self.prompt_end_layer)
                if isinstance(prompts, list):
                    if i >= self.prompt_start_layer and (
                            self.prompt_end_layer == -1 or i <= self.prompt_end_layer):
                        layer_prompt = prompts[i - self.prompt_start_layer]
                        if not isinstance(prompts[0], Adapter) and not isinstance(prompts[0],
                                                                                  AttnAdapter):
                            actual_attention_mask = attention_mask_with_prompts
                else:
                    if i == self.prompt_start_layer:
                        layer_prompt = prompts
                    actual_attention_mask = attention_mask_with_prompts

            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    prompts=layer_prompt,
                    prompt_start_layer=is_prompt_start_layer,
                    prompt_end_layer=is_prompt_end_layer
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = next_decoder_cache+ (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True, prompt_start_layer=0, prompt_end_layer=-1):
        super().__init__(config)
        self.config = config
        self.prompt_start_layer = prompt_start_layer
        self.prompt_end_layer = prompt_end_layer

        self.embeddings = BertEmbeddings(config)
        
        self.encoder = BertEncoder(config, prompt_start_layer=prompt_start_layer,
                                        prompt_end_layer=prompt_end_layer)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()
 

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    
    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: device, is_decoder: bool) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=False,
        mode='multi_modal',
        prompts=None
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif encoder_embeds is not None:    
            input_shape = encoder_embeds.size()[:-1]
            batch_size, seq_length = input_shape 
            device = encoder_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or encoder_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape,
                                                                                 device, is_decoder)

        if prompts is not None:
            if isinstance(prompts, list):
                if isinstance(prompts[0], list):
                    prompt_length = prompts[0][0].shape[1]
                elif isinstance(prompts[0], Adapter) or isinstance(prompts[0], AttnAdapter):
                    prompt_length = 0
                else:
                    prompt_length = prompts[0].shape[1]
            else:
                prompt_length = prompts.shape[1]
            if prompt_length > 0:
                attention_mask_with_prompts = torch.cat([torch.ones((batch_size, prompt_length), device=device),
                                                         attention_mask], dim=1)
                input_shape_with_prompts = torch.Size([batch_size, input_shape[1] + prompt_length])
                extended_attention_mask_with_prompts = self.text_model.get_extended_attention_mask(attention_mask_with_prompts,
                                                                                                    input_shape_with_prompts)
                extended_attention_mask = [extended_attention_mask, extended_attention_mask_with_prompts]
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            
            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:    
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        if encoder_embeds is None:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
        else:
            embedding_output = encoder_embeds
            
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mode=mode,
            prompts=prompts
        )
        sequence_output = encoder_outputs[0]
        if prompts is not None and self.prompt_end_layer == -1:
            if isinstance(prompts, list):
                if not isinstance(prompts[0], list) and not isinstance(prompts[0], Adapter) and not isinstance(prompts[0], AttnAdapter):
                    sequence_output[:, 0, :] = sequence_output[:, prompts[0].shape[1], :]
            else:
                sequence_output[:, 0, :] = sequence_output[:, prompts.shape[1], :]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertForMaskedLM(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, prompt_start_layer=0, prompt_end_layer=-1):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False,
                              prompt_start_layer=prompt_start_layer,
                              prompt_end_layer=prompt_end_layer)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="bert-base-uncased",
    #     output_type=MaskedLMOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=False,
        mode='multi_modal',
        soft_labels=None,
        alpha=0,
        return_logits=False,
        prompts=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        print(f"mode is {mode}")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_embeds=encoder_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
            mode=mode,
            prompts=prompts
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        
        if return_logits:
            return prediction_scores

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        
        if soft_labels is not None:
            loss_distill = -torch.sum(F.log_softmax(prediction_scores, dim=1)*soft_labels,dim=-1)
            loss_distill = loss_distill[labels!=-100].mean()
            masked_lm_loss = (1-alpha)*masked_lm_loss + alpha*loss_distill

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            #hidden_states=outputs.hidden_states,
            hidden_states=outputs.last_hidden_state,      
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
