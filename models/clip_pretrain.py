from models.xbert import BertConfig, BertForMaskedLM
from transformers import BertTokenizer
import transformers
transformers.logging.set_verbosity_error()

import torch
from torch import nn
import torch.nn.functional as F
from models.model_utils import create_vit
import torch.distributed as dist
import math
from transformers import ChineseCLIPModel, ChineseCLIPProcessor, CLIPModel, CLIPProcessor
from models.vit import ChineseCLIPVitWrapper
from models.xbert import ChineseCLIPBertTokenizerWrapper, ChineseCLIPBertMaskedLMWrapper
from models.prompt_pool import UnifiedPrompt
from models.adapter import UnifiedAdapter

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import copy
import numpy as np

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma =nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) 

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), \
                F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out

class CLIP_Pretrain(nn.Module):
    def __init__(self, config,                
                 med_config = '/mnt/log/code/CTP/configs/albef_bert_chinese_config.json',
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                    
                 embed_dim = 256,   
                 mode=None,
                 chinese_clip=False,
                 clip=False,
                 hybrid_clip=False,
                 pretrained_proj=False,
                 text_prompt_per_task=0,
                 visual_prompt_per_task=0,
                 prompt_init="random",
                 text_deep_prompt=False,
                 visual_deep_prompt=False,
                 prompts_start_layer=0,
                 prompts_end_layer=-1,
                 unified_prompt=False,
                 unified_adapter=False,
                 task_num=1,
                 unified_prompt_pool_per_task=10,
                 unified_prompt_length=4,
                 fine_tune=False,
                 two_stage=False,
                 convnorm=False,
                 ):
        super().__init__()
        self.config = config
        self.max_words = config['max_words']
        self.chinese_clip = chinese_clip
        self.clip = clip
        self.hybrid_clip = hybrid_clip
        self.two_stage = two_stage
        self.convnorm = convnorm
        self.is_stage_one = True

        # Prompt tuning config
        self.text_prompt_num = 0
        self.visual_prompt_num = 0
        self.text_prompt_per_task = text_prompt_per_task
        self.visual_prompt_per_task = visual_prompt_per_task
        self.text_prompt_list = nn.ParameterList()
        self.visual_prompt_list = nn.ParameterList()
        self.unified_prompt = unified_prompt
        self.unified_prompt_pool = None
        self.unified_adapter = unified_adapter
        self.unified_adapter_pool = None
        self.visual_adapter_current_task = None
        self.text_adapter_current_task = None
        self.task_num = task_num
        self.fine_tune = fine_tune
        self.unified_prompt_pool_per_task = unified_prompt_pool_per_task
        self.unified_prompt_length = unified_prompt_length
        self.prompt_init = prompt_init
        self.text_deep_prompt = text_deep_prompt
        self.visual_deep_prompt = visual_deep_prompt
        self.text_layers = 12
        self.visual_layers = 12
        self.prompts_start_layer = prompts_start_layer
        self.prompts_end_layer = prompts_end_layer
        self.current_task_index = -1

        self.pretrained_proj = pretrained_proj
        if self.chinese_clip or self.clip:
            if self.chinese_clip:
                model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
                # pretrained_model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
                # model = ChineseCLIPModel(pretrained_model.config)
                processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
            else:
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
                # pretrained_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
                # model = CLIPModel(pretrained_model.config)
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

            self.visual_layers = model.vision_model.config.num_hidden_layers
            self.text_layers = model.text_model.config.num_hidden_layers

            self.visual_encoder = ChineseCLIPVitWrapper(model, pretrained_proj=self.pretrained_proj,
                                                        prompt_start_layer=prompts_start_layer,
                                                        prompt_end_layer=prompts_end_layer)
            if self.two_stage or self.convnorm:
                self.finetune_visual_encoder = ChineseCLIPVitWrapper(model,
                                                            pretrained_proj=self.pretrained_proj)
                if self.prompts_start_layer > 0:
                    for p in self.finetune_visual_encoder.vision_model.embeddings.parameters():
                        p.requires_grad = False
                    for p in self.finetune_visual_encoder.vision_model.pre_layrnorm.parameters():
                        p.requires_grad = False



            # cpy_model = copy.deepcopy(model)
            if self.unified_prompt:
                self.fixed_visual_encoder = ChineseCLIPVitWrapper(model, pretrained_proj=True)
                for p in self.fixed_visual_encoder.parameters():
                    p.requires_grad = False


            self.tokenizer = ChineseCLIPBertTokenizerWrapper(processor)
            self.text_mlm_encoder = ChineseCLIPBertMaskedLMWrapper(model, pretrained_proj=self.pretrained_proj, chinese_clip=self.chinese_clip,
                                                                    prompt_start_layer = prompts_start_layer,
                                                                    prompt_end_layer = prompts_end_layer)
            self.text_encoder = self.text_mlm_encoder.bert
            if self.two_stage or self.convnorm:
                self.finetune_text_mlm_encoder = ChineseCLIPBertMaskedLMWrapper(model,
                                                                       pretrained_proj=self.pretrained_proj,
                                                                       chinese_clip=self.chinese_clip)
                self.finetune_text_encoder = self.finetune_text_mlm_encoder.bert


            if self.unified_prompt:
                self.fixed_text_mlm_encoder = ChineseCLIPBertMaskedLMWrapper(model, pretrained_proj=True,
                                                                       chinese_clip=self.chinese_clip)
                for p in self.fixed_text_mlm_encoder.parameters():
                    p.requires_grad = False
                self.fixed_text_encoder = self.fixed_text_mlm_encoder.bert

            vision_width = model.vision_model.config.hidden_size
            text_width = self.text_encoder.config.hidden_size
            self.visual_width = vision_width
            self.text_width = text_width

            if self.unified_prompt:
                assert text_width == vision_width
                self.unified_prompt_pool = UnifiedPrompt(text_width, self.task_num,
                                                         prompt_length=self.unified_prompt_length,
                                                         prompt_per_task=self.unified_prompt_pool_per_task,
                                                         key_dim=512)
            if self.unified_adapter or self.two_stage or self.convnorm:
                assert text_width == vision_width
                self.unified_adapter_pool = UnifiedAdapter(text_width, adapter_start_layer=self.prompts_start_layer, convnorm=self.convnorm)

            if self.pretrained_proj:
                self.vision_proj = nn.Identity()
                self.text_proj = nn.Identity()
            else:
                if self.text_prompt_per_task > 0:
                    self.text_proj = nn.Identity()
                else:
                    if mode == 'LUCIR':
                        self.text_proj = CosineLinear(text_width, embed_dim)
                    else:
                        self.text_proj = nn.Linear(text_width, embed_dim)
                if self.visual_prompt_per_task > 0:
                    self.vision_proj = nn.Identity()
                else:
                    if mode == 'LUCIR':
                        self.vision_proj = CosineLinear(vision_width, embed_dim)
                    else:
                        self.vision_proj = nn.Linear(vision_width, embed_dim)


        elif self.hybrid_clip:
            self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt,
                                                          vit_ckpt_layer, 0, prompts_start_layer, prompts_end_layer)
            if vit == 'base':
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu", check_hash=True)
                state_dict = checkpoint["model"]
                msg = self.visual_encoder.load_state_dict(state_dict, strict=False)

            self.tokenizer = BertTokenizer.from_pretrained(
                'bert-base-chinese')  # Download ‘bert-base-chinese’ in advance and save it to a local path
            encoder_config = BertConfig.from_json_file(med_config)
            encoder_config.encoder_width = vision_width

            self.text_mlm_encoder = BertForMaskedLM.from_pretrained('bert-base-chinese',
                                                                    config=encoder_config)
            self.text_encoder = self.text_mlm_encoder.bert

            self.visual_layers = 12
            self.text_layers = self.text_mlm_encoder.config.num_hidden_layers


            text_width = self.text_encoder.config.hidden_size
            self.visual_width = vision_width
            self.text_width = text_width

            if self.unified_prompt:
                assert text_width == vision_width
                self.unified_prompt_pool = UnifiedPrompt(text_width, self.task_num,
                                                         prompt_length=self.unified_prompt_length,
                                                         prompt_per_task=self.unified_prompt_pool_per_task,
                                                         key_dim=512)
            if self.unified_adapter:
                assert text_width == vision_width
                self.unified_adapter_pool = UnifiedAdapter(text_width)

            if mode == 'LUCIR':
                self.vision_proj = CosineLinear(vision_width, embed_dim)
                self.text_proj = CosineLinear(text_width, embed_dim)
            else:
                self.vision_proj = nn.Linear(vision_width, embed_dim)
                self.text_proj = nn.Linear(text_width, embed_dim)

        else:
            self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
            if vit=='base':
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu", check_hash=True)
                state_dict = checkpoint["model"]
                msg = self.visual_encoder.load_state_dict(state_dict,strict=False)

            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese') #Download ‘bert-base-chinese’ in advance and save it to a local path
            encoder_config = BertConfig.from_json_file(med_config)
            encoder_config.encoder_width = vision_width

            self.text_mlm_encoder = BertForMaskedLM.from_pretrained('bert-base-chinese',config=encoder_config)
            self.text_encoder = self.text_mlm_encoder.bert

            text_width = self.text_encoder.config.hidden_size

            self.visual_width = vision_width
            self.text_width = text_width

            if mode == 'LUCIR':
                self.vision_proj = CosineLinear(vision_width, embed_dim)
                self.text_proj = CosineLinear(text_width, embed_dim)
            else:
                self.vision_proj = nn.Linear(vision_width, embed_dim)
                self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(0.07 * torch.ones([])).requires_grad_(False)
        self.distill_temp = nn.Parameter(1.0 * torch.ones([])).requires_grad_(False)
        self.mlm_probability = 0.15
        self.momentum = 0.995

        self.queue_size = 1024
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def next_task(self):
        self.current_task_index += 1
        if self.unified_adapter:
            self.unified_adapter_pool.next_task(device=self.text_mlm_encoder.device)
            self.visual_adapter_current_task = []
            for i in range(self.visual_layers):
                self.visual_adapter_current_task.append(self.unified_adapter_pool.get_visual_adapter(i))

            self.text_adapter_current_task = []
            for i in range(self.text_layers):
                self.text_adapter_current_task.append(self.unified_adapter_pool.get_text_adapter(i))

            # for name, p in self.text_mlm_encoder.named_parameters():
            #     if 'intermediate' in name or ('output' in name and 'attention' not in name):
            #         p.requires_grad = False
            # for name, p in self.visual_encoder.named_parameters():
            #     if 'mlp' in name:
            #         p.requires_grad = False
            #
            # for name, p in self.text_mlm_encoder.named_parameters():
            #     if 'attention' in name:
            #         p.requires_grad = False
            # for name, p in self.visual_encoder.named_parameters():
            #     if 'self_attn' in name:
            #         p.requires_grad = False

            if self.current_task_index == 0 and not self.fine_tune:
                for p in self.text_mlm_encoder.parameters():
                    p.requires_grad = False
                for p in self.visual_encoder.parameters():
                    p.requires_grad = False

            if self.current_task_index == 0 and (self.two_stage or self.convnorm):
                for p in self.text_mlm_encoder.parameters():
                    p.requires_grad = False
                for p in self.visual_encoder.parameters():
                    p.requires_grad = False

                for name, p in self.finetune_text_mlm_encoder.named_parameters():
                    if 'attention' in name and 'LayerNorm' not in name:
                        p.requires_grad = True
                for name, p in self.finetune_visual_encoder.named_parameters():
                    if 'self_attn' in name:
                        p.requires_grad = True

                # for name, p in self.finetune_text_mlm_encoder.named_parameters():
                #     if 'intermediate' in name or ('output' in name and 'attention' not in name) or ('attention' in name and 'LayerNorm' in name):
                #         p.requires_grad = False
                # for name, p in self.finetune_visual_encoder.named_parameters():
                #     if 'mlp' in name or ('self_attn' in name and 'layer_norm' in name):
                #         p.requires_grad = False

                pass

        elif self.unified_prompt:
            self.unified_prompt_pool.next_task()
            if self.current_task_index == 0 and not self.fine_tune:
                for p in self.text_mlm_encoder.parameters():
                    p.requires_grad = False
                for p in self.visual_encoder.parameters():
                    p.requires_grad = False
        else:
            cls_factor = 0.01
            train_projection_layer = False
            if self.text_prompt_per_task > 0:
                if self.text_deep_prompt and len(self.text_prompt_list) == 0:
                    s = self.prompts_start_layer
                    e = self.text_layers-1 if self.prompts_end_layer == -1 else self.prompts_end_layer
                    for i in range(e-s+1):
                        self.text_prompt_list.append(nn.ParameterList())
                if not self.fine_tune:
                    for p in self.text_mlm_encoder.parameters():
                        p.requires_grad = False


                if train_projection_layer:
                    for p in self.text_mlm_encoder.bert.last_projection.parameters():
                        p.requires_grad = True

                if self.text_deep_prompt:
                    for i in range(len(self.text_prompt_list)):
                        for j in range(len(self.text_prompt_list[i])):
                            self.text_prompt_list[i][j].requires_grad = False
                else:
                    for i in range(len(self.text_prompt_list)):
                        self.text_prompt_list[i].requires_grad = False

                if self.prompt_init == "random":
                    if self.text_deep_prompt:
                        for i in range(len(self.text_prompt_list)):
                            prompts = nn.Parameter(torch.randn(1, self.text_prompt_per_task, self.text_width)
                                                   ).to(self.text_mlm_encoder.device)
                            prompts = F.layer_norm(prompts, prompts.shape[2:])
                            self.text_prompt_list[i].append(prompts)
                        self.text_prompt_num += self.text_prompt_per_task

                    else:
                        prompts = nn.Parameter(torch.randn(1, self.text_prompt_per_task, self.text_width)
                                               ).to(self.text_mlm_encoder.device)
                        prompts = F.layer_norm(prompts, prompts.shape[2:])
                        self.text_prompt_num += self.text_prompt_per_task
                        self.text_prompt_list.append(prompts)
                elif self.prompt_init == "cls_token":
                    if self.text_deep_prompt:
                        for i in range(len(self.text_prompt_list)):
                            prompts = nn.Parameter(cls_factor * torch.randn(1, self.text_prompt_per_task, self.text_width)
                                                   ).to(self.text_mlm_encoder.device)
                            if self.chinese_clip:
                                cls_token = copy.deepcopy(
                                    self.text_encoder.text_model.embeddings.word_embeddings.weight[
                                    self.tokenizer.cls_token_id, :]).detach()
                            elif self.clip:
                                cls_token = copy.deepcopy(
                                    self.text_encoder.text_model.embeddings.token_embedding.weight[
                                    self.tokenizer.eos_token_id, :]).detach()
                            else:
                                raise NotImplementedError
                            prompts += cls_token
                            prompts = F.layer_norm(prompts, prompts.shape[2:])
                            self.text_prompt_list[i].append(prompts)
                        self.text_prompt_num += self.text_prompt_per_task
                    else:
                        prompts = nn.Parameter(cls_factor * torch.randn(1, self.text_prompt_per_task, self.text_width)
                                               ).to(self.text_mlm_encoder.device)
                        if self.chinese_clip:
                            cls_token = copy.deepcopy(self.text_encoder.text_model.embeddings.word_embeddings.weight[self.tokenizer.cls_token_id, :]).detach()
                        elif self.clip:
                            cls_token = copy.deepcopy(self.text_encoder.text_model.embeddings.token_embedding.weight[self.tokenizer.eos_token_id, :]).detach()
                        else:
                            raise NotImplementedError
                        prompts += cls_token
                        prompts = F.layer_norm(prompts, prompts.shape[2:])
                        self.text_prompt_num += self.text_prompt_per_task
                        self.text_prompt_list.append(prompts)
                else:
                    raise NotImplementedError

            if self.visual_prompt_per_task > 0:
                if self.visual_deep_prompt and len(self.visual_prompt_list) == 0:
                    s = self.prompts_start_layer
                    e = self.visual_layers-1 if self.prompts_end_layer == -1 else self.prompts_end_layer
                    for i in range(e-s+1):
                        self.visual_prompt_list.append(nn.ParameterList())
                if not self.fine_tune:
                    for p in self.visual_encoder.parameters():
                        p.requires_grad = False

                if train_projection_layer:
                    for p in self.visual_encoder.clip_model.visual_projection.parameters():
                        p.requires_grad = True

                if self.visual_deep_prompt:
                    for i in range(len(self.visual_prompt_list)):
                        for j in range(len(self.visual_prompt_list[i])):
                            self.visual_prompt_list[i][j].requires_grad = False
                else:
                    for i in range(len(self.visual_prompt_list)):
                        self.visual_prompt_list[i].requires_grad = False
                if self.prompt_init == "random":
                    scale = self.visual_width ** -0.5
                    if self.visual_deep_prompt:
                        for i in range(len(self.visual_prompt_list)):
                            prompts = nn.Parameter(scale * torch.randn(1, self.visual_prompt_per_task, self.visual_width)
                                                   ).to(self.visual_encoder.device)
                            prompts = F.layer_norm(prompts, prompts.shape[2:])
                            self.visual_prompt_list[i].append(prompts)
                        self.visual_prompt_num += self.visual_prompt_per_task
                    else:
                        prompts = nn.Parameter(scale * torch.randn(1, self.visual_prompt_per_task, self.visual_width)
                                               ).to(self.visual_encoder.device)
                        prompts = F.layer_norm(prompts, prompts.shape[2:])
                        self.visual_prompt_num += self.visual_prompt_per_task
                        self.visual_prompt_list.append(prompts)
                elif self.prompt_init == "cls_token":
                    if self.visual_deep_prompt:
                        for i in range(len(self.visual_prompt_list)):
                            prompts = nn.Parameter(cls_factor * torch.randn(1, self.visual_prompt_per_task, self.visual_width)
                                                   ).to(self.visual_encoder.device)
                            cls_token = copy.deepcopy(
                                self.visual_encoder.clip_model.vision_model.embeddings.class_embedding).detach()
                            prompts += cls_token
                            prompts = F.layer_norm(prompts, prompts.shape[2:])
                            self.visual_prompt_list[i].append(prompts)
                        self.visual_prompt_num += self.visual_prompt_per_task
                    else:
                        prompts = nn.Parameter(cls_factor * torch.randn(1, self.visual_prompt_per_task, self.visual_width)
                                               ).to(self.visual_encoder.device)
                        cls_token = copy.deepcopy(self.visual_encoder.clip_model.vision_model.embeddings.class_embedding).detach()
                        prompts += cls_token
                        prompts = F.layer_norm(prompts, prompts.shape[2:])
                        self.visual_prompt_num += self.visual_prompt_per_task
                        self.visual_prompt_list.append(prompts)
                else:
                    raise NotImplementedError

        if self.fine_tune:
            # Fix mlp
            # if self.clip:
            #     for name, p in self.text_mlm_encoder.named_parameters():
            #         if 'mlp' in name:
            #             p.requires_grad = False
            # else:
            #     for name, p in self.text_mlm_encoder.named_parameters():
            #         if 'intermediate' in name or ('output' in name and 'attention' not in name):
            #             p.requires_grad = False
            # for name, p in self.visual_encoder.named_parameters():
            #     if 'mlp' in name:
            #         p.requires_grad = False

            # Fix attention
            # if self.clip:
            #     for name, p in self.text_mlm_encoder.named_parameters():
            #         if 'self_attn' in name:
            #             p.requires_grad = False
            # else:
            #     for name, p in self.text_mlm_encoder.named_parameters():
            #         if 'attention' in name:
            #             p.requires_grad = False
            # if self.hybrid_clip:
            #     for name, p in self.visual_encoder.named_parameters():
            #         if 'attn' in name:
            #             p.requires_grad = False
            # else:
            #     for name, p in self.visual_encoder.named_parameters():
            #         if 'self_attn' in name:
            #             p.requires_grad = False
            pass
    def get_raw_VL_feature(self, image,caption): #used in MAS
        visual_prompts = None
        if self.unified_adapter:
            visual_prompts = self.visual_adapter_current_task
        elif self.unified_prompt:
            with torch.no_grad():
                ref_image_embeds = self.fixed_visual_encoder(image)
                ref_image_feature = ref_image_embeds[:, 0, :]
            visual_prompts = []
            for i in range(self.visual_layers):
                visual_prompts.append(
                    self.unified_prompt_pool.get_visual_prompt(ref_image_feature, i))
        else:
            if self.visual_prompt_num > 0:
                if self.visual_deep_prompt:
                    visual_prompts = []
                    for i in range(len(self.visual_prompt_list)):
                        visual_prompts.append(torch.cat(list(self.visual_prompt_list[i]), dim=1))
                else:
                    visual_prompts = torch.cat(list(self.visual_prompt_list), dim=1)
        image_embeds = self.visual_encoder(image, prompts=visual_prompts)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        image_feat = self.vision_proj(image_embeds[:,0,:])

        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_words, 
                              return_tensors="pt").to(image.device)
        text_prompts = None
        if self.unified_adapter:
            text_prompts = self.text_adapter_current_task
        elif self.unified_prompt:
            with torch.no_grad():
                ref_text = self.tokenizer(caption, padding='max_length', truncation=True,
                                           max_length=self.max_words,
                                           return_tensors="pt").to(image.device)
                ref_mode = 'multi_modal'
                ref_text_output = self.fixed_text_encoder(ref_text.input_ids,
                                                           attention_mask=ref_text.attention_mask,
                                                           return_dict=True, mode=ref_mode)
                ref_text_feature = ref_text_output.last_hidden_state[:, 0, :]
            text_prompts = []
            for i in range(self.text_layers):
                text_prompts.append(self.unified_prompt_pool.get_text_prompt(ref_text_feature, i))
        else:
            if self.text_prompt_num > 0:
                if self.text_deep_prompt:
                    text_prompts = []
                    for i in range(len(self.text_prompt_list)):
                        text_prompts.append(torch.cat(list(self.text_prompt_list[i]), dim=1))
                else:
                    text_prompts = torch.cat(list(self.text_prompt_list), dim=1)
        mode = 'multi_modal' if (text_prompts is not None or self.pretrained_proj) else 'text'
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = mode, prompts=text_prompts)
        text_feat = self.text_proj(text_output.last_hidden_state[:,0,:])

        mlm_output = self.text_mlm_encoder.bert(text.input_ids, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,return_dict = True, prompts=text_prompts)
        fusion_out = mlm_output.last_hidden_state[:,0,:]
        return image_feat, text_feat, fusion_out
        
    def get_raw_feature(self, image,caption):
        visual_prompts = None
        if self.unified_adapter:
            visual_prompts = self.visual_adapter_current_task
        elif self.unified_prompt:
            with torch.no_grad():
                ref_image_embeds = self.fixed_visual_encoder(image)
                ref_image_feature = ref_image_embeds[:, 0, :]
            visual_prompts = []
            for i in range(self.visual_layers):
                visual_prompts.append(
                    self.unified_prompt_pool.get_visual_prompt(ref_image_feature, i))
        else:
            if self.visual_prompt_num > 0:
                if self.visual_deep_prompt:
                    visual_prompts = []
                    for i in range(len(self.visual_prompt_list)):
                        visual_prompts.append(torch.cat(list(self.visual_prompt_list[i]), dim=1))
                else:
                    visual_prompts = torch.cat(list(self.visual_prompt_list), dim=1)
        image_embeds = self.visual_encoder(image, prompts= visual_prompts)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        image_feature = self.vision_proj(image_embeds[:,0,:])
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_words, 
                              return_tensors="pt").to(image.device)
        text_prompts = None
        if self.unified_adapter:
            text_prompts = self.text_adapter_current_task
        elif self.unified_prompt:
            with torch.no_grad():
                ref_text = self.tokenizer(caption, padding='max_length', truncation=True,
                                          max_length=self.max_words,
                                          return_tensors="pt").to(image.device)
                ref_mode = 'multi_modal'
                ref_text_output = self.fixed_text_encoder(ref_text.input_ids,
                                                          attention_mask=ref_text.attention_mask,
                                                          return_dict=True, mode=ref_mode)
                ref_text_feature = ref_text_output.last_hidden_state[:, 0, :]
            text_prompts = []
            for i in range(self.text_layers):
                text_prompts.append(self.unified_prompt_pool.get_text_prompt(ref_text_feature, i))
        else:
            if self.text_prompt_num > 0:
                if self.text_deep_prompt:
                    text_prompts = []
                    for i in range(len(self.text_prompt_list)):
                        text_prompts.append(torch.cat(list(self.text_prompt_list[i]), dim=1))
                else:
                    text_prompts = torch.cat(list(self.text_prompt_list), dim=1)
        mode = 'multi_modal' if (text_prompts is not None or self.pretrained_proj) else 'text'
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,
                                        return_dict = True, mode = mode, prompts= text_prompts)
        text_feature = self.text_proj(text_output.last_hidden_state[:,0,:])
        return image_feature, text_feature, image_embeds, image_atts, text, text_output

    def get_feature(self, image,caption):
        image_feature, text_feature, image_embeds, image_atts, text, text_output = self.get_raw_feature(image,caption)
        image_feat = F.normalize(image_feature,dim=-1) 
        text_feat = F.normalize(text_feature,dim=-1)
        return image_feat, text_feat, image_embeds, image_atts, text, text_output
       
    def get_VL_feature(self, image,caption): 
        #get the multimodal fusion feature
        visual_prompts = None
        if self.unified_adapter:
            visual_prompts = self.visual_adapter_current_task
        elif self.unified_prompt:
            with torch.no_grad():
                ref_image_embeds = self.fixed_visual_encoder(image)
                ref_image_feature = ref_image_embeds[:, 0, :]
            visual_prompts = []
            for i in range(self.visual_layers):
                visual_prompts.append(
                    self.unified_prompt_pool.get_visual_prompt(ref_image_feature, i))
        else:
            if self.visual_prompt_num > 0:
                if self.visual_deep_prompt:
                    visual_prompts = []
                    for i in range(len(self.visual_prompt_list)):
                        visual_prompts.append(torch.cat(list(self.visual_prompt_list[i]), dim=1))
                else:
                    visual_prompts = torch.cat(list(self.visual_prompt_list), dim=1)
        image_embeds = self.visual_encoder(image, prompts= visual_prompts)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_words, 
                              return_tensors="pt").to(image.device)
        text_prompts = None
        if self.unified_adapter:
            text_prompts = self.text_adapter_current_task
        elif self.unified_prompt:
            with torch.no_grad():
                ref_text = self.tokenizer(caption, padding='max_length', truncation=True,
                                          max_length=self.max_words,
                                          return_tensors="pt").to(image.device)
                ref_mode = 'multi_modal'
                ref_text_output = self.fixed_text_encoder(ref_text.input_ids,
                                                          attention_mask=ref_text.attention_mask,
                                                          return_dict=True, mode=ref_mode)
                ref_text_feature = ref_text_output.last_hidden_state[:, 0, :]
            text_prompts = []
            for i in range(self.text_layers):
                text_prompts.append(self.unified_prompt_pool.get_text_prompt(ref_text_feature, i))
        else:
            if self.text_prompt_num > 0:
                if self.text_deep_prompt:
                    text_prompts = []
                    for i in range(len(self.text_prompt_list)):
                        text_prompts.append(torch.cat(list(self.text_prompt_list[i]), dim=1))
                else:
                    text_prompts = torch.cat(list(self.text_prompt_list), dim=1)
        mlm_output = self.text_mlm_encoder.bert(text.input_ids, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,return_dict = True, prompts= text_prompts)
        
        fusion_out = mlm_output.last_hidden_state[:,0,:]
        fusion_out = F.normalize(fusion_out,dim=-1)
        return fusion_out

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def get_mlm_loss(self,text,image_embeds,image_atts ,device):        
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, len(self.tokenizer), device, targets=labels,
                                    probability_matrix = probability_matrix)
        text_prompts = None
        if self.text_prompt_num > 0:
            if self.text_deep_prompt:
                text_prompts = []
                for i in range(len(self.text_prompt_list)):
                    text_prompts.append(torch.cat(list(self.text_prompt_list[i]), dim=1))
                labels = torch.cat([-100 * torch.ones(labels.shape[0], text_prompts[0].shape[1], dtype=torch.long).to(labels.device),
                                    labels], dim=1)
            else:
                text_prompts = torch.cat(list(self.text_prompt_list), dim=1)
                labels = torch.cat([-100 * torch.ones(labels.shape[0], text_prompts.shape[1], dtype=torch.long).to(labels.device),
                                    labels], dim=1)

        mlm_output = self.text_mlm_encoder(input_ids = input_ids, 
                                    attention_mask = text.attention_mask,
                                    encoder_hidden_states = image_embeds,
                                    encoder_attention_mask = image_atts,      
                                    return_dict = True,
                                    labels = labels, prompts= text_prompts
                                    ) 
        return mlm_output, input_ids, labels
    
    def distill_mlm(self, logit_mlm, ref_logits, labels):
        temp =self.distill_temp
        loss_mlm_dis = -torch.sum(F.log_softmax(logit_mlm/temp, dim=-1)*F.softmax(ref_logits/temp,dim=-1),dim=-1)
        loss_mlm_dis = loss_mlm_dis[labels!=-100].mean()
        return loss_mlm_dis

    def stage_one(self, image, caption):
        image_embeds = self.finetune_visual_encoder(image, prompts=None)
        image_feature = self.vision_proj(image_embeds[:, 0, :])
        text = self.tokenizer(caption, padding='max_length', truncation=True,
                              max_length=self.max_words,
                              return_tensors="pt").to(image.device)
        text_output = self.finetune_text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='multi_modal', prompts=None)
        text_feature = self.text_proj(text_output.last_hidden_state[:, 0, :])

        image_feat = F.normalize(image_feature, dim=-1)
        text_feat = F.normalize(text_feature, dim=-1)

        batch_size = image.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long,
                              device=image.device) + batch_size * dist.get_rank()

        sim_i2t = image_feat @ all_gather_with_grad(text_feat).T
        sim_t2i = text_feat @ all_gather_with_grad(image_feat).T

        loss_i2t = nn.CrossEntropyLoss()(sim_i2t / self.temp, labels)
        loss_t2i = nn.CrossEntropyLoss()(sim_t2i / self.temp, labels)

        loss_ita = (loss_i2t + loss_t2i) / 2

        return loss_ita

    def to_stage_two_pretrained(self):
        self.is_stage_one = False
        self.visual_encoder = self.finetune_visual_encoder
        self.text_mlm_encoder = self.finetune_text_mlm_encoder
        self.text_encoder = self.finetune_text_encoder

        for p in self.finetune_visual_encoder.parameters(): p.requires_grad = False
        for p in self.finetune_text_mlm_encoder.parameters(): p.requires_grad = False
        for p in self.visual_encoder.parameters(): p.requires_grad = False
        for p in self.text_mlm_encoder.parameters(): p.requires_grad = False

    @torch.no_grad()
    def stage_two_feat(self, data_loader, device):
        # return
        # feat_dir = "convnorm_feat"
        feat_dir = "/mnt/log/code/CTP_convnorm_feat"
        rank = dist.get_rank()
        if rank == 0:
            if not os.path.exists(feat_dir): os.mkdir(feat_dir)
        dist.barrier()

        for i, batch in enumerate(data_loader):
            print(f"{i}-th iter")
            if i * batch[1].shape[0] * dist.get_world_size() >= 8 * 512:
                break

            id, image, caption = batch

            image = image.to(device, non_blocking=True)
            _, finetune_visual_hidden_states = self.finetune_visual_encoder(image,
                                                                       prompts=None,
                                                                       output_hidden_states=True)

            for j in range(12):
                vis_norm = self.finetune_visual_encoder.vision_model.encoder.encoder.layers[j].encoder_layer.layer_norm1
                vis_attn = self.finetune_visual_encoder.vision_model.encoder.encoder.layers[j].encoder_layer.self_attn
                fixed_vis_attn = self.visual_encoder.vision_model.encoder.encoder.layers[j].encoder_layer.self_attn
                # fixed_vis_attn = self.fixed_visual_encoder.vision_model.encoder.encoder.layers[j].encoder_layer.self_attn

                vis_x = vis_norm(finetune_visual_hidden_states[j])
                vis_y_q, vis_y_k, vis_y_v, vis_x_p, vis_y_p = vis_attn.get_intermediate(vis_x)

                fixed_vis_y_q, fixed_vis_y_k, fixed_vis_y_v, fixed_vis_x_p, fixed_vis_y_p = fixed_vis_attn.get_intermediate(vis_x)

                np.save(f"{feat_dir}/feat_visual_x_batch{i}_layer{j}_rank{rank}", vis_x.detach().cpu().numpy())
                np.save(f"{feat_dir}/feat_visual_q_batch{i}_layer{j}_rank{rank}", (vis_y_q - fixed_vis_y_q).detach().cpu().numpy())
                np.save(f"{feat_dir}/feat_visual_k_batch{i}_layer{j}_rank{rank}", (vis_y_k - fixed_vis_y_k).detach().cpu().numpy())
                np.save(f"{feat_dir}/feat_visual_v_batch{i}_layer{j}_rank{rank}", (vis_y_v - fixed_vis_y_v).detach().cpu().numpy())
                np.save(f"{feat_dir}/feat_visual_p_x_batch{i}_layer{j}_rank{rank}", vis_x_p.detach().cpu().numpy())
                np.save(f"{feat_dir}/feat_visual_p_y_batch{i}_layer{j}_rank{rank}", (vis_y_p - fixed_vis_y_p).detach().cpu().numpy())
            del vis_x, vis_y_q, vis_y_k, vis_y_v, vis_x_p, vis_y_p, fixed_vis_y_q, fixed_vis_y_k, fixed_vis_y_v, fixed_vis_x_p, fixed_vis_y_p
            text = self.tokenizer(caption, padding='max_length', truncation=True,
                                  max_length=self.max_words,
                                  return_tensors="pt").to(image.device)
            text_output = self.finetune_text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                                     return_dict=True, mode='multi_modal', prompts=None,
                                                     output_hidden_states=True)
            finetune_text_hidden_states = text_output.hidden_states

            attn_mask = self.finetune_text_encoder.text_model.get_extended_attention_mask(
                text.attention_mask, text.input_ids.shape)
            for j in range(12):
                text_attn = self.finetune_text_encoder.text_model.encoder.clip_text_encoder.layer[j].layer_module.attention.attention.self.attention
                text_x = finetune_text_hidden_states[j]
                text_y_q, text_y_k, text_y_v = text_attn.query(text_x), text_attn.key(text_x), text_attn.value(text_x)

                fixed_text_attn = self.text_encoder.text_model.encoder.clip_text_encoder.layer[
                    j].layer_module.attention.attention.self.attention
                # fixed_text_attn = self.fixed_text_encoder.text_model.encoder.clip_text_encoder.layer[
                #     j].layer_module.attention.attention.self.attention
                fixed_text_y_q, fixed_text_y_k, fixed_text_y_v = fixed_text_attn.query(text_x), fixed_text_attn.key(
                    text_x), fixed_text_attn.value(text_x)

                text_p_x, text_p_y = \
                self.finetune_text_encoder.text_model.encoder.clip_text_encoder.layer[
                    j].layer_module.attention.get_intermediate(text_x, attn_mask)

                fixed_text_p_x, fixed_text_p_y = \
                self.text_encoder.text_model.encoder.clip_text_encoder.layer[
                    j].layer_module.attention.get_intermediate(text_x, attn_mask)

                np.save(f"{feat_dir}/feat_text_x_batch{i}_layer{j}_rank{rank}", text_x.detach().cpu().numpy())
                np.save(f"{feat_dir}/feat_text_q_batch{i}_layer{j}_rank{rank}", (text_y_q - fixed_text_y_q).detach().cpu().numpy())
                np.save(f"{feat_dir}/feat_text_k_batch{i}_layer{j}_rank{rank}", (text_y_k - fixed_text_y_k).detach().cpu().numpy())
                np.save(f"{feat_dir}/feat_text_v_batch{i}_layer{j}_rank{rank}", (text_y_v - fixed_text_y_v).detach().cpu().numpy())
                np.save(f"{feat_dir}/feat_text_p_x_batch{i}_layer{j}_rank{rank}", text_p_x.detach().cpu().numpy())
                np.save(f"{feat_dir}/feat_text_p_y_batch{i}_layer{j}_rank{rank}", (text_p_y - fixed_text_p_y).detach().cpu().numpy())

            del text_x, text_y_q, text_y_k, text_y_v, fixed_text_y_q, fixed_text_y_k, fixed_text_y_v, text_p_x, text_p_y, fixed_text_p_x, fixed_text_p_y


    def stage_two_pca(self):
        rank = dist.get_rank()
        # feat_dir = "convnorm_feat"
        feat_dir = "/mnt/log/code/CTP_convnorm_feat"
        if rank == 0:
            for l in range(12):
                print(f"PCA visual init for layer {l}")
                feat_visual_x = []
                for fname in os.listdir(feat_dir):
                    if "feat_visual_x" in fname and f"layer{l}_" in fname:
                        feat = np.load(f"{feat_dir}/"+fname)
                        feat_visual_x.append(feat.reshape(-1, 768))
                feat_visual_x = np.concatenate(feat_visual_x, axis=0)

                self.unified_adapter_pool.visual_adapter_pool[l].init_convnorm(x=feat_visual_x)
                del feat_visual_x

                feat_visual_q = []
                for fname in os.listdir(feat_dir):
                    if "feat_visual_q" in fname and f"layer{l}_" in fname:
                        feat = np.load(f"{feat_dir}/"+fname)
                        feat_visual_q.append(feat.reshape(-1, 768))
                feat_visual_q = np.concatenate(feat_visual_q, axis=0)
                self.unified_adapter_pool.visual_adapter_pool[l].init_convnorm(q=feat_visual_q)
                del feat_visual_q

                feat_visual_k = []
                for fname in os.listdir(feat_dir):
                    if "feat_visual_k" in fname and f"layer{l}_" in fname:
                        feat = np.load(f"{feat_dir}/"+fname)
                        feat_visual_k.append(feat.reshape(-1, 768))
                feat_visual_k = np.concatenate(feat_visual_k, axis=0)

                self.unified_adapter_pool.visual_adapter_pool[l].init_convnorm(k=feat_visual_k)
                del feat_visual_k

                feat_visual_v = []
                for fname in os.listdir(feat_dir):
                    if "feat_visual_v" in fname and f"layer{l}_" in fname:
                        feat = np.load(f"{feat_dir}/"+fname)
                        feat_visual_v.append(feat.reshape(-1, 768))
                feat_visual_v = np.concatenate(feat_visual_v, axis=0)

                self.unified_adapter_pool.visual_adapter_pool[l].init_convnorm(v=feat_visual_v)
                del feat_visual_v

                feat_visual_p_x = []
                for fname in os.listdir(feat_dir):
                    if "feat_visual_p_x" in fname and f"layer{l}_" in fname:
                        feat = np.load(f"{feat_dir}/"+fname)
                        feat_visual_p_x.append(feat.reshape(-1, 768))
                feat_visual_p_x = np.concatenate(feat_visual_p_x, axis=0)

                self.unified_adapter_pool.visual_adapter_pool[l].init_convnorm(px=feat_visual_p_x)
                del feat_visual_p_x

                feat_visual_p_y = []
                for fname in os.listdir(feat_dir):
                    if "feat_visual_p_y" in fname and f"layer{l}_" in fname:
                        feat = np.load(f"{feat_dir}/"+fname)
                        feat_visual_p_y.append(feat.reshape(-1, 768))
                feat_visual_p_y = np.concatenate(feat_visual_p_y, axis=0)

                self.unified_adapter_pool.visual_adapter_pool[l].init_convnorm(py=feat_visual_p_y)
                del feat_visual_p_y

                print(f"PCA text init for layer {l}")
                feat_text_x = []
                for fname in os.listdir(feat_dir):
                    if "feat_text_x" in fname and f"layer{l}_" in fname:
                        feat = np.load(f"{feat_dir}/" + fname)
                        feat_text_x.append(feat.reshape(-1, 768))
                feat_text_x = np.concatenate(feat_text_x, axis=0)

                self.unified_adapter_pool.text_adapter_pool[l].init_convnorm(x=feat_text_x)
                del feat_text_x

                feat_text_q = []
                for fname in os.listdir(feat_dir):
                    if "feat_text_q" in fname and f"layer{l}_" in fname:
                        feat = np.load(f"{feat_dir}/" + fname)
                        feat_text_q.append(feat.reshape(-1, 768))
                feat_text_q = np.concatenate(feat_text_q, axis=0)
                self.unified_adapter_pool.text_adapter_pool[l].init_convnorm(q=feat_text_q)
                del feat_text_q

                feat_text_k = []
                for fname in os.listdir(feat_dir):
                    if "feat_text_k" in fname and f"layer{l}_" in fname:
                        feat = np.load(f"{feat_dir}/" + fname)
                        feat_text_k.append(feat.reshape(-1, 768))
                feat_text_k = np.concatenate(feat_text_k, axis=0)

                self.unified_adapter_pool.text_adapter_pool[l].init_convnorm(k=feat_text_k)
                del feat_text_k

                feat_text_v = []
                for fname in os.listdir(feat_dir):
                    if "feat_text_v" in fname and f"layer{l}_" in fname:
                        feat = np.load(f"{feat_dir}/" + fname)
                        feat_text_v.append(feat.reshape(-1, 768))
                feat_text_v = np.concatenate(feat_text_v, axis=0)

                self.unified_adapter_pool.text_adapter_pool[l].init_convnorm(v=feat_text_v)
                del feat_text_v

                feat_text_p_x = []
                for fname in os.listdir(feat_dir):
                    if "feat_text_p_x" in fname and f"layer{l}_" in fname:
                        feat = np.load(f"{feat_dir}/" + fname)
                        feat_text_p_x.append(feat.reshape(-1, 768))
                feat_text_p_x = np.concatenate(feat_text_p_x, axis=0)

                self.unified_adapter_pool.text_adapter_pool[l].init_convnorm(px=feat_text_p_x)
                del feat_text_p_x

                feat_text_p_y = []
                for fname in os.listdir(feat_dir):
                    if "feat_text_p_y" in fname and f"layer{l}_" in fname:
                        feat = np.load(f"{feat_dir}/" + fname)
                        feat_text_p_y.append(feat.reshape(-1, 768))
                feat_text_p_y = np.concatenate(feat_text_p_y, axis=0)

                self.unified_adapter_pool.text_adapter_pool[l].init_convnorm(py=feat_text_p_y)
                del feat_text_p_y

        dist.barrier()

    def to_stage_three(self):
        self.is_stage_one = False
        for p in self.finetune_visual_encoder.parameters(): p.requires_grad = False
        for p in self.finetune_text_mlm_encoder.parameters(): p.requires_grad = False

        # for p in self.unified_adapter_pool.text_adapter_pool.parameters(): p.requires_grad = False
        # for p in self.unified_adapter_pool.visual_adapter_pool.parameters(): p.requires_grad = False

    def to_stage_two(self):
        self.is_stage_one = False
        for p in self.finetune_visual_encoder.parameters(): p.requires_grad = False
        for p in self.finetune_text_mlm_encoder.parameters(): p.requires_grad = False

        visual_list = self.finetune_visual_encoder.get_attn_weights()
        text_list = self.finetune_text_encoder.get_attn_weights()
        diff = True
        if diff:
            fixed_visual_list = self.visual_encoder.get_attn_weights()
            fixed_text_list = self.fixed_text_encoder.get_attn_weights()
            with torch.no_grad():
                for i in range(len(fixed_visual_list)):
                    for j in range(len(fixed_visual_list[i])):
                        tmp_layer = copy.deepcopy(visual_list[i][j])
                        tmp_layer.weight.copy_(visual_list[i][j].weight - fixed_visual_list[i][j].weight)
                        tmp_layer.bias.copy_(visual_list[i][j].bias - fixed_visual_list[i][j].bias)
                        visual_list[i][j] = tmp_layer
                for i in range(len(fixed_text_list)):
                    for j in range(len(fixed_text_list[i])):
                        tmp_layer = copy.deepcopy(text_list[i][j])
                        tmp_layer.weight.copy_(text_list[i][j].weight - fixed_text_list[i][j].weight)
                        tmp_layer.bias.copy_(text_list[i][j].bias - fixed_text_list[i][j].bias)
                        text_list[i][j] = tmp_layer

        print("Init adapters with pca")
        self.unified_adapter_pool.init_with_qkv_list(visual_list, text_list)

    def stage_two(self, image, caption):
        with torch.no_grad():
            image_embeds, finetune_visual_hidden_states = self.finetune_visual_encoder(image, prompts=None, output_hidden_states=True)
            image_feature = self.vision_proj(image_embeds[:, 0, :])
            text = self.tokenizer(caption, padding='max_length', truncation=True,
                                  max_length=self.max_words,
                                  return_tensors="pt").to(image.device)
            text_output = self.finetune_text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                                     return_dict=True, mode='multi_modal', prompts=None, output_hidden_states=True)
            finetune_text_hidden_states = text_output.hidden_states
            text_feature = self.text_proj(text_output.last_hidden_state[:, 0, :])

            finetune_image_feat = F.normalize(image_feature, dim=-1)
            finetune_text_feat = F.normalize(text_feature, dim=-1)

        visual_prompts = self.visual_adapter_current_task
        image_embeds, visual_hidden_states = self.visual_encoder(image, prompts=visual_prompts, output_hidden_states=True)
        image_feature = self.vision_proj(image_embeds[:, 0, :])
        text = self.tokenizer(caption, padding='max_length', truncation=True,
                              max_length=self.max_words,
                              return_tensors="pt").to(image.device)
        text_prompts = self.text_adapter_current_task
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='multi_modal', prompts=text_prompts, output_hidden_states=True)
        text_hidden_states = text_output.hidden_states
        text_feature = self.text_proj(text_output.last_hidden_state[:, 0, :])
        image_feat = F.normalize(image_feature, dim=-1)
        text_feat = F.normalize(text_feature, dim=-1)

        loss_dist = nn.MSELoss()(image_feat, finetune_image_feat) + nn.MSELoss()(text_feat, finetune_text_feat)

        batch_size = image.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long,
                              device=image.device) + batch_size * dist.get_rank()

        sim_i2t = image_feat @ all_gather_with_grad(text_feat).T
        sim_t2i = text_feat @ all_gather_with_grad(image_feat).T

        loss_i2t = nn.CrossEntropyLoss()(sim_i2t / self.temp, labels)
        loss_t2i = nn.CrossEntropyLoss()(sim_t2i / self.temp, labels)

        loss_ita = (loss_i2t + loss_t2i) / 2

        return loss_ita, loss_dist

        # Distillation losses, not very useful

        # N_deep = (len(finetune_visual_hidden_states) - self.prompts_start_layer)
        # for i in range(self.prompts_start_layer, len(finetune_visual_hidden_states)):
        #     for j in range(3):
        #         loss_deep_dist = loss_deep_dist + nn.MSELoss()(finetune_visual_hidden_states[i][j][:, 0, :],
        #                                                        visual_hidden_states[i][j][:, 0, :]) / N_deep
        #         loss_deep_dist = loss_deep_dist + nn.MSELoss()(finetune_text_hidden_states[i][j][:, :, 0, :],
        #                                                        text_hidden_states[i][j][:, :, 0, :]) / N_deep

        # N_deep = (len(finetune_visual_hidden_states) - self.prompts_start_layer - 1)
        # for i in range(1 + self.prompts_start_layer, len(finetune_visual_hidden_states)):
        #     loss_deep_dist = loss_deep_dist + nn.MSELoss()(
        #         finetune_visual_hidden_states[i][:, 0, :],
        #         visual_hidden_states[i][:, 0, :]) / N_deep
        #     loss_deep_dist = loss_deep_dist + nn.MSELoss()(
        #         finetune_text_hidden_states[i][:, 0, :],
        #         text_hidden_states[i][:, 0, :]) / N_deep
        #
        # return loss_dist + loss_deep_dist


        # with torch.no_grad():
        #     image_embeds = self.finetune_visual_encoder(image, prompts=None)
        #     image_feature = self.vision_proj(image_embeds[:, 0, :])
        #     text = self.tokenizer(caption, padding='max_length', truncation=True,
        #                           max_length=self.max_words,
        #                           return_tensors="pt").to(image.device)
        #     text_output = self.finetune_text_encoder(text.input_ids, attention_mask=text.attention_mask,
        #                                              return_dict=True, mode='multi_modal', prompts=None)
        #     text_feature = self.text_proj(text_output.last_hidden_state[:, 0, :])
        #
        #     finetune_image_feat = F.normalize(image_feature, dim=-1)
        #     finetune_text_feat = F.normalize(text_feature, dim=-1)
        #
        # visual_prompts = self.visual_adapter_current_task
        # image_embeds = self.visual_encoder(image, prompts=visual_prompts)
        # image_feature = self.vision_proj(image_embeds[:, 0, :])
        # text = self.tokenizer(caption, padding='max_length', truncation=True,
        #                       max_length=self.max_words,
        #                       return_tensors="pt").to(image.device)
        # text_prompts = self.text_adapter_current_task
        # text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
        #                                 return_dict=True, mode='multi_modal', prompts=text_prompts)
        # text_feature = self.text_proj(text_output.last_hidden_state[:, 0, :])
        # image_feat = F.normalize(image_feature, dim=-1)
        # text_feat = F.normalize(text_feature, dim=-1)
        #
        # loss_dist = nn.MSELoss()(image_feat, finetune_image_feat) + nn.MSELoss()(text_feat, finetune_text_feat)
        #
        # return loss_dist
    
    def finetune_forward(self, image, caption, mlm_loss=True, orth_loss=False):
        raw_image_feat, raw_text_feat, image_embeds, image_atts, text, text_output = self.get_feature(image,caption)
        batch_size = image.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=image.device) + batch_size * dist.get_rank()

        sim_i2t = raw_image_feat @ all_gather_with_grad(raw_text_feat).T
        sim_t2i = raw_text_feat @ all_gather_with_grad(raw_image_feat).T

        loss_i2t = nn.CrossEntropyLoss()(sim_i2t / self.temp, labels)
        loss_t2i = nn.CrossEntropyLoss()(sim_t2i / self.temp, labels)

        loss_ita = (loss_i2t + loss_t2i) / 2
        if mlm_loss:
            mlm_output, input_ids_new, labels_new = self.get_mlm_loss(text, image_embeds, image_atts ,image.device)
            loss_mlm = mlm_output.loss
            return loss_ita, loss_mlm
        else:
            if orth_loss:
                assert self.unified_prompt
                loss_orth = self.unified_prompt_pool.orth_loss()
                return loss_ita, loss_orth
            return loss_ita

    def distill_from_pretrained(self, image, caption, mlm_loss=False):
        raw_image_feat, raw_text_feat, image_embeds, image_atts, text, text_output = self.get_feature(image,caption)
        batch_size = image.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=image.device) + batch_size * dist.get_rank()

        sim_i2t = raw_image_feat @ all_gather_with_grad(raw_text_feat).T
        sim_t2i = raw_text_feat @ all_gather_with_grad(raw_image_feat).T

        mask_index = torch.arange(batch_size * dist.get_rank(), batch_size * (dist.get_rank() + 1)).unsqueeze_(-1).to(
            image.device)

        # sim_i2i = (raw_image_feat @ all_gather_with_grad(raw_image_feat).T).scatter(1, mask_index, -1000)
        # sim_t2t = (raw_text_feat @ all_gather_with_grad(raw_text_feat).T).scatter(1, mask_index, -1000)


        loss_i2t = nn.CrossEntropyLoss()(sim_i2t / self.temp, labels)
        loss_t2i = nn.CrossEntropyLoss()(sim_t2i / self.temp, labels)

        loss_ita = (loss_i2t + loss_t2i) / 2

        with torch.no_grad():
            ref_image_embeds = self.fixed_visual_encoder(image)
            ref_image_feature = ref_image_embeds[:, 0, :]
            ref_text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_words,
                                  return_tensors="pt").to(image.device)
            mode = 'multi_modal'
            ref_text_output = self.fixed_text_encoder(ref_text.input_ids, attention_mask=ref_text.attention_mask,
                                            return_dict=True, mode=mode)
            ref_text_feature = ref_text_output.last_hidden_state[:, 0, :]

            sim_i2t_ref = (ref_image_feature @ concat_all_gather(ref_text_feature.contiguous()).T)
            sim_t2i_ref = (ref_text_feature @ concat_all_gather(ref_image_feature.contiguous()).T)

            # sim_i2i_ref = (image_feature @ concat_all_gather(image_feature).T).scatter(1, mask_index, -1000)
            # sim_t2t_ref = (text_feature @ concat_all_gather(text_feature).T).scatter(1, mask_index, -1000)

        # cross modal
        loss_ita_dis_i2t = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(sim_i2t / self.temp, dim=1),
                                                               F.softmax(sim_i2t_ref / self.temp, dim=1))
        loss_ita_dis_t2i = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(sim_t2i / self.temp, dim=1),
                                                               F.softmax(sim_t2i_ref / self.temp, dim=1))
        loss_ita_dis = (loss_ita_dis_i2t + loss_ita_dis_t2i) / 2
        # same modal
        # loss_cos_i = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(sim_i2i / self.temp, dim=1),
        #                                                  F.softmax(sim_i2i_ref / self.temp, dim=1))
        # loss_cos_t = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(sim_t2t / self.temp, dim=1),
        #                                                  F.softmax(sim_t2t_ref / self.temp, dim=1))
        # loss_ita_dis += (loss_cos_i + loss_cos_t) / 2


        if mlm_loss:
            mlm_output, input_ids_new, labels_new = self.get_mlm_loss(text, image_embeds, image_atts ,image.device)
            loss_mlm = mlm_output.loss
            return loss_ita, loss_ita_dis, loss_mlm
        else:
            return loss_ita, loss_ita_dis


    def LWF_forward(self, image, caption, iteration, ref_model=None):
        raw_image_feature, raw_text_feature, image_embeds, image_atts, text, text_output = self.get_raw_feature(image,caption)

        raw_image_feat = F.normalize(raw_image_feature,dim=-1)  
        raw_text_feat = F.normalize(raw_text_feature,dim=-1)  

        mlm_output, input_ids_new, labels_new = self.get_mlm_loss(text, image_embeds, image_atts ,image.device)
        loss_mlm_new = mlm_output.loss
        loss_mlm = loss_mlm_new 

        loss_ita, loss_ita_dis, loss_mlm_dis = 0*loss_mlm, 0*loss_mlm, 0*loss_mlm

        batch_size = image.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=image.device) + batch_size * dist.get_rank()

        sim_i2t = raw_image_feat @ all_gather_with_grad(raw_text_feat).T  
        sim_t2i = raw_text_feat @ all_gather_with_grad(raw_image_feat).T 

        loss_i2t = nn.CrossEntropyLoss()(sim_i2t/self.temp, labels)
        loss_t2i = nn.CrossEntropyLoss()(sim_t2i/self.temp, labels)

        loss_ita = (loss_i2t+loss_t2i)/2

        if iteration >0:
            with torch.no_grad():
                ref_image_feature, ref_text_feature, ref_image_embeds, ref_image_atts, ref_text, ref_text_output = ref_model.get_raw_feature(image,caption)
                ref_mlm_output = ref_model.text_mlm_encoder(input_ids = input_ids_new, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = ref_image_embeds,
                                       encoder_attention_mask = ref_image_atts,      
                                       return_dict = True,
                                       labels = labels_new,   
                                      ) 

            loss_i_dis = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(raw_image_feature/self.temp, dim=1),F.softmax(ref_image_feature/self.temp, dim=1))
            loss_t_dis = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(raw_text_feature/self.temp, dim=1),F.softmax(ref_text_feature/self.temp, dim=1))
            loss_ita_dis  = (loss_i_dis + loss_t_dis) /2

            loss_mlm_dis = self.distill_mlm(mlm_output.logits,ref_mlm_output.logits, labels_new) 

        return loss_ita, loss_mlm, loss_ita_dis, loss_mlm_dis

    def LUCIR_forward(self, image, caption, iteration, ref_model):
        raw_image_feat, raw_text_feat, image_embeds, image_atts, text, text_output = self.get_feature(image,caption)   
        mlm_output, input_ids_new, labels_new = self.get_mlm_loss(text, image_embeds, image_atts ,image.device)
        loss_mlm_new = mlm_output.loss
        loss_mlm = loss_mlm_new 

        batch_size = image.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=image.device) + batch_size * dist.get_rank()

        sim_i2t = raw_image_feat @ all_gather_with_grad(raw_text_feat).T  
        sim_t2i = raw_text_feat @ all_gather_with_grad(raw_image_feat).T 

        loss_i2t = nn.CrossEntropyLoss()(sim_i2t/self.temp, labels)
        loss_t2i = nn.CrossEntropyLoss()(sim_t2i/self.temp, labels)

        loss_ita = (loss_i2t+loss_t2i)/2
        loss_ita_dis, loss_mlm_dis = 0*loss_ita, 0*loss_ita

        if iteration >0:
            with torch.no_grad():
                ref_image_feat, ref_text_feat, ref_image_embeds, ref_image_atts, ref_text, ref_text_output = ref_model.get_feature(image,caption)
                ref_mlm_output = ref_model.text_mlm_encoder(input_ids = input_ids_new, 
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = ref_image_embeds,
                                       encoder_attention_mask = ref_image_atts,      
                                       return_dict = True,
                                       labels = labels_new,   
                                      ) 

            loss_cos_i = 1- torch.cosine_similarity(F.normalize(image_embeds,dim=-1), F.normalize(ref_image_embeds,dim=-1)).mean() 
            loss_cos_t = 1- torch.cosine_similarity(F.normalize(text_output.last_hidden_state,dim=-1), F.normalize(ref_text_output.last_hidden_state,dim=-1)).mean()  

            loss_cos = (loss_cos_i+ loss_cos_t)/2
            loss_ita_dis += loss_cos
            loss_mlm_dis = self.distill_mlm(mlm_output.logits,ref_mlm_output.logits,labels_new ) 
        return loss_ita, loss_mlm, loss_ita_dis, loss_mlm_dis

        
    def CEMA_init(self, image, caption, momentum_model):
        raw_image_feat, raw_text_feat, image_embeds, image_atts, text, text_output = self.get_feature(image,caption)
        mlm_output, input_ids_new, labels_new = self.get_mlm_loss(text, image_embeds, image_atts ,image.device)
        loss_mlm = mlm_output.loss 

        with torch.no_grad():
            model_pairs = [[self.visual_encoder,momentum_model.visual_encoder],
                        [self.vision_proj,momentum_model.vision_proj],
                        [self.text_mlm_encoder,momentum_model.text_mlm_encoder],
                        [self.text_proj,momentum_model.text_proj],
                    ]
            self._momentum_update(model_pairs, momentum=self.momentum)
            image_feat_m, text_feat_m, image_embeds_m, image_atts_m, text_m, text_output_m = momentum_model.get_feature(image,caption) 
            image_feat_all =  image_feat_m.t()          
            text_feat_all = text_feat_m.t()
            mlm_output_m = momentum_model.text_mlm_encoder(input_ids = input_ids_new, 
                                    attention_mask = text.attention_mask,
                                    encoder_hidden_states = image_embeds_m,
                                    encoder_attention_mask = image_atts_m,      
                                    return_dict = True,
                                    labels = labels_new,   
                                    )
     
        sim_i2t_md = raw_image_feat @ text_feat_all / self.temp
        sim_t2i_md = raw_text_feat @ image_feat_all / self.temp

        sim_targets = torch.zeros(sim_i2t_md.size()).to(image.device)
        sim_targets.fill_diagonal_(1)          

        loss_i2t_md = -torch.sum(F.log_softmax(sim_i2t_md, dim=1)*sim_targets,dim=1).mean()
        loss_t2i_md = -torch.sum(F.log_softmax(sim_t2i_md, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_i2t_md+loss_t2i_md)/2

        batch_size = image.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=image.device) + batch_size * dist.get_rank()

        sim_i2t = raw_image_feat @ all_gather_with_grad(raw_text_feat).T  
        sim_t2i = raw_text_feat @ all_gather_with_grad(raw_image_feat).T 

        loss_i2t = nn.CrossEntropyLoss()(sim_i2t/self.temp, labels)
        loss_t2i = nn.CrossEntropyLoss()(sim_t2i/self.temp, labels)

        loss_ita += (loss_i2t+loss_t2i)/2
        return loss_ita, loss_mlm

    def CEMA(self, image, caption, ref_model, momentum_model):
        raw_image_feat, raw_text_feat, image_embeds, image_atts, text, text_output = self.get_feature(image,caption)
        mlm_output, input_ids_new, labels_new = self.get_mlm_loss(text, image_embeds, image_atts ,image.device)
        loss_mlm = mlm_output.loss 

        loss_ita_dis, loss_mlm_dis = 0*loss_mlm, 0*loss_mlm

        batch_size = image.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long, device=image.device) + batch_size * dist.get_rank()

        sim_i2t = raw_image_feat @ all_gather_with_grad(raw_text_feat).T  
        sim_t2i = raw_text_feat @ all_gather_with_grad(raw_image_feat).T 

        loss_i2t = nn.CrossEntropyLoss()(sim_i2t/self.temp, labels)
        loss_t2i = nn.CrossEntropyLoss()(sim_t2i/self.temp, labels)

        loss_ita = (loss_i2t+loss_t2i)/2

        mask_index = torch.arange(batch_size * dist.get_rank(), batch_size * (dist.get_rank()+1)).unsqueeze_(-1).to(image.device)

        sim_i2i = (raw_image_feat @ all_gather_with_grad(raw_image_feat).T).scatter(1,mask_index,-1000)
        sim_t2t = (raw_text_feat @ all_gather_with_grad(raw_text_feat).T).scatter(1,mask_index,-1000)

        with torch.no_grad():
            model_pairs = [[self.visual_encoder,ref_model.visual_encoder,momentum_model.visual_encoder],
                        [self.vision_proj,ref_model.vision_proj,momentum_model.vision_proj],
                        [self.text_mlm_encoder,ref_model.text_mlm_encoder,momentum_model.text_mlm_encoder],
                        [self.text_proj,ref_model.text_proj,momentum_model.text_proj],
                    ]
            self._momentum_update_three(model_pairs,momentum=0.9)
            image_feat_m, text_feat_m, image_embeds_m, image_atts_m, text_m, text_output_m = momentum_model.get_feature(image,caption) 
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)              
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)
            
            mlm_output_m = momentum_model.text_mlm_encoder(input_ids = input_ids_new, 
                                    attention_mask = text.attention_mask,
                                    encoder_hidden_states = image_embeds_m,
                                    encoder_attention_mask = image_atts_m,      
                                    return_dict = True,
                                    labels = labels_new,   
                                    )

            sim_i2t_mom = image_feat_m @ text_feat_all / self.temp  
            sim_targets = torch.zeros(sim_i2t_mom.size()).to(image.device)
            sim_targets.fill_diagonal_(1)     
   
        sim_i2t_md = raw_image_feat @ text_feat_all / self.temp
        sim_t2i_md = raw_text_feat @ image_feat_all / self.temp
        loss_i2t_md = -torch.sum(F.log_softmax(sim_i2t_md, dim=1)*sim_targets,dim=1).mean()
        loss_t2i_md = -torch.sum(F.log_softmax(sim_t2i_md, dim=1)*sim_targets,dim=1).mean() 

        loss_ita_dis += (loss_i2t_md+loss_t2i_md)/2
        loss_mlm_dis += self.distill_mlm(mlm_output.logits,mlm_output_m.logits, labels_new)

        self._dequeue_and_enqueue(image_feat_m, text_feat_m) 

        with torch.no_grad():
            ref_image_feat, ref_text_feat, ref_image_embeds, ref_image_atts, ref_text, ref_text_output = ref_model.get_feature(image,caption)
            sim_i2t_ref = (ref_image_feat @ concat_all_gather(ref_text_feat).T)
            sim_t2i_ref = (ref_text_feat @ concat_all_gather(ref_image_feat).T)

            sim_i2i_ref = (ref_image_feat @ concat_all_gather(ref_image_feat).T).scatter(1,mask_index,-1000)
            sim_t2t_ref = (ref_text_feat @ concat_all_gather(ref_text_feat).T).scatter(1,mask_index,-1000)
  
        #cross modal
        loss_ita_dis_i2t = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(sim_i2t/self.temp, dim=1),F.softmax(sim_i2t_ref/self.temp, dim=1))
        loss_ita_dis_t2i = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(sim_t2i/self.temp, dim=1),F.softmax(sim_t2i_ref/self.temp, dim=1))
        loss_ita_dis += (loss_ita_dis_i2t + loss_ita_dis_t2i) /2
        #same modal
        loss_cos_i = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(sim_i2i/self.temp, dim=1),F.softmax(sim_i2i_ref/self.temp, dim=1))
        loss_cos_t = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(sim_t2t/self.temp, dim=1),F.softmax(sim_t2t_ref/self.temp, dim=1))
        loss_ita_dis += (loss_cos_i + loss_cos_t) /2
               
        return loss_ita, loss_mlm, loss_ita_dis, loss_mlm_dis

   
    def forward(self, mode, image, caption, iteration=0, epoch=0, ref_model=None, momentum_model=None):
        if mode == 'finetune': 
            loss_ita, loss_mlm= self.finetune_forward(image, caption)
            return loss_ita, loss_mlm
        elif mode == 'finetune_prompt' or mode == 'convnorm_stage_three':
            loss_ita = self.finetune_forward(image, caption, mlm_loss=False)
            return loss_ita
        elif mode == 'stage_one' or mode == 'convnorm_stage_one':
            loss_ita = self.stage_one(image, caption)
            return loss_ita
        elif mode == 'stage_two':
            loss_ita = self.stage_two(image, caption)
            return loss_ita
        elif mode == 'finetune_prompt_orth':
            loss_ita, loss_orth = self.finetune_forward(image, caption, mlm_loss=False, orth_loss=True)
            return loss_ita, loss_orth
        elif mode == 'LWF':
            loss_ita, loss_mlm, loss_dis, loss_mlm_dis = self.LWF_forward(image, caption, iteration, ref_model)
            return loss_ita, loss_mlm, loss_dis, loss_mlm_dis   
        elif mode == 'LUCIR':
            loss_ita, loss_mlm, loss_dis, loss_mlm_dis= self.LUCIR_forward(image, caption, iteration, ref_model)
            return loss_ita, loss_mlm, loss_dis, loss_mlm_dis
        elif mode == 'CTP':
            loss_ita, loss_mlm, loss_dis, loss_mlm_dis= self.CEMA(image, caption, ref_model, momentum_model)
            return loss_ita, loss_mlm, loss_dis, loss_mlm_dis
        elif mode == 'CTP_init':
            loss_ita, loss_mlm= self.CEMA_init(image, caption, momentum_model)
            return loss_ita, loss_mlm
        elif mode == 'distill_from_pretrained':
            loss_ita, loss_ita_dis = self.distill_from_pretrained(image, caption, mlm_loss=False)
            return loss_ita, loss_ita_dis

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self,model_pairs, momentum):
        for model_pair in model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * momentum + param.data * (1. - momentum)

    @torch.no_grad()        
    def _momentum_update_three(self,model_pairs, momentum):
        for model_pair in model_pairs:           
            for param, param_r, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters(), model_pair[2].parameters()):
                param_m.data = param_m.data * momentum + param.data * (1. - momentum)/2 + param_r.data * (1. - momentum)/2


    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 


def clip_pretrain(**kwargs):
    model = CLIP_Pretrain(**kwargs)
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output     
    
class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)
