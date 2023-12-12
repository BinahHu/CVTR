import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import timm
import torch
import model.myclip as clip
from model.mytransformer import VisionTransformer, TextTransformerWithPrompt
import copy

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class CLIPModel(BaseModel):
    def __init__(self, pretrained_model_name="ViT-B/16", image_prompts_per_task=4, text_prompts_per_task=4,
                 embd_dim=768, prompt_strategy="mean_per_task", clip_strategy="cls_token", continual=False,
                 prompt_init="random", seq=False, **kwargs):
        super(CLIPModel, self).__init__()
        if "ckpt_" in pretrained_model_name:
            pretrained_model_name = pretrained_model_name.split("arch_")[1]
            arch, model_path = pretrained_model_name.split("_ckpt_")
            self.clip_model, _ = clip.load(arch)
            self.load_ckpt(model_path)
        else:
            self.clip_model, _ = clip.load(pretrained_model_name)
        self.image_prompts_per_task = image_prompts_per_task
        self.text_prompts_per_task = text_prompts_per_task
        self.continual = continual
        self.old_model = None
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.embd_dim = embd_dim
        self.prompt_strategy = prompt_strategy
        self.clip_strategy = clip_strategy
        self.current_task_id = -1
        self.image_prompt_num = 0
        self.text_prompt_num = 0
        self.prompt_init = prompt_init
        self.seq = seq

    def load_ckpt(self, model_path):
        state = torch.load(model_path)
        state_dict = state['state_dict']
        self.clip_model.text_prompt_list.append(nn.Parameter(torch.zeros_like(state_dict['clip_model.text_prompt_list.0'])))
        self.clip_model.visual.prompt_list.append(nn.Parameter(torch.zeros_like(state_dict['clip_model.visual.prompt_list.0'])))
        self.load_state_dict(state_dict)
        self.clip_model.process_ckpt()
    def input_preprocess(self, data):
        caption = data["text"]
        data["text"] = clip.tokenize(caption, truncate=True)
        return data
    def prepare_decouple(self):
        pass

    def after_decouple(self):
        pass
    def next_task(self, updates):
        self.current_task_id += 1
        if self.current_task_id > 0 and self.continual:
            self.old_model = copy.deepcopy(self)
            for p in self.old_model.parameters():
                p.requires_grad = False
        self.clip_model.add_prompt(vision_prompt_num=self.image_prompts_per_task, text_prompt_num=self.text_prompts_per_task,
                                   prompt_init=self.prompt_init, seq=self.seq)
        if self.seq and self.image_prompt_num > 0:
            return
        self.image_prompt_num += self.image_prompts_per_task
        self.text_prompt_num += self.text_prompts_per_task

    def compute_text(self, text_tokens):
        text_features = self.clip_model.encode_text(text_tokens)
        text_prompts = None
        if self.text_prompt_num > 0:
            text_prompts = text_features[:, :self.text_prompt_num, :]
            if self.prompt_strategy == 'mean_per_task':
                B, N, D = text_prompts.shape
                text_prompts = text_prompts.view(B, self.text_prompts_per_task, -1, D).mean(dim=1)
            text_features = text_features[:, -1, :]
        if self.clip_strategy == "last_prompt":
            text_features = text_prompts[:, -1, :]
        return text_features

    def compute_image(self, image_data):
        image_prompts = None
        image_features = self.clip_model.encode_image(image_data)
        if self.image_prompt_num > 0:
            image_prompts = image_features[:, :self.image_prompt_num, :]
            if self.prompt_strategy == 'mean_per_task':
                B, N, D = image_prompts.shape
                image_prompts = image_prompts.view(B, self.text_prompts_per_task, -1, D).mean(dim=1)
            image_features = image_features[:, -1, :]
        if self.clip_strategy == "last_prompt":
            image_features = image_prompts[:, -1, :]
        return image_features

    def forward(self, data, visual_only=False, text_only=False):
        if visual_only:
            image = data["visual"]
            image_embeddings= self.compute_image(image)
            return {"visual_embeddings": image_embeddings,
                    "reference_feature": image_embeddings}

        if text_only:
            text = data["text"]
            text_embeddings= self.compute_text(text)
            return {"text_embeddings": text_embeddings,
                    "reference_feature": text_embeddings}

        image = data["visual"]
        image_embeddings = self.compute_image(image)
        text = data["text"]
        text_embeddings = self.compute_text(text)

        old_text_embeddings = None
        old_image_embeddings = None
        if self.continual and self.current_task_id > 0:
            old_image_embeddings = self.old_model.compute_image(image)
            old_text_embeddings = self.old_model.compute_text(text)

        return {"text_embeddings": text_embeddings, "visual_embeddings": image_embeddings,
                "old_text_embeddings": old_text_embeddings, "old_visual_embeddings": old_image_embeddings,
                "reference_feature": image_embeddings}


class VisualTextTransformers(BaseModel):
    def __init__(self, pretrained_vision_model_name="vit_base_patch16_224",
                 pretrained_text_model_name="bert-base-uncased",
                 image_prompts_per_task=4, text_prompts_per_task=4,
                 embd_dim=768, prompt_strategy="mean", prompt_init="random", **kwargs):
        super(VisualTextTransformers, self).__init__()
        self.visual = VisionTransformer(pretrained_model=pretrained_vision_model_name)
        self.text = TextTransformerWithPrompt(pretrained_model_name=pretrained_text_model_name)
        self.prompt_init = prompt_init

        self.image_prompts_per_task = image_prompts_per_task
        self.text_prompts_per_task = text_prompts_per_task
        for p in self.visual.parameters():
            p.requires_grad = False
        for p in self.text.parameters():
            p.requires_grad = False

        self.embd_dim = embd_dim
        self.prompt_strategy = prompt_strategy
        self.current_task_id = -1
        self.image_prompt_num = 0
        self.text_prompt_num = 0

    def input_preprocess(self, data):
        prompt_place_holder = "[CLS] " * self.text_prompt_num
        caption = [str(prompt_place_holder + t) for t in data["text"]]
        data["text"] = self.text.tokenizer(caption, return_tensors='pt', padding=True)
        return data
    def prepare_decouple(self):
        pass

    def after_decouple(self):
        pass
    def next_task(self, updates):
        self.current_task_id += 1
        self.visual.add_prompt(prompts_num=self.image_prompts_per_task, prompt_init=self.prompt_init)
        self.text.add_prompt(prompts_num=self.text_prompts_per_task, prompt_init=self.prompt_init)
        self.image_prompt_num += self.image_prompts_per_task
        self.text_prompt_num += self.text_prompts_per_task

    def compute_text(self, text_inputs):
        text_output = self.text.encode_text(text_inputs)
        text_features = text_output['last_hidden_state']
        text_prompts = None
        if self.text_prompt_num > 0:
            text_prompts = text_features[:, :self.text_prompt_num, :]
            text_features = text_features[:, -1, :]
        else:
            text_features = text_features[:, 0, :]
        return text_features, text_prompts

    def compute_image(self, image_data):
        image_prompts = None
        image_features = self.visual.forward_features(image_data)
        if self.image_prompt_num > 0:
            image_prompts = image_features[:, :self.text_prompt_num, :]
            image_features = image_features[:, -1, :]
        return image_features, image_prompts

    def forward(self, data, visual_only=False, text_only=False):
        if visual_only:
            image = data["visual"]
            image_embeddings, image_prompts = self.compute_image(image)
            return {"visual_embeddings": image_embeddings,
                    "reference_feature": image_embeddings}

        if text_only:
            text = data["text"]
            text_embeddings, text_prompts = self.compute_text(text)
            return {"text_embeddings": text_embeddings,
                    "reference_feature": text_embeddings}

        image = data["visual"]
        image_embeddings, image_prompts = self.compute_image(image)
        text = data["text"]
        text_embeddings, text_prompts = self.compute_text(text)
        # print(f"text prompts shape is {text_prompts.shape}, video prompts shape is {video_prompts.shape}")

        # text_embeddings = text_prompts.mean(dim=1)
        # video_embeddings = video_prompts.mean(dim=1)

        # text_embeddings = torch.cat([text_embeddings, text_prompts.view(text_prompts.shape[0], -1)], dim=-1)
        # video_embeddings = torch.cat([video_embeddings, video_prompts.view(video_prompts.shape[0], -1)], dim=-1)


        old_text_embeddings = None
        old_image_embeddings = None
        # if self.continual and self.current_task_id > 0:
        #     old_image_embeddings = self.old_model.compute_image(image)
        #     old_text_embeddings = self.old_model.compute_text(text)

        return {"text_embeddings": text_embeddings, "visual_embeddings": image_embeddings,
                "old_text_embeddings": old_text_embeddings, "old_visual_embeddings": old_image_embeddings,
                "reference_feature": image_embeddings}

