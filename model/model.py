import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import timm
import torch
import model.myclip as clip

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
    def __init__(self, pretrained_model_name="ViT-B/16", video_prompts_per_task=4, text_prompts_per_task=4,
                 pop_num=4, embd_dim=768, prompt_cls_strategy="mean", clip_strategy="mean", **kwargs):
        super().__init__()
        self.clip_model, _ = clip.load(pretrained_model_name)
        self.video_prompt_per_task = video_prompts_per_task
        self.text_prompts_per_task = text_prompts_per_task
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.num_categories = 0
        self.embd_dim = embd_dim
        self.pop_num = pop_num
        self.prompt_cls_strategy = prompt_cls_strategy
        self.clip_strategy = clip_strategy
        self.current_task_id = -1
        self.video_prompt_num = 0
        self.text_prompt_num = 0

    def input_preprocess(self, data):
        caption = data["caption"]
        data["caption"] = clip.tokenize(caption)
        return data
    def prepare_decouple(self):
        pass

    def after_decouple(self):
        pass
    def next_task(self, updates):
        new_categories = updates["new_categories"]
        self.current_task_id += 1
        self.num_categories += new_categories
        self.clip_model.add_prompt(vision_prompt_num=self.video_prompt_per_task, text_prompt_num=self.text_prompts_per_task)
        self.video_prompt_num += self.video_prompt_per_task
        self.text_prompt_num += self.text_prompts_per_task
        if self.prompt_cls_strategy == "mean":
            self.pop_head = nn.Linear(self.embd_dim, self.current_task_id + 1).to(self.clip_model.visual.positional_embedding.device)
        elif self.prompt_cls_strategy == "cat":
            self.pop_head = nn.Linear(self.pop_num * self.embd_dim,
                                      self.current_task_id + 1).to(self.clip_model.visual.positional_embedding.device)
        elif self.prompt_cls_strategy == "mean_per_task":
            self.pop_head = nn.Linear(self.embd_dim,
                                      self.current_task_id + 1).to(self.clip_model.visual.positional_embedding.device)
        else:
            raise NotImplementedError

    def compute_text(self, text_tokens):
        text_features = self.clip_model.encode_text(text_tokens)
        text_prompts = None
        if self.text_prompt_num > 0:
            text_prompts = text_features[:, :self.text_prompt_num, :]
            text_features = text_prompts[:, -1, :]
        return text_features, text_prompts

    def compute_video(self, video_data):
        B, F, C, H, W = video_data.shape
        video_data_as_images = video_data.view(B * F, C, H, W)
        video_prompts = None
        image_features = self.clip_model.encode_image(video_data_as_images)
        if self.video_prompt_num > 0:
            video_prompts = image_features[:, :self.video_prompt_num, :]
            video_prompts = video_prompts.view(B, F, self.video_prompt_num, -1).mean(dim=1)
            video_features = image_features[:, -1, :]
            video_features = video_features.view(B, F, -1).mean(dim=1)
        else:
            video_features = image_features.view(B, F, -1).mean(dim=1)
        return video_features, video_prompts

    def forward(self, data):
        video = data["video"]
        text = data["caption"]

        text_embeddings, text_prompts = self.compute_text(text)
        video_embeddings, video_prompts = self.compute_video(video)
        # print(f"text prompts shape is {text_prompts.shape}, video prompts shape is {video_prompts.shape}")

        # text_embeddings = text_prompts.mean(dim=1)
        # video_embeddings = video_prompts.mean(dim=1)

        # text_embeddings = torch.cat([text_embeddings, text_prompts.view(text_prompts.shape[0], -1)], dim=-1)
        # video_embeddings = torch.cat([video_embeddings, video_prompts.view(video_prompts.shape[0], -1)], dim=-1)

        return {"text_embeddings": text_embeddings, "video_embeddings": video_embeddings,
                "text_prompts": text_prompts, "video_prompts": video_prompts,
                "reference_feature": video_embeddings}

