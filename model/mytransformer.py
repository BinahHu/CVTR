import copy
import math
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import timm
from timm.models.helpers import named_apply
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_

from transformers import AutoTokenizer, AutoModel

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', pretrained_model="vit_base_patch16_224"):
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
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Continual Learning and prompt parameters
        self.prompt_num = 0
        self.prompt_list = nn.ParameterList()

        if pretrained_model is None:
            self.init_weights(weight_init)
        else:
            self.load_pretrained(pretrained_model)

    def add_prompt(self, prompts_num=0, freeze_old_prompts=True, prompt_init="random"):
        if freeze_old_prompts:
            for p in self.prompt_list.parameters():
                p.requires_grad = False
        if prompts_num > 0:
            scale = self.embed_dim ** -0.5
            if prompt_init == "cls_token":
                prompts = nn.Parameter(0.1 * torch.randn(prompts_num, self.embed_dim)).to(
                    self.cls_token.device)
                prompts += copy.deepcopy(self.cls_token).detach().squeeze(0)
            else:
                prompts = nn.Parameter(scale * torch.randn(prompts_num, self.embed_dim)).to(self.cls_token.device)
            self.prompt_list.append(prompts)
            self.prompt_num += prompts_num

    def load_pretrained(self, pretrained_model_name):
        if pretrained_model_name == "vit_base_patch16_224":
            pretrained_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=True)
        else:
            raise NotImplementedError

        param_dict = dict(self.named_parameters())
        pretrained_param_dict = dict(pretrained_model.named_parameters())
        for k in param_dict:
            if 'prompt' in k:
                continue
            assert k in pretrained_param_dict
            param_dict[k] = copy.deepcopy(pretrained_param_dict[k])

        self.num_classes = pretrained_model.num_classes
        self.num_features = pretrained_model.num_features
        self.num_tokens = pretrained_model.num_tokens

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        if self.prompt_num > 0:
            prompt = torch.cat(list(self.prompt_list))
            x = torch.cat([prompt.to(x.dtype) + torch.zeros(x.shape[0], self.prompt_num, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)

        x = self.blocks(x)
        x = self.norm(x)

        if self.dist_token is None:
            if self.prompt_num > 0:
                return self.pre_logits(x[:, :self.prompt_num + 1, :])
            else:
                return self.pre_logits(x[:, 0, :])
        else:
            return x[:, self.prompt_num], x[:, self.prompt_num+1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class PROMPTEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = True):
        super(PROMPTEmbedding, self).__init__()
        self.wte = wte
        self.random_range = random_range
        self.initialize_from_vocab = initialize_from_vocab
        self.prompt_num = 0
        self.prompt_list = nn.ParameterList()

    def initialize_embedding(self,
                             wte: nn.Embedding,
                             prompts_num: int = 10,
                             random_range: float = 0.5,
                             initialize_from_vocab: bool = True,
                             cls_token_init: bool = False):
        if cls_token_init:
            return self.wte.weight[101].clone().detach().unsqueeze(0) + torch.FloatTensor(prompts_num, wte.weight.size(1)).uniform_(-0.1, 0.1).to(wte.weight.device)
        if initialize_from_vocab:
            return self.wte.weight[:prompts_num].clone().detach()
        return torch.FloatTensor(wte.weight.size(1), prompts_num).uniform_(-random_range, random_range).to(wte.weight.device)

    def forward(self, tokens):
        input_embedding = self.wte(tokens[:, self.prompt_num:])
        if self.prompt_num == 0:
            return input_embedding
        learned_embedding = torch.cat(list(self.prompt_list), dim=0)
        learned_embedding = learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], dim=1)

    def add_prompt(self, prompts_num=0, freeze_old_prompts=True, prompt_init="random"):
        if freeze_old_prompts:
            for p in self.prompt_list.parameters():
                p.requires_grad = False
        if prompts_num > 0:
            if prompt_init == "cls_token":
                prompts = nn.Parameter(self.initialize_embedding(self.wte, prompts_num,
                                                                 self.random_range, self.initialize_from_vocab,
                                                                 cls_token_init=True))
            else:
                prompts = nn.Parameter(self.initialize_embedding(self.wte, prompts_num,
                                                             self.random_range, self.initialize_from_vocab))
            self.prompt_list.append(prompts)
            self.prompt_num += prompts_num

class TextTransformerWithPrompt(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased"):
        super(TextTransformerWithPrompt, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        text_model = AutoModel.from_pretrained(pretrained_model_name)

        prompt_emb = PROMPTEmbedding(text_model.get_input_embeddings(),
                                     initialize_from_vocab=True)
        text_model.set_input_embeddings(prompt_emb)
        self.text_model = text_model
        self.prompt_emb = prompt_emb

    def encode_text(self, text_data):
        return self.text_model(**text_data)

    def add_prompt(self, prompts_num=0, freeze_old_prompts=True, prompt_init="random"):
        self.prompt_emb.add_prompt(prompts_num=prompts_num, freeze_old_prompts=freeze_old_prompts, prompt_init=prompt_init)
        self.text_model.set_input_embeddings(self.prompt_emb)