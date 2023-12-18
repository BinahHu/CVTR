import transformers
transformers.logging.set_verbosity_error()

import torch
from torch import nn
import torch.nn.functional as F
from sklearn import decomposition
import numpy as np

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import copy

class AttnAdapter(nn.Module):
    def __init__(self, feat_dim, reduce_factor=8, convnorm=False):
        super().__init__()
        self.feat_dim = feat_dim
        self.reduce_factor = reduce_factor
        self.task_count = 0
        self.convnorm = convnorm
        if convnorm:
            self.downsample_block_shared = nn.Linear(feat_dim, feat_dim // reduce_factor)
        else:
            self.downsample_block_q = nn.Linear(feat_dim, feat_dim // reduce_factor)
            self.downsample_block_k = nn.Linear(feat_dim, feat_dim // reduce_factor)
            self.downsample_block_v = nn.Linear(feat_dim, feat_dim // reduce_factor)
        self.downsample_block_proj = nn.Linear(feat_dim, feat_dim // reduce_factor)
        self.upsample_block_q = nn.Linear(feat_dim // reduce_factor, feat_dim)
        self.upsample_block_k = nn.Linear(feat_dim // reduce_factor, feat_dim)
        self.upsample_block_v = nn.Linear(feat_dim // reduce_factor, feat_dim)
        self.upsample_block_proj = nn.Linear(feat_dim // reduce_factor, feat_dim)

        # self.s_q = nn.Parameter(torch.ones(feat_dim))
        # self.s_k = nn.Parameter(torch.ones(feat_dim))
        # self.s_v = nn.Parameter(torch.ones(feat_dim))

        # nn.init.zeros_(self.downsample_block_q.weight)
        # nn.init.zeros_(self.downsample_block_q.bias)
        # nn.init.zeros_(self.upsample_block_q.weight)
        # nn.init.zeros_(self.upsample_block_q.bias)
        #
        # nn.init.zeros_(self.downsample_block_k.weight)
        # nn.init.zeros_(self.downsample_block_k.bias)
        # nn.init.zeros_(self.upsample_block_k.weight)
        # nn.init.zeros_(self.upsample_block_k.bias)
        #
        # nn.init.zeros_(self.downsample_block_v.weight)
        # nn.init.zeros_(self.downsample_block_v.bias)
        # nn.init.zeros_(self.upsample_block_v.weight)
        # nn.init.zeros_(self.upsample_block_v.bias)


        self.act_fn = nn.GELU()

    @torch.no_grad()
    def init_convnorm(self, x=None, q=None, k=None, v=None, px=None, py=None):
        self.act_fn = nn.Identity()
        # debug:
        # return
        if x is not None:
            name = "x"
            data = x
            block = self.downsample_block_shared
            downsample_block = True
        elif px is not None:
            name = "px"
            data = px
            block = self.downsample_block_proj
            downsample_block = True
        elif q is not None:
            name = "q"
            data = q
            block = self.upsample_block_q
            downsample_block = False
        elif k is not None:
            name = "k"
            data = k
            block = self.upsample_block_k
            downsample_block = False
        elif v is not None:
            name = "v"
            data = v
            block = self.upsample_block_v
            downsample_block = False
        elif py is not None:
            name = "py"
            data = py
            block = self.upsample_block_proj
            downsample_block = False
        else:
            raise AttributeError

        pca = decomposition.PCA(n_components=self.feat_dim // self.reduce_factor)
        pca.fit(data)
        mean = pca.mean_
        comp = pca.components_
        sig = pca.singular_values_
        if downsample_block:
            w = np.dot(np.diag(1/sig), comp)
            block.weight.copy_(torch.tensor(w))
            block.bias.copy_(torch.tensor(-np.dot(w, mean)))
        else:
            w = np.dot(np.diag(sig), comp).T
            block.weight.copy_(torch.tensor(w))
            block.bias.copy_(torch.tensor(mean))



    def init_with_qkv(self, q, k, v, proj=None):
        # Linear y = xA^T + b
        with torch.no_grad():
            qu, qs, qv = torch.pca_lowrank(q.weight, self.feat_dim // self.reduce_factor,
                                           center=False)
            qds = torch.sqrt(qs)
            qds = torch.diag(qds)
            self.downsample_block_q.weight.copy_(qds @ qv.T)
            nn.init.zeros_(self.downsample_block_q.bias)
            self.upsample_block_q.weight.copy_(qu @ qds)
            self.upsample_block_q.bias.copy_(q.bias)

            ku, ks, kv = torch.pca_lowrank(k.weight, self.feat_dim // self.reduce_factor,
                                           center=False)
            kds = torch.sqrt(ks)
            kds = torch.diag(kds)
            self.downsample_block_k.weight.copy_(kds @ kv.T)
            nn.init.zeros_(self.downsample_block_k.bias)
            self.upsample_block_k.weight.copy_(ku @ kds)
            self.upsample_block_k.bias.copy_(k.bias)

            vu, vs, vv = torch.pca_lowrank(v.weight, self.feat_dim // self.reduce_factor,
                                           center=False)
            vds = torch.sqrt(vs)
            vds = torch.diag(vds)
            self.downsample_block_v.weight.copy_(vds @ vv.T)
            nn.init.zeros_(self.downsample_block_v.bias)
            self.upsample_block_v.weight.copy_(vu @ vds)
            self.upsample_block_v.bias.copy_(v.bias)

            if proj is not None:
                proju, projs, projv = torch.pca_lowrank(proj.weight, self.feat_dim // self.reduce_factor,
                                               center=False)
                projds = torch.sqrt(projs)
                projds = torch.diag(projds)
                self.downsample_block_proj.weight.copy_(projds @ projv.T)
                nn.init.zeros_(self.downsample_block_proj.bias)
                self.upsample_block_proj.weight.copy_(proju @ projds)
                self.upsample_block_proj.bias.copy_(proj.bias)

        self.act_fn = nn.Identity()

    def forward(self, q, k, v, proj=None):

        if proj is not None:
            proj_res = self.downsample_block_proj(proj)
            proj_res = self.act_fn(proj_res)
            proj_res = self.upsample_block_proj(proj_res)
            return proj_res

        if self.convnorm:
            shared_x = self.downsample_block_shared(q)
            shared_x = self.act_fn(shared_x)
            q_res = self.upsample_block_q(shared_x)
            k_res = self.upsample_block_k(shared_x)
            v_res = self.upsample_block_v(shared_x)
        else:
            q_res = self.downsample_block_q(q)
            q_res = self.act_fn(q_res)
            q_res = self.upsample_block_q(q_res)
            # q_res = q_res * self.s_q

            k_res = self.downsample_block_k(k)
            k_res = self.act_fn(k_res)
            k_res = self.upsample_block_k(k_res)
            # k_res = k_res * self.s_k

            # v_res = 0
            v_res = self.downsample_block_v(v)
            v_res = self.act_fn(v_res)
            v_res = self.upsample_block_v(v_res)
            # v_res = v_res * self.s_v

        return q_res, k_res, v_res

class Adapter(nn.Module):
    def __init__(self, feat_dim, reduce_factor=1):
        super().__init__()
        self.task_count = 0
        self.downsample_block = nn.Linear(feat_dim, feat_dim // reduce_factor)
        self.upsample_block = nn.Linear(feat_dim // reduce_factor, feat_dim)
        self.act_fn = nn.GELU()

        # nn.init.zeros_(self.downsample_block.weight)
        # nn.init.zeros_(self.downsample_block.bias)
        #
        # nn.init.zeros_(self.upsample_block.weight)
        # nn.init.zeros_(self.upsample_block.bias)

    def forward(self, x):
        x = self.downsample_block(x)
        x = self.act_fn(x)
        x = self.upsample_block(x)
        return x

class AdapterPool(nn.Module):
    def __init__(self, feat_dim, reduce_factor=8, convnorm=False):
        super().__init__()
        self.task_count = 0
        self.feat_dim = feat_dim
        self.reduce_factor = reduce_factor
        self.convnorm = convnorm
        self.adapter_list = nn.ModuleList()

    def next_task(self, device, freeze_old=True):
        if self.task_count == 0:
            self.adapter_list.append(AttnAdapter(self.feat_dim, self.reduce_factor, self.convnorm).to(device))
        else:
            new_adapter = copy.deepcopy(self.adapter_list[-1].detach()).to(device)
            for p in new_adapter.parameters():
                p.requires_grad = True
            if freeze_old:
                for p in self.adapter_list[-1].parameters():
                    p.requires_grad = False
            self.adapter_list.append(new_adapter)
        self.task_count += 1

    def get_adapter(self):
        return self.adapter_list[self.task_count - 1]

    def init_with_qkv(self, q, k, v, proj=None):
        self.adapter_list[self.task_count - 1].init_with_qkv(q, k, v, proj)

    def init_convnorm(self, x=None, q=None, k=None, v=None, px=None, py=None):
        self.adapter_list[self.task_count - 1].init_convnorm(x, q, k, v, px, py)


class UnifiedAdapter(nn.Module):
    def __init__(self, feat_dim, reduce_factor=4, unified_thd_layer=12, num_layers=12, adapter_start_layer=0,
                 convnorm=False):
        super().__init__()
        self.adapter_start_layer = adapter_start_layer
        self.unified_thd_layer = unified_thd_layer
        self.num_layers = num_layers
        self.visual_adapter_pool = nn.ModuleList()
        self.text_adapter_pool = nn.ModuleList()
        self.unified_adapter_pool = nn.ModuleList()


        for i in range(unified_thd_layer - adapter_start_layer):
            self.visual_adapter_pool.append(AdapterPool(feat_dim, reduce_factor, convnorm))
            self.text_adapter_pool.append(AdapterPool(feat_dim, reduce_factor, convnorm))
        for i in range(num_layers - unified_thd_layer):
            self.unified_adapter_pool.append(AdapterPool(feat_dim, reduce_factor, convnorm))

    def next_task(self, device):
        for i in range(self.unified_thd_layer - self.adapter_start_layer):
            self.visual_adapter_pool[i].next_task(device)
            self.text_adapter_pool[i].next_task(device)
        for i in range(self.num_layers - self.unified_thd_layer):
            self.unified_adapter_pool[i].next_task(device)

    def get_visual_adapter(self, l):
        if l < self.unified_thd_layer:
            return self.visual_adapter_pool[l - self.adapter_start_layer].get_adapter()
        else:
            return self.unified_adapter_pool[l - self.unified_thd_layer].get_adapter()

    def get_text_adapter(self, l):
        if l < self.unified_thd_layer:
            return self.text_adapter_pool[l - self.adapter_start_layer].get_adapter()
        else:
            return self.unified_adapter_pool[l - self.unified_thd_layer].get_adapter()

    def init_with_qkv_list(self, visual_list, text_list):
        assert self.unified_thd_layer == self.num_layers
        assert len(visual_list) == self.num_layers
        for l in range(self.adapter_start_layer, self.num_layers):
            visual_proj = None if len(visual_list[l]) < 4 else visual_list[l][3]
            self.visual_adapter_pool[l - self.adapter_start_layer].init_with_qkv(
                visual_list[l][0], visual_list[l][1], visual_list[l][2], visual_proj)

            text_proj = None if len(text_list[l]) < 4 else text_list[l][3]
            self.text_adapter_pool[l - self.adapter_start_layer].init_with_qkv(
                text_list[l][0], text_list[l][1], text_list[l][2], text_proj)