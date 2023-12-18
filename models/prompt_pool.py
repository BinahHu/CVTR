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

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import copy

class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_per_task, prompt_length, ortho_mu=0.1, num_layers=12, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(prompt_per_task, prompt_length, ortho_mu, num_layers)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(prompt_per_task *  n_tasks, e_l, emb_d)
            k = tensor_prompt(prompt_per_task *  n_tasks, self.key_d)
            a = tensor_prompt(prompt_per_task *  n_tasks, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}', self.tensor2list(p, self.n_tasks))
            setattr(self, f'e_k_{e}', self.tensor2list(k, self.n_tasks))
            setattr(self, f'e_a_{e}', self.tensor2list(a, self.n_tasks))

    def tensor2list(self, t, n_tasks):
        p_list = nn.ParameterList()
        for i in range(n_tasks):
            s = i * self.e_prompt_per_task
            f = (i+1) * self.e_prompt_per_task
            p_list.append(nn.Parameter(t[s:f].detach().clone()))
        return p_list

    def list_gram_schmidt(self, p_list):
        grad_list = []
        for p in p_list:
            grad_list.append(p.requires_grad)
        t = torch.cat(list(p_list), dim=0)
        t = self.gram_schmidt(t)
        new_p_list = self.tensor2list(t, self.n_tasks)
        for i in range(self.n_tasks):
            new_p_list[i].requires_grad = grad_list[i]
        return new_p_list


    def _init_smart(self, prompt_per_task, prompt_length, ortho_mu, num_layers):

        # prompt basic param
        self.e_prompt_per_task = prompt_per_task
        self.e_p_length = prompt_length
        self.e_layers = list(range(num_layers))

        # strenth of ortho penalty
        self.ortho_mu = ortho_mu

    def next_task(self):
        if self.task_count > 0:
            for i in range(self.task_count):
                for e in self.e_layers:
                    K = getattr(self, f'e_k_{e}')
                    A = getattr(self, f'e_a_{e}')
                    P = getattr(self, f'e_p_{e}')
                    K[i].requires_grad = False
                    A[i].requires_grad = False
                    P[i].requires_grad = False

            for e in self.e_layers:
                K = getattr(self, f'e_k_{e}')
                A = getattr(self, f'e_a_{e}')
                P = getattr(self, f'e_p_{e}')
                k = self.list_gram_schmidt(K)
                a = self.list_gram_schmidt(A)
                p = self.list_gram_schmidt(P)
                setattr(self, f'e_p_{e}', p)
                setattr(self, f'e_k_{e}', k)
                setattr(self, f'e_a_{e}', a)

        self.task_count += 1

    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        pool_size = vv.shape[0]
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        s = (self.task_count - 1) * self.e_prompt_per_task
        f = self.task_count * self.e_prompt_per_task
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

    def forward(self, x_querry, l):

        # e prompts
        if l in self.e_layers:
            B, C = x_querry.shape

            K = getattr(self, f'e_k_{l}')
            A = getattr(self, f'e_a_{l}')
            p = getattr(self, f'e_p_{l}')
            f = self.task_count * self.e_prompt_per_task

            K = torch.cat(list(K), dim=0)
            A = torch.cat(list(A), dim=0)
            p = torch.cat(list(p), dim=0)

            K = K[0:f]
            A = A[0:f]
            p = p[0:f]

            # with attention and cosine sim\
            # k: prompt pool size, d: feature dim, b: batch size, l: prompt length
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length / 2)
            Ek = P_[:, :i, :]
            Ev = P_[:, i:, :]

            return [Ek, Ev]
        else:
            return None


    def orth_loss(self):
        loss_list = []
        for e in self.e_layers:
            K = getattr(self, f'e_k_{e}')
            A = getattr(self, f'e_a_{e}')
            P = getattr(self, f'e_p_{e}')
            K = torch.cat(list(K), dim=0)
            A = torch.cat(list(A), dim=0)
            P = torch.cat(list(P), dim=0)
            f = self.task_count * self.e_prompt_per_task
            K = K[0:f]
            A = A[0:f]
            P = P[0:f]
            loss = ortho_penalty(K) * self.ortho_mu
            loss += ortho_penalty(A) * self.ortho_mu
            loss += ortho_penalty(P.view(P.shape[0], -1)) * self.ortho_mu
            loss_list.append(loss)
        return sum(loss_list) / len(loss_list)


class UnifiedPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_length, prompt_per_task=10, unified_thd_layer=9,
                 ortho_mu=0.1, num_layers=12, key_dim=768):
        super().__init__()
        self.unified_thd_layer = unified_thd_layer
        shallow_layers = unified_thd_layer
        deep_layers = num_layers - unified_thd_layer
        self.visual_prompt_pool = CodaPrompt(emb_d, n_tasks, prompt_per_task, prompt_length,
                 ortho_mu=ortho_mu, num_layers=shallow_layers, key_dim=key_dim)

        self.text_prompt_pool = CodaPrompt(emb_d, n_tasks, prompt_per_task, prompt_length,
                 ortho_mu=ortho_mu, num_layers=shallow_layers, key_dim=key_dim)

        self.unfied_prompt_pool = CodaPrompt(emb_d, n_tasks, prompt_per_task, prompt_length,
                 ortho_mu=ortho_mu, num_layers=deep_layers, key_dim=key_dim)

    def next_task(self):
        self.visual_prompt_pool.next_task()
        self.text_prompt_pool.next_task()
        self.unfied_prompt_pool.next_task()

    def get_visual_prompt(self, x_querry, l):
        if l < self.unified_thd_layer:
            return self.visual_prompt_pool(x_querry, l)
        else:
            return self.unfied_prompt_pool(x_querry, l - self.unified_thd_layer)

    def get_text_prompt(self, x_querry, l):
        if l < self.unified_thd_layer:
            return self.text_prompt_pool(x_querry, l)
        else:
            return self.unfied_prompt_pool(x_querry, l - self.unified_thd_layer)


    def orth_loss(self):
        loss = self.visual_prompt_pool.orth_loss()
        loss += self.text_prompt_pool.orth_loss()
        loss += self.unfied_prompt_pool.orth_loss()
        return loss / 3

def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p

def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda()) ** 2).mean()


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
