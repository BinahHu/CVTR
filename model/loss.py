import torch.nn.functional as F
import torch
from utils import AllGather_multi, sim_matrix

def nll_loss(output, target):
    return F.nll_loss(output, target)

def last_10_ce_loss(output):
    label = output["label"]
    task_index = output["task_index"]
    prompt = output["prompt_res"]
    L = prompt.shape[1]
    is_train = output["is_train"]
    if L > 10 and is_train:
        label -= (prompt.shape[1] - 10)
        prompt = prompt[:, -10:]
    loss = F.cross_entropy(prompt, label)
    return {"loss_backward": loss, "loss": loss.item()}
def ce_loss(output):
    label = output["label"]
    task_index = output["task_index"]
    prompt = output["prompt_res"]
    loss = F.cross_entropy(prompt, label)
    return {"loss_backward": loss, "loss": loss.item()}
def bce_loss(output):
    label = output["label"]
    task_index = output["task_index"]
    prompt = output["prompt_res"]
    loss = F.binary_cross_entropy_with_logits(
        prompt,
        torch.eye(prompt.shape[1])[label].to(label.device)
    )
    return {"loss_backward": loss, "loss": loss.item()}

def video_text_contrastive_loss(output):
    temperature = 0.05
    video_embeddings = output["video_embeddings"].contiguous()
    text_embeddings = output["text_embeddings"].contiguous()
    text_prompts = output["text_prompts"]
    video_prompts = output["video_prompts"]

    gather_func = AllGather_multi.apply
    video_embeddings = gather_func(video_embeddings)
    text_embeddings = gather_func(text_embeddings)
    output = sim_matrix(text_embeddings, video_embeddings)

    i_sm = F.softmax(output / temperature, dim=1)
    j_sm = F.softmax(output / temperature, dim=1)

    idiag = torch.log(torch.diagonal(i_sm))
    loss_i = idiag.sum() / len(idiag)

    jdiag = torch.log(torch.diagonal(j_sm))
    loss_j = jdiag.sum() / len(jdiag)

    loss = - loss_i - loss_j

    return {"loss_backward": loss, "loss": loss.item()}


def bce_with_task_loss(output):
    label = output["label"]
    task_index = output["task_index"]
    prompt = output["prompt_res"]
    loss = F.binary_cross_entropy_with_logits(
        prompt,
        torch.eye(prompt.shape[1])[label].to(label.device)
    )

    pop = output["pop_res"]
    if pop.shape[1] > 1:
        loss += F.cross_entropy(pop, task_index)

    return {"loss_backward": loss, "loss": loss.item()}

def bce_with_task_loss_with_aux(output):
    label = output["label"]
    task_index = output["task_index"]
    prompt = output["prompt_res"]
    loss = F.binary_cross_entropy_with_logits(
        prompt,
        torch.eye(prompt.shape[1])[label].to(label.device)
    )

    pop = output["pop_res"]
    if pop.shape[1] > 1:
        loss += F.cross_entropy(pop, task_index)

    aux_res = output["aux_res"]
    if aux_res == None or (not output["is_decouple"]):
        return {"loss_backward": loss, "loss": loss.item()}
    all_cls_num = prompt.shape[1]
    aux_cls_num = aux_res.shape[1]
    if aux_cls_num > all_cls_num:
        return {"loss_backward": loss, "loss": loss.item()}
    new_cls_num = aux_cls_num - 1
    aux_label = label - all_cls_num + new_cls_num
    aux_label[aux_label < 0] = new_cls_num
    loss += F.cross_entropy(aux_res, aux_label)

    return {"loss_backward": loss, "loss": loss.item()}