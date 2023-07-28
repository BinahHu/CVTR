import torch
import os
from utils import sim_matrix

global_video_embeddings = []
global_text_embeddings = []

def img_cls_accuracy(output):
    label = output["label"]
    task_index = output["task_index"]
    prompt = output["prompt_res"]
    with torch.no_grad():
        pred = torch.argmax(prompt, dim=1)
        assert pred.shape[0] == len(label)
        correct = 0
        correct += torch.sum(pred == label).item()
    return correct / len(label)

def text2video_retrieval_accuracy(output):
    world_size = int(os.environ['WORLD_SIZE'])
    video_embeddings = output["video_embeddings"].contiguous()
    text_embeddings = output["text_embeddings"].contiguous()
    text_prompts = output["text_prompts"]
    video_prompts = output["video_prompts"]

    video_embeddings_all = [torch.zeros_like(video_embeddings) for _ in range(world_size)]
    text_embeddings_all = [torch.zeros_like(text_embeddings) for _ in range(world_size)]
    torch.distributed.all_gather(video_embeddings_all, video_embeddings)
    torch.distributed.all_gather(text_embeddings_all, text_embeddings)
    video_embeddings_all = torch.cat(video_embeddings_all, dim=0)
    text_embeddings_all = torch.cat(text_embeddings_all, dim=0)

    global global_video_embeddings
    global global_text_embeddings
    is_train = output["is_train"]
    if is_train:
        global_video_embeddings = []
        global_text_embeddings = []
    else:
        global_video_embeddings.append(video_embeddings_all.detach().cpu())
        global_text_embeddings.append(text_embeddings_all.detach().cpu())
        video_embeddings_all = torch.cat(global_video_embeddings).float()
        text_embeddings_all = torch.cat(global_text_embeddings).float()

    data_pred = sim_matrix(text_embeddings_all, video_embeddings_all)
    data_pred = data_pred.argmax(dim=-1)
    data_gt = torch.arange(data_pred.shape[0]).to(data_pred.device)

    return (data_pred == data_gt).sum() / data_pred.shape[0]

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
