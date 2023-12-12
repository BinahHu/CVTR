import numpy as np
import torch
import os
from utils import sim_matrix

global_video_embeddings = []
global_text_embeddings = []


def visualtext_retrieval_accuracy(data):
    visual_feats = data["visual_feats"]
    text_feats = data["text_feats"]
    visual2text_idx_record = data["visual2text_record"]
    text2visual_idx_record = data["text2visual_record"]
    is_eval = data["is_eval"]

    v2t_list = []
    t2v_list = []
    if is_eval:
        visual_embeddings_all = visual_feats
        text_embeddings_all = text_feats
        for d in visual2text_idx_record:
            v2t_list.append(d[1])
        for d in text2visual_idx_record:
            t2v_list.append(int(d[1]))
    else:
        # Do not sync across different GPUs, but this is just training acc, should be fine.
        visual_embeddings_all = visual_feats
        text_embeddings_all = text_feats
        for d in visual2text_idx_record:
            tmp = []
            for txt_id in d[1]:
                for i in range(len(text2visual_idx_record)):
                    if txt_id == int(text2visual_idx_record[i][0]):
                        tmp.append(i)
                        break
            v2t_list.append(tmp)
        for d in text2visual_idx_record:
            for i in range(len(visual2text_idx_record)):
                if int(d[1]) == int(visual2text_idx_record[i][0]):
                    t2v_list.append(i)
                    break


    visual2text = sim_matrix(visual_embeddings_all, text_embeddings_all).detach().cpu().numpy()
    positive_text_rank = np.zeros(visual2text.shape[0])
    for i, score in enumerate(visual2text):
        inds = np.argsort(score)[::-1]
        rank = inds.shape[0] + 1
        for j in v2t_list[i]:
            rk = np.where(inds == j)[0][0]
            rank = min(rank, rk)
        positive_text_rank[i] = rank
    v2t_top1_acc = 100.0 * len(np.where(positive_text_rank < 1)[0]) / len(positive_text_rank)
    v2t_top5_acc = 100.0 * len(np.where(positive_text_rank < 5)[0]) / len(positive_text_rank)

    text2visual = sim_matrix(text_embeddings_all, visual_embeddings_all).detach().cpu().numpy()
    positive_visual_rank = np.zeros(text2visual.shape[0])
    for i, score in enumerate(text2visual):
        inds = np.argsort(score)[::-1]
        j = t2v_list[i]
        rank = np.where(inds == j)[0][0]
        positive_visual_rank[i] = rank
    t2v_top1_acc = 100.0 * len(np.where(positive_visual_rank < 1)[0]) / len(positive_visual_rank)
    t2v_top5_acc = 100.0 * len(np.where(positive_visual_rank < 5)[0]) / len(positive_visual_rank)

    return {"visual2text_retrieval_accuracy_top1": v2t_top1_acc,
            "visual2text_retrieval_accuracy_top5": v2t_top5_acc,
            "text2visual_retrieval_accuracy_top1": t2v_top1_acc,
            "text2visual_retrieval_accuracy_top5": t2v_top5_acc}

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
