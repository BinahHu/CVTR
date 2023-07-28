import os
import json
import random


import data_loader.data_loaders as module_data
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize
from PIL import Image
import copy
import torch
import pandas as pd
import numpy as np
from utils.video_reader import RawVideoExtractorCV2


class ContinualMSRVTTDataset(Dataset):
    def __init__(self, config, dataset_type, meta_root_dir, dataset_root_dir,
                 feature_framerate, max_frames, slice_framepos, frame_order, task_split):
        self.config = config
        self.logger = config.get_logger("MSRVTT", verbosity=1)
        self.meta_data, self.task_split_list = self.build_meta_data(dataset_type, meta_root_dir, dataset_root_dir, task_split)
        self.memory_index = []
        self.categorywise_memory_index = []
        self.memory_size = config["memory"]["size"]
        self.current_task_index = -1
        self.max_frames = max_frames
        self.slice_framepos = slice_framepos
        self.frame_order = frame_order

        self.rawVideoExtractor = RawVideoExtractorCV2(framerate=feature_framerate, size=224)

    def build_meta_data(self, dataset_type, meta_root_dir, dataset_root_dir, task_split):
        meta_root_dir = meta_root_dir
        dataset_root_dir = dataset_root_dir
        dataset_type = dataset_type
        meta_file = "MSRVTT_data.json"
        if dataset_type == "train":
            index_file = "MSRVTT_train.9k.csv"
        elif dataset_type == "val":
            index_file = "MSRVTT_JSFUSION_test.csv"
        elif dataset_type == "test":
            index_file = "MSRVTT_JSFUSION_test.csv"
        else:
            raise NotImplementedError

        # Task split should be an array of array, which indicates the classes in each task
        self.task_split = task_split
        all_categories = sum([len(self.task_split[i]) for i in range(len(task_split))])
        self.category_mapping = [0 for _ in range(all_categories)]
        p = 0
        for i in range(len(task_split)):
            for j in range(len(task_split[i])):
                self.category_mapping[task_split[i][j]] = p
                p += 1

        with open(os.path.join(meta_root_dir, meta_file), "r") as f:
            raw_meta_data = json.load(f)
        self.task_index = []
        vid_dict = {}
        video_data = raw_meta_data["videos"]
        text_data = raw_meta_data["sentences"]
        meta_data = []
        for i, video in enumerate(video_data):
            cat = int(video['category'])
            video_path = video['video_id']
            vid_dict[video_path] = i
            for k in range(len(self.task_split)):
                if cat in self.task_split[k]:
                    self.task_index.append(k)
                    break
            meta_data.append([os.path.join(dataset_root_dir, video_path + ".mp4"), [], cat])
        if dataset_type == "train":
            for i, text in enumerate(text_data):
                caption = text['caption']
                video_path = text['video_id']
                meta_index = vid_dict[video_path]
                meta_data[meta_index][1].append(caption)

        task_split_list = []
        for _ in range(len(self.task_split)):
            task_split_list.append([])
        index_data = pd.read_csv(os.path.join(meta_root_dir, index_file))
        for i in range(len(index_data)):
            video_path = index_data['video_id'].values[i]
            meta_index = vid_dict[video_path]
            task_label = int(self.task_index[meta_index])

            if dataset_type == "train":
                task_split_list[task_label].append(meta_index)
            else:
                caption = index_data['sentence'].values[i]
                meta_data[meta_index][1].append(caption)
                for k in range(task_label, len(self.task_split)):
                    task_split_list[k].append(meta_index)
        self.current_task_split = task_split_list[0]
        return meta_data, task_split_list

    def next_task(self):
        # Goto next task and build the dataloader
        self.current_task_index += 1
        self.current_task_split = self.task_split_list[self.current_task_index] + self.memory_index

        data_loader = self.config.init_obj('data_loader', module_data, dataset=self)
        updates = {"new_categories": len(self.task_split[self.current_task_index])}
        return updates, data_loader

    def update_memory(self, extra_memory_index_list):
        all_categories = sum([len(self.task_split[i]) for i in range(self.current_task_index + 1)])
        bound = self.memory_size // all_categories
        self.memory_index = []
        for i, prev_memory_index in enumerate(self.categorywise_memory_index):
            self.categorywise_memory_index[i] = prev_memory_index[:min(bound, len(prev_memory_index))]
            self.memory_index += self.categorywise_memory_index[i]
        for extra_memory_index in extra_memory_index_list:
            self.categorywise_memory_index.append(extra_memory_index)
            self.memory_index += extra_memory_index
        self.current_task_split = self.task_split_list[self.current_task_index] + self.memory_index

    def build_decouple(self):
        selected_categories = self.task_split[self.current_task_index]
        max_prev_per_category = 0
        for prev_memory_index in self.categorywise_memory_index:
            max_prev_per_category = max(max_prev_per_category, len(prev_memory_index))
        curr_index = self.task_split_list[self.current_task_index]
        curr_categorywise_index = []
        selected_category_dict = {}
        for i, c in enumerate(selected_categories):
            curr_categorywise_index.append([])
            selected_category_dict[c] = i
        for idx in curr_index:
            _, _, cat = self.meta_data[idx]
            assert cat in selected_category_dict
            cid = selected_category_dict[cat]
            curr_categorywise_index[cid].append(idx)
        balanced_index = []
        for class_wise_index in curr_categorywise_index:
            random.shuffle(class_wise_index)
            class_wise_index = class_wise_index[:min(max_prev_per_category, len(class_wise_index))]
            balanced_index += class_wise_index
        balanced_index += self.memory_index
        random.shuffle(balanced_index)
        self.current_task_split = balanced_index

        data_loader = self.config.init_obj('data_loader', module_data, dataset=self)

        return {}, data_loader


    def build_memory(self):
        selected_categories = self.task_split[self.current_task_index]
        all_classes = sum([len(self.task_split[i]) for i in range(self.current_task_index + 1)])
        bound = self.memory_size // all_classes
        bound_list = []
        categorywise_datasets = []
        for selected_cat in selected_categories:
            dataset_cpy = copy.deepcopy(self)
            task_split = []
            for idx in dataset_cpy.current_task_split:
                _, _, cat = self.meta_data[idx]
                if cat == selected_cat:
                    task_split.append(idx)
            dataset_cpy.current_task_split = task_split
            categorywise_datasets.append(dataset_cpy)
            bound_list.append(bound)

        loader_list = [torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=8,
                                    pin_memory=True, drop_last=False, shuffle=False) for dataset in categorywise_datasets]

        return loader_list, bound_list

    def _get_raw_video(self, video_path):

        # Pair x L x T x 3 x H x W
        video = np.zeros((self.max_frames, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)
        video_mask = np.zeros((self.max_frames,), dtype=np.int32)
        raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
        raw_video_data = raw_video_data['video']
        if len(raw_video_data.shape) > 3:
            raw_video_data_clip = raw_video_data
            # L x T x 3 x H x W
            raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
            if self.max_frames < raw_video_slice.shape[0]:
                if self.slice_framepos == 0:
                    video_slice = raw_video_slice[:self.max_frames, ...]
                elif self.slice_framepos == 1:
                    video_slice = raw_video_slice[-self.max_frames:, ...]
                else:
                    sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                    video_slice = raw_video_slice[sample_indx, ...]
            else:
                video_slice = raw_video_slice

            video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

            slice_len = video_slice.shape[0]
            if slice_len < 1:
                pass
            else:
                video[:slice_len, ...] = video_slice
                video_mask[:slice_len] = [1] * slice_len
        else:
            self.logger.info(f"raw video shape is {raw_video_data.shape} video path: {video_path} error.")

        return video, video_mask


    def __len__(self):
        return len(self.current_task_split)

    def __getitem__(self, index):
        actual_index = self.current_task_split[index]
        video_path, text_list, cat = self.meta_data[actual_index]
        if len(text_list) > 1:
            rind = random.randint(0, len(text_list) - 1)
            caption = text_list[rind]
        else:
            caption = text_list[0]
        task_index = self.task_index[actual_index]
        cat = self.category_mapping[cat]
        video, video_mask = self._get_raw_video(video_path)

        return {"video": video, "video_mask": video_mask, "caption": caption, "category_index": cat, "task_index": task_index, "meta_index": actual_index}

if __name__ == "__main__":
    config = {"memory":{"size":2000}, "dataset_type": "train", "meta_root_dir": "/mnt/datasets/MSRVTT/videos/msrvtt_data",
              "dataset_root_dir":"/mnt/datasets/MSRVTT/videos/all", "max_frames": 12,
              "slice_framepos": 2, "frame_order": 0, "feature_framerate": 1,
              "task_split": [[0, 1, 2, 3, 4],
                             [5, 6, 7, 8, 9],
                             [10, 11, 12, 13, 14],
                             [15, 16, 17, 18, 19]]}
    d = ContinualMSRVTTDataset(config, config["dataset_type"], config["meta_root_dir"], config["dataset_root_dir"],
                               config["feature_framerate"], config["max_frames"],
                               config["slice_framepos"], config["frame_order"], config["task_split"])
