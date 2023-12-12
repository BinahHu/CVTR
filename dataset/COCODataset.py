import os
import json
import random


import data_loader.data_loaders as module_data
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize, InterpolationMode
from PIL import Image
import copy
import torch
import torchvision
import numpy as np


class ContinualCOCODataset(Dataset):
    def __init__(self, config, dataset_type, dataset_root_dir, few_shot, joint=False, task_num=10,
                 random_split=False, random_seed=123):
        self.config = config
        self.few_shot = few_shot
        self.dataset_type = dataset_type
        self.image_text_flag = "image"
        self.joint = joint
        self.task_num = task_num

        self.random_split = random_split
        self.random_seed = random_seed

        # These two lists store the actual path/data of image and text
        self.image_list = []
        self.text_list = []

        # These two lists store the indexes of image/text in the above lists for each task
        self.image_index_split_list = []
        self.text_index_split_list = []

        # Indexs for image and text in current task
        self.current_image_split = []
        self.current_text_split = []

        self.build_meta_data(dataset_type, dataset_root_dir)

        self.image_memory_index = []
        self.text_memory_index = []
        self.memory_size = config["memory"]["size"]
        self.current_task_index = -1

        self.vis_transform = None
        if dataset_type == "train":
            self.vis_transform = Compose([
                RandomResizedCrop(224),
                RandomHorizontalFlip(),
                ColorJitter(brightness=63 / 255),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        else:
            self.vis_transform = Compose([
                Resize(224, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(224),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

    def build_meta_data(self, dataset_type, dataset_root_dir):
        meta_data_list = json.load(open(os.path.join(dataset_root_dir, f'task_meta_data_split{self.task_num}.json')))
        if self.random_split:
            meta_data_all = []
            for dl in meta_data_list:
                meta_data_all += dl[dataset_type]
            random.seed(self.random_seed)
            random.shuffle(meta_data_all)
            data_idx = 0
            for i in range(len(meta_data_list)):
                l = len(meta_data_list[i][dataset_type])
                meta_data_list[i][dataset_type] = meta_data_all[data_idx:data_idx+l]
                data_idx += l

        img_idx = 0
        txt_idx = 0
        for i, meta_data in enumerate(meta_data_list):
            meta_data = meta_data[dataset_type]
            image_list = []
            text_list = []
            for data in meta_data:
                self.image_list.append(os.path.join(dataset_root_dir, data['file_path']))
                img_item = {}
                img_item['text_idxs'] = []
                img_item['image_idx'] = img_idx
                for caption in data['captions']:
                    self.text_list.append(caption)
                    text_item = {}
                    text_item['image_idx'] = img_idx
                    text_item['text_idx'] = txt_idx
                    img_item['text_idxs'].append(txt_idx)
                    text_list.append(text_item)
                    txt_idx += 1
                image_list.append(img_item)
                img_idx += 1
            if dataset_type == "train" and (not self.joint):
                self.image_index_split_list.append(image_list)
                self.text_index_split_list.append(text_list)
            else:
                prev_image_list = [] if len(self.image_index_split_list) == 0 else self.image_index_split_list[-1]
                prev_text_list = [] if len(self.text_index_split_list) == 0 else self.text_index_split_list[-1]
                self.image_index_split_list.append(prev_image_list + image_list)
                self.text_index_split_list.append(prev_text_list + text_list)

        if self.few_shot > 0 and dataset_type == "train":
            for j in range(len(self.text_index_split_list)):
                text_split = copy.deepcopy(self.text_index_split_list[j])
                random.shuffle(text_split)
                self.text_index_split_list[j] = text_split[:min(self.few_shot, len(text_split))]

        if self.joint:
            self.image_index_split_list = [self.image_index_split_list[-1]]
            self.text_index_split_list = [self.text_index_split_list[-1]]

        self.current_image_split = self.image_index_split_list[0]
        self.current_text_split = self.text_index_split_list[0]


    def next_task(self):
        # Goto next task and build the dataloader
        self.current_task_index += 1
        if self.dataset_type == "train":
            self.current_image_split = self.image_index_split_list[self.current_task_index] + self.image_memory_index
            self.current_text_split = self.text_index_split_list[self.current_task_index] + self.text_memory_index
        else:
            self.current_image_split = self.image_index_split_list[self.current_task_index]
            self.current_text_split = self.text_index_split_list[self.current_task_index]

        data_loader = self.config.init_obj('data_loader', module_data, dataset=self)
        #data_loader = module_data.ContinualLoader(dataset=self, batch_size=2)
        updates = {}
        return updates, data_loader

    def update_memory(self, extra_image_memory_index, extra_text_memory_index):
        assert self.dataset_type == "train", f"Only training dataset can update memory, the current dataset type is {self.dataset_type}"
        T = self.current_task_index + 1
        bound = (self.memory_size // T) * (T-1)
        self.image_memory_index += extra_image_memory_index
        self.text_memory_index = self.text_memory_index[:bound] + extra_text_memory_index

        self.current_image_split = self.image_index_split_list[self.current_task_index] + self.image_memory_index
        self.current_text_split = self.text_index_split_list[self.current_task_index] + self.text_memory_index

    def build_decouple(self):
        assert self.dataset_type == "train", f"Only training dataset can build decouple, the current dataset type is {self.dataset_type}"
        balanced_image_index = self.image_memory_index + self.current_image_split

        T = self.current_task_index + 1
        bound = len(self.text_memory_index) // (T-1)
        current_text_split = copy.deepcopy(self.current_text_split)
        random.shuffle(current_text_split)
        current_text_split = current_text_split[:min(bound, len(current_text_split))]
        balanced_text_index = self.text_memory_index + current_text_split

        self.current_image_split = balanced_image_index
        self.current_text_split = balanced_text_index

        data_loader = self.config.init_obj('data_loader', module_data, dataset=self)

        return {}, data_loader

    def switch_flag(self, flag):
        if flag == "visual":
            flag = "image"
        self.image_text_flag = flag

    def __len__(self):
        if self.dataset_type == "train":
            return len(self.current_text_split)
        else:
            return len(self.current_image_split) if self.image_text_flag == "image" else len(self.current_text_split)

    def __getitem__(self, index):
        # For training set, we load the text and its corresponding image
        # For validation/test set, image and text are loaded separately

        if self.dataset_type == "train":
            text_item = self.current_text_split[index]
            image_idx = text_item['image_idx']
            text_idx = text_item['text_idx']

            img = self.image_list[image_idx]
            text = self.text_list[text_idx]
            img = Image.open(img).convert("RGB")

            if self.vis_transform is not None:
                img = self.vis_transform(img)

            return {
                "visual": img,
                "text": text,
                "visual_idx": image_idx,
                "text_idx": text_idx
            }
        else:
            if self.image_text_flag == "image":
                image_item = self.current_image_split[index]
                image_idx = image_item["image_idx"]
                text_idxs = image_item["text_idxs"]
                PADDING_CONSTANT = 10
                L = len(text_idxs)
                text_idxs += [-1] * (PADDING_CONSTANT - L)
                img = self.image_list[image_idx]
                img = Image.open(img).convert("RGB")

                if self.vis_transform is not None:
                    img = self.vis_transform(img)

                return {
                    "visual": img,
                    "visual_idx": image_idx,
                    "text_idxs": np.array(text_idxs),
                    "text_idxs_length": L
                }
            else:
                text_item = self.current_text_split[index]
                image_idx = text_item['image_idx']
                text_idx = text_item['text_idx']

                text = self.text_list[text_idx]

                return {
                    "text": text,
                    "visual_idx": image_idx,
                    "text_idx": text_idx,
                }

if __name__ == "__main__":
    config = {"memory": {"size": 2000}, "dataset_type": "train", "dataset_root_dir": "/mnt/datasets/COCO"}
    d = ContinualCOCODataset(config, config["dataset_type"], config["dataset_root_dir"], -1)
    updates, data_loader = d.next_task()
    for data in data_loader:
        imgs = data["image"]
        txts = data["text"]
        print(imgs.shape)
        print(txts)
        c = input()
