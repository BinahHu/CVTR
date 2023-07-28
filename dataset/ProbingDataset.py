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

dataset_folders = ["caltech-101", "dtd", "eurosat", "fgvc_aircraft", "food-101",
                   "imagenet", "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101"]

imagenet_datasets = ["imagenet", "imagenet-adversarial", "imagenet-rendition", "imagenet-sketch", "imagenetv2"]

class ProbingDataset(Dataset):
    # Load the 11 datasets to check if the performances of original CLIP model is remained after continual learning
    def __init__(self, config, dataset_type, dataset_root_dir, few_shot, **kwargs):
        self.config = config
        self.few_shot = few_shot
        self.current_task_index = -1
        self.task_split = []
        self.task_index = []
        self.current_task_split = []
        self.cls_num_per_task = []
        self.meta_data = []
        self.build_meta_data(dataset_type, dataset_root_dir)
        # if int(os.environ['LOCAL_RANK']) == 0:
        #     print(f"Dataset type is {dataset_type}")
        #     print(f"Overall sample num is {len(self.meta_data)}")
        #     for i in range(11):
        #         print(f"Task {i} sample num is {len(self.task_split[i])}, sample per cls is {len(self.task_split[i]) // self.cls_num_per_task[i]}")
        if dataset_type == "train":
            self.transform = Compose([RandomResizedCrop(224), RandomHorizontalFlip(), ColorJitter(brightness=63 / 255),
                                      ToTensor(), Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
        else:
            self.transform = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC), CenterCrop(224),
                                     ToTensor(),
                                     Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

    def build_meta_data(self, dataset_type, dataset_root_dir):
        meta_data = []
        task_split = []
        task_index = []
        cls_num_per_task = []
        meta_idx = 0
        for dataset_id, dataset_dir in enumerate(dataset_folders):
            is_imagenet = (dataset_dir == "imagenet")
            cur_task_split = []
            cls_num = -1
            dataset_dir = os.path.join(dataset_root_dir, dataset_dir)
            split_file_name = ""
            for fn in os.listdir(dataset_dir):
                if fn.startswith("split_zhou"):
                    split_file_name = fn
                    break
            split_file_path = os.path.join(dataset_dir, split_file_name)
            with open(split_file_path, "r") as f:
                split_dir = json.load(f)
                data_items = split_dir[dataset_type]
                for data in data_items:
                    if is_imagenet:
                        meta_data.append([os.path.join(dataset_dir, data[0]), data[1], data[2]])
                    else:
                        meta_data.append([os.path.join(os.path.join(dataset_dir, "images"), data[0]), data[1], data[2]])
                    cur_task_split.append(meta_idx)
                    meta_idx += 1
                    task_index.append(dataset_id)
                    cls_num = max(cls_num, data[1])
            # Downsample food-101 and imagenet
            if dataset_id in [4,5] and dataset_type == "train":
                random.shuffle(cur_task_split)
                cur_task_split = cur_task_split[:min(100 * (cls_num + 1), len(cur_task_split))]

            cls_num_per_task.append(cls_num + 1)
            task_split.append(cur_task_split)


        if self.few_shot > 0:
            for j in range(len(task_split)):
                random.shuffle(task_split[j])
                task_split[j] = task_split[j][:min(self.few_shot * cls_num_per_task[j], len(task_split[j]))]


        self.task_split = task_split
        self.task_index = task_index
        self.current_task_split = task_split[0]
        self.cls_num_per_task = cls_num_per_task
        self.meta_data = meta_data

    def next_task(self):
        # Goto next task and build the dataloader
        self.current_task_index += 1
        self.current_task_split = self.task_split[self.current_task_index]

        data_loader = self.config.init_obj('data_loader', module_data, dataset=self)
        #data_loader = module_data.ContinualLoader(dataset=self, batch_size=64, num_workers=8)
        updates = {"new_classes": self.cls_num_per_task[self.current_task_index],
                   "dataset_name": dataset_folders[self.current_task_index]}

        return updates, data_loader

    def update_memory(self, extra_memory_index_list):
        pass

    def build_decouple(self):
        return None, None


    def build_memory(self):
        return None, None


    def __len__(self):
        return len(self.current_task_split)

    def __getitem__(self, index):
        actual_index = self.current_task_split[index]
        img_path, label, cls_name = self.meta_data[actual_index]
        task_index = self.task_index[actual_index]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {"img": img, "label": label, "task_index": task_index, "meta_index": actual_index}


if __name__ == "__main__":
    config = {"dataset_type": "train",
              "dataset_root_dir" : "/mnt/datasets/CLIPdatasets", "few_shot":-1 }
    d = ProbingDataset(config, config["dataset_type"], config["dataset_root_dir"], config["few_shot"])