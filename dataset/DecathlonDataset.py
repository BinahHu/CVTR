import os
import json
import random


import data_loader.data_loaders as module_data
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize
from PIL import Image
import copy
import torch
import torchvision
import numpy as np


class ContinualDecathlonDataset(Dataset):
    def __init__(self, config, dataset_type, meta_root_dir, dataset_root_dir, task_split, few_shot):
        self.config = config
        self.few_shot = few_shot
        self.meta_data, self.task_split_list = self.build_meta_data(dataset_type, meta_root_dir, dataset_root_dir, task_split)
        self.memory_index = []
        self.classwise_memory_index = []
        self.memory_size = config["memory"]["size"]
        self.current_task_index = -1

        if dataset_type == "train":
            self.transform = Compose([RandomResizedCrop(224), RandomHorizontalFlip(), ColorJitter(brightness=63 / 255),
                                      ToTensor(), Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])
        else:
            self.transform = Compose([Resize(256), CenterCrop(224),
                                      ToTensor(), Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])

    def build_meta_data(self, dataset_type, meta_root_dir, dataset_root_dir, task_split):
        dataset_name = dataset_root_dir.split('/')[-1]
        meta_file = os.path.join(meta_root_dir, f"{dataset_name}_{dataset_type}.json")
        meta_data = []

        all_classes = len(os.listdir(os.path.join(dataset_root_dir, dataset_type)))
        class_orders = list(range(all_classes))
        random.seed(123)
        random.shuffle(class_orders)
        task_num = 10
        if dataset_name == 'svhn':
            task_num = 5
        task_step = all_classes // task_num
        task_split = []
        cat2task = {}
        for i in range(task_num-1):
            task_split.append(class_orders[i * task_step : i * task_step + task_step])
            for j in class_orders[i * task_step : i * task_step + task_step]:
                cat2task[j] = i
        task_split.append(class_orders[(task_num-1) * task_step:])
        for j in class_orders[(task_num-1) * task_step:]:
            cat2task[j] = task_num - 1

        self.task_split = task_split
        self.label_mapping = [0 for i in range(all_classes)]
        for i in range(len(class_orders)):
            cls_cat = class_orders[i]
            self.label_mapping[cls_cat] = i

        self.task_index = []
        task_split_list = []
        for _ in range(len(self.task_split)):
            task_split_list.append([])

        raw_meta_file = json.load(open(meta_file, 'r'))
        imgs = raw_meta_file["images"]
        annotations = raw_meta_file["annotations"]
        assert len(imgs) == len(annotations)

        for i in range(len(imgs)):
            img_info = imgs[i]
            annotation_info = annotations[i]
            img_path = img_info["file_name"]
            img_path = os.path.join(dataset_root_dir, "/".join(img_path.split('/')[2:]))
            cat_id = annotation_info["category_id"] % 10000 - 1
            meta_data.append([img_path, cat_id])
            task_id = cat2task[cat_id]
            self.task_index.append(task_id)
            if dataset_type == "train":
                task_split_list[task_id].append(i)
            else:
                for k in range(task_id, len(self.task_split)):
                    task_split_list[k].append(i)

        if self.few_shot > 0:
            for j in range(len(self.task_split)):
                random.shuffle(task_split_list[j])
                task_split_list[j] = task_split_list[j][:min(self.few_shot * len(self.task_split[j]), len(task_split_list[j]))]
        self.current_task_split = task_split_list[0]
        return meta_data, task_split_list

    def next_task(self):
        # Goto next task and build the dataloader
        self.current_task_index += 1
        self.current_task_split = self.task_split_list[self.current_task_index] + self.memory_index

        data_loader = self.config.init_obj('data_loader', module_data, dataset=self)
        updates = {"new_classes": len(self.task_split[self.current_task_index])}
        return updates, data_loader

    def update_memory(self, extra_memory_index_list):
        all_classes = sum([len(self.task_split[i]) for i in range(self.current_task_index + 1)])
        bound = self.memory_size // all_classes
        self.memory_index = []
        for i, prev_memory_index in enumerate(self.classwise_memory_index):
            self.classwise_memory_index[i] = prev_memory_index[:min(bound, len(prev_memory_index))]
            self.memory_index += self.classwise_memory_index[i]
        for extra_memory_index in extra_memory_index_list:
            self.classwise_memory_index.append(extra_memory_index)
            self.memory_index += extra_memory_index
        self.current_task_split = self.task_split_list[self.current_task_index] + self.memory_index

    def build_decouple(self):
        selected_classes = self.task_split[self.current_task_index]
        max_prev_per_class = 0
        for prev_memory_index in self.classwise_memory_index:
            max_prev_per_class = max(max_prev_per_class, len(prev_memory_index))
        curr_index = self.task_split_list[self.current_task_index]
        curr_classwise_index = []
        selected_classes_dict = {}
        for i, c in enumerate(selected_classes):
            curr_classwise_index.append([])
            selected_classes_dict[c] = i
        for idx in curr_index:
            _, label = self.meta_data[idx]
            assert label in selected_classes_dict
            cid = selected_classes_dict[label]
            curr_classwise_index[cid].append(idx)
        balanced_index = []
        for class_wise_index in curr_classwise_index:
            random.shuffle(class_wise_index)
            class_wise_index = class_wise_index[:min(max_prev_per_class, len(class_wise_index))]
            balanced_index += class_wise_index
        balanced_index += self.memory_index
        random.shuffle(balanced_index)
        self.current_task_split = balanced_index

        data_loader = self.config.init_obj('data_loader', module_data, dataset=self)

        return {}, data_loader


    def build_memory(self):
        selected_classes = self.task_split[self.current_task_index]
        all_classes = sum([len(self.task_split[i]) for i in range(self.current_task_index + 1)])
        bound = self.memory_size // all_classes
        bound_list = []
        classwise_datasets = []
        for cls in selected_classes:
            dataset_cpy = copy.deepcopy(self)
            task_split = []
            for idx in dataset_cpy.current_task_split:
                _, label = self.meta_data[idx]
                if label == cls:
                    task_split.append(idx)
            dataset_cpy.current_task_split = task_split
            classwise_datasets.append(dataset_cpy)
            bound_list.append(bound)

        loader_list = [torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=10,
                                    pin_memory=True, drop_last=False, shuffle=False) for dataset in classwise_datasets]

        return loader_list, bound_list


    def __len__(self):
        return len(self.current_task_split)

    def __getitem__(self, index):
        actual_index = self.current_task_split[index]
        img_path, label = self.meta_data[actual_index]
        label = self.label_mapping[label]
        task_index = self.task_index[actual_index]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {"img": img, "label": label, "task_index": task_index, "meta_index": actual_index}