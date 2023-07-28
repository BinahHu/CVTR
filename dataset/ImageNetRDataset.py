import os
import json
import random


import data_loader.data_loaders as module_data
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize
from PIL import Image
import copy
import torch


class ContinualImageNetRDataset(Dataset):
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
                                      ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            self.transform = Compose([Resize(256), CenterCrop(224),
                                      ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def build_meta_data(self, dataset_type, meta_root_dir, dataset_root_dir, task_split):
        meta_root_dir = meta_root_dir
        dataset_root_dir = dataset_root_dir
        dataset_type = dataset_type
        if dataset_type == "train":
            meta_file = "train.json"
        elif dataset_type == "val":
            meta_file = "val.json"
        elif dataset_type == "test":
            meta_file = "test.json"
        else:
            raise NotImplementedError

        # Class split should be an array of array, which indicates the classes in each task
        self.task_split = task_split
        all_classes = sum([len(self.task_split[i]) for i in range(len(task_split))])
        self.label_mapping = [0 for _ in range(all_classes)]
        p = 0
        for i in range(len(task_split)):
            for j in range(len(task_split[i])):
                self.label_mapping[task_split[i][j]] = p
                p += 1

        self.task_index = []
        task_split_list = []
        for _ in range(len(self.task_split)):
            task_split_list.append([])
        with open(os.path.join(meta_root_dir, meta_file), "r") as f:
            meta_data = json.load(f)

        for i, data in enumerate(meta_data):
            img_path, label = data
            img_path = os.path.join(dataset_root_dir, img_path)
            meta_data[i][0] = img_path
            for j in range(len(self.task_split)):
                if label in self.task_split[j]:
                    self.task_index.append(j)
                    if dataset_type == "train":
                        task_split_list[j].append(i)
                    else:
                        for k in range(j, len(self.task_split)):
                            task_split_list[k].append(i)
                    break
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

        loader_list = [torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=10,
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