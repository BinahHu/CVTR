import os
import json
import random


import data_loader.data_loaders as module_data
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize, InterpolationMode
from torchvision.transforms._transforms_video import RandomResizedCropVideo, RandomHorizontalFlipVideo
from PIL import Image
import copy
import torch
import torchvision
import numpy as np
import cv2


class ContinualEgoClipDataset(Dataset):
    def __init__(self, config, dataset_type, dataset_root_dir, few_shot, joint=False, random_split=False, random_seed=123):
        self.config = config
        self.few_shot = few_shot
        self.dataset_type = dataset_type
        self.video_text_flag = "video"
        self.joint = joint

        self.random_split = random_split
        self.random_seed = random_seed

        # These two lists store the actual path/data of video and text
        self.video_list = []
        self.text_list = []

        # These two lists store the indexes of video/text in the above lists for each task
        self.video_index_split_list = []
        self.text_index_split_list = []

        # Indexs for video and text in current task
        self.current_video_split = []
        self.current_text_split = []

        self.build_meta_data(dataset_type, dataset_root_dir)

        self.video_memory_index = []
        self.text_memory_index = []
        self.memory_size = config["memory"]["size"]
        self.current_task_index = -1

        self.vis_transform = None
        if dataset_type == "train":
            self.vis_transform = Compose([
                RandomResizedCropVideo(224, scale=(0.5, 1.0)),
                RandomHorizontalFlipVideo(),
                ColorJitter(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.vis_transform = Compose([
                Resize(256),
                CenterCrop(256),
                Resize(224),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def build_meta_data(self, dataset_type, dataset_root_dir):
        meta_data_list = json.load(open(os.path.join(dataset_root_dir, 'task_meta_data_split10.json')))
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

        video_idx = 0
        txt_idx = 0
        for i, meta_data in enumerate(meta_data_list):
            meta_data = meta_data[dataset_type]
            video_list = []
            text_list = []
            for data in meta_data:
                self.video_list.append({'dir': os.path.join(dataset_root_dir, data['video_dir']),
                                        'start_time': data['start_time'], 'end_time': data['end_time']})
                self.text_list.append(data['caption'])

                img_item = {}
                img_item['video_idx'] = video_idx
                img_item['text_idx'] = txt_idx

                text_item = {}
                text_item['video_idx'] = video_idx
                text_item['text_idx'] = txt_idx

                text_list.append(text_item)
                video_list.append(img_item)
                txt_idx += 1
                video_idx += 1

            if dataset_type == "train" and (not self.joint):
                self.video_index_split_list.append(video_list)
                self.text_index_split_list.append(text_list)
                self.frame_sample = 'rand'
            else:
                prev_video_list = [] if len(self.video_index_split_list) == 0 else self.video_index_split_list[-1]
                prev_text_list = [] if len(self.text_index_split_list) == 0 else self.text_index_split_list[-1]
                self.video_index_split_list.append(prev_video_list + video_list)
                self.text_index_split_list.append(prev_text_list + text_list)
                self.frame_sample = 'uniform'

        if self.few_shot > 0 and dataset_type == "train":
            for j in range(len(self.text_index_split_list)):
                text_split = copy.deepcopy(self.text_index_split_list[j])
                random.shuffle(text_split)
                self.text_index_split_list[j] = text_split[:min(self.few_shot, len(text_split))]

        if self.joint:
            self.video_index_split_list = [self.video_index_split_list[-1]]
            self.text_index_split_list = [self.text_index_split_list[-1]]

        self.current_video_split = self.video_index_split_list[0]
        self.current_text_split = self.text_index_split_list[0]


    def next_task(self):
        # Goto next task and build the dataloader
        self.current_task_index += 1
        if self.dataset_type == "train":
            self.current_video_split = self.video_index_split_list[self.current_task_index] + self.video_memory_index
            self.current_text_split = self.text_index_split_list[self.current_task_index] + self.text_memory_index
        else:
            self.current_video_split = self.video_index_split_list[self.current_task_index]
            self.current_text_split = self.text_index_split_list[self.current_task_index]

        data_loader = self.config.init_obj('data_loader', module_data, dataset=self)
        # data_loader = module_data.ContinualLoader(dataset=self, batch_size=2)
        updates = {}
        return updates, data_loader

    def update_memory(self, extra_video_memory_index, extra_text_memory_index):
        assert self.dataset_type == "train", f"Only training dataset can update memory, the current dataset type is {self.dataset_type}"
        T = self.current_task_index + 1
        bound = (self.memory_size // T) * (T-1)
        self.video_memory_index += extra_video_memory_index
        self.text_memory_index = self.text_memory_index[:bound] + extra_text_memory_index

        self.current_video_split = self.video_index_split_list[self.current_task_index] + self.video_memory_index
        self.current_text_split = self.text_index_split_list[self.current_task_index] + self.text_memory_index

    def build_decouple(self):
        assert self.dataset_type == "train", f"Only training dataset can build decouple, the current dataset type is {self.dataset_type}"
        balanced_video_index = self.video_memory_index + self.current_video_split

        T = self.current_task_index + 1
        bound = len(self.text_memory_index) // (T-1)
        current_text_split = copy.deepcopy(self.current_text_split)
        random.shuffle(current_text_split)
        current_text_split = current_text_split[:min(bound, len(current_text_split))]
        balanced_text_index = self.text_memory_index + current_text_split

        self.current_video_split = balanced_video_index
        self.current_text_split = balanced_text_index

        data_loader = self.config.init_obj('data_loader', module_data, dataset=self)

        return {}, data_loader

    def switch_flag(self, flag):
        if flag == "visual":
            flag = "video"
        self.video_text_flag = flag

    def __len__(self):
        if self.dataset_type == "train":
            return len(self.current_text_split)
        else:
            return len(self.current_video_split) if self.video_text_flag == "video" else len(self.current_text_split)

    def sample_frames_start_end(self, num_frames, start, end, sample='rand', fix_start=None):
        intervals = np.linspace(start=start, stop=end, num=num_frames + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        elif fix_start is not None:
            frame_idxs = [x[0] + fix_start for x in ranges]
        elif sample == 'uniform':
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        return frame_idxs

    def get_video_frames(self, video_item):
        video_dir = video_item['dir']
        start_time = max(float(video_item['start_time']), 0)
        end_time = max(float(video_item['end_time']), 0)

        chunk_start_id = int(start_time // 600)
        chunk_end_id = int(end_time // 600)

        full_video_start_fp = os.path.join(video_dir, str(chunk_start_id) + ".mp4")
        full_video_end_fp = os.path.join(video_dir, str(chunk_end_id) + ".mp4")
        bound_sec = (chunk_start_id + 1) * 600

        if full_video_start_fp == full_video_end_fp:
            cap1 = cv2.VideoCapture(full_video_start_fp)
            cap2 = cap1
            vlen1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
            vlen2 = vlen1
            assert (cap1.isOpened())
            assert (cap1.get(cv2.CAP_PROP_FPS) == 30)
        else:  # some clips may span two segments.
            cap1 = cv2.VideoCapture(full_video_start_fp)
            cap2 = cv2.VideoCapture(full_video_end_fp)
            vlen1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
            vlen2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
            assert (cap1.isOpened())
            assert (cap2.isOpened())
            assert (cap1.get(cv2.CAP_PROP_FPS) == 30)
            assert (cap2.get(cv2.CAP_PROP_FPS) == 30)

        start_f = max(0, int(start_time * 30))
        end_f = max(0, int(end_time * 30))
        bound_f = int(bound_sec * 30)
        num_frames = 4
        frame_idxs = self.sample_frames_start_end(num_frames, start_f, end_f, sample=self.frame_sample)

        frames = []
        success_idxs = []
        for index in frame_idxs:
            _index = index % (600 * 30)
            if index > bound_f:  # frame from the last video
                _index = min(_index, vlen2)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, _index - 1)
                ret, frame = cap2.read()
            else:  # frame from the first video
                _index = min(_index, vlen1)
                cap1.set(cv2.CAP_PROP_POS_FRAMES, _index - 1)
                ret, frame = cap1.read()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if ret:
                # frame = Image.fromarray(frame)
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1)
                frames.append(frame)
                success_idxs.append(index)

        while len(frames) < num_frames:  # complete the frame
            frames.append(frames[-1])

        frames = torch.stack(frames).float() / 255

        cap1.release()
        cap2.release()

        if self.vis_transform is not None:
            # frames = [self.vis_transform(frame) for frame in frames]
            frames = self.vis_transform(frames)
        # video = torch.stack(frames)
        video = frames

        return video


    def __getitem__(self, index):
        # For training set, we load the text and its corresponding video
        # For validation/test set, video and text are loaded separately

        if self.dataset_type == "train":
            text_item = self.current_text_split[index]
            video_idx = text_item['video_idx']
            text_idx = text_item['text_idx']
            text = self.text_list[text_idx]
            video_item = self.video_list[video_idx]
            video = self.get_video_frames(video_item)

            return {
                "visual": video,
                "text": text,
                "visual_idx": video_idx,
                "text_idx": text_idx
            }
        else:
            if self.video_text_flag == "video":
                video_item = self.current_video_split[index]
                video_idx = video_item["video_idx"]
                text_idxs = [video_item["text_idx"]]
                L = len(text_idxs)
                video_item = self.video_list[video_idx]
                video = self.get_video_frames(video_item)

                return {
                    "visual": video,
                    "visual_idx": video_idx,
                    "text_idxs": np.array(text_idxs),
                    "text_idxs_length": L
                }
            else:
                text_item = self.current_text_split[index]
                video_idx = text_item['video_idx']
                text_idx = text_item['text_idx']

                text = self.text_list[text_idx]

                return {
                    "text": text,
                    "visual_idx": video_idx,
                    "text_idx": text_idx,
                }

if __name__ == "__main__":
    config = {"memory": {"size": 0}, "dataset_type": "train", "dataset_root_dir": "/mnt/ego4d-256"}
    d = ContinualEgoClipDataset(config, config["dataset_type"], config["dataset_root_dir"], -1)
    updates, data_loader = d.next_task()
    for data in data_loader:
        video = data["visual"]
        txts = data["text"]
        print(video.shape)
        print(txts)
        c = input()
