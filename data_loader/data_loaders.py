import os
from torch.utils.data import DataLoader, DistributedSampler


class ContinualLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=1, **kwargs):
        self.dataset = dataset
        self.dataset_type = dataset.dataset_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        if self.dataset_type == "train":
            world_size = int(os.environ['WORLD_SIZE'])
            rank = int(os.environ['RANK'])
            sampler = DistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
            num_workers = 0
            self.num_workers = 0
            super().__init__(self.dataset, batch_size=batch_size, pin_memory=False, num_workers=num_workers,
                             drop_last=False, sampler=sampler)
        else:
            # For validation/test dataset, to avoid the annoying drop_last problem, we do not use distributed sampler
            sampler = None
            super().__init__(self.dataset, batch_size=batch_size, pin_memory=False, num_workers=num_workers,
                             drop_last=False, shuffle=shuffle, sampler=sampler)

