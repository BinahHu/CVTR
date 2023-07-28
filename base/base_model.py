import torch.nn as nn
import numpy as np
from abc import abstractmethod
import torch

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, *inputs):
        super(BaseModel, self).__init__()
        self.place_holder_param = nn.Parameter(torch.zeros(1))
        self.prob_mode = False

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def activate_prob_mode(self):
        """
        Enter prob mode
        """
        raise NotImplementedError

    def prob_next_task(self, *updates):
        """
        Next task in prob mode
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
