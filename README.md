# CVTR

Ongoing project of Zhiyuan Hu about Continual Visual Text Retrieval

The repository is inherited from the CTP(*"[CTP: Towards Vision-Language Continual Pretraining via Compatible Momentum Contrast and Topology Preservation](https://arxiv.org/abs/2308.07146)"*.) paper and its [code repo](https://github.com/KevinLight831/CTP) 

## Introduction
Continual learning has long been focused on closed-set problems like image classification, with a relatively small scale backbone and small scale datasets. Real world problems, are typically open set, large scale. This project would like to extend the continual learning literature to a more realistic scenario, with the usage of large scale foundation models.


## Setup and Environments

See *requirements.txt*.
I am using the nautilus system of UCSD to train the model, a sample configuration *cvtr_job.yaml* is included. Please use *push.sh* to upload the code to nautilus server.

## Datasets and configuration
Follow the CTP paper, we use the [P9D](https://github.com/KevinLight831/P9D) image-text retrieval dataset to do experiments. We are also attempting to do some experiments on the [Ego4d]( https://ego4d-data.org/), or more specifically, the [EgoClip]( https://github.com/showlab/EgoVLP) benchmark.


## Current process and findings
Different from CTP, we want to use methods like prompt tuning or adapters, lora to fine tune a foundation model with small parameters. The foundation model we use is ChineseCLIP on P9D.

Our initial results show that the prompt tuning method is not working well, no matter how we tune the prompts or use other prompt tuning methods (like deep prompt, CODA-prompt). The prompt tuning model is even worse than the zero-shot ChineseCLIP model. However, we find that if we fine tune the entire ChineseCLIP model, the performances will be much better than zero-shot. 

Our analysis goes like this: prompt tuning can change the output of a transformer by changing the attentions. However, the projection layers (kqv projection layers and the MLP layers) are not changed. We can thus assume the feature space of the transformer is not changed too much, prompt tuning is biasing the model towards some subspace of the original feature space. For classification tasks, this is fine, since classification only needs a boundary. For retrieval tasks, the model needs to indeed learn a better feature space. That might be the reason that prompt tuning does not work but finetuning can work.

However, some recent works like [VoP]( https://arxiv.org/abs/2211.12764) show that prompt tuning can work in video-text retrieval tasks, so there might be something wrong in our analysis and experiments about prompt tuning.

Anyway we try to use adapter based methods like residual adapter or [ConvNorm]( https://github.com/liyunsheng13/Efficient-Multi-Domain-Network-Learning-by-Covariance-Normalization?tab=readme-ov-file). Theoretically these methods should be able to perform similarly to the finetuned model, but in our case there is still a gap between these adapter based methods and finetuned model.

Also, the previous experiments are all conducted in joint training setting, we need to first ensure that the joint model is better than zero-shot model, and then move to continual learning setting.

## TO list

- [ ] Make adapter based method work
- [ ] Check if prompt tuning is indeed not working
- [ ] Run the experiments in continual learning setting
