import argparse
import collections
import torch
import numpy as np
import dataset as module_dataset
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import ContinualTrainer
from utils import prepare_device
import copy
import os


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config, args):
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='tcp://{}:{}'.format(
                                             args.master_address, args.master_port),
                                         rank=args.rank, world_size=args.world_size)

    logger = config.get_logger('train')

    # setup data_loader instances
    config['dataset']['args']['dataset_type'] = "train"
    train_dataset = config.init_obj("dataset", module_dataset, config=config)
    val_config = copy.deepcopy(config)
    val_config['dataset']['args']['dataset_type'] = "val"
    val_config["data_loader"]["args"]["shuffle"] = False
    valid_dataset = val_config.init_obj("dataset", module_dataset, config=val_config)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    #logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    device = torch.device(f'cuda:{args.local_rank}')
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainer = ContinualTrainer(config, criterion, metrics, train_dataset, valid_dataset, model, device)
    if args.eval:
        trainer.eval()
    else:
        trainer.train()


if __name__ == '__main__':
    try:    # with ddp
        master_address = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'])
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
    except:  # for debug only
        master_address = 9339
        master_port = 1
        world_size = 1
        rank = 0
        local_rank = 0

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-k', '--local_rank', type=int, default=local_rank)
    args.add_argument('-ma', '--master_address', default=master_address)
    args.add_argument('-mp', '--master_port', type=int, default=master_port)
    args.add_argument('-ws', '--world_size', type=int, default=world_size)
    args.add_argument('-rk', '--rank', type=int, default=rank)
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-eval', '--eval', action='store_true',
                      help='evaluation')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    args = args.parse_args()
    main(config, args)
