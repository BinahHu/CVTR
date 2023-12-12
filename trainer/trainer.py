import copy
import os
import os.path

import numpy as np
import torch
from torchvision.utils import make_grid
from numpy import inf
from utils import inf_loop, MetricTracker, memory_builder
from logger import TensorboardWriter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
class ContinualTrainer:
    global_debug_flag = False
    """
    Trainer class
    """
    def __init__(self, config, criterion, metrics, train_dataset, val_dataset, model, device,
                 iter_per_epoch=None):
        self.config = config
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metrics
        self.device = device
        self.checkpoint_dir = config["trainer"]["save_dir"]
        self.data_loader = None
        self.valid_data_loader = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.do_validation = self.valid_data_loader is not None
        self.val_step = config["trainer"].get("val_step", -1)
        self.lr_scheduler = None
        self.optimizer = None
        self.epochs = None

        # Continual group index
        self.group_index = config["continual"]["group_idx"]
        self.current_task_id = -1
        self.start_task_id = 0

        self.task_num = config["continual"]["task_num"]
        self.decouple = config["continual"].get("decouple", False)
        self.decouple_group_index = config["continual"].get("decouple_group_index", -1)

        # Memory setup
        self.memory_size = config["memory"]["size"]
        self.memory_strategy = config["memory"]["memory_strategy"]

        # Trainer setup
        cfg_trainer = config['trainer']
        self.trainer_group_args = cfg_trainer["group_args"]
        self.start_epoch = 1
        self.save_period = cfg_trainer['save_period']

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        self.iter_per_epoch = iter_per_epoch

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        metrics = []
        for m in self.metric_ftns:
            if m.__name__ == "visualtext_retrieval_accuracy":
                metrics += ["visual2text_retrieval_accuracy_top1", "visual2text_retrieval_accuracy_top5",
                            "text2visual_retrieval_accuracy_top1", "text2visual_retrieval_accuracy_top5"]
            else:
                metrics.append(m.__name__)
        self.incremental_metrics = MetricTracker(*metrics, writer=self.writer)
        self.train_metrics = MetricTracker('loss', *metrics, writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *metrics, writer=self.writer)

    def next_task(self):
        updates, train_dataloader = self.train_dataset.next_task()
        if hasattr(self.model, "module"):
            self.model.module.next_task(updates)
        else:
            self.model.next_task(updates)
        updates, val_dataloader = self.val_dataset.next_task()
        self.current_task_id += 1

        self.data_loader = train_dataloader
        self.valid_data_loader = val_dataloader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = self.config["trainer"].get("log_step", int(np.sqrt(self.data_loader.batch_size)))

        if self.iter_per_epoch is None:
            # epoch-based training
            self.len_epoch = len(train_dataloader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(train_dataloader)
            self.len_epoch = self.iter_per_epoch

        # Build group args
        args_group_index = self.group_index[self.current_task_id]
        self.epochs = self.trainer_group_args[args_group_index]["epochs"]
        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.config.init_obj('optimizer', torch.optim, params=trainable_params, index=args_group_index)
        lr_scheduler = self.config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer=optimizer, index=args_group_index)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def build_decouple(self):
        decouple_train_dataset = copy.deepcopy(self.train_dataset)
        updates, decouple_train_loader = decouple_train_dataset.build_decouple()
        if self.iter_per_epoch is None:
            # epoch-based training
            len_epoch = len(decouple_train_loader)
        else:
            # iteration-based training
            decouple_train_loader = inf_loop(decouple_train_loader)
            len_epoch = self.iter_per_epoch

        self.decouple_epochs = self.trainer_group_args[self.decouple_group_index]["epochs"]
        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        if hasattr(self.model, "module"):
            self.model.module.prepare_decouple()
        else:
            self.model.prepare_decouple()
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.config.init_obj('optimizer', torch.optim, params=trainable_params, index=self.decouple_group_index)
        lr_scheduler = self.config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer=optimizer,
                                            index=self.decouple_group_index)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        return decouple_train_loader, len_epoch

    def eval(self):
        self.incremental_metrics.reset()
        eval_start_idx = 0
        eval_end_idx = 10
        epoch = 170
        for task_id in range(0, eval_start_idx):
            self.next_task()
        for task_id in range(eval_start_idx, eval_end_idx):
            self.next_task()
            exper_name = self.config['name']
            model_checkpoint_dir = os.path.join(self.checkpoint_dir, f"{exper_name}")
            task_checkpoint_dir = os.path.join(model_checkpoint_dir, f"task{task_id}")
            filename = f"{task_checkpoint_dir}/checkpoint-epoch{epoch}.pth"
            self._resume_checkpoint(filename, bypass_optimizer=True)

            self.model.eval()
            self.valid_metrics.reset()
            taskwise_acc = [0 for _ in range(10)]
            taskwise_cnt = [0 for _ in range(10)]
            with torch.no_grad():
                for batch_idx, data in enumerate(self.valid_data_loader):
                    if hasattr(self.model, "module"):
                        data = self.model.module.input_preprocess(data)
                    else:
                        data = self.model.input_preprocess(data)

                    data = {k: v.to(self.device) for k, v in data.items()}

                    output = self.model(data)
                    output.update(data)
                    label = output["label"]
                    prompt = output["prompt_res"]
                    with torch.no_grad():
                        pred = torch.argmax(prompt, dim=1)
                        assert pred.shape[0] == len(label)
                        for i in range(len(label)):
                            task_idx = label[i] // 10
                            taskwise_cnt[task_idx] += 1
                            taskwise_acc[task_idx] += (pred[i] == label[i])
            for i in range(task_id + 1):
                if self.local_rank == 0:
                    self.logger.info(f"In task {task_id}, the acc of task {i} is {taskwise_acc[i] / taskwise_cnt[i] * 100}")

    def train(self):
        """
        Full training logic
        """
        zero_shot_test = False
        if zero_shot_test:
            for task_id in range(self.start_task_id, self.task_num):
                updates, train_dataloader = self.train_dataset.next_task()
                updates, val_dataloader = self.val_dataset.next_task()
                self.current_task_id += 1
                self.data_loader = train_dataloader
                self.valid_data_loader = val_dataloader
            if self.local_rank == 0:
                val_log = self._valid_epoch(self.epochs, self.valid_data_loader)
                for k, v in val_log.items():
                    self.logger.info(f"Validation zero shot. {k} : {v}")
            return


        self.incremental_metrics.reset()
        for task_id in range(self.start_task_id, self.task_num):
            self.next_task()
            not_improved_count = 0
            for epoch in range(self.start_epoch, self.epochs + 1):
                result = self._train_epoch(epoch, self.data_loader, self.valid_data_loader, self.len_epoch)

                # save logged informations into log dict
                log = {'task': self.current_task_id, 'epoch': epoch}
                log.update(result)

                # print logged informations to the screen
                for key, value in log.items():
                    if self.local_rank == 0:
                        self.logger.info('    {:15s}: {}'.format(str(key), value))

            if self.local_rank == 0:
                self._save_checkpoint(task_id, self.epochs, save_best=False)

            # Decouple stage, the build_decouple() interface will change the dataset, so we need to make a copy first
            self.decouple_epochs = 0
            if self.decouple and task_id > 0 and self.memory_size > 0:
                decouple_train_loader, len_epoch = self.build_decouple()
                for epoch in range(self.epochs+1, self.epochs + self.decouple_epochs + 1):
                    result = self._train_epoch(epoch, decouple_train_loader, self.valid_data_loader, len_epoch, is_decouple=True)

                    # save logged informations into log dict
                    log = {'task': self.current_task_id, 'epoch': epoch}
                    log.update(result)

                    # print logged informations to the screen
                    for key, value in log.items():
                        if self.local_rank == 0:
                            self.logger.info('    {:15s}: {}'.format(str(key), value))
                if hasattr(self.model, "module"):
                    self.model.module.after_decouple()
                else:
                    self.model.after_decouple()

            if self.local_rank == 0:
                val_log = self._valid_epoch(self.epochs, self.valid_data_loader)
                for k, v in val_log.items():
                    if k != "loss":
                        self.incremental_metrics.update({k: v})
                for k, v in val_log.items():
                    self.logger.info(f"Validation after task {task_id}. {k} : {v}")
                self._save_checkpoint(task_id, self.epochs + self.decouple_epochs, save_best=False)

            # Build memory, build_memory() interfcce won't change the dataset itself
            if self.memory_size > 0:
                dataloader_list, bound_list = self.train_dataset.build_memory()
                memory_builder.build_memory(self.config, dataloader_list, bound_list, self.train_dataset,
                                            self.model, self.device, self.local_rank)

        final_incremental_metrics = self.incremental_metrics.result()
        if self.local_rank == 0:
            for k, v in final_incremental_metrics.items():
                self.logger.info(f"Final Incremental Metrics: {k} : {v}")

    def _train_epoch(self, epoch, train_loader, val_loader, len_epoch, is_decouple=False):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(train_loader):
            if self.global_debug_flag and batch_idx == 1:
               print(f"break for debug")
               break
            if hasattr(self.model, "module"):
                data = self.model.module.input_preprocess(data)
            else:
                data = self.model.input_preprocess(data)


            visual2text_idx_record = []
            id2pos = {}
            text2visual_idx_record = []

            for i in range(len(data["text_idx"])):
                text2visual_idx_record.append([data["text_idx"][i], data["visual_idx"][i]])
                if data["visual_idx"][i] not in id2pos:
                    id2pos[data["visual_idx"][i]] = len(visual2text_idx_record)
                    visual2text_idx_record.append([data["visual_idx"][i], [data["text_idx"][i]]])
                else:
                    visual2text_idx_record[id2pos[data["visual_idx"][i]]][1].append(data["text_idx"][i])

            if isinstance(data["text"], dict):
                text_data = dict((k, data["text"][k].to(self.device)) for k in data["text"])
                data = {"visual": data["visual"].to(self.device), "text": text_data}
            else:
                data = {"visual": data["visual"].to(self.device), "text": data["text"].to(self.device)}

            self.optimizer.zero_grad()
            output = self.model(data)

            metric_input = {"visual_feats": output["visual_embeddings"], "text_feats": output["text_embeddings"],
                            "old_visual_feats": output["old_visual_embeddings"], "old_text_feats": output["old_text_embeddings"],
                            "visual2text_record": visual2text_idx_record, "text2visual_record": text2visual_idx_record,
                            "is_eval": False, "is_decouple": is_decouple}

            loss = self.criterion(metric_input)
            if hasattr(self.model, "zero_shot"):
               if not self.model.zero_shot:
                   loss["loss_backward"].backward()
            else:
                loss["loss_backward"].backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len_epoch + batch_idx)
            for k, v in loss.items():
                if k != "loss_backward":
                    self.train_metrics.update({k: v})
            for met in self.metric_ftns:
                self.train_metrics.update(met(metric_input))

            if batch_idx % self.log_step == 0 and self.local_rank == 0:
                log_str = 'Train Task {} Epoch: {} {}: '.format(self.current_task_id, epoch, self._progress(batch_idx, train_loader, len_epoch))
                for k, v in loss.items():
                    if k != "loss_backward":
                        log_str += "{}: {:.6f}; ".format(k, v)

                self.logger.debug(log_str)
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation and self.val_step != -1 and (epoch % self.val_step == 0) and self.local_rank == 0:
            val_log = self._valid_epoch(epoch, val_loader)
            for k, v in val_log.items():
                self.logger.info(f"Validation at epoch {epoch}. {k} : {v}")
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch, val_loader, write_res=True):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        self.valid_metrics.reset()

        # Do validation only for GPU0 as the validation dataset is not distributed
        if self.local_rank != 0:
            return

        self.model.eval()
        visual_feats = []
        text_feats = []
        visual2text_idx_record = []
        text2visual_idx_record = []
        with torch.no_grad():
            val_loader.dataset.switch_flag("visual")
            for batch_idx, data in enumerate(val_loader):
                if self.global_debug_flag and batch_idx == 1:
                   print(f"break for debug in val")
                   break

                for i in range(len(data["visual_idx"])):
                    l = data["text_idxs_length"][i]
                    text_idxs = data["text_idxs"][i, :l].tolist()
                    visual2text_idx_record.append([data["visual_idx"][i], text_idxs])

                data = {"visual": data["visual"].to(self.device)}

                output = self.model(data, visual_only=True)
                visual_feats.append(output["visual_embeddings"])

            val_loader.dataset.switch_flag("text")
            for batch_idx, data in enumerate(val_loader):
                if self.global_debug_flag and batch_idx == 1:
                   print(f"break for debug in val")
                   break

                for i in range(len(data["text_idx"])):
                    text2visual_idx_record.append([data["text_idx"][i], data["visual_idx"][i]])

                if hasattr(self.model, "module"):
                    data = self.model.module.input_preprocess(data)
                else:
                    data = self.model.input_preprocess(data)

                if isinstance(data["text"], dict):
                    text_data = dict((k, data["text"][k].to(self.device)) for k in data["text"])
                    data = {"text": text_data}
                else:
                    data = {"text": data["text"].to(self.device)}

                output = self.model(data, text_only=True)
                text_feats.append(output["text_embeddings"])

            visual_feats = torch.cat(visual_feats, dim=0)
            text_feats = torch.cat(text_feats, dim=0)

            metric_input = {"visual_feats": visual_feats, "text_feats": text_feats,
                            "visual2text_record": visual2text_idx_record, "text2visual_record": text2visual_idx_record,
                            "is_eval": True}
            if self.global_debug_flag and batch_idx == 1:
                print(f"bypass evaluation")
            else:
                for met in self.metric_ftns:
                    self.valid_metrics.update(met(metric_input))

        return self.valid_metrics.result()

    def _progress(self, batch_idx, data_loader, len_epoch):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(data_loader, 'n_samples'):
            current = batch_idx * data_loader.batch_size
            total = data_loader.n_samples
        else:
            current = batch_idx
            total = len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _save_checkpoint(self, task_id, epoch, save_best=False):
        """
        Saving checkpoints

        :param task_id: current task number
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'task_id': task_id,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        exper_name = self.config['name']
        model_checkpoint_dir = os.path.join(self.checkpoint_dir, f"{exper_name}")
        if not os.path.exists(model_checkpoint_dir):
            os.mkdir(model_checkpoint_dir)
        task_checkpoint_dir = os.path.join(model_checkpoint_dir, f"task{task_id}")
        if not os.path.exists(task_checkpoint_dir):
            os.mkdir(task_checkpoint_dir)
        filename = f"{task_checkpoint_dir}/checkpoint-epoch{epoch}.pth"
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path, bypass_optimizer=False):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        prev_task_id = self.start_task_id
        self.start_task_id = checkpoint['task_id'] + 1
        self.start_epoch = checkpoint['epoch'] + 1

        #TODO: In continual learning model arch will change in different tasks, should first call next_task()
        # for several times and then load model and optimizer

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])
        if bypass_optimizer:
            self.logger.info("Bypass optimizer")
            return

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
