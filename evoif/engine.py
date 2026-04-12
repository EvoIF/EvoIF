import torch
from torch.utils import data as torch_data
# from torch.utils.tensorboard import SummaryWriter
from torchdrug import core, data, utils
from torchdrug.utils import comm
from itertools import islice
import os
from datetime import datetime


class CustomEngine(core.Engine):
    """
    Custom Engine class that supports custom collate_fn and TensorBoard logging
    
    Inherits from torchdrug.core.Engine and overrides train and evaluate methods 
    to support custom collate_fn and TensorBoard integration
    """
    
    def __init__(self, task, train_set, valid_set, test_set, optimizer, scheduler=None, 
                 gpus=None, batch_size=1, gradient_interval=1, num_workers=64, 
                 logger="logging", log_interval=100, collate_fn=None, prefetch_factor=16,
                 log_dir=None):
        """
        Initialize CustomEngine with TensorBoard support
        
        Parameters:
            collate_fn (callable, optional): custom collate function
            prefetch_factor (int, optional): number of batches loaded in memory by each worker
            log_dir (str, optional): directory for TensorBoard logs. If None, will use runs/current_time
            other parameters are the same as torchdrug.core.Engine
        """
        self.collate_fn = collate_fn
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.iteration = 0
        self.eval_step = 0
        
        # Initialize TensorBoard writer
        if log_dir is None:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join('runs', current_time)
        # self.writer = SummaryWriter(log_dir=log_dir)
        
        super().__init__(task, train_set, valid_set, test_set, optimizer, scheduler,
                        gpus, batch_size, gradient_interval, num_workers, logger, log_interval)
    
    def train(self, num_epoch=1, batch_per_epoch=None, log=True):
        """
        Train the model using custom collate_fn with TensorBoard logging
        
        Parameters:
            num_epoch (int, optional): number of training epochs
            batch_per_epoch (int, optional): number of batches per epoch
            log (bool, optional): whether to log to TensorBoard
        """
        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        
        # Create DataLoader with custom collate_fn
        dataloader = data.DataLoader(
            self.train_set, 
            self.batch_size, 
            sampler=sampler, 
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn
        )
        
        batch_per_epoch = batch_per_epoch or len(dataloader)
        model = self.model
        model.split = "train"
        
        if self.world_size > 1:
            if self.device.type == "cuda":
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[self.device], find_unused_parameters=True
                )
            else:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, find_unused_parameters=True
                )
        model.train()

        for epoch in self.meter(num_epoch):
            sampler.set_epoch(epoch)

            metrics = []
            start_id = 0
            # The last gradient update may contain fewer batches than gradient_interval
            gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)

            for batch_id, batch in enumerate(islice(dataloader, batch_per_epoch)):
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)

                loss, metric = model(batch)
                if not loss.requires_grad:
                    raise RuntimeError("Loss doesn't require grad. Did you define any loss in the task?")
                loss = loss / gradient_interval
                loss.backward()
                metrics.append(metric)

                if batch_id - start_id + 1 == gradient_interval:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    metric = utils.stack(metrics, dim=0)
                    metric = utils.mean(metric, dim=0)
                    if self.world_size > 1:
                        metric = comm.reduce(metric, op="mean")
                    
                    # # Log to TensorBoard
                    # if log and comm.get_rank() == 0:
                    #     self._log_to_tensorboard(metric, epoch, batch_id, prefix="train")
                    
                    self.meter.update(metric)
                    metrics = []
                    start_id = batch_id + 1
                    gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)

            if self.scheduler:
                self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, split, log=True):
        """
        Evaluate the model using custom collate_fn with TensorBoard logging
        
        Parameters:
            split (str): dataset split to evaluate, can be 'train', 'valid' or 'test'
            log (bool, optional): whether to log metrics to TensorBoard
            
        Returns:
            dict: evaluation metrics
        """
        # if self.world_size > 1 and comm.get_rank() != 0:
        #     return {}
        if comm.get_rank() == 0:
            import logging
            logger = logging.getLogger(__name__)
            from torchdrug.utils import pretty
            logger.warning(pretty.separator)
            logger.warning("Evaluate on %s" % split)
            
        test_set = getattr(self, "%s_set" % split)
        sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)
        
        # Create DataLoader with custom collate_fn
        dataloader = data.DataLoader(
            test_set, 
            self.batch_size, 
            sampler=sampler, 
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn
        )
        
        model = self.model
        model.split = split

        model.eval()
        preds = []
        targets = []
        for batch in dataloader:
            if self.device.type == "cuda":
                batch = utils.cuda(batch, device=self.device)

            pred,target= model.predict_and_target(batch)
            preds.append(pred)
            targets.append(target)
        
        pred = utils.cat(preds)
        target = utils.cat(targets)

        metric = model.evaluate(pred, target)
        
        # Log to TensorBoard
        if log and comm.get_rank() == 0:
            # self._log_to_tensorboard(metric, self.eval_step, 0, prefix=split)
            self.eval_step += 1
            
        if log:
            self.meter.log(metric, category="%s/epoch" % split)
        
        return metric
    
    