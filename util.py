import os
import time
import math  
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch.utils import data as torch_data
from torch import distributed as dist

from torchdrug import core, utils
from torchdrug.utils import comm
from evoif.engine import CustomEngine
from evoif.dataset import custom_collate_fn

logger = logging.getLogger(__file__)

def zeropower_via_newtonschulz5(G, steps):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.float32)
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        wd=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            muon_params = []
            adamw_params = []
            for p in group['params']:
                name = next((name for name, param in group['named_params'] if param is p), None)
                
                if name is not None and p.ndim >= 2 and "embed" not in name and "lm_head" not in name:
                    muon_params.append(p)
                    self.state[p]["use_muon"] = True
                else:
                    adamw_params.append(p)
                    self.state[p]["use_muon"] = False
                    
            group['muon_params'] = muon_params
            group['adamw_params'] = adamw_params

    def adjust_lr_for_muon(self, lr, param_shape):
        if len(param_shape) < 2:
            return lr
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        return lr * adjusted_ratio

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Muon parameters
            params = group.get('muon_params', [])
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            
            for p in params:
                if not self.state[p].get("use_muon", False):
                    continue
                    
                g = p.grad
                if g is None:
                    continue
                    
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                elif g.ndim < 2:
                    g = g.view(1, -1)
                    
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                    
                if g.ndim != 2:
                    g = g.view(1, -1)
                    
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adjusted_lr)

            # AdamW parameters
            params = group.get('adamw_params', [])
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]
            
            for p in params:
                if self.state[p].get("use_muon", False):
                    continue
                    
                g = p.grad
                if g is None:
                    continue
                    
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                    
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss

def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def create_working_directory(cfg):
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "null")
    file_name = "%s_working_dir.tmp" % slurm_job_id
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    if isinstance(cfg.task.get("model"), dict):
        model_class = cfg.task.model["class"]
    else:
        model_class = "Placeholder"
    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.task["class"], cfg.dataset["class"], model_class, slurm_job_id,
                               time.strftime("%Y-%m-%d-%H-%M-%S"))

    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    ast = env.parse(raw)
    vars = meta.find_undeclared_variables(ast)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def build_solver(cfg, dataset):
    generator = torch.Generator().manual_seed(0)
    lengths = [int(len(dataset) * cfg.split[0]), int(len(dataset) * cfg.split[1])]
    lengths.append(len(dataset) - sum(lengths))
    train_set, valid_set, test_set = torch_data.random_split(dataset, lengths, generator=generator)
    if comm.get_rank() == 0:
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    task = core.Configurable.load_config_dict(cfg.task)

    if "fix_sequence_model" in cfg:
        model = task.model
        assert cfg.task.model["class"] == "FusionNetwork"
        for p in model.sequence_model.parameters():
            p.requires_grad = False
  
    named_params = list(task.named_parameters())
    all_params = [p for _, p in named_params if p.requires_grad]
    param_groups = [{
        'params': all_params,
        'named_params': named_params  
    }]
    
    # use muon+adaw optimizer
    optimizer = Muon(
        param_groups,  
        lr=1e-3,       
        wd=0.1,        
        momentum=0.95,  
        nesterov=True,  
        ns_steps=5,     
        adamw_betas=(0.9, 0.95),  
        adamw_eps=1e-8   
    )
    
  

    cfg.engine['collate_fn'] = custom_collate_fn
    if 'prefetch_factor' not in cfg.engine:
        cfg.engine['prefetch_factor'] = 2
    solver = CustomEngine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)

    if cfg.get("checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.checkpoint)
        solver.load(cfg.checkpoint,strict = False,load_optimizer=False)
        # solver.load(cfg.checkpoint)
    
        
    
    return solver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, default="null")
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars

