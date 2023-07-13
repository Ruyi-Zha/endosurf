import os
import os.path as osp
import torch
from tqdm import tqdm, trange
from shutil import copyfile
import numpy as np
import imageio.v2 as iio

import sys
sys.path.append("./")

from src.trainer.utils import load_config, CustomSummaryWritter
from src.dataset import Dataset


class Trainer(object):
    """Trainer object for endoscopy reconstruction.
    """
    def __init__(self, cfg_dir, mode="train"):
        cfg = load_config(cfg_dir)
        self.cfg_dir = cfg_dir
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tqdm.write(f"[Mode] {mode}")
        
        # Dataset
        data_cfg = cfg["data"].copy()
        data_cfg["device"] = self.device
        self.dset = Dataset(data_cfg)
        
        # Experiment
        self.proj_name = None
        self.exp_name = None
        self.exp_dir = None
        self.ckpt_dir = None
        self.ckpt_dir_backup = None
        self.init_exp()
        tqdm.write(f"[Experiment] exp_dir: {self.exp_dir}")
        
        # Renderer and network
        self.render_cfg = None
        self.renderer = None
        self.init_renderer()
        
        # Train
        self.n_iter = 0
        self.train_cfg = None
        self.resume = False
        self.init_train()
        
        # Optimizer
        self.optimizer = None
        self.optim_cfg = None
        self.lr_init = None
        self.init_optimizer()
        
        # Checkpoints
        self.step_start = 1
        if mode != "train":
            assert osp.exists(self.ckpt_dir), f"[Load checkpoints failed] {self.ckpt_dir}."
            self.load_checkpoint()
            tqdm.write(f"[Load checkpoints] {self.ckpt_dir}.")
        else:
            copyfile(self.cfg_dir, osp.join(self.exp_dir, "cfg.yml"))
            if self.resume and osp.exists(self.ckpt_dir):
                tqdm.write(f"[Load checkpoints] {self.ckpt_dir}.")
                self.load_checkpoint()
            else:
                tqdm.write(f"[No checkpoint loaded] New training!")
        
        # Logging
        log_cfg = cfg["log"].copy()
        self.i_eval = log_cfg["i_eval"]
        self.i_save = log_cfg["i_save"]
        writer_cfg = log_cfg["summary_writer"].copy()
        writer_cfg.update({
            "exp_dir": self.exp_dir,
            "proj_name": self.proj_name,
            "exp_name": self.exp_name,
        })
        self.writer = None
        if mode == "train":
            self.writer = CustomSummaryWritter(writer_cfg, cfg)
    
    def start(self):
        """
        Main loop.
        """
        tqdm.write(f"[Start training] Iterations: {self.n_iter}")
        t = trange(self.step_start, self.n_iter+1, leave=True)
        for i_iter in t:
            
            # Evaluate
            if self.i_eval > 0 and (i_iter == 1 or i_iter % self.i_eval == 0 or i_iter == self.n_iter):
                self.renderer.eval()
                with torch.no_grad():
                    self.eval(global_step=i_iter)
                    
            # Train
            self.renderer.train()
            loss_train = self.train_step(global_step=i_iter)
            desc = f"TRAIN|loss:{loss_train:.5g}|" 
            t.set_description(desc)
                    
            # Update lrate
            self.update_learning_rate(i_iter)
                
            # Save
            if self.i_save > 0 and (i_iter % self.i_save == 0 or i_iter == self.n_iter):
                if os.path.exists(self.ckpt_dir):
                    copyfile(self.ckpt_dir, self.ckpt_dir_backup)
                tqdm.write(f'SAVE|iter:{i_iter}/{self.n_iter}|path:{self.ckpt_dir}')
                self.save_checkpoint(i_iter)
                
        print(f'Training complete!')
        
    def init_exp(self):
        """Initialize experiment setting.
        """
        pass
        
    def load_checkpoint(self):
        """Load checkpoint.
        """
        pass
    
    def save_checkpoint(self, global_step):
        """Save checkpoint.
        """
        pass
    
    def train_step(self, global_step):
        """
        Training step
        """
        pass
    
    def eval(self, global_step):
        """
        Evaluation step
        """
        pass
    
    def update_learning_rate(self, global_step):
        """
        Update learning rate.
        """
        pass
    
    def init_renderer(self):
        """Initialize renderer.
        """
        pass
    
    def init_optimizer(self):
        """Initialize optimizer.
        """
        pass
    
    def init_train(self):
        """Initialize training.
        """
        pass