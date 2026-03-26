from math import sqrt
from random import choice
from pathlib import Path
from shutil import rmtree
import wandb

from beartype import beartype

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import make_grid, save_image

import torchvision.transforms as T

from laq_model.optimizer import get_optimizer

from ema_pytorch import EMA

import json
import glob
import os
import numpy as np

from laq_model.data import ImageVideoDataset, BridgeVideoDataset

from accelerate import Accelerator, DistributedDataParallelKwargs

from einops import rearrange


def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data



def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def build_prefix(entry): 
    p = Path(entry["traj_path"]) 
    traj_name = p.name 
    traj_group = p.parent.name 
    raw_dir = p.parent.parent 
    date_dir = raw_dir.parent.name 
    collection_id = raw_dir.parent.parent.name 
    task_name = raw_dir.parent.parent.parent.name 
    environment = entry.get("environment", "unknown_env") 
    
    return f"{environment}/{task_name}/{collection_id}/{traj_group}/{date_dir}/{traj_name}"


# main trainer class

@beartype
class LAQTrainerWithDepth(nn.Module):
    def __init__(
        self,
        vae,
        *,
        num_train_steps,
        batch_size,
        folder,
        traj_info=None,
        train_on_images = False,
        lr = 3e-4,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        discr_max_grad_norm = None,
        save_results_every = 50,
        save_model_every = 9998,
        results_folder = './results',
        use_ema = True,
        ema_update_after_step = 0,
        ema_update_every = 1,
        accelerate_kwargs: dict = dict(),
        weights = None,
        offsets = None,
    ):
        super().__init__()

        image_size = vae.image_size

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)
        self.accelerator = Accelerator(**accelerate_kwargs, kwargs_handlers=[ddp_kwargs])

        self.vae = vae
        self.results_folder_str = results_folder
        self.lr = lr

        self.use_ema = use_ema
        if self.is_main and use_ema:
            self.ema_vae = EMA(vae, update_after_step = ema_update_after_step, update_every = ema_update_every)

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        self.vae.discr = None # this seems to be missing

        if exists(self.vae.discr):
            all_parameters = set(vae.parameters())
            discr_parameters = set(vae.discr.parameters())
            vae_parameters = all_parameters - discr_parameters

            self.vae_parameters = vae_parameters

            self.optim = get_optimizer(vae_parameters, lr = lr, wd = wd)
            self.discr_optim = get_optimizer(discr_parameters, lr = lr, wd = wd)
        else:
            self.vae_parameters  = set(vae.parameters())
            self.optim = get_optimizer(self.vae_parameters, lr = lr, wd = wd)

        self.max_grad_norm = max_grad_norm
        self.discr_max_grad_norm = discr_max_grad_norm

        # create dataset
        self.train_on_images = train_on_images
        
        
        # sthv2 training
        # self.ds = ImageVideoDataset(folder, image_size, offset=offsets)

        # bridge data v2 training

        with open("/rds/general/user/mf822/home/FinalYearProject/LAPA-Testing/train_meta.json", "r") as f:
            train_entries = json.load(f)

        with open("/rds/general/user/mf822/home/FinalYearProject/LAPA-Testing/val_meta.json", "r") as f:
            val_entries = json.load(f)
     
        envs = ["toykitchen1", "toykitchen2", "toykitchen5", "toykitchen7"]
        train_entries = [t for t in train_entries if t["environment"] in envs]
        val_entries = [t for t in val_entries if t["environment"] in envs]

        stats = np.load("/rds/general/user/mf822/home/FinalYearProject/LAPA-Testing/depth_stats.npy", allow_pickle=True).item()

        global_depth_stats = (stats["mean_single"], stats["std_single"])

        print(f"Train entries: {len(train_entries)}")
        print(f"Val entries: {len(val_entries)}")

        self.ds = BridgeVideoDataset(
            image_size,
            offset=offsets,
            traj_entries=train_entries,
            use_multiview=True,
            altview=True,
            use_depth=True,
            depth_norm="global",
            global_depth_stats=global_depth_stats
        )

        self.valid_ds = BridgeVideoDataset(
            image_size,
            offset=offsets,
            traj_entries=val_entries,
            use_multiview=True,
            use_depth=True,
            depth_norm="global",
            global_depth_stats=global_depth_stats
        )

        self.dl = DataLoader(
            self.ds,
            batch_size = batch_size,
            shuffle=True,
            num_workers=6,  # or more depending on your CPU cores
            pin_memory=True,  # Helps with faster data transfer to GPU
            prefetch_factor=4,
            persistent_workers=True
            )

        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size = batch_size,
            num_workers = 4)

        if exists(self.vae.discr):
            (
                self.vae,
                self.optim,
                self.discr_optim,
                self.dl
            ) = self.accelerator.prepare(
                self.vae,
                self.optim,
                self.discr_optim,
                self.dl
            )
        else:
            (
                self.vae,
                self.optim,
                self.dl
            ) = self.accelerator.prepare(
                self.vae,
                self.optim,
                self.dl
            )

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)


        self.results_folder.mkdir(parents = True, exist_ok = True)

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        if exists(self.vae.discr):
            pkg = dict(
                model = self.accelerator.get_state_dict(self.vae),
                optim = self.optim.state_dict(),
                discr_optim = self.discr_optim.state_dict(),
                steps = self.steps.item()
            )
        else:
            pkg = dict(
                model=self.accelerator.get_state_dict(self.vae),
                optim=self.optim.state_dict(),
                steps=self.steps.item()
            )

        # Save DataLoader state
        pkg['dl_iter_state'] = self.get_dl_state(self.dl_iter)

        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        vae = self.accelerator.unwrap_model(self.vae)
        vae.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])
        if exists(self.vae.discr):
            self.discr_optim.load_state_dict(pkg['discr_optim'])


    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.vae.train()

        # logs

        logs = {}

        # update vae (generator)

        for _ in range(self.grad_accum_every):
            img = next(self.dl_iter)
            img = img.to(device)

            # with self.accelerator.autocast():
            loss, num_unique_indices = self.vae(
                img,
                step=steps,
            )

            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})
            accum_log(logs, {'num_unique_indices': num_unique_indices})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.vae.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        if self.is_main:  # Ensure only the main process logs in a distributed setting
            wandb.log(logs)

        if self.is_main and self.use_ema:
            self.ema_vae.update()

        if self.is_main and not (steps % self.save_results_every):
            self.print(f'{steps}: saving to {str(self.results_folder)}')

            unwrapped_vae = self.accelerator.unwrap_model(self.vae)
            vaes_to_evaluate = ((unwrapped_vae, str(steps)),)

            if self.use_ema:
                vaes_to_evaluate = ((self.ema_vae.ema_model, f'{steps}.ema'),) + vaes_to_evaluate

            for model, filename in vaes_to_evaluate:
                model.eval()

                valid_data = next(self.valid_dl_iter)


                valid_data = valid_data.to(device)

                recons = model(valid_data, return_recons_only = True)


                if self.train_on_images:
                    imgs_and_recons = torch.stack((valid_data, recons), dim = 0)
                    # imgs_and_recons = torch.stack((valid_data, recons), dim = 0)
                    imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

                    imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
                    grid = make_grid(imgs_and_recons, nrow = 2, normalize = True, value_range = (0, 1))

                    logs['reconstructions'] = grid
                    save_image(grid, str(self.results_folder / f'{filename}.png'))
                else:
                    # valid_data shape: (B, C, 2, H, W)
                    # recons shape:     (B, C, H, W) or similar

                    rgb = valid_data[:, :3]      # (B, 3, 2, H, W)
                    depth = valid_data[:, 3:4]   # (B, 1, 2, H, W)

                    rgb_t0 = rgb[:, :, 0]
                    rgb_t1 = rgb[:, :, 1]

                    depth_t0 = depth[:, :, 0]
                    depth_t1 = depth[:, :, 1]

                    # Expand depth → fake RGB for visualization
                    depth_t0_vis = depth_t0.repeat(1, 3, 1, 1)
                    depth_t1_vis = depth_t1.repeat(1, 3, 1, 1)

                    # If recon has depth channel
                    if recons.shape[1] == 4:
                        recon_rgb = recons[:, :3]
                        recon_depth = recons[:, 3:4]
                        recon_depth_vis = recon_depth.repeat(1, 3, 1, 1)
                    else:
                        recon_rgb = recons
                        recon_depth_vis = None

                    # --- Build rows ---

                    row_rgb = torch.stack([rgb_t0, rgb_t1, recon_rgb], dim=1)  
                    # (B, 3_images, 3, H, W)

                    row_rgb = rearrange(row_rgb, 'b n c h w -> (b n) c h w')

                    if recon_depth_vis is not None:
                        row_depth = torch.stack([depth_t0_vis, depth_t1_vis, recon_depth_vis], dim=1)
                    else:
                        row_depth = torch.stack([depth_t0_vis, depth_t1_vis], dim=1)

                    row_depth = rearrange(row_depth, 'b n c h w -> (b n) c h w')

                    # Combine RGB + Depth rows
                    imgs_and_recons = torch.cat([row_rgb, row_depth], dim=0)

                    imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)

                    grid = make_grid(
                        imgs_and_recons,
                        nrow=3,              # 3 columns: t0, t1, recon
                        normalize=True,
                        value_range=(0, 1)
                    )

                    logs['reconstructions'] = grid
                    save_image(grid, str(self.results_folder / f'{filename}.png'))
                    
        # save model every so often

        # self.accelerator.wait_for_everyone()

        if self.is_main and not (steps % self.save_model_every):
            # self.save(str(self.results_folder / f'vae.pt'))
            state_dict = self.vae.state_dict()
            model_path = str(self.results_folder / f'vae.{steps}.pt')
            torch.save(state_dict, model_path)

            if self.use_ema:
                ema_state_dict = self.ema_vae.state_dict()
                model_path = str(self.results_folder / f'vae.{steps}.ema.pt')
                torch.save(ema_state_dict, model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def train(self, log_fn = noop):
        device = next(self.vae.parameters()).device
        if self.accelerator.is_main_process:
            wandb.init(project='phenaki_cnn',name=self.results_folder_str.split('/')[-1], config={
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "num_train_steps": self.num_train_steps,
            })

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')
        if self.accelerator.is_main_process:
            wandb.finish()  
