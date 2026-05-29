import torch
import json
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision.utils as vutils
from torchvision.transforms.functional import normalize
from PIL import Image
import glob
import os
import torch.nn.functional as F
from pytorch_msssim import ssim as compute_ssim
import lpips

from laq_model import LatentActionQuantization

DEPTH_MEAN = 0.882886777966528
DEPTH_STD = 0.38224160713129973

# -------------------------
# Utilities
# -------------------------

def frame_index(path):
    name = os.path.basename(path)
    digits = ''.join(filter(str.isdigit, name))
    return int(digits) if digits else 0


def hamming_distance(a, b):
    return np.mean(a != b)


# -------------------------
# Dataset
# -------------------------

class BridgeComparisonDatasetSingle(Dataset):
    def __init__(self, traj_entries, image_size=256, offset=5):
        self.offset = offset

        self.rgb_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

        self.depth_transform = T.Compose([
            T.Resize((image_size, image_size),
                     interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        self.samples = []

        for entry in traj_entries:
            traj_path = entry["traj_path"]
            rgb_views = entry.get("rgb_views", {})

            if "images0" not in rgb_views:
                continue 

            view_names = sorted([v for v in rgb_views.keys()])

            frames_per_view = {}
            depth_per_view = {}

            valid = True

            for view in view_names:
                view_dir = os.path.join(traj_path, view)

                rgb_imgs = sorted(
                    glob.glob(os.path.join(view_dir, "im_*.jpg")) +
                    glob.glob(os.path.join(view_dir, "im_*.png")),
                    key=frame_index
                )

                depth_imgs = sorted(
                    glob.glob(os.path.join(view_dir, "im_*_depth.npy")),
                    key=frame_index
                )

                if len(rgb_imgs) <= offset:
                    valid = False
                    break

                if len(depth_imgs) != len(rgb_imgs):
                    valid = False
                    break

                frames_per_view[view] = rgb_imgs
                depth_per_view[view] = depth_imgs

            if valid:
                self.samples.append({
                    "traj_path": traj_path,
                    "environment": entry.get("environment", "unknown"),
                    "task": entry.get("task", "unknown"),
                    "datetime": entry.get("datetime", "unknown"),
                    "views": view_names,
                    "rgb": frames_per_view,
                    "depth": depth_per_view
                })
            else:
                print(f"Invalid sample {traj_path}")

        print(f"Loaded {len(self.samples)} trajectories")

    def __len__(self):
        return len(self.samples) * 5

    def _normalize_depth(self, depth):
        depth = depth.astype(np.float32)

        mean = DEPTH_MEAN
        std = DEPTH_STD

        return (depth - mean) / std

    def __getitem__(self, index):
        traj_idx = index % len(self.samples)
        sample = self.samples[traj_idx]

        v = sample["views"][0]

        frames = sample["rgb"][v]

        i = np.random.randint(0, len(frames) - self.offset)

        def load_pair(frames, depth_frames):
            frame_t = frames[i]
            frame_t_plus = frames[i + self.offset]

            img1 = Image.open(frame_t)
            img2 = Image.open(frame_t_plus)

            x1_rgb = self.rgb_transform(img1)
            x2_rgb = self.rgb_transform(img2)

            d1 = np.load(depth_frames[i]).astype(np.float32)
            d2 = np.load(depth_frames[i + self.offset]).astype(np.float32)

            d1 = self._normalize_depth(d1)
            d2 = self._normalize_depth(d2)

            d1 = Image.fromarray(d1, mode="F")
            d2 = Image.fromarray(d2, mode="F")

            x1_d = self.depth_transform(d1)
            x2_d = self.depth_transform(d2)

            rgb_pair = torch.stack([x1_rgb, x2_rgb], dim=1)
            rgbd_pair = torch.stack(
                [torch.cat([x1_rgb, x1_d], dim=0),
                 torch.cat([x2_rgb, x2_d], dim=0)],
                dim=1
            )

            return rgb_pair, rgbd_pair, frame_t, frame_t_plus

        rgb, rgbd, f_t, f_tp = load_pair(frames, sample["depth"][v])

        metadata = {
            "traj_path": sample["traj_path"],
            "environment": sample["environment"],
            "task": sample["task"],
            "datetime": sample["datetime"],

            "view": v,
            "frame_t": f_t,
            "frame_t_plus": f_tp,

            "view2": "none",
            "frame_t_v2": 0,
            "frame_t_plus_v2": 0,

            "view3": "none",
            "frame_t_v3": 0,
            "frame_t_plus_v3": 0,

            "frame_idx": i
        }

        return rgb, rgbd, 0, 0, 0, 0, metadata

class BridgeComparisonDataset(Dataset):
    def __init__(self, traj_entries, image_size=256, offset=5):
        self.offset = offset

        self.rgb_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

        self.depth_transform = T.Compose([
            T.Resize((image_size, image_size),
                     interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        self.samples = []

        for entry in traj_entries:
            traj_path = entry["traj_path"]
            rgb_views = entry.get("rgb_views", {})

            if len(rgb_views) < 3:
                continue 

            view_names = sorted([v for v in rgb_views.keys()])

            frames_per_view = {}
            depth_per_view = {}

            valid = True

            for view in view_names:
                view_dir = os.path.join(traj_path, view)

                rgb_imgs = sorted(
                    glob.glob(os.path.join(view_dir, "im_*.jpg")) +
                    glob.glob(os.path.join(view_dir, "im_*.png")),
                    key=frame_index
                )

                depth_imgs = sorted(
                    glob.glob(os.path.join(view_dir, "im_*_depth.npy")),
                    key=frame_index
                )

                if len(rgb_imgs) <= offset:
                    valid = False
                    break

                if len(depth_imgs) != len(rgb_imgs):
                    valid = False
                    break

                frames_per_view[view] = rgb_imgs
                depth_per_view[view] = depth_imgs

            if valid:
                self.samples.append({
                    "traj_path": traj_path,
                    "environment": entry.get("environment", "unknown"),
                    "task": entry.get("task", "unknown"),
                    "datetime": entry.get("datetime", "unknown"),
                    "views": view_names,
                    "rgb": frames_per_view,
                    "depth": depth_per_view
                })
            else:
                print(f"Invalid sample {traj_path}")

        print(f"Loaded {len(self.samples)} trajectories")

    def __len__(self):
        return len(self.samples) * 5

    def _normalize_depth(self, depth):
        depth = depth.astype(np.float32)

        mean = DEPTH_MEAN
        std = DEPTH_STD

        return (depth - mean) / std

    def __getitem__(self, index):
        traj_idx = index % len(self.samples)
        sample = self.samples[traj_idx]

        v1, v2, v3 = sample["views"]

        frames1 = sample["rgb"][v1]
        frames2 = sample["rgb"][v2]
        frames3 = sample["rgb"][v3]

        i = np.random.randint(0, len(frames1) - self.offset)

        def load_pair(frames, depth_frames):
            frame_t = frames[i]
            frame_t_plus = frames[i + self.offset]

            img1 = Image.open(frame_t)
            img2 = Image.open(frame_t_plus)

            x1_rgb = self.rgb_transform(img1)
            x2_rgb = self.rgb_transform(img2)

            d1 = np.load(depth_frames[i]).astype(np.float32)
            d2 = np.load(depth_frames[i + self.offset]).astype(np.float32)

            d1 = self._normalize_depth(d1)
            d2 = self._normalize_depth(d2)

            d1 = Image.fromarray(d1, mode="F")
            d2 = Image.fromarray(d2, mode="F")

            x1_d = self.depth_transform(d1)
            x2_d = self.depth_transform(d2)

            rgb_pair = torch.stack([x1_rgb, x2_rgb], dim=1)
            rgbd_pair = torch.stack(
                [torch.cat([x1_rgb, x1_d], dim=0),
                 torch.cat([x2_rgb, x2_d], dim=0)],
                dim=1
            )

            return rgb_pair, rgbd_pair, frame_t, frame_t_plus

        rgb_v1, rgbd_v1, f1_t, f1_tp = load_pair(frames1, sample["depth"][v1])
        rgb_v2, rgbd_v2, f2_t, f2_tp = load_pair(frames2, sample["depth"][v2])
        rgb_v3, rgbd_v3, f3_t, f3_tp = load_pair(frames3, sample["depth"][v3])

        metadata = {
            "traj_path": sample["traj_path"],
            "environment": sample["environment"],
            "task": sample["task"],
            "datetime": sample["datetime"],

            "view": v1,
            "frame_t": f1_t,
            "frame_t_plus": f1_tp,

            "view2": v2,
            "frame_t_v2": f2_t,
            "frame_t_plus_v2": f2_tp,

            "view3": v3,
            "frame_t_v3": f3_t,
            "frame_t_plus_v3": f3_tp,

            "frame_idx": i
        }

        return rgb_v1, rgbd_v1, rgb_v2, rgbd_v2, rgb_v3, rgbd_v3, metadata
    

def compare_models(
    laq_rgb, 
    laq_rgbd,
    device,
    dataloader, 
    max_samples,
    num_vis_samples=0,
    vis_dir=None,
    multiview=True):

    os.makedirs(vis_dir, exist_ok=True)
    vis_saved = 0
    total = 0
    rows = []

    stats = {
        "agree": {
            "hamming_rgb": {10: [], 20: [], 12: []},
            "mse_view_rgb": {10: [], 20: [], 12: []},
            "hamming_rgbd": {10: [], 20: [], 12: []},
            "mse_view_rgbd": {10: [], 20: [], 12: []},
        },
        "recon": {
            "mse_rgb": {0: [], 1: [], 2: []},
            "mse_rgbd": {0: [], 1: [], 2: []},
            "ssim_rgb": {0: [], 1: [], 2: []},
            "ssim_rgbd": {0: [], 1: [], 2: []},
            "lpips_rgb": {0: [], 1: [], 2: []},
            "lpips_rgbd": {0: [], 1: [], 2: []},
        }
    }

    lpips_fn = lpips.LPIPS(net='alex').to(device)

    with torch.no_grad():
        for rgb_v1, rgbd_v1, rgb_v2, rgbd_v2, rgb_v3, rgbd_v3, batch_meta in dataloader:

            if total >= max_samples:
                break

            rgb_v1 = rgb_v1.to(device)
            rgbd_v1 = rgbd_v1.to(device)
            if multiview:
                rgb_v2 = rgb_v2.to(device)
                rgbd_v2 = rgbd_v2.to(device)
                rgb_v3 = rgb_v3.to(device)
                rgbd_v3 = rgbd_v3.to(device)

            # ---- Forward passes ----
            codes_rgb_v1 = laq_rgb(rgb_v1, return_only_codebook_ids=True)
            emb_rgb_v1   = laq_rgb(rgb_v1, return_embeddings=True)
            recon_rgb_v1 = laq_rgb(rgb_v1, return_recons_only=True)

            codes_rgbd_v1 = laq_rgbd(rgbd_v1, return_only_codebook_ids=True)
            emb_rgbd_v1   = laq_rgbd(rgbd_v1, return_embeddings=True)
            recon_rgbd_v1 = laq_rgbd(rgbd_v1, return_recons_only=True)

            if multiview:
                codes_rgb_v2 = laq_rgb(rgb_v2, return_only_codebook_ids=True)
                emb_rgb_v2   = laq_rgb(rgb_v2, return_embeddings=True)
                recon_rgb_v2 = laq_rgb(rgb_v2, return_recons_only=True)

                codes_rgbd_v2 = laq_rgbd(rgbd_v2, return_only_codebook_ids=True)
                emb_rgbd_v2   = laq_rgbd(rgbd_v2, return_embeddings=True)
                recon_rgbd_v2 = laq_rgbd(rgbd_v2, return_recons_only=True)

                codes_rgb_v3 = laq_rgb(rgb_v3, return_only_codebook_ids=True)
                emb_rgb_v3   = laq_rgb(rgb_v3, return_embeddings=True)
                recon_rgb_v3 = laq_rgb(rgb_v3, return_recons_only=True)

                codes_rgbd_v3 = laq_rgbd(rgbd_v3, return_only_codebook_ids=True)
                emb_rgbd_v3   = laq_rgbd(rgbd_v3, return_embeddings=True)
                recon_rgbd_v3 = laq_rgbd(rgbd_v3, return_recons_only=True)

            # ---- Move to CPU once ----
            codes_rgb_v1  = codes_rgb_v1.cpu().numpy()
            codes_rgbd_v1 = codes_rgbd_v1.cpu().numpy()
            emb_rgb_v1    = emb_rgb_v1.cpu().numpy()
            emb_rgbd_v1   = emb_rgbd_v1.cpu().numpy()

            if multiview:
                codes_rgb_v2  = codes_rgb_v2.cpu().numpy()
                codes_rgbd_v2 = codes_rgbd_v2.cpu().numpy()
                emb_rgb_v2    = emb_rgb_v2.cpu().numpy()
                emb_rgbd_v2   = emb_rgbd_v2.cpu().numpy()

                codes_rgb_v3  = codes_rgb_v3.cpu().numpy()
                codes_rgbd_v3 = codes_rgbd_v3.cpu().numpy()
                emb_rgb_v3    = emb_rgb_v3.cpu().numpy()
                emb_rgbd_v3   = emb_rgbd_v3.cpu().numpy()

            batch_size = len(codes_rgb_v1)

            for i in range(batch_size):

                if total >= max_samples:
                    break

                meta = {k: batch_meta[k][i] for k in batch_meta}
                
                if multiview:
                    # ---- View agreement (RGB v1 vs v2) ----
                    hamming_rgb_v12 = hamming_distance(codes_rgb_v1[i], codes_rgb_v2[i])
                    mse_view_rgb_v12 = np.mean((emb_rgb_v1[i] - emb_rgb_v2[i]) ** 2)

                    # ---- View agreement (RGBD v1 vs v2) ----
                    hamming_rgbd_v12 = hamming_distance(codes_rgbd_v1[i], codes_rgbd_v2[i])
                    mse_view_rgbd_v12 = np.mean((emb_rgbd_v1[i] - emb_rgbd_v2[i]) ** 2)

                     # ---- View agreement (RGB v1 vs v3) ----
                    hamming_rgb_v13 = hamming_distance(codes_rgb_v1[i], codes_rgb_v3[i])
                    mse_view_rgb_v13 = np.mean((emb_rgb_v1[i] - emb_rgb_v3[i]) ** 2)

                    # ---- View agreement (RGBD v1 vs v3) ----
                    hamming_rgbd_v13 = hamming_distance(codes_rgbd_v1[i], codes_rgbd_v3[i])
                    mse_view_rgbd_v13 = np.mean((emb_rgbd_v3[i] - emb_rgbd_v2[i]) ** 2)

                     # ---- View agreement (RGB v2 vs v3) ----
                    hamming_rgb_v23 = hamming_distance(codes_rgb_v2[i], codes_rgb_v3[i])
                    mse_view_rgb_v23 = np.mean((emb_rgb_v2[i] - emb_rgb_v3[i]) ** 2)

                    # ---- View agreement (RGBD v2 vs v3) ----
                    hamming_rgbd_v23 = hamming_distance(codes_rgbd_v2[i], codes_rgbd_v3[i])
                    mse_view_rgbd_v23 = np.mean((emb_rgbd_v2[i] - emb_rgbd_v3[i]) ** 2)
                
                def recon_losses_rgb(x, xhat):
                    recon_mse = torch.mean(
                            (xhat - x) ** 2
                    ).item()

                    # --- SSIM ---
                    ssim_loss = compute_ssim(
                            xhat.unsqueeze(0),
                            x.unsqueeze(0),
                            data_range=1.0
                    ).item()
    
                    # --- LPIPS ---
                    lpips_loss = lpips_fn(
                            xhat.unsqueeze(0) * 2 - 1,
                            x.unsqueeze(0) * 2 - 1
                    ).item()

                    return recon_mse, ssim_loss, lpips_loss
                
                def recon_losses_rgbd(x, xhat):
                    recon_mse = torch.mean(
                            (xhat - x) ** 2
                    ).item()
    
                    # --- SSIM ---
                    ssim_loss = compute_ssim(
                            xhat.unsqueeze(0)[:, :3],
                            x.unsqueeze(0)[:, :3],
                            data_range=1.0
                    ).item()

                    # --- LPIPS ---
                    lpips_loss = lpips_fn(
                            xhat.unsqueeze(0)[:, :3] * 2 - 1,
                            x.unsqueeze(0)[:, :3] * 2 - 1
                    ).item()
                    
                    return recon_mse, ssim_loss, lpips_loss

                mse_recon_rgb_v1, ssim_rgb_v1, lpips_rgb_v1 = recon_losses_rgb(rgb_v1[i,:,1], recon_rgb_v1[i])
                mse_recon_rgbd_v1, ssim_rgbd_v1, lpips_rgbd_v1 = recon_losses_rgbd(rgbd_v1[i,:,1], recon_rgbd_v1[i])

                if multiview:
                    mse_recon_rgb_v2, ssim_rgb_v2, lpips_rgb_v2 = recon_losses_rgb(rgb_v2[i,:,1], recon_rgb_v2[i])
                    mse_recon_rgbd_v2, ssim_rgbd_v2, lpips_rgbd_v2 = recon_losses_rgbd(rgbd_v2[i,:,1], recon_rgbd_v2[i])

                    mse_recon_rgb_v3, ssim_rgb_v3, lpips_rgb_v3 = recon_losses_rgb(rgb_v3[i,:,1], recon_rgb_v3[i])
                    mse_recon_rgbd_v3, ssim_rgbd_v3, lpips_rgbd_v3 = recon_losses_rgbd(rgbd_v3[i,:,1], recon_rgbd_v3[i])

                    if vis_saved < num_vis_samples:
                        save_reconstruction_grid(
                            vis_saved,
                            vis_dir,
                            multiview,
                            rgb_v1[i],
                            recon_rgb_v1[i],
                            rgbd_v1[i],
                            recon_rgbd_v1[i],
                            rgb_v2[i],
                            recon_rgb_v2[i],
                            rgbd_v2[i],
                            recon_rgbd_v2[i]
                        )
                        vis_saved += 1
                else:
                    if vis_saved < num_vis_samples:
                        save_reconstruction_grid(
                            vis_saved,
                            vis_dir,
                            multiview,
                            rgb_v1[i],
                            recon_rgb_v1[i],
                            rgbd_v1[i],
                            recon_rgbd_v1[i]
                        )
                        vis_saved += 1

                    mse_recon_rgb_v2, ssim_rgb_v2, lpips_rgb_v2 = -1, -1, -1
                    mse_recon_rgbd_v2, ssim_rgbd_v2, lpips_rgbd_v2 = -1, -1, -1

                    mse_recon_rgb_v3, ssim_rgb_v3, lpips_rgb_v3 = -1, -1, -1
                    mse_recon_rgbd_v3, ssim_rgbd_v3, lpips_rgbd_v3 = -1, -1, -1

                    hamming_rgb_v12 = -1
                    hamming_rgb_v13 = -1
                    hamming_rgb_v23 = -1
                    mse_view_rgb_v12 = -1
                    mse_view_rgb_v13 = -1
                    mse_view_rgb_v23 = -1
                    hamming_rgbd_v12 = -1
                    hamming_rgbd_v13 = -1
                    hamming_rgbd_v23 = -1
                    mse_view_rgbd_v12 = -1
                    mse_view_rgbd_v13 = -1
                    mse_view_rgbd_v23 = -1

                stats["agree"]["hamming_rgb"][10].append(hamming_rgb_v12)
                stats["agree"]["mse_view_rgb"][10].append(mse_view_rgb_v12)
                stats["agree"]["hamming_rgbd"][10].append(hamming_rgbd_v12)
                stats["agree"]["mse_view_rgbd"][10].append(mse_view_rgbd_v12)
                stats["agree"]["hamming_rgb"][20].append(hamming_rgb_v13)
                stats["agree"]["mse_view_rgb"][20].append(mse_view_rgb_v13)
                stats["agree"]["hamming_rgbd"][20].append(hamming_rgbd_v13)
                stats["agree"]["mse_view_rgbd"][20].append(mse_view_rgbd_v13)
                stats["agree"]["hamming_rgb"][12].append(hamming_rgb_v23)
                stats["agree"]["mse_view_rgb"][12].append(mse_view_rgb_v23)
                stats["agree"]["hamming_rgbd"][12].append(hamming_rgbd_v23)
                stats["agree"]["mse_view_rgbd"][12].append(mse_view_rgbd_v23)

                stats["recon"]["mse_rgb"][0].append(mse_recon_rgb_v1)
                stats["recon"]["mse_rgbd"][0].append(mse_recon_rgbd_v1)
                stats["recon"]["mse_rgb"][1].append(mse_recon_rgb_v2)
                stats["recon"]["mse_rgbd"][1].append(mse_recon_rgbd_v2)
                stats["recon"]["mse_rgb"][2].append(mse_recon_rgb_v3)
                stats["recon"]["mse_rgbd"][2].append(mse_recon_rgbd_v3)
                stats["recon"]["ssim_rgb"][0].append(ssim_rgb_v1)
                stats["recon"]["ssim_rgbd"][0].append(ssim_rgbd_v1)
                stats["recon"]["ssim_rgb"][1].append(ssim_rgb_v2)
                stats["recon"]["ssim_rgbd"][1].append(ssim_rgbd_v2)
                stats["recon"]["ssim_rgb"][2].append(ssim_rgb_v3)
                stats["recon"]["ssim_rgbd"][2].append(ssim_rgbd_v3)
                stats["recon"]["lpips_rgb"][0].append(lpips_rgb_v1)
                stats["recon"]["lpips_rgbd"][0].append(lpips_rgbd_v1)
                stats["recon"]["lpips_rgb"][1].append(lpips_rgb_v2)
                stats["recon"]["lpips_rgbd"][1].append(lpips_rgbd_v2)
                stats["recon"]["lpips_rgb"][2].append(lpips_rgb_v3)
                stats["recon"]["lpips_rgbd"][2].append(lpips_rgbd_v3)

                row = {
                    # --- metadata ---
                    "traj_path": meta["traj_path"],
                    "environment": meta["environment"],
                    "task": meta["task"],
                    "datetime": meta["datetime"],

                    "view_v1": meta["view"],
                    "view_v2": meta["view2"],
                    "view_v3": meta["view3"],
                    "frame_t": meta["frame_t"],
                    "frame_t_plus": meta["frame_t_plus"],

                    # --- cross-view (pair-specific) ---
                    "ham_rgb_01": float(hamming_rgb_v12),
                    "ham_rgb_02": float(hamming_rgb_v13),
                    "ham_rgb_12": float(hamming_rgb_v23),
                    "ham_rgbd_01": float(hamming_rgbd_v12),
                    "ham_rgbd_02": float(hamming_rgbd_v13),
                    "ham_rgbd_12": float(hamming_rgbd_v23),

                    "mse_view_rgb_01": float(mse_view_rgb_v12),
                    "mse_view_rgb_02": float(mse_view_rgb_v13),
                    "mse_view_rgb_12": float(mse_view_rgb_v23),
                    "mse_view_rgbd_01": float(mse_view_rgbd_v12),
                    "mse_view_rgbd_02": float(mse_view_rgbd_v13),
                    "mse_view_rgbd_12": float(mse_view_rgbd_v23),

                    # --- reconstruction RGB ---
                    "mse_rgb_v1": float(mse_recon_rgb_v1),
                    "ssim_rgb_v1": float(ssim_rgb_v1),
                    "lpips_rgb_v1": float(lpips_rgb_v1),

                    "mse_rgb_v2": float(mse_recon_rgb_v2),
                    "ssim_rgb_v2": float(ssim_rgb_v2),
                    "lpips_rgb_v2": float(lpips_rgb_v2),

                    "mse_rgb_v3": float(mse_recon_rgb_v3),
                    "ssim_rgb_v3": float(ssim_rgb_v3),
                    "lpips_rgb_v3": float(lpips_rgb_v3),

                    # --- reconstruction RGBD (as currently defined) ---
                    "mse_rgbd_v1": float(mse_recon_rgbd_v1),
                    "ssim_rgbd_v1": float(ssim_rgbd_v1),
                    "lpips_rgbd_v1": float(lpips_rgbd_v1),
        
                    "mse_rgbd_v2": float(mse_recon_rgbd_v2),
                    "ssim_rgbd_v2": float(ssim_rgbd_v2),
                    "lpips_rgbd_v2": float(lpips_rgbd_v2),

                    "mse_rgbd_v3": float(mse_recon_rgbd_v3),
                    "ssim_rgbd_v3": float(ssim_rgbd_v3),
                    "lpips_rgbd_v3": float(lpips_rgbd_v3),
                }

                for j, code in enumerate(codes_rgb_v1[i]):
                    row[f"rgb_v1_d{j}"] = int(code)

                for j, code in enumerate(codes_rgbd_v1[i]):
                    row[f"rgbd_v1_d{j}"] = int(code)

                if multiview:
                    for j, code in enumerate(codes_rgb_v2[i]):
                        row[f"rgb_v2_d{j}"] = int(code)

                    for j, code in enumerate(codes_rgbd_v2[i]):
                        row[f"rgbd_v2_d{j}"] = int(code)

                    for j, code in enumerate(codes_rgb_v3[i]):
                        row[f"rgb_v3_d{j}"] = int(code)

                    for j, code in enumerate(codes_rgbd_v3[i]):
                        row[f"rgbd_v3_d{j}"] = int(code)

                rows.append(row)

                total += 1

    print(f"\nProcessed {total} samples")
    return stats, rows


def save_reconstruction_grid(
    idx,
    save_dir,
    multiview,
    rgb_v1,
    recon_rgb_v1,
    rgbd_v1,
    recon_rgbd_v1,
    rgb_v2=None,
    recon_rgb_v2=None,
    rgbd_v2=None,
    recon_rgbd_v2=None
):
    """
    Creates 4-row grid:
    Row 1: V1 RGB  (x1 | x2 | x2_hat)
    Row 2: V1 Depth
    Row 3: V2 RGB
    Row 4: V2 Depth
    """

    def split_rgb(pair):
        return pair[:,0], pair[:,1]

    def split_depth(pair):
        return pair[3,0].unsqueeze(0), pair[3,1].unsqueeze(0)

    def split_rgb_depth(pair):
        return pair[:3,0].squeeze(0), pair[:3,1].squeeze(0)

    def norm_depth(x):
        return x.repeat(3,1,1)

    # ---- View 1 ----
    v1_x1_rgb, v1_x2_rgb = split_rgb(rgb_v1)
    v1_x2_hat_rgb = recon_rgb_v1

    v1_x1_rgbd, v1_x2_rgbd = split_rgb_depth(rgbd_v1)
    v1_x2_hat_rgbd = recon_rgbd_v1[:3]

    v1_x1_depth, v1_x2_depth = split_depth(rgbd_v1)
    v1_x2_hat_depth = recon_rgbd_v1[3].unsqueeze(0)

    v1_x1_depth = norm_depth(v1_x1_depth)
    v1_x2_depth = norm_depth(v1_x2_depth)
    v1_x2_hat_depth = norm_depth(v1_x2_hat_depth)

    # ---- View 2 ----
    if multiview:
        v2_x1_rgb, v2_x2_rgb = split_rgb(rgb_v2)
        v2_x2_hat_rgb = recon_rgb_v2

        v2_x1_rgbd, v2_x2_rgbd = split_rgb_depth(rgbd_v2)
        v2_x2_hat_rgbd = recon_rgbd_v2[:3]

        v2_x1_depth, v2_x2_depth = split_depth(rgbd_v2)
        v2_x2_hat_depth = recon_rgbd_v2[3].unsqueeze(0)

        v2_x1_depth = norm_depth(v2_x1_depth)
        v2_x2_depth = norm_depth(v2_x2_depth)
        v2_x2_hat_depth = norm_depth(v2_x2_hat_depth)

    # ---- Stack rows ----
    row1 = torch.stack([v1_x1_rgb, v1_x2_rgb, v1_x2_hat_rgb], dim=0)
    row2 = torch.stack([v1_x1_rgbd, v1_x2_rgbd, v1_x2_hat_rgbd], dim=0)
    row3 = torch.stack([v1_x1_depth, v1_x2_depth, v1_x2_hat_depth], dim=0)
    if multiview:
        row4 = torch.stack([v2_x1_rgb, v2_x2_rgb, v2_x2_hat_rgb], dim=0)
        row5 = torch.stack([v2_x1_rgbd, v2_x2_rgbd, v2_x2_hat_rgbd], dim=0)
        row6 = torch.stack([v2_x1_depth, v2_x2_depth, v2_x2_hat_depth], dim=0)

        grid = torch.cat([row1, row2, row3, row4, row5, row6], dim=0)
    else:
        grid = torch.cat([row1, row2, row3], dim=0)

    vutils.save_image(
        grid,
        os.path.join(save_dir, f"sample_{idx}.png"),
        nrow=3,
        value_range=(-1,1)
    )




# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--val_meta", type=str, required=True)
    parser.add_argument("--train_meta", type=str, required=True)
    parser.add_argument("--all_meta", type=str, required=True)
    parser.add_argument("--rgb_ckpt", type=str, required=True)
    parser.add_argument("--rgbd_ckpt", type=str, required=True)
    parser.add_argument("--codebook_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--offset", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_vis_samples", type=int, default=10)
    parser.add_argument("--output_root", type=str, default="model_comparison_results")

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    with open(args.val_meta, "r") as f:
        val_entries = json.load(f)
    
    with open(args.train_meta, "r") as f:
        train_entries = json.load(f)

    with open(args.all_meta, "r") as f:
        all_entries = json.load(f)

    def create_dataloader(entries):
        ds = BridgeComparisonDataset(entries, offset=args.offset)

        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )
        return dl

    tk1_entries = [
        t for t in val_entries
        if t["environment"] == "toykitchen1"
    ]

    tk1_loader = create_dataloader(tk1_entries)

    print("ID toykitchen1 validation entries: ", len(tk1_entries))

    tk1_train_entries = [
        t for t in train_entries
        if t["environment"] == "toykitchen1"
    ]

    tk1_train_loader = create_dataloader(tk1_train_entries)

    print("ID toykitchen1 train entries: ", len(tk1_train_entries))

    tk2_entries = [
        t for t in val_entries
        if t["environment"] == "toykitchen2"
    ]
    
    tk2_loader = create_dataloader(tk2_entries)

    print("ID toykitchen2 validation entries: ", len(tk2_entries))

    tk2_train_entries = [
        t for t in train_entries
        if t["environment"] == "toykitchen2"
    ]

    tk2_train_loader = create_dataloader(tk2_train_entries)

    print("ID toykitchen2 train entries: ", len(tk2_train_entries))

    tk5_entries = [
        t for t in val_entries
        if t["environment"] == "toykitchen5"
    ]

    tk5_loader = create_dataloader(tk5_entries)

    print("ID toykitchen5 validation entries: ", len(tk5_entries))

    tk5_train_entries = [
        t for t in train_entries
        if t["environment"] == "toykitchen5"
    ]

    tk5_train_loader = create_dataloader(tk5_train_entries)

    print("ID toykitchen5 train entries: ", len(tk5_train_entries))

    tk7_entries = [
        t for t in val_entries
        if t["environment"] == "toykitchen7"
    ]

    tk7_loader = create_dataloader(tk7_entries)

    print("ID toykitchen7 validation entries: ", len(tk7_entries))
    
    tk7_train_entries = [
        t for t in train_entries
        if t["environment"] == "toykitchen7"
    ]

    tk7_train_loader = create_dataloader(tk7_train_entries)

    print("ID toykitchen7 train entries: ", len(tk7_train_entries))

    ood_env_entries = [
        t for t in all_entries
        if t["environment"] == "toykitchen6"
    ]

    ood_env_loader = create_dataloader(ood_env_entries)

    print("OOD environment entries: ", len(ood_env_entries))

    # ood_emb_entries = [
    #     t for t in all_entries
    #     if t["environment"] == "dt_toykitchen2"
    # ]

    # ood_emb_dataset = BridgeComparisonDatasetSingle(ood_emb_entries, offset=args.offset)

    # ood_emb_loader = DataLoader(
    #     ood_emb_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=4
    # )

    # print("OOD embodiment entries: ", len(ood_emb_entries))

    dataloaders = {
        'Val_tk1' : tk1_loader,
        'Val_tk2' : tk2_loader,
        'Val_tk5' : tk5_loader,
        'Val_tk7' : tk7_loader,
        'Train_tk1' : tk1_train_loader,
        'Train_tk2' : tk2_train_loader,
        'Train_tk5' : tk5_train_loader,
        'Train_tk7' : tk7_train_loader,
        'OOD_env' : ood_env_loader
        # 'OOD_emb' : ood_emb_loader
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model: ", args.rgb_ckpt)

    laq_rgb = LatentActionQuantization(
        dim=1024,
        quant_dim=32,
        codebook_size=args.codebook_size,
        image_size=256,
        patch_size=32,
        spatial_depth=8,
        temporal_depth=8,
        dim_head=64,
        heads=16,
        code_seq_len=4,
    ).to(device)

    laq_rgb.load(args.rgb_ckpt)
    laq_rgb.eval()

    print("Loading model: ", args.rgbd_ckpt)

    laq_rgbd = LatentActionQuantization(
        dim=1024,
        quant_dim=32,
        codebook_size=args.codebook_size,
        image_size=256,
        patch_size=32,
        spatial_depth=8,
        temporal_depth=8,
        dim_head=64,
        heads=16,
        code_seq_len=4,
        channels=4
    ).to(device)

    laq_rgbd.load(args.rgbd_ckpt)
    laq_rgbd.eval()

    print("\n==== RESULTS ====")
    for dataset_name, dataloader in dataloaders.items():

        dataset_dir = os.path.join(args.output_root, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        json_path = os.path.join(dataset_dir, "stats.jsonl")
        vis_dir = os.path.join(dataset_dir, "vis")

        mv = dataset_name != "OOD_emb"

        stats, rows = compare_models(
            laq_rgb, 
            laq_rgbd,
            device,
            dataloader, 
            args.num_samples,
            num_vis_samples=args.num_vis_samples,
            vis_dir=vis_dir,
            multiview=mv
        )

        print(f"\n--- {dataset_name} ---")
        #for group_name, group in stats.items():
        #    print(f"\n{group_name.upper()}")
        #    for metric_name, values in group.items():
        #        print(
        #            f"{metric_name}: "
        #            f"mean={np.mean(values):.4f}, "
        #            f"std={np.std(values):.4f}"
        #        )

        for group_name, group in stats.items():
            print(f"\n{group_name.upper()}")

            for metric_name, metric_dict in group.items():

                # metric_dict is now a dict: key -> list
                for key, values in metric_dict.items():

                    if len(values) == 0:
                        continue

                    print(
                        f"{metric_name}[{key}]: "
                        f"mean={np.mean(values):.4f}, "
                        f"std={np.std(values):.4f}"
                    )

        # Tag each row with dataset name
        with open(json_path, "w") as f_json:
            for row in rows:
                f_json.write(json.dumps(row) + "\n")

        print("\nSaved:")
        print(json_path)


if __name__ == "__main__":
    main()
