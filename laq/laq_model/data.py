from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader

from torchvision import transforms as T

import os
import random

import glob
import tarfile
import io
import numpy as np


def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def pair(val):
    return val if isinstance(val, tuple) else (val, val)

'''
This is the dataset class for Sthv2 dataset.
The dataset is a list of folders, each folder contains a sequence of frames.
You have to change the dataset class to fit your dataset for custom training.
'''

class ImageVideoDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        offset=5,
    ):
        super().__init__()
        
        self.folder = folder
        self.folder_list = os.listdir(folder)
        self.image_size = image_size
      
        self.offset = offset

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.ToTensor(),
        ])


    def __len__(self):
        return len(self.folder_list) ## length of folder list is not exact number of frames; TODO: change this to actual number of frames
    
    def __getitem__(self, index):
        try :
            offset = self.offset
            
            folder = self.folder_list[index]
            img_list = os.listdir(os.path.join(self.folder, folder))

            img_list = sorted(img_list, key=lambda x: int(x.split('.')[0][4:]))
            ## pick random frame 
            first_frame_idx = random.randint(0, len(img_list)-1)
            first_frame_idx = min(first_frame_idx, len(img_list)-1)
            second_frame_idx = min(first_frame_idx + offset, len(img_list)-1)
            
            first_path = os.path.join(self.folder, folder, img_list[first_frame_idx])
            second_path = os.path.join(self.folder, folder, img_list[second_frame_idx])
                    
            img = Image.open(first_path)
            next_img = Image.open(second_path)
            
            transform_img = self.transform(img).unsqueeze(1)
            next_transform_img = self.transform(next_img).unsqueeze(1)
            
            cat_img = torch.cat([transform_img, next_transform_img], dim=1)
            return cat_img
        except :
            print("error", index)
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))


def frame_index(path):
    name = os.path.basename(path)
    digits = ''.join(filter(str.isdigit, name))
    return int(digits) if digits else 0

class BridgeVideoDataset(Dataset):
    def __init__(
        self,
        image_size,
        offset=5,
        traj_entries=None,
        use_multiview=True,
        altview=False,
        use_depth=False,
        depth_norm="per_sample",
        global_depth_stats=None
    ):
        self.offset = offset
        self.image_size = image_size
        self.use_multiview = use_multiview
        self.use_depth = use_depth
        self.depth_norm = depth_norm
        self.global_depth_stats = global_depth_stats

        self.rgb_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.ToTensor(),
        ])
        
        self.depth_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        self.samples = []

        env_view = {"toykitchen1": "images1",
                    "toykitchen2": "images2",
                    "toykitchen5": "images1",
                    "toykitchen7": "images0"
        }

        for entry in traj_entries:
            traj_path = entry["traj_path"]
            rgb_views = entry.get("rgb_views", {}).items()


            for view_name, frame_count in rgb_views:
                if frame_count <= offset:
                    continue

                if not self.use_multiview and view_name != "images0":
                    continue

                if altview and view_name != env_view[entry["environment"]]:
                    continue

                view_dir = os.path.join(traj_path, view_name)

                rgb_imgs = sorted(
                    glob.glob(os.path.join(view_dir, "im_*.jpg")) +
                    glob.glob(os.path.join(view_dir, "im_*.png")),
                    key=frame_index
                )

                if len(rgb_imgs) <= offset:
                    continue

                sample = {"rgb": rgb_imgs}

                if use_depth:
                    depth_paths = sorted(
                        glob.glob(os.path.join(view_dir, "im_*_depth.npy")),
                        key=frame_index
                    )

                    if len(depth_paths) != len(rgb_imgs):
                        print(f"Skipping {traj_path}/{view_name} (depth mismatch)")
                        continue

                    sample["depth"] = depth_paths

                self.samples.append(sample)

        print(f"Loaded {len(self.samples)} trajectory-view samples")

    def _normalize_depth(self, depth):
        depth = depth.astype(np.float32)

        if self.depth_norm == "per_sample":
            mean = depth.mean()
            std = depth.std() + 1e-6
        else:
            mean, std = self.global_depth_stats

        return (depth - mean) / std

    def __getitem__(self, index):
        sample = self.samples[index]
        rgb_list = sample["rgb"]

        i = random.randint(0, len(rgb_list) - self.offset - 1)

        rgb1 = Image.open(rgb_list[i])
        rgb2 = Image.open(rgb_list[i + self.offset])

        x1_rgb = self.rgb_transform(rgb1)
        x2_rgb = self.rgb_transform(rgb2)

        if self.use_depth:
            depth_list = sample["depth"]

            depth1 = self._normalize_depth(np.load(depth_list[i]))
            depth2 = self._normalize_depth(np.load(depth_list[i + self.offset]))

            depth1_pil = Image.fromarray(depth1, mode="F")
            depth2_pil = Image.fromarray(depth2, mode="F")

            x1_depth = self.depth_transform(depth1_pil)
            x2_depth = self.depth_transform(depth2_pil)

            x1 = torch.cat([x1_rgb, x1_depth], dim=0)
            x2 = torch.cat([x2_rgb, x2_depth], dim=0)

            return torch.stack([x1, x2], dim=1)

        return torch.stack([x1_rgb, x2_rgb], dim=1)

    def __len__(self):
        return len(self.samples)


class OldBridgeVideoDataset(Dataset):
    def __init__(self, image_size, offset=5, traj_entries=None):
        self.offset = offset
        self.image_size = image_size

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.ToTensor(),
        ])

        self.samples = []

        for entry in traj_entries:
            traj_path = entry["traj_path"]
            rgb_views = entry.get("rgb_views", {})

            views = []

            for view_name, frame_count in rgb_views.items():
                if frame_count <= offset:
                    continue

                view_dir = os.path.join(traj_path, view_name)

                imgs = sorted(
                    glob.glob(os.path.join(view_dir, "*.jpg")) +
                    glob.glob(os.path.join(view_dir, "*.png")),
                    key=frame_index
                )

                if len(imgs) > offset:
                    views.append(imgs)

            if len(views) >= 1:
                self.samples.append(views)

        print(f"Loaded {len(self.samples)} trajectories")

    def __getitem__(self, index):
        views = self.samples[index]

        # Random camera view
        view = random.choice(views)

        i = random.randint(0, len(view) - self.offset - 1)

        img1 = Image.open(view[i])
        img2 = Image.open(view[i + self.offset])

        x1 = self.transform(img1)
        x2 = self.transform(img2)

        return torch.stack([x1, x2], dim=1)  # (C, 2, H, W)

    def __len__(self):
        return len(self.samples)


class BridgeTarVideoDataset(Dataset):
    def __init__(
        self,
        tar_paths,
        image_size,
        offset=5,
        use_depth=False,
        use_multiview_depth=False,
        allowed_prefixes=None,
        depth_norm="per_sample",  # or "global"
        global_depth_stats=None   # (mean, std) if global
    ):
        self.offset = offset
        self.image_size = image_size
        self.use_depth = use_depth
        self.use_multiview_depth = use_multiview_depth
        self.allowed_prefixes = allowed_prefixes
        self.depth_norm = depth_norm
        self.global_depth_stats = global_depth_stats

        self.rgb_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
        ])
        self.depth_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])


        self.tar_cache = {}

        self.samples = []  # [(tar_path, traj_prefix, view_name, frame_indices)]

        print("Indexing tar shards...", flush=True)

        for tar_path in tar_paths:
           
            with tarfile.open(tar_path, "r") as tar:
            
                traj_view_frames = {}

                for m in tar:
                    if not m.isfile():
                        continue

                    parts = m.name.split("/")
                    if len(parts) < 2:
                        continue

                    traj_prefix = "/".join(parts[:-2])

                    if self.allowed_prefixes is not None and traj_prefix not in self.allowed_prefixes:
                        continue

                    view_name = parts[-2]
                    filename = parts[-1]

                    if filename.startswith("rgb_"):
                        frame_id = int(filename.split("_")[1].split(".")[0])

                        key = (tar_path, traj_prefix, view_name)
                        traj_view_frames.setdefault(key, []).append(frame_id)

                for (tar_path, traj_prefix, view_name), frame_ids in traj_view_frames.items():
                    frame_ids = sorted(frame_ids)

                    if len(frame_ids) > offset:
                        self.samples.append(
                            (tar_path, traj_prefix, view_name, frame_ids)
                        )

        print(f"Indexed {len(self.samples)} trajectory-view samples", flush=True)

    def _get_tar(self, tar_path):
        # Re-initialize cache if we're in a new worker process
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else -1
    
        if not hasattr(self, '_tar_cache_worker_id') or self._tar_cache_worker_id != worker_id:
            self.tar_cache = {}
            self._tar_cache_worker_id = worker_id
    
        if tar_path not in self.tar_cache:
            self.tar_cache[tar_path] = tarfile.open(tar_path, "r")
        return self.tar_cache[tar_path]

    def _normalize_depth(self, depth):
        depth = depth.astype(np.float32)

        if self.depth_norm == "per_sample":
            mean = depth.mean()
            std = depth.std() + 1e-6
        else:
            assert self.global_depth_stats is not None, \
                "global_depth_stats required for global normalization"
            mean, std = self.global_depth_stats

        return (depth - mean) / std

    def _safe_extract(self, tar, path):
        try:
            f = tar.extractfile(path)
            if f is None:
                return None
            return f.read()
        except KeyError:
            return None

    def __getitem__(self, index):
        tar_path, traj_prefix, view_name, frame_ids = self.samples[index]

        i = random.randint(0, len(frame_ids) - self.offset - 1)

        f1 = frame_ids[i]
        f2 = frame_ids[i + self.offset]

        rgb_key1 = f"{traj_prefix}/{view_name}/rgb_{f1:04d}.png"
        rgb_key2 = f"{traj_prefix}/{view_name}/rgb_{f2:04d}.png"
        
        if self.use_depth:
            depth_type = "depth_multiview" if self.use_multiview_depth else "depth_single"
            depth_key1 = f"{traj_prefix}/{view_name}/{depth_type}_{f1:04d}.npy"
            depth_key2 = f"{traj_prefix}/{view_name}/{depth_type}_{f2:04d}.npy"

        tar = self._get_tar(tar_path)

        rgb_bytes1 = self._safe_extract(tar, rgb_key1)
        rgb_bytes2 = self._safe_extract(tar, rgb_key2)

        if rgb_bytes1 is None or rgb_bytes2 is None:
            raise RuntimeError(f"Missing RGB frame in {tar_path}")

        rgb1 = Image.open(io.BytesIO(rgb_bytes1)).convert("RGB")
        rgb2 = Image.open(io.BytesIO(rgb_bytes2)).convert("RGB")

        if self.use_depth:
            depth_bytes1 = self._safe_extract(tar, depth_key1)
            depth_bytes2 = self._safe_extract(tar, depth_key2)

            if depth_bytes1 is None or depth_bytes2 is None:
                raise RuntimeError(f"Missing depth frame in {tar_path}")

            depth1 = self._normalize_depth(np.load(io.BytesIO(depth_bytes1)))
            depth2 = self._normalize_depth(np.load(io.BytesIO(depth_bytes2)))
            depth1_pil = Image.fromarray(depth1, mode='F')
            depth2_pil = Image.fromarray(depth2, mode='F')
            
        x1_rgb = self.rgb_transform(rgb1)
        x2_rgb = self.rgb_transform(rgb2)
        
        if self.use_depth:
            x1_depth = self.depth_transform(depth1_pil)
            x2_depth = self.depth_transform(depth2_pil)

            x1 = torch.cat([x1_rgb, x1_depth], dim=0)
            x2 = torch.cat([x2_rgb, x2_depth], dim=0)

            return torch.stack([x1, x2], dim=1)

        return torch.stack([x1_rgb, x2_rgb], dim=1)

    def __len__(self):
        return len(self.samples)
