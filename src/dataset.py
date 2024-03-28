
import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from torch.utils.data import DataLoader
from sorcery import dict_of
import os
from scipy import io
import random


def _blur_down(img, scale=0.25, ksize=(3, 3), sigma=(1.5, 1.5)):
    blur = gaussian_blur2d(img, ksize, sigma)
    return F.interpolate(blur, scale_factor=scale, mode="bicubic")


class NBUDataset(Dataset):
    def __init__(self, data_dir, split="train", ori_test=False):
        self.data_dir = data_dir
        self.data_dir = os.path.expanduser(data_dir)
        self.split = split
        self.ori_test = ori_test
        random_seed = 42
        train_ratio = 0.5
        val_ratio = 0.2

        mat_dir = os.path.join(self.data_dir, "MS_256")
        mat_files = os.listdir(mat_dir)
        num_files = len(mat_files)

        random.seed(random_seed)
        np.random.seed(random_seed)
        random_sequence = list(range(num_files))
        random.shuffle(random_sequence)

        train_idx = int(num_files * train_ratio)
        val_idx = train_idx + int(num_files * val_ratio)

        train_mat_files_files = [mat_files[idx] for idx in random_sequence[:train_idx]]
        val_mat_files_files = [mat_files[idx] for idx in random_sequence[train_idx:val_idx]]
        test_mat_files_files = [mat_files[idx] for idx in random_sequence[val_idx:]]

        if self.split == "train":
            self.mat_files = train_mat_files_files
        elif self.split == "test":
            self.mat_files = test_mat_files_files
        elif self.split == "val":
            self.mat_files = val_mat_files_files
        else:
            raise RuntimeError("Wrong split.")

        if "gaofen" in data_dir.lower():
            self.max_val = 1023
        else:
            self.max_val = 2047

        print(f"split={self.split}", len(self.mat_files), self.mat_files[:5])

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):

        max_val = self.max_val

        mat_ms = io.loadmat(os.path.join(self.data_dir, "MS_256", self.mat_files[idx]))
        mat_pan = io.loadmat(os.path.join(self.data_dir, "PAN_1024", self.mat_files[idx]))

        # Some keys may be different.
        key_ms = "imgMS" if "imgMS" in mat_ms.keys() else "I_MS"
        key_pan = "imgPAN" if "imgPAN" in mat_pan.keys() else "I_PAN"
        if "imgPAN" not in mat_pan.keys() and "I_PAN" not in mat_pan.keys():
            key_pan = "block"
        ms = torch.from_numpy((mat_ms[key_ms] / max_val).astype(np.float32)).permute(2, 0, 1)
        pan = torch.from_numpy((mat_pan[key_pan] / max_val).astype(np.float32)).unsqueeze(0)
        gt = ms

        if self.split == "train" or self.split == "val" or not self.ori_test:
            ms = _blur_down(ms.unsqueeze(0)).squeeze(0)
            pan = _blur_down(pan.unsqueeze(0)).squeeze(0)

        up_ms = F.interpolate(ms.unsqueeze(0), (pan.shape[-2], pan.shape[-1]), mode="bicubic").squeeze(0)

        inp_dict = dict_of(ms, pan, up_ms, gt)

        return inp_dict


class plNBUDataset(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers=4, pin_memory=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_train = NBUDataset(data_dir, split="train")
        self.dataset_val = NBUDataset(data_dir, split="val")
        self.dataset_test_ori = NBUDataset(data_dir, split="test", ori_test=True)
        self.dataset_test = NBUDataset(data_dir, split="test", ori_test=False)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self):
        return [DataLoader(
            self.dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        ),
            DataLoader(
            self.dataset_test_ori,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )]
