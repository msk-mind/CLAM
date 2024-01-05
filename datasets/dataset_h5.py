import logging
import math
import os
import pdb
import pickle
import re
from random import randrange
from typing import Optional

import fsspec
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tiffslide import TiffSlide
from torch.utils.data import DataLoader, Dataset, sampler
from torchvision import models, transforms, utils

log = logging.getLogger(__name__)


def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    trnsfrms_val = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    return trnsfrms_val


class Whole_Slide_Bag(Dataset):
    def __init__(
        self,
        file_path,
        pretrained=False,
        custom_transforms=None,
        target_patch_size=-1,
    ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.pretrained = pretrained
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None

        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            raise NotImplementedError  # self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f["imgs"]
            self.length = len(dset)

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file["imgs"]
        for name, value in dset.attrs.items():
            log.info(f"{name}, {value}")

        log.info(f"pretrained: {self.pretrained}")
        log.info(f"transformations: {self.roi_transforms}")
        if self.target_patch_size is not None:
            log.info(f"target_size: {self.target_patch_size}")

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as hdf5_file:
            img = hdf5_file["imgs"][idx]
            coord = hdf5_file["coords"][idx]

        img = Image.fromarray(img)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord


class Whole_Slide_Bag_FP(Dataset):
    def __init__(
        self,
        file_path,
        wsi_path,
        pretrained=False,
        custom_transforms=None,
        target_patch_size: Optional[int] = None,
        storage_options: dict = {},
    ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            target_patch_size (int): Custom defined image size before embedding
        """
        self.pretrained = pretrained
        self.wsi_path = wsi_path
        self.storage_options = storage_options
        if not custom_transforms and target_patch_size:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
            self.roi_transforms.transforms.insert(
                0, transforms.CenterCrop(target_patch_size)
            )
        else:
            raise NotImplementedError  # self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f["coords"]
            self.patch_level = f["coords"].attrs["patch_level"]
            self.patch_size = f["coords"].attrs["patch_size"]
            self.length = len(dset)
            if target_patch_size:
                self.target_patch_size = (target_patch_size,) * 2
            else:
                self.target_patch_size = None
        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file["coords"]
        for name, value in dset.attrs.items():
            log.info(f"{name}, {value}")

        log.info("\nfeature extraction settings")
        log.info(f"target patch size: {self.target_patch_size}")
        log.info(f"pretrained: {self.pretrained}")
        log.info(f"transformations: {self.roi_transforms}")

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as hdf5_file:
            coord = hdf5_file["coords"][idx]
        with fsspec.open(self.wsi_path, **self.storage_options) as f:
            wsi = TiffSlide(f)
            img = wsi.read_region(
                coord, self.patch_level, (self.patch_size, self.patch_size)
            ).convert("RGB")
            if self.target_patch_size is not None:
                img = img.resize(self.target_patch_size)
            img = self.roi_transforms(img).unsqueeze(0)

        return img, coord


class Dataset_All_Bags(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df["slide_id"][idx]
