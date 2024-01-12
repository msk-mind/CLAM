#!/usr/bin/env python

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import h5py
import hydra
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from torch.utils.data import DataLoader

from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models.resnet_custom import resnet50_baseline
from utils.file_utils import save_hdf5
from utils.utils import collate_features


@dataclass
class ExtractFeaturesConfig:
    slide_path: str = MISSING
    h5_path: str = MISSING
    output_prefix: str = MISSING
    model_name: str = "resnet50"
    model_path: str = ""
    ctranspath_path: str = ""
    batch_size: int = 8
    print_every: int = 20
    target_patch_size: Optional[int] = None
    use_gpu: bool = True
    num_workers: int = 16
    storage_options: dict = field(default_factory=dict)
    defaults: List[Any] = field(
        default_factory=lambda: ["_self_", "config", "extract_features_config"]
    )


log = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="base_extract_features_config", node=ExtractFeaturesConfig)


@hydra.main(
    version_base=None, config_path=".", config_name="base_extract_features_config"
)
def extract_features(cfg: ExtractFeaturesConfig):
    Path(cfg.output_prefix).parent.mkdir(parents=True, exist_ok=True)
    output_h5_path = cfg.output_prefix + ".h5"
    output_pt_path = cfg.output_prefix + ".pt"
    device = (
        torch.device("cuda")
        if cfg.use_gpu and torch.cuda.is_available()
        else torch.device("cpu")
    )

    log.info("loading model checkpoint")
    pretrained = False
    if cfg.model_name == "resnet50":
        pretrained = True
        model = resnet50_baseline(pretrained=True)
    elif cfg.model_name == "ctranspath":
        pretrained = True
        sys.path.append(cfg.ctranspath_path)
        model = torch.load(cfg.model_path)
        cfg.target_patch_size = 224
    elif cfg.model_name == "vit16":
        from hipt_model_utils import get_vit256

        model = get_vit256(pretrained_weights=cfg.model_path)
    else:
        log.error(f"Unsupported model {cfg.model_name}")

    model = model.to(device)
    # print_network(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    dataset = Whole_Slide_Bag_FP(
        file_path=cfg.h5_path,
        wsi_path=cfg.slide_path,
        pretrained=pretrained,
        target_patch_size=cfg.target_patch_size,
        storage_options=OmegaConf.to_container(cfg.storage_options),
    )

    kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.use_gpu and torch.cuda.is_available(),
    }
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        **kwargs,
        collate_fn=collate_features,
    )

    mode = "w"
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            if count % cfg.print_every == 0:
                log.info(
                    "batch {}/{}, {} files processed".format(
                        count, len(loader), count * cfg.batch_size
                    )
                )
            batch = batch.to(device, non_blocking=True)

            features = model(batch)
            features = features.cpu().numpy()

            asset_dict = {"features": features, "coords": coords}
            save_hdf5(output_h5_path, asset_dict, attr_dict=None, mode=mode)
            mode = "a"

    with h5py.File(output_h5_path, "r") as file:
        features = file["features"][:]
        log.info(f"features size: {features.shape}")
        log.info(f"coordinates size: {file['coords'].shape}")
        features = torch.from_numpy(features)
        torch.save(features, output_pt_path)


if __name__ == "__main__":
    extract_features()
