#!/usr/bin/env python

import torch
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models.resnet_custom import resnet50_baseline
from torch.utils.data import DataLoader
from utils.utils import collate_features
from utils.file_utils import save_hdf5
import openslide
import logging
from typing import Optional
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import sys

@dataclass
class ExtractFeaturesConfig:
    slide_path: str
    h5_path: str
    output_prefix: str
    model_name: str = "resnet50"
    model_path: str = ""
    ctranspath_path: str = ""
    batch_size: int = 8
    print_every: int = 20
    target_patch_size: Optional[int] = None
    use_gpu: bool = True
    num_workers: int = 16


log = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="extract_features_config", node=ExtractFeaturesConfig)

@hydra.main(version_base=None, config_name="extract_features_config")
def compute_w_loader(cfg: ExtractFeaturesConfig):
    wsi = openslide.open_slide(cfg.slide_path)
    output_h5_path = cfg.output_prefix + ".h5"
    device = torch.device('cuda') if cfg.use_gpu and torch.cuda.is_available() else torch.device('cpu')

    log.info('loading model checkpoint')
    pretrained = False
    if cfg.model_name == 'resnet50':
        pretrained = True
        model = resnet50_baseline(pretrained=True)
    elif cfg.model_name == 'ctranspath':
        pretrained = True
        sys.path.append(cfg.ctranspath_path)
        model = torch.load(cfg.model_path)
        cfg.target_patch_size = 224
    elif cfg.model_name == 'vit16':
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
        wsi=wsi,
        pretrained=pretrained,
        target_patch_size=cfg.target_patch_size)

    x, y = dataset[0]
    kwargs = {'num_workers': cfg.num_workers, 'pin_memory': cfg.use_gpu and torch.cuda.is_available()}
    loader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, **kwargs, collate_fn=collate_features)

    mode = 'w'
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            if count % cfg.print_every == 0:
                log.info('batch {}/{}, {} files processed'.format(count, len(loader), count * cfg.batch_size))
            batch = batch.to(device, non_blocking=True)

            features = model(batch)
            features = features.cpu().numpy()

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_h5_path, asset_dict, attr_dict= None, mode=mode)
            mode = 'a'

if __name__ == "__main__":
    compute_w_loader()

