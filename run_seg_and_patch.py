#!/usr/bin/env python

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import fsspec
import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from tiffslide import TiffSlide

from wsi_core.batch_process_utils import initialize_df
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords


@dataclass
class FilterConfig:
    a_t: int = 1
    a_h: int = 1
    max_n_holes: int = 2


@dataclass
class SegConfig:
    seg_level: int = -1
    sthresh: int = 15
    mthresh: int = 11
    close: int = 2
    use_otsu: bool = False
    filter_config: FilterConfig = field(default_factory=FilterConfig)


@dataclass
class VisConfig:
    vis_level: int = -1
    line_thickness: int = 50


@dataclass
class PatchConfig:
    patch_size: int = 256
    step_size: int = 246
    patch_level: int = 0
    contour_fn: str = "four_pt"
    use_padding: bool = True
    custom_downsample: int = 1


@dataclass
class StitchConfig:
    downscale: int = 64
    bg_color: tuple = (0, 0, 0)
    alpha: int = -1


@dataclass
class SegPatchConfig:
    slide_path: str = MISSING
    output_prefix: str = MISSING
    requested_magnification: Optional[int] = None
    seg_config: SegConfig = field(default_factory=SegConfig)
    vis_config: VisConfig = field(default_factory=VisConfig)
    patch_config: PatchConfig = field(default_factory=PatchConfig)
    stitch_config: StitchConfig = field(default_factory=StitchConfig)
    seg: bool = True
    save_mask: bool = True
    stitch: bool = True
    patch: bool = True
    storage_options: dict = field(default_factory=dict)
    defaults: List[Any] = field(
        default_factory=lambda: ["_self_", "config", "seg_patch_config"]
    )


log = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="base_seg_patch_config", node=SegPatchConfig)


@hydra.main(version_base=None, config_path=".", config_name="base_seg_patch_config")
def seg_and_patch(cfg: SegPatchConfig):
    slide_id = Path(cfg.slide_path).stem
    WSI_object = WholeSlideImage(slide_id, path=cfg.slide_path, storage_options=cfg.storage_options)
    Path(cfg.output_prefix).parent.mkdir(parents=True, exist_ok=True)

    slide_mag = int(float(WSI_object.wsi.properties["aperio.AppMag"]))
    if cfg.requested_magnification:
        custom_downsample = slide_mag / cfg.requested_magnification
        cfg.patch_config.custom_downsample = custom_downsample

    if cfg.vis_config.vis_level < 0:
        if len(WSI_object.level_dim) == 1:
            cfg.vis_config.vis_level = 0
        else:
            wsi = WSI_object.getOpenSlide()
            best_level = wsi.get_best_level_for_downsample(64)
            cfg.vis_config.vis_level = best_level

    if cfg.seg_config.seg_level < 0:
        if len(WSI_object.level_dim) == 1:
            cfg.seg_config.seg_level = 0
        else:
            wsi = WSI_object.getOpenSlide()
            best_level = wsi.get_best_level_for_downsample(64)
            cfg.seg_config.seg_level = best_level

    w, h = WSI_object.level_dim[cfg.seg_config.seg_level]
    if w * h > 1e8:
        raise ValueError(
            "level_dim {} x {} is likely too large for successful segmentation, aborting".format(
                w, h
            )
        )

    if cfg.seg:
        log.info("Running segmentation...")
        WSI_object.segmentTissue(**OmegaConf.to_container(cfg.seg_config, resolve=True))

    if cfg.save_mask:
        mask = WSI_object.visWSI(**OmegaConf.to_container(cfg.vis_config, resolve=True))
        mask_path = cfg.output_prefix + ".mask.jpg"
        log.info(f"Saving mask to {mask_path}...")
        mask.save(mask_path)

    if cfg.patch:
        log.info("Patching...")
        h5file = WSI_object.process_contours(
            cfg.output_prefix + ".h5",
            **OmegaConf.to_container(cfg.patch_config, resolve=True),
        )

    if cfg.stitch:
        log.info("Stitching...")
        file_path = cfg.output_prefix + ".h5"
        if Path(file_path).is_file():
            heatmap = StitchCoords(
                file_path,
                WSI_object,
                **OmegaConf.to_container(cfg.stitch_config, resolve=True),
            )
            stitch_path = cfg.output_prefix + ".stitch.jpg"
            heatmap.save(stitch_path)
            log.info(f"Stitches written to {stitch_path}")
        else:
            log.info("No patches found")



if __name__ == "__main__":
    seg_and_patch()
