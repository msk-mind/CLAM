import argparse
import os
import pdb
import random
import sys
import time
from math import floor

import h5py
import numpy as np
import openslide
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models.resnet_custom import resnet50_baseline
import argparse
import fsspec
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from pathlib import Path
from PIL import Image
import h5py
from tiffslide import TiffSlide
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(
        file_path, output_path, wsi_path, model,
        batch_size = 8, verbose = 0, print_every=20, pretrained=True,
        target_patch_size=-1, slide_file_path=None,
    storage_options={},
):
    """
    args:
        file_path: directory of bag (.h5 file)
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        batch_size: batch_size for computing features in batches
        verbose: level of feedback
        pretrained: use weights pretrained on imagenet
        custom_downsample: custom defined downscale factor of image patches
        target_patch_size: custom defined, rescaled image size before embedding
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi_path=wsi_path, pretrained=pretrained,
        target_patch_size=target_patch_size, storage_options=storage_options)
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features,
                        worker_init_fn=dataset.worker_init)
    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path,len(loader)))

    mode = 'w'
    fs, output_path = fsspec.core.url_to_fs(output_path., **storage_options)
    simplecache_fs = fsspec.filesystem('simplecache', fs=fs)
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            if count % print_every == 0:
                print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
            batch = batch.to(device, non_blocking=True)

            features = model(batch)
            features = features.cpu().numpy()

            asset_dict = {'features': features, 'coords': coords}

            with simplecache_fs.open(output_path, "ab+") as output_f:
                save_hdf5(output_f, asset_dict, attr_dict= None, mode=mode)
            mode = 'a'

    return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--s3_storage_key', type = str, default=None)
parser.add_argument('--s3_storage_secret', type = str, default=None)
parser.add_argument('--s3_endpoint_url', type = str, default=None)
args = parser.parse_args()


if __name__ == '__main__':

    print('initializing dataset')

    storage_options = {"key" : args.s3_storage_key,
                       "secret" : args.s3_storage_secret,
                       "client_kwargs" : {'endpoint_url' : args.s3_endpoint_url}}

    slides_filesystem, slides_path = fsspec.core.url_to_fs(args.data_slide_dir, **storage_options)
    src_filesystem, src_path = fsspec.core.url_to_fs(args.data_h5_dir, **storage_options)
    dst_filesystem, dst_path = fsspec.core.url_to_fs(args.feat_dir, **storage_options)
    src_filesystem = fsspec.filesystem('simplecache', fs=src_filesystem)
    dst_filesystem = fsspec.filesystem('simplecache', fs=src_filesystem)

    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)

    print(dst_path)
    dst_filesystem.makedirs(str(Path(dst_path)), exist_ok=True)
    dst_filesystem.makedirs(str(Path((dst_path + '/pt_files'))), exist_ok=True)
    dst_filesystem.makedirs(str(Path((dst_path + '/h5_files'))), exist_ok=True)

    print('loading model checkpoint')
    model = resnet50_baseline(pretrained=True)
    model = model.to(device)

    # print_network(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()
    total = len(bags_dataset)

    for bag_candidate_idx in range(total):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id+'.h5'

        h5_file_path = str(Path(src_path + '/patches') / bag_name)
        slide_file_path = os.path.join(args.data_slide_dir,f"{slide_id}{args.slide_ext}")

        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        if not args.no_auto_skip and dst_filesystem.isfile(str(Path(dst_path + '/pt_files') / f'{slide_id}.pt')):
            print('skipped {}'.format(slide_id))
            continue

        output_path = str(Path(dst_path + '/h5_files') / bag_name)
        time_start = time.time()

        print("h5 file path: ", h5_file_path)
        print("output path: ", output_path)

        output_file_path = compute_w_loader(
            h5_file_path,
            output_path,
            None,  # initializing this via worker_init
            model,
            src_filesystem,
            dst_filesystem,
            batch_size = args.batch_size,
            verbose = 1,
            print_every = 20,
            target_patch_size=args.target_patch_size,
            slide_file_path=slide_file_path,
            storage_options=storage_options
        )

        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

        with dst_filesystem.open(str(Path(output_path)), "rb") as output_f:
            file = h5py.File(output_f, "r")
            features = file['features'][:]
            print('features size: ', features.shape)
            print('coordinates size: ', file['coords'].shape)
            file.close()

        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)

        pt_output_path = str(Path(dst_path + '/pt_files') / f'{bag_base}.pt')
        with dst_filesystem.open(pt_output_path, 'wb') as pt_f:
            torch.save(features, pt_f)


