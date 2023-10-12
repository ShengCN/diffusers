import os
from os.path import join
import h5py
import numpy as np
from glob import glob
import pandas as pd
import torch

from torch.utils.data import Dataset


class SSN_Dataset(Dataset):
    def __init__(self, ds_dict: dict, pixel_transform, condition_transform, tokenizer):
        """ Init
        :param ds_dict: {'hdf5': path} 
        """        
        keys = ['ds_root']
        for k in keys:
            assert k in ds_dict, 'Not find key {}'.format(k)

        ds_root = ds_dict['ds_root']
        ds_meta = join(ds_root, 'data.csv')

        assert os.path.exists(ds_root), 'Not find {}'.format(ds_root)
        assert os.path.exists(ds_meta), 'Not find {}'.format(ds_meta)

        self.files = glob(join(ds_root, 'data', '*.npz'))
        self.files.sort()
        self.N = len(self.files)
        self.df = pd.read_csv(ds_meta)

        self.pixel_transform = pixel_transform
        self.condition_transform = condition_transform
        self.tokenizer = tokenizer


    def __len__(self):
        return self.N


    def __getitem__(self, idx):
        prompt = 'shadow'

        buffers = np.load(self.files[idx])
        mask    = buffers['mask']
        ibl     = buffers['ibl']
        shadow  = buffers['shadow']

        iblx = self.df.at[idx, 'ibl_x']
        ibly = self.df.at[idx, 'ibl_y']
        captions = "{} {}".format(iblx, ibly)

        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        # Normalize source images to [0, 1].
        # conditioning_pixel_values = self.condition_transform(conditioning_pixel_values)
        conditioning_pixel_values = np.repeat(mask[..., None], 3, axis=2)

        # Normalize target images to [-1, 1].
        # pixel_values = np.repeat(shadow[..., None], 3, axis=2)
        # pixel_values = self.pixel_transform(pixel_values)
        pixel_values = np.repeat(shadow[..., None] * 2.0 - 1.0, 3, axis=2)

        conditioning_pixel_values = torch.tensor(conditioning_pixel_values.transpose(2,0,1))
        pixel_values = torch.tensor(pixel_values.transpose(2,0,1))

        # return dict(jpg=target, txt=prompt, hint=source)
        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "ibl": ibl,
            "input_ids": torch.squeeze(inputs.input_ids),
        }
