import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os


class FUNSRDataset(Dataset):
    def __init__(self, data_dir, split_file, mode='train'):
        """
        Args:
            data_dir: Đường dẫn folder chứa .npy (processed/pointclouds)
            split_file: Đường dẫn file split_data.json
            mode: 'train', 'val', hoặc 'test'
        """
        self.data_dir = data_dir
        with open(split_file, 'r') as f:
            splits = json.load(f)

        self.file_list = splits.get(mode, [])
        if not self.file_list:
            print(f"⚠️ Cảnh báo: Tập {mode} rỗng!")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        path = os.path.join(self.data_dir, fname)

        # Load point cloud
        points = np.load(path)  # Shape (N, 3)
        return torch.from_numpy(points).float(), fname