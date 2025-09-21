import torch
import numpy as np
from torch.utils.data import Dataset

from sweep_surf_detect.Data.sweep_surf import SweepSurf


class SweepSurfDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        num_curve_ctrlpts: int = 3,
        is_curve_closed: bool = False,
        num_plane_curve_ctrlpts: int = 3,
        is_plane_curve_closed: bool = False,
        epoch_size: int = 10000,
        num_sample_surf_pts=10000,
        split: str = "train",
        dtype=torch.float32,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.num_curve_ctrlpts = num_curve_ctrlpts
        self.is_curve_closed = is_curve_closed
        self.num_plane_curve_ctrlpts = num_plane_curve_ctrlpts
        self.is_plane_curve_closed = is_plane_curve_closed
        self.epoch_size = epoch_size
        self.num_sample_surf_pts = num_sample_surf_pts
        self.split = split
        self.dtype = dtype

        self.output_error = False
        return

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):
        index = index % self.epoch_size

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        curve_ctrlpts = np.random.randn(self.num_curve_ctrlpts, 3)

        plane_curve_ctrlpts = np.random.randn(self.num_plane_curve_ctrlpts, 2)

        sweep_surf = SweepSurf(
            curve_ctrlpts,
            self.is_curve_closed,
            plane_curve_ctrlpts,
            self.is_plane_curve_closed,
        )

        random_t = np.random.rand(self.num_sample_surf_pts, 2)

        sample_pts = []
        for i in range(random_t.shape[0]):
            sample_point = sweep_surf.querySurfPoint(random_t[i][0], random_t[i][1])
            sample_pts.append(sample_point)

        sample_pts = np.array(sample_pts)

        data = {
            "pts": torch.from_numpy(sample_pts).float(),
            "t": torch.from_numpy(random_t).float(),
        }

        return data
