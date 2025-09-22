import os
import torch
import numpy as np
from typing import Union

from sweep_surf_detect.Data.sweep_surf import SweepSurf
from sweep_surf_detect.Model.sweep_surf_ptv3 import SweepSurfPTv3


class Detector(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        use_ema: bool = False,
        device: str = "cuda:0",
        dtype=torch.float32,
    ) -> None:
        self.use_ema = use_ema
        self.device = device
        self.dtype = dtype

        self.num_curve_ctrlpts = 3
        self.is_curve_closed = False
        self.num_plane_curve_ctrlpts = 3
        self.is_plane_curve_closed = False
        self.epoch_size = 10000
        self.num_sample_surf_pts = 10000

        self.latent_dim = 128

        self.model = SweepSurfPTv3(
            latent_dim=self.latent_dim,
        ).to(self.device, dtype=self.dtype)

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model_file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        model_dict = torch.load(
            model_file_path, map_location=torch.device(self.device), weights_only=False
        )

        if self.use_ema:
            self.model.load_state_dict(model_dict["ema_model"])
        else:
            self.model.load_state_dict(model_dict["model"])

        print("[INFO][Detector::loadModel]")
        print("\t load model success!")
        print("\t model_file_path:", model_file_path)
        return True

    @torch.no_grad()
    def detect(self, pts: torch.Tensor) -> torch.Tensor:
        self.model.eval()

        data_dict = {"pts": pts}

        result_dict = self.model(data_dict)

        pred_t = result_dict["t"]

        return pred_t

    @torch.no_grad()
    def detectRandomSweepSurf(self) -> torch.Tensor:
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

        pts = (
            torch.from_numpy(sample_pts).unsqueeze(0).to(self.device, dtype=self.dtype)
        )

        pred_t = self.detect(pts)[0].detach().cpu().numpy()

        return
