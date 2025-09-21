import torch
from torch import nn

from pointcept.models.point_transformer_v3.point_transformer_v3m2_sonata import (
    PointTransformerV3,
)

from sweep_surf_detect.Model.point_embed import PointEmbed


class SweepSurfPTv3(nn.Module):
    def __init__(
        self,
        latent_dim: int = 512,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.point_embed = PointEmbed(dim=self.latent_dim)

        self.ptv3_encoder = PointTransformerV3(self.latent_dim)

        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )
        return

    def encode(self, pts: torch.Tensor) -> torch.Tensor:
        B, N, _ = pts.shape

        flattened_pts = pts.view(-1, 3)
        batch_indices = (
            torch.arange(B).unsqueeze(1).expand(B, N).reshape(-1).to(pts.device)
        )
        pts_feature = self.point_embed(flattened_pts.unsqueeze(0)).squeeze(0)

        ptv3_data = {
            "coord": flattened_pts,
            "feat": pts_feature,
            "batch": batch_indices,
            "grid_size": 0.01,
        }

        point = self.ptv3_encoder(ptv3_data)

        # batch = point.batch
        feature = point.feat

        feature = feature.view(B, N, -1)

        return feature

    def decode(self, feature: torch.Tensor) -> torch.Tensor:
        t = self.decoder(feature)
        return t

    def forward(self, data_dict: dict) -> dict:
        pts = data_dict["pts"]

        feature = self.encode(pts)

        t = self.decoder(feature)

        result_dict = {
            "t": t,
        }

        return result_dict
