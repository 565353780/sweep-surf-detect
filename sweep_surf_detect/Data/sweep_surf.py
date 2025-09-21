import numpy as np

from sweep_surf_detect.Data.curve import Curve
from sweep_surf_detect.Data.plane_curve import PlaneCurve


class SweepSurf(object):
    def __init__(
        self,
        curve_points: np.ndarray,
        is_curve_closed: bool,
        plane_curve_points: np.ndarray,
        is_plane_curve_closed: bool,
    ) -> None:
        self.curve = Curve(
            curve_points,
            is_curve_closed,
        )

        self.plane_curve = PlaneCurve(
            plane_curve_points,
            is_plane_curve_closed,
        )
        return

    def querySurfPoint(self, curve_t: float, plane_curve_t: float) -> np.ndarray:
        plane_point = self.plane_curve.queryPoint(plane_curve_t)

        transform = self.curve.queryTransform(curve_t)

        local_point = np.array([plane_point[0], plane_point[1], 0.0, 1.0])

        world_point = transform @ local_point

        return world_point[:3]
