import numpy as np

from sweep_surf_detect.Data.sweep_surf import SweepSurf
from sweep_surf_detect.Method.render import visualize_sweep_surface


def test():
    num_curve_ctrlpts = 3
    is_curve_closed = False
    curve_t_samples = 100
    num_plane_curve_ctrlpts = 3
    is_plane_curve_closed = False
    plane_curve_t_samples = 100

    curve_ctrlpts = np.random.randn(num_curve_ctrlpts, 3)

    plane_curve_ctrlpts = np.random.randn(num_plane_curve_ctrlpts, 2)

    sweep_surf = SweepSurf(
        curve_ctrlpts,
        is_curve_closed,
        plane_curve_ctrlpts,
        is_plane_curve_closed,
    )

    visualize_sweep_surface(
        sweep_surf,
        curve_t_samples,
        plane_curve_t_samples,
    )

    return True
