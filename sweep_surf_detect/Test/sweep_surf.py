import numpy as np
import open3d as o3d

from sweep_surf_detect.Data.sweep_surf import SweepSurf
from sweep_surf_detect.Method.render import visualize_sweep_surface


def test_random_sample():
    num_curve_ctrlpts = 3
    is_curve_closed = False
    num_plane_curve_ctrlpts = 3
    is_plane_curve_closed = False
    num_sample_surf_pts = 10000

    while True:
        curve_ctrlpts = np.random.randn(num_curve_ctrlpts, 3)

        plane_curve_ctrlpts = np.random.randn(num_plane_curve_ctrlpts, 2)

        sweep_surf = SweepSurf(
            curve_ctrlpts,
            is_curve_closed,
            plane_curve_ctrlpts,
            is_plane_curve_closed,
        )

        random_t = np.random.rand(num_sample_surf_pts, 2)

        sample_pts = []
        for i in range(random_t.shape[0]):
            sample_point = sweep_surf.querySurfPoint(random_t[i][0], random_t[i][1])
            sample_pts.append(sample_point)

        sample_pts = np.array(sample_pts)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sample_pts)

        o3d.visualization.draw_geometries([pcd])
    return True


def test():
    return test_random_sample()

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
