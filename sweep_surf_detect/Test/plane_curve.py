import numpy as np

from sweep_surf_detect.Data.plane_curve import PlaneCurve
from sweep_surf_detect.Method.render import plot_plane_curve_with_tangents


def test():
    num_control_points = 5
    is_closed = False
    t_samples = 100

    control_points = np.random.randn(num_control_points, 2)

    curve = PlaneCurve(control_points, is_closed)

    # 调用可视化函数
    plot_plane_curve_with_tangents(curve, samples=t_samples, tangent_len=0.1)
