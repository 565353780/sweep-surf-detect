import numpy as np

from sweep_surf_detect.Data.curve import Curve
from sweep_surf_detect.Method.render import plot_curve_with_frames


def test():
    num_control_points = 5
    is_closed = False
    t_samples = 100

    control_points = np.random.randn(num_control_points, 3)

    curve = Curve(control_points, is_closed)

    plot_curve_with_frames(curve, t_samples)
    return True
