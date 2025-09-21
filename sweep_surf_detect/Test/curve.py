import numpy as np

from sweep_surf_detect.Data.curve import Curve
from sweep_surf_detect.Method.render import plot_curve_with_frames


def test():
    num_control_points = 5
    is_closed = True
    t_samples = 100

    np.random.seed(42)  # 固定随机种子

    control_points = np.cumsum(np.random.randn(num_control_points, 3), axis=0)

    # 构造 Curve 实例
    curve = Curve(control_points, is_closed)

    # 可视化
    plot_curve_with_frames(curve, t_samples)
