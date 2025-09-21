import numpy as np
import matplotlib.pyplot as plt

from sweep_surf_detect.Data.curve import Curve


def visualize_curve_with_frames(curve: Curve, t_samples=30):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 曲线采样
    ts = np.linspace(0, 1, t_samples)
    pts = np.array([curve.evaluate(t) for t in ts])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], label="Interpolated Curve", color="blue")

    # 控制点
    ax.scatter(
        curve.control_points[:, 0],
        curve.control_points[:, 1],
        curve.control_points[:, 2],
        c="r",
        label="Control Points",
    )

    # 局部坐标系
    for t in np.linspace(0, 1, t_samples):
        T = curve.queryTransform(t)
        o = T[:3, 3]
        x, y, z = T[:3, 0], T[:3, 1], T[:3, 2]
        s = 0.2
        ax.quiver(*o, *(x * s), color="r")
        ax.quiver(*o, *(y * s), color="g")
        ax.quiver(*o, *(z * s), color="b")

    ax.set_title("Hermite-style Interpolated Curve with Tangents")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.show()
