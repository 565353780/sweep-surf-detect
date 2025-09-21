import numpy as np
import matplotlib.pyplot as plt

from sweep_surf_detect.Data.curve import Curve


def plot_curve_with_frames(curve: Curve, samples=30, axis_len=0.1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ts = np.linspace(0, 1, samples)
    points = np.array([curve._get_point(t) for t in ts])

    ax.plot(points[:, 0], points[:, 1], points[:, 2], "b-", label="Curve")

    for t in ts:
        T = curve.queryTransform(t)
        origin = T[:3, 3]
        x_axis = T[:3, 0] * axis_len
        y_axis = T[:3, 1] * axis_len
        z_axis = T[:3, 2] * axis_len

        ax.quiver(*origin, *x_axis, color="r", length=axis_len, normalize=True)
        ax.quiver(*origin, *y_axis, color="g", length=axis_len, normalize=True)
        ax.quiver(*origin, *z_axis, color="b", length=axis_len, normalize=True)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Curve with Local Coordinate Frames")
    ax.legend(["Curve", "X axis", "Y axis", "Z axis"])
    ax.set_box_aspect([1, 1, 1])
    plt.show()
