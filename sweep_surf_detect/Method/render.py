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


def plot_plane_curve_with_tangents(
    curve, samples=50, tangent_len=0.1, show_control_points=True
):
    """
    可视化 PlaneCurve：曲线 + 切线方向箭头

    参数:
        curve: PlaneCurve 实例
        samples: int, 采样点数量
        tangent_len: float, 每个切线箭头长度
        show_control_points: bool, 是否显示控制点
    """
    ts = np.linspace(0, 1, samples)
    pts = np.array([curve.queryPoint(t) for t in ts])
    tangents = np.array([curve.queryTangent(t) for t in ts])

    fig, ax = plt.subplots(figsize=(6, 6))

    # 画曲线
    ax.plot(pts[:, 0], pts[:, 1], "b-", label="Plane Curve")

    # 画切线箭头
    for p, t in zip(pts, tangents):
        ax.arrow(
            p[0],
            p[1],
            t[0] * tangent_len,
            t[1] * tangent_len,
            head_width=tangent_len * 0.2,
            head_length=tangent_len * 0.3,
            fc="r",
            ec="r",
        )

    # 控制点
    if show_control_points:
        ctrl_pts = curve.points
        ax.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], "ko--", label="Control Points")

    ax.set_aspect("equal")
    ax.set_title("PlaneCurve with Tangents")
    ax.grid(True)
    ax.legend()
    plt.show()


def visualize_sweep_surface(
    sweep_surf, num_curve_samples=100, num_profile_samples=50, color="lightblue"
):
    """
    可视化 SweepSurf 的扫略曲面（通过采样点构造网格）

    参数:
        sweep_surf: SweepSurf 实例
        num_curve_samples: 沿主曲线采样数（纵向）
        num_profile_samples: 截面曲线采样数（横向）
        color: 表面颜色
    """
    curve_ts = np.linspace(0, 1, num_curve_samples)
    profile_ts = np.linspace(0, 1, num_profile_samples)

    # 采样网格坐标
    X = np.zeros((num_curve_samples, num_profile_samples))
    Y = np.zeros((num_curve_samples, num_profile_samples))
    Z = np.zeros((num_curve_samples, num_profile_samples))

    for i, ct in enumerate(curve_ts):
        for j, pt in enumerate(profile_ts):
            p = sweep_surf.querySurfPoint(curve_t=ct, plane_curve_t=pt)
            X[i, j], Y[i, j], Z[i, j] = p

    # 绘制曲面
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X,
        Y,
        Z,
        color=color,
        edgecolor="k",
        linewidth=0.2,
        alpha=0.9,
        rstride=1,
        cstride=1,
    )

    ax.set_title("Sweep Surface Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.axis("auto")
    plt.tight_layout()
    plt.show()
