import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def ellipse(t, a=1, b=0.5):
    """参数t生成椭圆截面曲线"""
    x = a * np.cos(t)
    y = b * np.sin(t)
    return np.stack([x, y], axis=1)  # (N,2)


def helix(t, radius=3, pitch=0.5):
    """参数t生成螺旋轨迹曲线"""
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = pitch * t
    return np.stack([x, y, z], axis=1)  # (N,3)


def sweep_surface(section_curve, trajectory_curve):
    """
    section_curve: (M, 2) 截面二维曲线点
    trajectory_curve: (N, 3) 轨迹三维曲线点
    返回：
      surface_points: (N, M, 3) 扫略面点云
    """
    N = trajectory_curve.shape[0]
    M = section_curve.shape[0]
    surface_points = np.zeros((N, M, 3))

    # 计算轨迹曲线切线，用于确定截面曲线的平面方向
    tangent = np.gradient(trajectory_curve, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

    for i in range(N):
        # 构造局部坐标系：tangent为z轴方向，随机选择x轴方向
        z_axis = tangent[i]
        # 找一个非平行向量作为参考
        ref = np.array([0, 0, 1]) if abs(z_axis[2]) < 0.9 else np.array([1, 0, 0])
        x_axis = np.cross(ref, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        # 构造旋转矩阵
        R = np.stack([x_axis, y_axis, z_axis], axis=1)  # 3x3矩阵

        # 将二维截面点扩展到3D (x, y, 0) -> (x, y, 0)
        section_3d = np.concatenate([section_curve, np.zeros((M, 1))], axis=1)  # (M,3)

        # 旋转截面曲线
        rotated_section = section_3d @ R.T  # (M,3)

        # 平移到轨迹曲线上对应点
        surface_points[i] = rotated_section + trajectory_curve[i]

    return surface_points


def test():
    # 生成参数
    t_section = np.linspace(0, 2 * np.pi, 50)
    t_trajectory = np.linspace(0, 4 * np.pi, 100)

    section = ellipse(t_section)
    trajectory = helix(t_trajectory)

    surface = sweep_surface(section, trajectory)  # (N, M, 3)

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(surface.shape[0]):
        ax.plot(
            surface[i, :, 0], surface[i, :, 1], surface[i, :, 2], color="b", alpha=0.6
        )
    ax.set_box_aspect([1, 1, 1])
    plt.show()
    return True
