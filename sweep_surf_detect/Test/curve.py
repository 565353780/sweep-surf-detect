import numpy as np

from sweep_surf_detect.Data.curve import Curve
from sweep_surf_detect.Method.render import visualize_curve_with_frames


def generate_tangents(control_points, noise_scale=0.1):
    """
    基于控制点自动生成切向量，并添加轻微扰动以增加自然感
    control_points: (N, 3)
    noise_scale: 控制扰动强度（例如 0.1 表示10%扰动）
    返回：
        tangents: (N, 3) 每个点的方向向量（未缩放，已归一化）
    """
    control_points = np.array(control_points)
    N = len(control_points)
    tangents = np.zeros_like(control_points)

    for i in range(N):
        if i == 0:
            base_dir = control_points[1] - control_points[0]
        elif i == N - 1:
            base_dir = control_points[-1] - control_points[-2]
        else:
            base_dir = control_points[i + 1] - control_points[i - 1]

        # 添加轻微扰动（正态分布，均值0，标准差noise_scale）
        noise = np.random.normal(scale=noise_scale, size=3)

        # 加扰动后归一化
        tangent = base_dir + noise
        tangent /= np.linalg.norm(tangent)
        tangents[i] = tangent

    return tangents


# ---------------- 主程序：随机构造并可视化 ---------------- #


def test():
    num_control_points = 5
    noise_scale = 0.1
    closed = True
    t_samples = 100

    np.random.seed(42)  # 固定随机种子

    # 随机生成控制点和单位法向量
    control_points = np.cumsum(
        np.random.randn(num_control_points, 3), axis=0
    )  # 累加生成"平滑"路径
    tangents = generate_tangents(control_points, noise_scale)

    # 构造 Curve 实例
    curve = Curve(control_points, tangents)

    # 可视化
    visualize_curve_with_frames(curve, t_samples)
