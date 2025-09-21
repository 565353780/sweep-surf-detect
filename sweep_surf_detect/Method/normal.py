import numpy as np
from scipy.interpolate import CubicSpline


def generate_tangents(
    control_points: np.ndarray, noise_scale: float = 0.01, is_closed: bool = False
) -> np.ndarray:
    """
    生成控制点处的切向量
    Args:
        control_points: (N, 3) array
        noise_scale: 随机扰动大小
        is_closed: 是否闭合曲线
    Returns:
        tangents: (N, 3) array, 每个控制点的切线方向
    """
    N = len(control_points)
    tangents = np.zeros_like(control_points)

    for i in range(N):
        if is_closed:
            prev = control_points[(i - 1) % N]
            next = control_points[(i + 1) % N]
        else:
            if i == 0:
                prev = control_points[i]
                next = control_points[i + 1]
            elif i == N - 1:
                prev = control_points[i - 1]
                next = control_points[i]
            else:
                prev = control_points[i - 1]
                next = control_points[i + 1]

        direction = next - prev
        direction += np.random.normal(
            scale=noise_scale, size=direction.shape
        )  # 可控扰动
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])  # fallback
        else:
            direction /= norm

        tangents[i] = direction

    return tangents


def generate_tangent_angles(
    control_points: np.ndarray, is_closed: bool = False, noise_scale: float = 0.01
) -> np.ndarray:
    """
    根据控制点位置估计平滑的切线方向角度（单位：弧度）

    参数:
        control_points: (N,2) 控制点
        is_closed: bool，是否闭合曲线
        noise_scale: float，添加的随机扰动大小

    返回:
        angles: (N,) 每个控制点处的切线夹角（单位：弧度，范围 [-π, π]）
    """
    N = len(control_points)
    assert control_points.shape[1] == 2, "控制点必须是 Nx2"

    if is_closed:
        prev_idx = np.arange(-1, N - 1)
        next_idx = np.arange(1, N + 1) % N
    else:
        prev_idx = np.clip(np.arange(N) - 1, 0, N - 1)
        next_idx = np.clip(np.arange(N) + 1, 0, N - 1)

    tangents = control_points[next_idx] - control_points[prev_idx]
    tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)

    angles = np.arctan2(tangents[:, 1], tangents[:, 0])

    # unwrap angles to avoid discontinuities
    unwrapped = np.unwrap(angles)

    t = np.linspace(0, 1, N, endpoint=False if is_closed else True)

    if is_closed:
        # Append first value to close the loop
        unwrapped = np.append(unwrapped, unwrapped[0])
        t = np.append(t, 1.0)
        bc_type = "periodic"
    else:
        bc_type = "natural"

    angle_spline = CubicSpline(t, unwrapped, bc_type=bc_type)
    smoothed_angles = angle_spline(t[:-1] if is_closed else t)

    if noise_scale > 0:
        smoothed_angles += np.random.normal(
            scale=noise_scale, size=smoothed_angles.shape
        )

    # Wrap back to [-pi, pi]
    final_angles = (smoothed_angles + np.pi) % (2 * np.pi) - np.pi
    return final_angles
