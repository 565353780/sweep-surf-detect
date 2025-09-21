import numpy as np


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
