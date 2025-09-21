import numpy as np


class Curve:
    def __init__(self, control_points, tangents):
        """
        control_points: (N, 3) - 要插值经过的点
        tangents:       (N, 3) - 每个点的切线方向（可以不是单位向量，控制强度）
        """
        self.control_points = np.array(control_points)
        self.tangents = np.array(tangents)
        self.N = len(control_points)
        assert self.control_points.shape == self.tangents.shape

        # 每个段的参数范围（均匀参数化）
        self.segment_t = np.linspace(0, 1, self.N)

    def _hermite_basis(self, t):
        """返回 Hermite 基函数"""
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        return h00, h10, h01, h11

    def evaluate(self, t_global):
        """
        评估全局 t ∈ [0, 1] 上的曲线点
        """
        t_global = np.clip(t_global, 0.0, 1.0)
        num_segments = self.N - 1

        # 确定 t 落在哪个段
        segment_idx = min(int(t_global * num_segments), num_segments - 1)
        local_t = (t_global - self.segment_t[segment_idx]) / (
            self.segment_t[segment_idx + 1] - self.segment_t[segment_idx]
        )

        P0 = self.control_points[segment_idx]
        P1 = self.control_points[segment_idx + 1]
        T0 = self.tangents[segment_idx]
        T1 = self.tangents[segment_idx + 1]

        h00, h10, h01, h11 = self._hermite_basis(local_t)
        point = h00 * P0 + h10 * T0 + h01 * P1 + h11 * T1
        return point

    def derivative(self, t_global):
        """
        计算曲线的一阶导数（切线方向）
        """
        t_global = np.clip(t_global, 0.0, 1.0)
        num_segments = self.N - 1

        segment_idx = min(int(t_global * num_segments), num_segments - 1)
        local_t = (t_global - self.segment_t[segment_idx]) / (
            self.segment_t[segment_idx + 1] - self.segment_t[segment_idx]
        )

        P0 = self.control_points[segment_idx]
        P1 = self.control_points[segment_idx + 1]
        T0 = self.tangents[segment_idx]
        T1 = self.tangents[segment_idx + 1]

        # 导数的 Hermite 基函数
        h00 = 6 * local_t**2 - 6 * local_t
        h10 = 3 * local_t**2 - 4 * local_t + 1
        h01 = -6 * local_t**2 + 6 * local_t
        h11 = 3 * local_t**2 - 2 * local_t

        tangent = h00 * P0 + h10 * T0 + h01 * P1 + h11 * T1
        return tangent / np.linalg.norm(tangent)

    def queryTransform(self, t_global):
        """
        获取 t ∈ [0, 1] 的齐次变换矩阵（4x4），用于扫略面
        """
        position = self.evaluate(t_global)
        z_axis = self.derivative(t_global)

        # 法向方向：我们使用默认的初始法向 + 平滑并正交化
        # 简化：用固定初始方向投影到垂直切线空间
        default_normal = np.array([0, 0, 1])
        y_axis = default_normal - np.dot(default_normal, z_axis) * z_axis
        if np.linalg.norm(y_axis) < 1e-6:  # 说明 default_normal 与切向太接近
            default_normal = np.array([0, 1, 0])
            y_axis = default_normal - np.dot(default_normal, z_axis) * z_axis

        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        return T
