import numpy as np
from typing import Union
from scipy.interpolate import CubicSpline

from sweep_surf_detect.Method.normal import generate_tangents


class Curve:
    def __init__(
        self,
        points: np.ndarray,
        is_closed=False,
        tangents: Union[np.ndarray, None] = None,
        normals: Union[np.ndarray, None] = None,
    ):
        """
        points: (N,3) 控制点
        tangents: (N,3) 每个控制点处的切线方向（单位向量）
        normals: (N,3) 每个控制点处的X轴方向法线（单位向量）
        is_closed: bool，是否闭合曲线
        """
        self.points = points
        self.is_closed = is_closed

        self.N = len(points)

        if tangents is None:
            self.tangents = generate_tangents(
                self.points, noise_scale=0.01, is_closed=self.is_closed
            )
        else:
            self.tangents = tangents

        if normals is None:
            self.normals = np.random.randn(self.N, 3)
        else:
            self.normals = normals

        self._orthonormalize()

        self.ts = np.linspace(0, 1, self.N, endpoint=not is_closed)
        self._build_spline()
        return

    def _orthonormalize(self):
        """
        归一化tangents和normals，并正交化法线，使其垂直于切线
        """
        for i in range(self.N):
            t = self.tangents[i]
            n = self.normals[i]

            # 归一化
            t = t / np.linalg.norm(t)
            n = n / np.linalg.norm(n)

            # Gram-Schmidt正交化法线：去除法线在切线方向的分量
            n = n - np.dot(n, t) * t
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-8:
                # 如果法线接近平行切线，尝试生成一个垂直向量
                if abs(t[0]) < 0.9:
                    n = np.cross(t, [1, 0, 0])
                else:
                    n = np.cross(t, [0, 1, 0])
                n = n / np.linalg.norm(n)
            else:
                n = n / n_norm

            self.tangents[i] = t
            self.normals[i] = n

        # 闭合曲线时保证首尾切线方向一致（若需要）
        if self.is_closed:
            # 这里简单取首尾切线平均后归一化
            avg_t = self.tangents[0] + self.tangents[-1]
            avg_t /= np.linalg.norm(avg_t)
            self.tangents[0] = avg_t
            self.tangents[-1] = avg_t

            # 同样对法线进行调整，保持正交
            avg_n = self.normals[0] + self.normals[-1]
            # 去除法线在切线方向的分量
            avg_n = avg_n - np.dot(avg_n, avg_t) * avg_t
            avg_n /= np.linalg.norm(avg_n)
            self.normals[0] = avg_n
            self.normals[-1] = avg_n

    def _build_spline(self):
        if self.is_closed:
            # 周期样条，首尾点连接
            pts = np.vstack([self.points, self.points[0]])
            ns = np.vstack([self.normals, self.normals[0]])
            ts = np.linspace(0, 1, len(pts))
            bc_type = "periodic"
            tg = np.vstack([self.tangents, self.tangents[0]])
        else:
            pts = self.points
            ns = self.normals
            ts = self.ts
            bc_type = "not-a-knot"
            tg = self.tangents

        self.spline_x = CubicSpline(ts, pts[:, 0], bc_type=bc_type)
        self.spline_y = CubicSpline(ts, pts[:, 1], bc_type=bc_type)
        self.spline_z = CubicSpline(ts, pts[:, 2], bc_type=bc_type)

        self.norm_x = CubicSpline(ts, ns[:, 0], bc_type=bc_type)
        self.norm_y = CubicSpline(ts, ns[:, 1], bc_type=bc_type)
        self.norm_z = CubicSpline(ts, ns[:, 2], bc_type=bc_type)

        # 切线样条插值（如果有）
        self.tan_x = CubicSpline(ts, tg[:, 0], bc_type=bc_type)
        self.tan_y = CubicSpline(ts, tg[:, 1], bc_type=bc_type)
        self.tan_z = CubicSpline(ts, tg[:, 2], bc_type=bc_type)

    def _get_point(self, t):
        return np.array([self.spline_x(t), self.spline_y(t), self.spline_z(t)])

    def _get_normal(self, t):
        n = np.array([self.norm_x(t), self.norm_y(t), self.norm_z(t)])
        return n / np.linalg.norm(n)

    def _get_tangent(self, t):
        if self.tan_x is not None:
            # 用输入切线插值
            tg = np.array([self.tan_x(t), self.tan_y(t), self.tan_z(t)])
            return tg / np.linalg.norm(tg)
        else:
            # 通过导数计算切线
            dp = np.array(
                [
                    self.spline_x.derivative()(t),
                    self.spline_y.derivative()(t),
                    self.spline_z.derivative()(t),
                ]
            )
            return dp / np.linalg.norm(dp)

    def queryTransform(self, t: float) -> np.ndarray:
        t = t % 1.0 if self.is_closed else np.clip(t, 0, 1)
        p = self._get_point(t)
        tangent = self._get_tangent(t)
        x_axis = self._get_normal(t)

        # 处理x_axis和tangent接近平行的情况
        if abs(np.dot(x_axis, tangent)) > 0.999:
            y_axis_candidate = np.array([0, 1, 0])
            if abs(np.dot(tangent, y_axis_candidate)) > 0.999:
                y_axis_candidate = np.array([1, 0, 0])
            x_axis = np.cross(y_axis_candidate, tangent)
            x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross(tangent, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, tangent)

        R = np.column_stack((x_axis, y_axis, tangent))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T
