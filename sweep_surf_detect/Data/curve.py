import numpy as np
from scipy.interpolate import CubicSpline


class Curve:
    def __init__(
        self,
        points: np.ndarray,
        tangents: np.ndarray,
        normals: np.ndarray,
        is_closed=False,
    ):
        """
        points: (N,3) 控制点
        tangents: (N,3) 每个控制点处的切线方向（单位向量）
        normals: (N,3) 每个控制点处的X轴方向法线（单位向量）
        is_closed: bool，是否闭合曲线
        """
        self.points = points
        self.normals = normals
        self.tangents = tangents
        self.is_closed = is_closed
        self.N = len(points)

        self.ts = np.linspace(0, 1, self.N, endpoint=not is_closed)
        self._build_spline()

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
