import numpy as np
from typing import Union
from scipy.interpolate import CubicSpline

from sweep_surf_detect.Method.normal import generate_tangent_angles


class PlaneCurve:
    def __init__(
        self,
        points: np.ndarray,
        is_closed: bool = False,
        tangents_angle: Union[np.ndarray, None] = None,
    ):
        """
        points: (N,2) 控制点数组
        tangents_angle: (N,) 每个控制点处切线与X轴正向的夹角（单位：弧度）
        is_closed: 是否闭合
        """
        self.points = points
        self.is_closed = is_closed

        if tangents_angle is None:
            self.tangent_angles = generate_tangent_angles(self.points, is_closed)
        else:
            self.tangent_angles = tangents_angle

        self.N = len(points)

        assert points.shape[1] == 2, "points 应该是 Nx2 的二维数组"
        assert len(points) == len(self.tangent_angles), "控制点数与切线角度数不一致"

        # 参数 t 对应每个控制点的参数值
        self.ts = np.linspace(0, 1, self.N, endpoint=not is_closed)

        # 构建二维样条插值器
        self._build_spline()

    def _build_spline(self):
        if self.is_closed:
            pts = np.vstack([self.points, self.points[0]])
            ts = np.linspace(0, 1, len(pts))
            bc_type = "periodic"
        else:
            pts = self.points
            ts = self.ts
            bc_type = "not-a-knot"

        self.spline_x = CubicSpline(ts, pts[:, 0], bc_type=bc_type)
        self.spline_y = CubicSpline(ts, pts[:, 1], bc_type=bc_type)

    def queryPoint(self, t: float) -> np.ndarray:
        """
        查询参数t ∈ [0,1]处的平面点坐标，返回 shape=(3,)
        """
        t = t % 1.0 if self.is_closed else np.clip(t, 0, 1)
        x = self.spline_x(t)
        y = self.spline_y(t)
        return np.array([x, y, 0.0])

    def queryTangent(self, t: float) -> np.ndarray:
        """
        查询参数t处切线方向（单位向量），在XY平面上，返回 shape=(3,)
        """
        dx = self.spline_x.derivative()(t)
        dy = self.spline_y.derivative()(t)
        tangent = np.array([dx, dy, 0.0])
        return tangent / np.linalg.norm(tangent)
