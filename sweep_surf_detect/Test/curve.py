import numpy as np
from scipy.special import comb


def bezier_curve(control_points, t):
    """
    计算贝塞尔曲线点。
    control_points: (N, 3) 控制点坐标
    t: 标量或数组，参数范围[0,1]
    返回对应t的贝塞尔曲线点坐标
    """
    n = len(control_points) - 1
    point = np.zeros((len(t), 3))
    for i in range(n + 1):
        binomial = comb(n, i)
        bernstein = binomial * (t**i) * ((1 - t) ** (n - i))
        point += np.outer(bernstein, control_points[i])
    return point  # (len(t), 3)


def bezier_tangent(control_points, t):
    """
    计算贝塞尔曲线在t处的切线（导数）。
    """
    n = len(control_points) - 1
    tangent = np.zeros((len(t), 3))
    for i in range(n):
        binomial = comb(n - 1, i)
        bernstein = binomial * (t**i) * ((1 - t) ** (n - 1 - i))
        tangent += np.outer(bernstein, n * (control_points[i + 1] - control_points[i]))
    # 单位化切线
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return tangent


def smooth_interpolate_normals(normals, t, control_t=None):
    """
    对控制点法向量做平滑插值。
    normals: (N, 3) 控制点法向量
    t: 采样参数数组[0,1]
    control_t: 控制点对应的参数位置，默认均匀分布
    返回：(len(t), 3) 插值后的法向量，单位化
    """
    from scipy.interpolate import CubicSpline

    if control_t is None:
        control_t = np.linspace(0, 1, len(normals))
    normals = np.array(normals)

    # 对每个分量单独插值
    cs_x = CubicSpline(control_t, normals[:, 0], bc_type="natural")
    cs_y = CubicSpline(control_t, normals[:, 1], bc_type="natural")
    cs_z = CubicSpline(control_t, normals[:, 2], bc_type="natural")

    interp_normals = np.stack([cs_x(t), cs_y(t), cs_z(t)], axis=1)
    # 单位化
    interp_normals /= np.linalg.norm(interp_normals, axis=1, keepdims=True)
    return interp_normals


def construct_rotation_matrices(tangents, normals):
    """
    给定切线和法向量，构造局部坐标系旋转矩阵。
    tangent: (N,3)
    normal: (N,3)
    返回：(N,3,3) 旋转矩阵，列向量分别是x,y,z轴
    """
    N = tangents.shape[0]
    R_all = np.zeros((N, 3, 3))
    for i in range(N):
        z = tangents[i]
        y = normals[i]
        x = np.cross(y, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        y /= np.linalg.norm(y)
        R_all[i] = np.stack([x, y, z], axis=1)
    return R_all


def test():
    # 控制点和法向量定义
    control_pts = np.array([[0, 0, 0], [1, 2, 0], [3, 3, 0], [4, 0, 0]])
    control_normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])

    t_sample = np.linspace(0, 1, 100)
    curve_points = bezier_curve(control_pts, t_sample)  # 曲线点
    curve_tangents = bezier_tangent(control_pts, t_sample)  # 切线方向
    interp_normals = smooth_interpolate_normals(control_normals, t_sample)  # 法向插值
    rot_mats = construct_rotation_matrices(
        curve_tangents, interp_normals
    )  # 局部坐标系旋转矩阵

    # 测试输出
    print("曲线点形状:", curve_points.shape)
    print("旋转矩阵形状:", rot_mats.shape)
    return True
