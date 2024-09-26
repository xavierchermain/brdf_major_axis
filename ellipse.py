import taichi as ti


@ti.func
def eval_parametric(
    t: float, r_x: float, r_y: float, theta: float
) -> ti.math.vec2:
    cos_theta = ti.cos(theta)
    sin_theta = ti.sin(theta)
    cos_t = ti.cos(t)
    sin_t = ti.sin(t)
    return ti.math.vec2(
        r_x * cos_theta * cos_t - r_y * sin_theta * sin_t,
        r_x * sin_theta * cos_t + r_y * cos_theta * sin_t,
    )

@ti.kernel
def eval_parametric_v(a: float, b: float, theta: float, p: ti.template()):
    for i in p:
        t_i = i / p.shape[0] * 2.0 * ti.math.pi
        p[i] = eval_parametric(t_i, a, b, theta)


@ti.func
def to_covariance(
    r_x: float, r_y: float, theta: float, std_dev_scale: float
) -> ti.math.mat2:
    cos_theta = ti.cos(theta)
    sin_theta = ti.sin(theta)
    R = ti.math.mat2([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    S = ti.math.mat2([r_x**2, 0.0], [0.0, r_y**2]) * std_dev_scale
    return R @ S @ R.transpose()


@ti.kernel
def to_covariance_kernel(
    a: float, b: float, theta: float, std_dev_scale: float
) -> ti.math.mat2:
    return to_covariance(a, b, theta, std_dev_scale)
