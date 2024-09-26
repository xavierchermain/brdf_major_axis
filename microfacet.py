import taichi as ti

import direction
import ellipse
import grid
import limits
import mathext


@ti.func
def slope_to_normal(slope: ti.math.vec2) -> ti.math.vec3:
    deno_sqr = slope.x * slope.x + slope.y * slope.y + 1.0
    ti.math.clamp(deno_sqr, 0.001, limits.f32_max)
    deno = ti.math.sqrt(deno_sqr)
    normal = ti.math.vec3(-slope.x, -slope.y, 1.0)
    return normal / deno


@ti.kernel
def slope_to_normal_v(slope: ti.template(), normal: ti.template()):
    for i in slope:
        normal[i] = slope_to_normal(slope[i])


@ti.func
def normal_to_slope(normal: ti.math.vec3) -> ti.math.vec2:
    return ti.math.vec2(-normal.x, -normal.y) / normal.z


@ti.kernel
def normal_to_slope_v(normal: ti.template(), slope: ti.template()):
    for i in normal:
        slope[i] = normal_to_slope(normal[i])


@ti.func
def cov_to_alpha(cov: ti.math.mat2) -> ti.math.vec3:
    sigma_x = ti.sqrt(cov[0, 0])
    sigma_y = ti.sqrt(cov[1, 1])
    corr = cov[0, 1] / (sigma_x * sigma_y)
    return ti.math.vec3(ti.sqrt(2.0) * sigma_x, ti.sqrt(2.0) * sigma_y, corr)


@ti.func
def alpha_lin_to_std_dev(alpha_lin: ti.math.vec2) -> ti.math.vec2:
    return alpha_lin**2 / ti.math.sqrt(2)


@ti.kernel
def user_parameters_to_cov(
    alpha_lin_x: float, alpha_lin_y: float, theta: float
) -> ti.math.mat2:
    alpha_lin = ti.math.vec2(alpha_lin_x, alpha_lin_y)
    std_slope = alpha_lin_to_std_dev(alpha_lin)
    variance = std_slope**2

    cos_theta = ti.cos(theta)
    sin_theta = ti.sin(theta)
    R = ti.math.mat2([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    S = ti.math.mat2([variance.x, 0.0], [0.0, variance.y])
    return R @ S @ R.transpose()


@ti.kernel
def user_parameters_to_ellipse(
    alpha_lin_x: float, alpha_lin_y: float, k: float
) -> ti.math.vec2:
    alpha_lin = ti.math.vec2(alpha_lin_x, alpha_lin_y)
    std_slope = alpha_lin_to_std_dev(alpha_lin)
    r = std_slope * k
    return r


@ti.func
def GGX_eval_slope_space(x: ti.math.vec2, cov_mat: ti.math.mat2) -> float:
    cov_mat_inv = cov_mat.inverse()
    mat_vec = cov_mat_inv @ x
    scale = 2.0 * ti.math.pi * ti.sqrt(cov_mat.determinant())
    return 1.0 / (scale * (1.0 + 0.5 * ti.math.dot(x, mat_vec)) ** 2)


@ti.func
def GGX_eval_normal_space(wm: ti.math.vec3, cov_mat: ti.math.mat2) -> float:
    slope = normal_to_slope(wm)
    return GGX_eval_slope_space(slope, cov_mat) / wm.z**4


@ti.kernel
def GGX_eval_normal_space_v(
    wm: ti.template(),
    cov_mat: ti.math.mat2,
    GGX_eval_value: ti.template(),
):
    for i in wm:
        # *wm.z to have a normalized function, i.e., integral = 1
        GGX_eval_value[i] = GGX_eval_normal_space(wm[i], cov_mat) * wm[i].z


@ti.kernel
def GGX_eval_slope_kernel(x: ti.math.vec2, cov_mat: ti.math.mat2) -> float:
    return GGX_eval_slope_space(x, cov_mat)


@ti.kernel
def GGX_eval_in_box(
    cov_mat: ti.math.mat2,
    scalar_field: ti.template(),
    p_min: ti.math.vec2,
    p_max: ti.math.vec2,
):
    cell_sides_length = (p_max.x - p_min.x) / scalar_field.shape[0]
    for cell_2dindex in ti.grouped(scalar_field):
        x_i = grid.cell_center_2dpoint(cell_2dindex, p_min, cell_sides_length)
        x_i = x_i - ti.math.vec2(0.5)
        scalar_field[cell_2dindex] = GGX_eval_slope_space(x_i, cov_mat)


@ti.func
def compute_extrema_t0_and_t1(r_x: float, r_y: float, wo: ti.math.vec3) -> ti.math.vec2:
    """
    Solves for Equation 7 from the article, which corresponds to the implementation of Appendix A.

    Parameters
    ----------
    r_x : float
        The first radius of the ellipse representing the GGX distribution's confidence region in slope space.
    r_y : float
        The second radius of the ellipse representing the GGX distribution's confidence region in slope space.
    wo : vec3
        The view (or outgoing) direction vector.

    Returns
    -------
    vec2
        A 2D vector representing the parametric values of the closest and farthest points on the ellipse from the reflection direction.
    """

    EPSILON = 1e-5

    wo2 = wo.xy * wo.xy
    r2x = r_x * r_x
    r2y = r_y * r_y
    p0 = 2.0 * wo.x * wo.y * r_x * r_y
    p1 = wo2.x * (r2x * r2y + r2y) - wo2.y * (r2x * r2y + r2x) + r2x - r2y
    p2 = -p0 * (r2y + 1.0)
    p3 = p0 * p2 * (r2x + 1.0)

    is_not_special_case = ti.abs(p2) > EPSILON

    # Special case
    extremum0 = ti.math.step(0.0, p1) * ti.math.pi * 0.5
    extremum1 = extremum0 + ti.math.pi * 0.5
    if is_not_special_case:
        sqrt_val = ti.math.sqrt(p1 * p1 - p3)
        extremum0 = mathext.atan((p1 + sqrt_val) / p2)
        extremum1 = mathext.atan((p1 - sqrt_val) / p2)
    # extremum0: the closest
    # extremum1: the farthest
    return ti.math.vec2(extremum0, extremum1)


@ti.func
def clamp_major_axis(p: ti.math.vec3, view: ti.math.vec3) -> ti.math.vec2:
    """ "
    Clamp the major axis. Implementation of the Appendix B.
    """
    x = p.x
    y = p.y
    z = p.z
    x_o = view.x
    y_o = view.y
    z_o = view.z

    p0 = x * x_o + y * y_o
    p1 = z_o * (1 - z**2)
    sqrt_arg = ti.math.sqrt(p0**2 + z_o * p1)
    t_0 = 0.5 + z * (p0 - sqrt_arg) / (2 * p1)
    t_1 = 0.5 + z * (p0 + sqrt_arg) / (2 * p1)

    return ti.math.clamp(ti.math.vec2(t_0, t_1), 0.0, 1.0)


@ti.kernel
def compute_extrema(
    r_x: float,
    r_y: float,
    theta: float,
    wo: ti.math.vec3,
    extrema_slope: ti.template(),
    extrema_max_clamped_slope: ti.template(),
    extrema_normal: ti.template(),
    extrema_max_clamped_normal: ti.template(),
    extrema_reflected: ti.template(),
    extrema_max_clamped_reflected: ti.template(),
    extrema_shifted: ti.template(),
):
    ellipse_to_tangent = ti.math.mat2(
        [[ti.cos(theta), -ti.sin(theta)], [ti.sin(theta), ti.cos(theta)]]
    )
    tangent_to_ellipse = ellipse_to_tangent.transpose()
    wo_e = ti.math.vec3(tangent_to_ellipse @ wo.xy, wo.z)

    # The most technical part is inside this function
    ts = compute_extrema_t0_and_t1(r_x, r_y, wo_e)

    t = ti.math.vec4(0.0)
    # t[0] and t[1]: gives the closest
    t[0] = ts[0]
    t[1] = t[0] + ti.math.pi
    # t[2] and t[3]: gives the farthest
    t[2] = ts[1]
    t[3] = t[2] + ti.math.pi
    for i in range(4):
        extrema_slope[i] = ellipse.eval_parametric(t[i], r_x, r_y, 0.0)
        # Extrema in normal space
        extrema_normal[i] = slope_to_normal(extrema_slope[i])
        # Extrema in reflected space
        extrema_reflected[i] = -ti.math.reflect(wo_e, extrema_normal[i])

    tau = clamp_major_axis(extrema_normal[3], wo_e)
    extrema_clamped_0 = direction.nlerp(extrema_normal[2], extrema_normal[3], tau[0])
    extrema_clamped_1 = direction.nlerp(extrema_normal[2], extrema_normal[3], tau[1])
    extrema_clamped_0_reflected = -ti.math.reflect(wo_e, extrema_clamped_0)
    extrema_clamped_1_reflected = -ti.math.reflect(wo_e, extrema_clamped_1)

    # Appendix C. Shifted minimum extrema.
    angleDelta = tau[0] * (t[0] - t[2]) if tau[0] > 0 else (tau[1] - 1) * (t[0] - t[2])
    shifted_min_0_slope = ellipse.eval_parametric(t[0] + angleDelta, r_x, r_y, 0.0)
    shifted_min_0_reflected = -ti.math.reflect(
        wo_e, slope_to_normal(shifted_min_0_slope)
    )
    shifted_min_1_slope = ellipse.eval_parametric(
        t[0] - angleDelta + ti.math.pi, r_x, r_y, 0.0
    )
    shifted_min_1_reflected = -ti.math.reflect(
        wo_e, slope_to_normal(shifted_min_1_slope)
    )

    minor_axis_length = ti.math.length(
        shifted_min_0_reflected - shifted_min_1_reflected
    )
    major_axis_length = ti.math.length(
        extrema_clamped_0_reflected - extrema_clamped_1_reflected
    )
    relative_length = (major_axis_length - minor_axis_length) / major_axis_length
    major_axis_center = direction.nlerp(extrema_clamped_0, extrema_clamped_1, 0.5)

    for i in range(2):
        extrema_max_clamped_normal[i] = mathext.normalize_safe(
            major_axis_center
            + relative_length
            * ((extrema_clamped_0 if i == 0 else extrema_clamped_1) - major_axis_center)
        )
        # Maximum extrema clamped in slope space
        extrema_max_clamped_slope[i] = normal_to_slope(extrema_max_clamped_normal[i])
        # Maximum extrema clamped in reflected space
        extrema_max_clamped_reflected[i] = -ti.math.reflect(
            wo_e, extrema_max_clamped_normal[i]
        )

        # Get back to tangent space
        extrema_max_clamped_slope[i] = (
            ellipse_to_tangent @ extrema_max_clamped_slope[i].xy
        )
        extrema_max_clamped_normal[i] = ti.math.vec3(
            ellipse_to_tangent @ extrema_max_clamped_normal[i].xy,
            extrema_max_clamped_normal[i].z,
        )
        extrema_max_clamped_reflected[i] = ti.math.vec3(
            ellipse_to_tangent @ extrema_max_clamped_reflected[i].xy,
            extrema_max_clamped_reflected[i].z,
        )

    for i in range(4):
        # Get back to tangent space
        extrema_slope[i] = ellipse_to_tangent @ extrema_slope[i].xy
        extrema_normal[i] = ti.math.vec3(
            ellipse_to_tangent @ extrema_normal[i].xy, extrema_normal[i].z
        )
        extrema_reflected[i] = ti.math.vec3(
            ellipse_to_tangent @ extrema_reflected[i].xy, extrema_reflected[i].z
        )

    extrema_shifted[0] = ti.math.vec3(
        ellipse_to_tangent @ extrema_clamped_0_reflected.xy,
        extrema_clamped_0_reflected.z,
    )
    extrema_shifted[1] = ti.math.vec3(
        ellipse_to_tangent @ extrema_clamped_1_reflected.xy,
        extrema_clamped_1_reflected.z,
    )
    extrema_shifted[2] = ti.math.vec3(
        ellipse_to_tangent @ shifted_min_0_reflected.xy, shifted_min_0_reflected.z
    )
    extrema_shifted[3] = ti.math.vec3(
        ellipse_to_tangent @ shifted_min_1_reflected.xy, shifted_min_1_reflected.z
    )


@ti.func
def GGX_masking_lambda(sph_d: ti.math.vec2, cov_mat: ti.math.mat2) -> float:
    EPSILON = 1e-5
    d_theta = ti.math.clamp(sph_d[0], 0.0, ti.math.pi * 0.4999)
    d_phi = sph_d[1]
    NDF_param = cov_to_alpha(cov_mat)
    alpha = ti.math.vec2(NDF_param[0], NDF_param[1])
    correlation_factor = NDF_param[2]
    alpha_o = mathext.sqrt_safe(
        ti.cos(d_phi) ** 2 * alpha.x**2
        + ti.sin(d_phi) ** 2 * alpha.y**2
        + 2.0 * ti.cos(d_phi) * ti.sin(d_phi) * correlation_factor * alpha.x * alpha.y
    )
    alpha_o = ti.math.clamp(alpha_o, EPSILON, limits.f32_max)
    a = 1.0 / (alpha_o * ti.tan(d_theta))
    Lambda = (-1.0 + ti.sqrt(1.0 + 1.0 / a**2)) * 0.5

    if d_theta < EPSILON:
        Lambda = 0.0
    return Lambda


@ti.func
def GGX_masking_shadowing(
    wo: ti.math.vec3, wi: ti.math.vec3, cov_mat: ti.math.mat2
) -> float:
    sph_o = direction.cartesian_to_spherical(wo)
    sph_i = direction.cartesian_to_spherical(wi)

    lambda_o = GGX_masking_lambda(sph_o, cov_mat)
    lambda_i = GGX_masking_lambda(sph_i, cov_mat)

    return 1.0 / (1.0 + lambda_o + lambda_i)


@ti.func
def eval_BRDF(wo: ti.math.vec3, wi: ti.math.vec3, cov_mat: ti.math.mat2) -> float:
    value = 0.0

    wh = mathext.normalize_safe(wo + wi)

    D = GGX_eval_normal_space(wh, cov_mat)
    G = GGX_masking_shadowing(wo, wi, cov_mat)

    value = G * D / (4.0 * wo.z)

    # Local masking shadowing
    if ti.math.dot(wo, wh) <= 0.0 or ti.math.dot(wi, wh) <= 0.0:
        value = 0.0

    if wo.z <= 0.0 or wi.z <= 0.0 or wh.z <= 0.0:
        value = 0.0

    return value


@ti.kernel
def eval_BRDF_v(
    wo: ti.math.vec3,
    wi: ti.template(),
    cov_mat: ti.math.mat2,
    f_value: ti.template(),
):
    for i in wi:
        wi_i = wi[i]
        f_value[i] = eval_BRDF(wo, wi_i, cov_mat)
