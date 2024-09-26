import taichi as ti


@ti.func
def apply_to_3dpoint(T: ti.math.mat4, p: ti.math.vec3) -> ti.math.vec3:
    p_homogeneous = ti.math.vec4(p, 1.0)
    p_transformed = T @ p_homogeneous
    p_transformed /= p_transformed[3]
    return p_transformed[:3]


@ti.kernel
def translate_v(p: ti.template(), t: ti.math.vec2):
    for i in p:
        p[i] = p[i] + t


@ti.func
def rotate_x(theta: float) -> ti.math.mat4:
    cos_theta = ti.cos(theta)
    sin_theta = ti.sin(theta)
    return ti.math.mat4(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_theta, -sin_theta, 0.0],
            [0.0, sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@ti.kernel
def rotate_x_v(theta: float, vertices: ti.template()):
    T = rotate_x(theta)
    for i in vertices:
        vertices[i] = apply_to_3dpoint(T, vertices[i])
