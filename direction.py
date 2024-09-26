import taichi as ti

import mathext


@ti.func
def spherical_to_cartesian(d: ti.math.vec2) -> ti.math.vec3:
    theta = d[0]
    phi = d[1]
    return ti.math.vec3(
        ti.math.cos(phi) * ti.math.sin(theta),
        ti.math.sin(phi) * ti.math.sin(theta),
        ti.math.cos(theta),
    )


@ti.func
def cartesian_to_spherical(d: ti.math.vec3) -> ti.math.vec2:
    return ti.math.vec2(mathext.acos_safe(d.z), ti.math.atan2(d.y, d.x))


@ti.kernel
def spherical_to_cartesian_kernel(d: ti.math.vec2) -> ti.math.vec3:
    return spherical_to_cartesian(d)


@ti.kernel
def reflect_kernel(w: ti.math.vec3, n: ti.math.vec3) -> ti.math.vec3:
    return -ti.math.reflect(w, n)


@ti.kernel
def reflect_v(w: ti.math.vec3, n: ti.template(), wr: ti.template()):
    for i in n:
        wr[i] = -ti.math.reflect(w, n[i])


@ti.func
def nlerp(x, y, t):
    return mathext.normalize_safe(ti.math.mix(x, y, t))


@ti.kernel
def interpolate_using_nlerp(x: ti.math.vec3, y: ti.math.vec3, p: ti.template()):
    for i in p:
        t = i / (p.shape[0] - 1.0)
        p[i] = nlerp(x, y, t)
