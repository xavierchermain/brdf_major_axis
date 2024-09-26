import taichi as ti
import numpy as np

class3set20 = [102.0, 194.0, 165.0]
class3set21 = [252.0, 141.0, 98.0]
class3set22 = [141.0, 160.0, 203.0]
class3set2 = np.array([class3set20, class3set21, class3set22]) / 255.0

class3dark20 = [27.0, 158.0, 119.0]
class3dark21 = [217.0, 95.0, 2.0]
class3dark22 = [117.0, 112.0, 179.0]
class3dark2 = np.array([class3dark20, class3dark21, class3dark22]) / 255.0


@ti.func
def turbo(t: ti.f32) -> ti.math.vec3:
    """
    Source
    ------
    https://www.shadertoy.com/view/3lBXR3
    """

    c0 = ti.math.vec3(0.1140890109226559, 0.06288340699912215, 0.2248337216805064)
    c1 = ti.math.vec3(6.716419496985708, 3.182286745507602, 7.571581586103393)
    c2 = ti.math.vec3(-66.09402360453038, -4.9279827041226, -10.09439367561635)
    c3 = ti.math.vec3(228.7660791526501, 25.04986699771073, -91.54105330182436)
    c4 = ti.math.vec3(-334.8351565777451, -69.31749712757485, 288.5858850615712)
    c5 = ti.math.vec3(218.7637218434795, 67.52150567819112, -305.2045772184957)
    c6 = ti.math.vec3(-52.88903478218835, -21.54527364654712, 110.5174647748972)

    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))


@ti.kernel
def turbo_v(scalar: ti.template(), scalar_max: float, color: ti.template()):
    for i in ti.grouped(scalar):
        scalar_normalized = scalar[i] / scalar_max
        color[i] = turbo(scalar_normalized)


@ti.func
def viridis(t: ti.f32) -> ti.math.vec3:
    """
    Source
    ------
    https://www.shadertoy.com/view/WlfXRN
    """
    c0 = ti.math.vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061)
    c1 = ti.math.vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685)
    c2 = ti.math.vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659)
    c3 = ti.math.vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987)
    c4 = ti.math.vec3(6.228269936347081, 14.17993336680509, 56.69055260068105)
    c5 = ti.math.vec3(4.776384997670288, -13.74514537774601, -65.35303263337234)
    c6 = ti.math.vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832)

    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))


@ti.kernel
def viridis_v(scalar: ti.template(), scalar_max: float, color: ti.template()):
    for i in ti.grouped(scalar):
        scalar_normalized = scalar[i] / scalar_max
        color[i] = viridis(scalar_normalized)
