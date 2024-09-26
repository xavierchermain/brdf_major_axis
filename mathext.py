import taichi as ti


@ti.func
def atan(x):
    """
    atan is not available in the Taichi language.
    """
    atan_val = ti.math.atan2(x, 1.0)
    return atan_val


@ti.func
def acos_safe(x):
    x = ti.math.clamp(x, -1.0, 1.0)
    return ti.math.acos(x)


@ti.func
def sqrt_safe(x):
    x = ti.math.max(x, 0.0)
    return ti.math.sqrt(x)


@ti.func
def normalize_safe(x):
    x_normalized = ti.math.normalize(x)
    x_nan = ti.math.isnan(x_normalized)
    if x_nan.any():
        x_normalized.fill(0.0)
        x_normalized[0] = 1.0
    return x_normalized

@ti.kernel
def eval_max(scalar: ti.template()) -> float:
    max = 0.0
    for i in scalar:
        ti.atomic_max(max, scalar[i])
    return max
