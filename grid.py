import taichi as ti


@ti.func
def cell_center_2dpoint(cell_2dindex: ti.math.ivec2, origin, cell_sides_length: float):
    ret = origin + cell_sides_length * (cell_2dindex + 0.5)
    return ret
