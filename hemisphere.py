import taichi as ti

import direction


def create(segment_count: int, ring_count: int) -> tuple:
    # vertex and normal buffer allocation
    vertex_count = segment_count * (ring_count + 1)
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)
    normals = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)

    # index buffer allocation
    face_index_count = 6
    face_count = segment_count * ring_count
    index_count = face_count * face_index_count
    indices = ti.field(dtype=ti.i32, shape=index_count)
    _fill_buffers(segment_count, ring_count, vertices, indices, normals)
    return vertices, indices, normals


@ti.kernel
def _fill_buffers(
    segment_count: int,
    ring_count: int,
    vertices: ti.template(),
    indices: ti.template(),
    normals: ti.template(),
):
    for theta_i in range(ring_count + 1):
        theta = theta_i * ti.math.pi * 0.5 / ring_count
        # print(f"theta_i: {theta_i}")

        for phi_i in range(segment_count):
            phi = phi_i * 2.0 * ti.math.pi / segment_count

            dir = direction.spherical_to_cartesian(ti.math.vec2(theta, phi))

            buffer_index = theta_i * segment_count + phi_i
            # print(f"phi_i: {phi_i}")
            # print(f"buffer_index: {buffer_index}")
            vertices[buffer_index] = dir
            normals[buffer_index] = dir

            if theta_i < ring_count:
                index_0 = (theta_i + 0) * segment_count + (phi_i + 0)
                index_1 = (theta_i + 1) * segment_count + (phi_i + 0)
                index_2 = (theta_i + 0) * segment_count + (phi_i + 1)
                index_3 = (theta_i + 1) * segment_count + (phi_i + 1)

                indices[buffer_index * 6] = index_0
                indices[buffer_index * 6 + 1] = index_1
                indices[buffer_index * 6 + 2] = index_2
                indices[buffer_index * 6 + 3] = index_1
                indices[buffer_index * 6 + 4] = index_3
                indices[buffer_index * 6 + 5] = index_2
