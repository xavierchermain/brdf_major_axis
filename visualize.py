import numpy as np
import taichi as ti

# Local library
import color
import direction
import ellipse
import hemisphere
import mathext
import microfacet
import transform

# Select your architecture
arch = ti.cpu
# arch = ti.gpu

ti.init(default_fp=ti.f32, arch=arch, kernel_profiler=False, debug=False)


def visualize():
    # BEGIN Parameters

    # BRDF major axis sample count
    N = 8

    # `segment_count` and `ring_count` define the resolution of the mesh
    # Higher values will result in a more detailed hemisphere.
    # Reduce these values to increase runtime performance

    # Number of segments (slices) around the hemisphere
    segment_count = 2**9
    # Number of rings from the center to the outer edge
    ring_count = 2**8

    # Number of vertices used to represent the ellipse, i.e., the confidence
    # region boundary of the GGX distribution. This value defines the
    # resolution of the ellipse geometry.
    ellipse_vertex_count = 32

    # The ellipse scaling factor. Used to increase or decrease the area of the
    # slope distribution confidence region
    k = 1.0

    # END Parameters

    # Create a mesh representing the hemisphere
    # Generate the hemisphere mesh vertices, indices for drawing, and normal vectors
    hemisphere_vertices, hemisphere_indices, hemisphere_normals = hemisphere.create(
        segment_count, ring_count
    )

    # Allocate a field to store per-vertex colors for the hemisphere mesh
    # The field has 3 components per vertex, representing RGB values
    hemisphere_per_vertex_color = ti.Vector.field(
        3, dtype=ti.f32, shape=hemisphere_vertices.shape[0]
    )

    # Create copies of the hemisphere vertices and normals for visualization
    # The copies will be re-oriented to point "up" (along the positive y-axis)
    hemisphere_vertices_top = ti.Vector.field(
        n=3, dtype=float, shape=hemisphere_vertices.shape
    )
    hemisphere_normals_T = ti.Vector.field(
        n=3, dtype=float, shape=hemisphere_vertices.shape
    )

    # Copy the original vertices and normals to the visualization fields
    hemisphere_vertices_top.copy_from(hemisphere_vertices)
    hemisphere_normals_T.copy_from(hemisphere_normals)

    # Rotate the hemisphere around the x-axis by -90 degrees (pi/2 radians)
    # This re-orients the hemisphere so that the flat side faces down
    # and the curved surface points upwards.
    transform.rotate_x_v(-ti.math.pi * 0.5, hemisphere_vertices_top)
    transform.rotate_x_v(-ti.math.pi * 0.5, hemisphere_normals_T)

    # Define the resolution for storing the density of the GGX slope distribution.
    # The GGX density will be represented as a 2D scalar field with dimensions (720 x 720).
    GGX_density_shape_0 = 720
    GGX_density_shape = tuple([GGX_density_shape_0] * 2)
    GGX_density = ti.field(dtype=float, shape=GGX_density_shape)

    # A 1D field that will store either the normal distribution densities (in normal space)
    # or the cosine-weighted BRDF values (in reflected space) for each hemisphere vertex.
    hemispherical_density = ti.field(dtype=float, shape=hemisphere_vertices.shape[0])

    # Fields for storing geometric and shading data for the ellipse vertices,
    # in the three different spaces.
    ellipse_vertex_slope = ti.Vector.field(n=2, dtype=float, shape=ellipse_vertex_count)
    ellipse_vertex_normal = ti.Vector.field(
        n=3, dtype=float, shape=ellipse_vertex_count
    )
    ellipse_vertex_reflected = ti.Vector.field(
        n=3, dtype=float, shape=ellipse_vertex_count
    )

    # The color assigned to the ellipse vertices, extracted from a predefined
    # color palette.
    ellipse_vertices_color = tuple(color.class3dark2[2])

    # Confidence region extrema allocation
    extrema_slope = ti.Vector.field(n=2, dtype=float, shape=4)
    extrema_max_clamped_slope = ti.Vector.field(n=2, dtype=float, shape=2)
    extrema_normal = ti.Vector.field(n=3, dtype=float, shape=4)
    extrema_max_clamped_normal = ti.Vector.field(n=3, dtype=float, shape=2)
    extrema_reflected = ti.Vector.field(n=3, dtype=float, shape=4)
    extrema_max_clamped_reflected = ti.Vector.field(n=3, dtype=float, shape=2)
    extrema_per_vertex_color = ti.Vector.field(n=3, dtype=ti.f32, shape=4)

    # Set colors for the closest and farthest extrema points.
    # The closest points (index 0 and 1) are colored green.
    extrema_per_vertex_color[0] = ti.math.vec3(color.class3dark2[0])
    extrema_per_vertex_color[1] = extrema_per_vertex_color[0]

    # The farthest points (index 2 and 3) are colored orange.
    extrema_per_vertex_color[2] = ti.math.vec3(color.class3dark2[1])
    extrema_per_vertex_color[3] = extrema_per_vertex_color[2]

    # Define the colors for the maximum clamped extrema and the major axis samples.
    # These are stored as tuples using a bright orange color.
    extrema_clamped_color = tuple(color.class3set2[1])
    major_axis_sample_color = tuple(color.class3set2[1])

    # Allocate fields for storing the colors of shifted extrema points.
    extrema_shifted_per_vertex_color = ti.Vector.field(n=3, dtype=ti.f32, shape=4)

    # Set colors for shifted extrema points.
    # The farther shifted points (index 0 and 1) are colored yellow.
    extrema_shifted_per_vertex_color[0] = ti.math.vec3(1, 1, 0)
    extrema_shifted_per_vertex_color[1] = extrema_shifted_per_vertex_color[0]

    # The closest shifted points (index 2 and 3) are colored magenta.
    extrema_shifted_per_vertex_color[2] = ti.math.vec3(1, 0, 1)
    extrema_shifted_per_vertex_color[3] = extrema_shifted_per_vertex_color[2]

    # Shifted extrema allocation
    extrema_shifted = ti.Vector.field(n=3, dtype=float, shape=4)

    # BRDF major axis sample allocation
    env_sample_slope = ti.Vector.field(n=2, dtype=float, shape=N)
    env_sample_normal = ti.Vector.field(n=3, dtype=float, shape=N)
    env_sample_reflected = ti.Vector.field(n=3, dtype=float, shape=N)

    # Roughness parameter in [0.045, 1]. See Lagarde and de Rousiers 2014
    # Also called linear roughness
    # It is the roughness exposed to the user.
    # We use the alpha_lin symbol in the article for this parameter
    alpha_lin = 0.52
    # Anisotropy value in (-1., 1.).
    # We use the Kulla and Conty 2017 parameterization.
    # We use the eta symbol for this parameter in the article.
    anisotropy = 0.445
    # Anisotropic orientation, in radians
    # In the article, it is U bar. It is illustrated in Fig. 5 of the article.
    anisotropic_orientation = 1.1

    # Ellipse radii
    r_x = None
    r_y = None

    # Anisotropic roughness deduced from alpha_lin and ansitropy
    alpha_lin_x = None
    alpha_lin_y = None

    # Outgoing light direction, a.k.a. view vector
    # Spherical parametrization
    theta_o = 0.722
    phi_o = 2.059

    # Vertices for outgoing and reflection directions
    wo_vertices = ti.Vector.field(n=3, dtype=float, shape=2)
    wr_vertices = ti.Vector.field(n=3, dtype=float, shape=2)

    # Vertices for surface tangent, bitangent and normal
    scale_axes = 3.0
    tangent_vertices = ti.Vector.field(n=3, dtype=float, shape=2)
    tangent_vertices[0] = ti.math.vec3(-1.0, 0.0, 0.0) * scale_axes
    tangent_vertices[1] = ti.math.vec3(1.0, 0.0, 0.0) * scale_axes
    transform.rotate_x_v(-ti.math.pi * 0.5, tangent_vertices)
    bitangent_vertices = ti.Vector.field(n=3, dtype=float, shape=2)
    bitangent_vertices[0] = ti.math.vec3(0.0, -1.0, 0.0) * scale_axes
    bitangent_vertices[1] = ti.math.vec3(0.0, 1.0, 0.0) * scale_axes
    transform.rotate_x_v(-ti.math.pi * 0.5, bitangent_vertices)
    normal_vertices = ti.Vector.field(n=3, dtype=float, shape=2)
    normal_vertices[0] = ti.math.vec3(0.0)
    normal_vertices[1] = ti.math.vec3(0.0, 0.0, 1.0) * 1.3
    transform.rotate_x_v(-ti.math.pi * 0.5, normal_vertices)

    # The image resolution is the shape of the GGX density shape.
    image_res = GGX_density_shape
    image = ti.Vector.field(n=3, dtype=float, shape=image_res)

    window = ti.ui.Window("BRDF Major Axis Sampling", image_res)
    canvas = window.get_canvas()
    scene = window.get_scene()
    gui = window.get_gui()
    camera = ti.ui.Camera()
    camera_position = np.array([2.0, 4.0, 2.0]) * 0.6
    camera.position(camera_position[0], camera_position[1], camera_position[2])
    camera.lookat(0.0, 0.1, 0.0)
    camera.fov(45)

    circle_radius = 0.01
    particle_radius = 0.01

    visualize_slope = 0
    visualize_normal = 0
    visualize_reflected = 1
    is_clicked_slope = 0
    is_clicked_normal = 0
    is_clicked_reflected = 0

    show_gui = True
    show_hemisphere = True
    show_lines_and_particles = True
    take_screenshot = False
    screenshot_counter = 0

    while window.running:
        canvas.set_background_color((1.0, 1.0, 1.0))
        scene.set_camera(camera)
        scene.ambient_light((1.0, 1.0, 1.0))

        # keyboard event processing
        if window.get_event(ti.ui.PRESS):
            if window.event.key == "g":
                show_gui = not show_gui
            if window.event.key == "h":
                show_hemisphere = not show_hemisphere
            if window.event.key == "p":
                show_lines_and_particles = not show_lines_and_particles
            if window.event.key == "i":
                take_screenshot = True

        if show_gui:
            with gui.sub_window("Parameters", 0.05, 0.05, 0.41, 0.35) as w:
                theta_o = w.slider_float("theta_o", theta_o, 0, ti.math.pi * 0.49)
                phi_o = w.slider_float("phi_o", phi_o, 0, 2.0 * ti.math.pi)
                alpha_lin = w.slider_float("alpha_lin", alpha_lin, 0.045, 1.0)
                anisotropy = w.slider_float("anisotropy", anisotropy, -0.95, 0.95)
                alpha_lin_x = alpha_lin * (1 + anisotropy)
                alpha_lin_y = alpha_lin * (1 - anisotropy)
                anisotropic_orientation = w.slider_float(
                    "theta", anisotropic_orientation, 0.0, ti.math.pi
                )
                k = w.slider_float("k", k, 0.25, 3.0)
                is_clicked_slope = gui.button("Slope space")
                is_clicked_normal = gui.button("Normal space")
                is_clicked_reflected = gui.button("Reflected space")

        if is_clicked_slope or is_clicked_normal or is_clicked_reflected:
            visualize_slope = is_clicked_slope
            visualize_normal = is_clicked_normal
            visualize_reflected = is_clicked_reflected

        camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.RMB)

        # Update view vector data
        wo = direction.spherical_to_cartesian_kernel(ti.math.vec2(theta_o, phi_o))
        wo_vertices[1] = wo * 2.0
        transform.rotate_x_v(-ti.math.pi * 0.5, wo_vertices)

        # Update reflected view vector data
        wr_vertices[1] = direction.reflect_kernel(wo, ti.math.vec3(0.0, 0.0, 1.0)) * 2.0
        transform.rotate_x_v(-ti.math.pi * 0.5, wr_vertices)

        cov_mat = microfacet.user_parameters_to_cov(
            alpha_lin_x, alpha_lin_y, anisotropic_orientation
        )
        r_x, r_y = microfacet.user_parameters_to_ellipse(alpha_lin_x, alpha_lin_y, k)

        # GGX confidence region in slope space
        ellipse.eval_parametric_v(
            r_x, r_y, anisotropic_orientation, ellipse_vertex_slope
        )
        # GGX confidence region in normal space
        microfacet.slope_to_normal_v(ellipse_vertex_slope, ellipse_vertex_normal)
        # GGX confidence region in reflected space
        direction.reflect_v(wo, ellipse_vertex_normal, ellipse_vertex_reflected)

        # Get all extrema, for all spaces
        microfacet.compute_extrema(
            r_x,
            r_y,
            anisotropic_orientation,
            wo,
            extrema_slope,
            extrema_max_clamped_slope,
            extrema_normal,
            extrema_max_clamped_normal,
            extrema_reflected,
            extrema_max_clamped_reflected,
            extrema_shifted,
        )
        # Get major axis samples from maximum extrema clamped
        direction.interpolate_using_nlerp(
            extrema_max_clamped_normal[0],
            extrema_max_clamped_normal[1],
            env_sample_normal,
        )
        # Convert to slope and reflected space
        microfacet.normal_to_slope_v(env_sample_normal, env_sample_slope)
        direction.reflect_v(wo, env_sample_normal, env_sample_reflected)

        if visualize_normal:
            if show_hemisphere:
                # Evaluation of the GGX distribution in the hemispherical space
                microfacet.GGX_eval_normal_space_v(
                    hemisphere_vertices, cov_mat, hemispherical_density
                )
                BRDF_max = mathext.eval_max(hemispherical_density)
                # Visualize GGX density
                color.viridis_v(
                    hemispherical_density, BRDF_max, hemisphere_per_vertex_color
                )
                scene.mesh(
                    hemisphere_vertices_top,
                    hemisphere_indices,
                    hemisphere_normals_T,
                    per_vertex_color=hemisphere_per_vertex_color,
                )
            if show_lines_and_particles:
                # Visualize ellipse vertices
                transform.rotate_x_v(-ti.math.pi * 0.5, ellipse_vertex_normal)
                scene.particles(
                    ellipse_vertex_normal,
                    radius=particle_radius,
                    color=ellipse_vertices_color,
                )
                # Visualize ellipse extrema
                transform.rotate_x_v(-ti.math.pi * 0.5, extrema_normal)
                scene.particles(
                    extrema_normal,
                    radius=particle_radius * 2,
                    per_vertex_color=extrema_per_vertex_color,
                )
                # Visualize maximum extrema clamped
                transform.rotate_x_v(-ti.math.pi * 0.5, extrema_max_clamped_normal)
                scene.particles(
                    extrema_max_clamped_normal,
                    radius=particle_radius * 2,
                    color=extrema_clamped_color,
                )
                # Visualize environment samples
                transform.rotate_x_v(-ti.math.pi * 0.5, env_sample_normal)
                scene.particles(
                    env_sample_normal,
                    radius=particle_radius * 2,
                    color=major_axis_sample_color,
                )
                scene.lines(tangent_vertices, width=5, color=(0.0, 0.0, 0.0))
                scene.lines(bitangent_vertices, width=5, color=(0.0, 0.0, 0.0))
                scene.lines(normal_vertices, width=5, color=(0.0, 0.0, 0.0))

        if visualize_reflected:
            if show_hemisphere:
                # Evaluation BRDF hemispherical space
                microfacet.eval_BRDF_v(
                    wo, hemisphere_vertices, cov_mat, hemispherical_density
                )
                BRDF_max = mathext.eval_max(hemispherical_density)
                # Visualize BRDF
                color.viridis_v(
                    hemispherical_density, BRDF_max, hemisphere_per_vertex_color
                )
                scene.mesh(
                    hemisphere_vertices_top,
                    hemisphere_indices,
                    hemisphere_normals_T,
                    per_vertex_color=hemisphere_per_vertex_color,
                )

            if show_lines_and_particles:
                # Visualize ellipse vertices
                transform.rotate_x_v(-ti.math.pi * 0.5, ellipse_vertex_reflected)
                scene.particles(
                    ellipse_vertex_reflected,
                    radius=particle_radius,
                    color=ellipse_vertices_color,
                )
                # Visualize ellipse extrema
                transform.rotate_x_v(-ti.math.pi * 0.5, extrema_reflected)
                scene.particles(
                    extrema_reflected,
                    radius=particle_radius * 2,
                    per_vertex_color=extrema_per_vertex_color,
                )
                transform.rotate_x_v(-ti.math.pi * 0.5, extrema_shifted)
                scene.particles(
                    extrema_shifted,
                    radius=particle_radius * 1.9,
                    per_vertex_color=extrema_shifted_per_vertex_color,
                )
                # Visualize maximum extrema shifted
                transform.rotate_x_v(-ti.math.pi * 0.5, extrema_max_clamped_reflected)
                scene.particles(
                    extrema_max_clamped_reflected,
                    radius=particle_radius * 2,
                    color=extrema_clamped_color,
                )
                # Visualize environment samples
                transform.rotate_x_v(-ti.math.pi * 0.5, env_sample_reflected)
                scene.particles(
                    env_sample_reflected,
                    radius=particle_radius * 2,
                    color=major_axis_sample_color,
                )

                scene.lines(wo_vertices, width=5, color=tuple(color.class3dark2[0]))
                scene.lines(wr_vertices, width=5, color=tuple(color.class3dark2[1]))
                scene.lines(tangent_vertices, width=5, color=(0.0, 0.0, 0.0))
                scene.lines(bitangent_vertices, width=5, color=(0.0, 0.0, 0.0))
                scene.lines(normal_vertices, width=5, color=(0.0, 0.0, 0.0))

        if visualize_slope:
            # Evaluation GGX slope space
            ggx_max = microfacet.GGX_eval_slope_kernel(ti.math.vec2(0.0), cov_mat)
            microfacet.GGX_eval_in_box(
                cov_mat,
                GGX_density,
                ti.math.vec2(0.0),
                ti.math.vec2(1.0),
            )
            color.viridis_v(GGX_density, ggx_max, image)
            canvas.set_image(image)

            if show_lines_and_particles:
                # Ellipse vertices
                transform.translate_v(ellipse_vertex_slope, ti.math.vec2(0.5))
                canvas.circles(
                    ellipse_vertex_slope, circle_radius, ellipse_vertices_color
                )
                # Ellipse extrema
                transform.translate_v(extrema_slope, ti.math.vec2(0.5))
                canvas.circles(
                    extrema_slope,
                    circle_radius * 2.0,
                    per_vertex_color=extrema_per_vertex_color,
                )
                # Visualize clamped maximum extrema
                transform.translate_v(extrema_max_clamped_slope, ti.math.vec2(0.5))
                canvas.circles(
                    extrema_max_clamped_slope,
                    circle_radius * 2.0,
                    color=extrema_clamped_color,
                )
                # Visualize environment samples
                transform.translate_v(env_sample_slope, ti.math.vec2(0.5))
                canvas.circles(
                    env_sample_slope,
                    circle_radius * 2.0,
                    color=major_axis_sample_color,
                )

        canvas.scene(scene)

        if take_screenshot:
            window.save_image(f"{screenshot_counter}.png")
            screenshot_counter += 1
            take_screenshot = False
        else:
            window.show()


if __name__ == "__main__":
    visualize()
