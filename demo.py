from pathlib import Path

import cv2 as cv
import numpy as np
import taichi as ti
from PIL import Image

import atmos as at


try:
    ti.init(arch=ti.cuda)
except Exception:
    try:
        ti.init(arch=ti.vulkan)
    except Exception:
        ti.init(arch=ti.cpu)


# Load once, relative to this file so the demo works from any working directory.
with Image.open(Path(__file__).parent / "assets" / "earth.jpg") as image:
    texture_srgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
texture_linear = np.where(texture_srgb <= 0.04045, texture_srgb / 12.92,
                          ((texture_srgb + 0.055) / 1.055) ** 2.4)
earth_texture = ti.Vector.field(3, dtype=ti.f32, shape=texture_linear.shape[:2])
earth_texture.from_numpy(texture_linear)
with Image.open(Path(__file__).parent / "assets" / "clouds.jpg") as image:
    cloud_mask = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
cloud_texture = ti.field(dtype=ti.f32, shape=cloud_mask.shape)
cloud_texture.from_numpy(cloud_mask)
CLOUD_ALTITUDE_KM = 6.0
CLOUD_OPACITY = 0.9
CLOUD_ALBEDO = 0.8
EARTH_ROTATION_DEG = 0.0  # Longitude facing the default camera; north is +Y.

SUN_ANGLE_RAD = 0.2
SUN_DIRECTION = np.array([0, np.sin(SUN_ANGLE_RAD), np.cos(SUN_ANGLE_RAD)])

FOCAL_LENGTH_M = 50 * 0.001
SENSOR_WIDTH_M = 36 * 0.001
SENSOR_HEIGHT_M = 36 * 0.001
X_RESOLUTION = 1024
Y_RESOLUTION = 1024
FOCAL_LENGTH_PIXELS_X = FOCAL_LENGTH_M / (SENSOR_WIDTH_M / X_RESOLUTION)
FOCAL_LENGTH_PIXELS_Y = FOCAL_LENGTH_M / (SENSOR_HEIGHT_M / Y_RESOLUTION)

# +Y is up; the camera looks along its -Z axis.
CAMERA_Y_ANGLE_RAD = 3.14 / 2
CAMERA_X_ANGLE_RAD = 0
rotation_y = np.array([
    [np.cos(CAMERA_Y_ANGLE_RAD), 0, np.sin(CAMERA_Y_ANGLE_RAD)],
    [0, 1, 0],
    [-np.sin(CAMERA_Y_ANGLE_RAD), 0, np.cos(CAMERA_Y_ANGLE_RAD)],
])
rotation_x = np.array([
    [1, 0, 0],
    [0, np.cos(CAMERA_X_ANGLE_RAD), -np.sin(CAMERA_X_ANGLE_RAD)],
    [0, np.sin(CAMERA_X_ANGLE_RAD), np.cos(CAMERA_X_ANGLE_RAD)],
])
world_to_camera = rotation_x @ rotation_y
CAMERA_TO_WORLD = world_to_camera.T
CAMERA_POSITION_KM = np.array([-30000, 600, 500])


@ti.func
def sample_globe(texture: ti.template(), normal):
    longitude = ti.atan2(-normal.z, -normal.x) + EARTH_ROTATION_DEG * np.pi / 180.0
    u = (longitude / (2.0 * np.pi) + 0.5) % 1.0
    v = 0.5 - ti.asin(ti.math.clamp(normal.y, -1.0, 1.0)) / np.pi
    height, width = texture.shape
    x = u * width - 0.5
    y = ti.math.clamp(v * height - 0.5, 0.0, height - 1.0)
    x0, y0 = ti.cast(ti.floor(x), ti.i32), ti.cast(ti.floor(y), ti.i32)
    fx, fy = x - x0, y - y0
    # Bilinear filtering wraps across the date line and clamps at the poles.
    top = ti.math.mix(texture[y0, x0 % width], texture[y0, (x0 + 1) % width], fx)
    bottom = ti.math.mix(texture[ti.min(y0 + 1, height - 1), x0 % width],
                         texture[ti.min(y0 + 1, height - 1), (x0 + 1) % width], fx)
    return ti.math.mix(top, bottom, fy)


@ti.func
def textured_earth(pos, ray, sun_dir):
    color = at.vec3(0.0)
    entry, exit = at._sphere_interval(pos, ray, at.EARTH_RADIUS_KM)
    if 0.0 <= entry <= exit:
        normal = (pos + entry * ray).normalized()
        albedo = sample_globe(earth_texture, normal)
        color = albedo * ti.max(normal.dot(sun_dir), 0.0) / np.pi
    return color


@ti.func
def add_clouds(pos, ray, sun_dir, background, atmosphere=True):
    radius = at.EARTH_RADIUS_KM + CLOUD_ALTITUDE_KM
    entry, exit = at._sphere_interval(pos, ray, radius)
    distance = entry
    if entry < 0.0:
        distance = exit
    ground_entry, ground_exit = at._sphere_interval(pos, ray, at.EARTH_RADIUS_KM)
    ground_blocks = 0.0 <= ground_entry <= ground_exit and ground_entry < distance
    color = background
    if entry <= exit and distance >= 0.0 and not ground_blocks:
        point = pos + distance * ray
        normal = point.normalized()
        opacity = ti.math.clamp(sample_globe(cloud_texture, normal) * CLOUD_OPACITY, 0.0, 1.0)
        if opacity > 0.0:
            haze = at.vec3(0.0)
            view_transmittance = at.vec3(1.0)
            sunlight = at.vec3(1.0)
            if atmosphere:
                # Stop the foreground atmosphere at the cloud, preserving haze in front.
                haze, view_transmittance, _ = at._atmos(
                    pos, ray, sun_dir, distance)
                sunlight = at._sun_transmittance(point, sun_dir)
            reflected = CLOUD_ALBEDO * ti.max(normal.dot(sun_dir), 0.0) / np.pi
            color = ti.math.mix(background, haze + view_transmittance * sunlight * reflected, opacity)
    return color


@ti.kernel
def render_kernel(
    pos: at.vec3,
    sun_dir: at.vec3,
    camera_to_world: ti.math.mat3,
    x_res: ti.i32,
    y_res: ti.i32,
    focal_x: ti.f32,
    focal_y: ti.f32,
    textured: ti.i32,
    atmosphere: ti.i32,
    out_color: ti.types.ndarray(dtype=ti.f32, ndim=3),
    out_debug: ti.types.ndarray(dtype=ti.f32, ndim=3),
):
    for j, i in ti.ndrange(y_res, x_res):
        x = (ti.cast(i, ti.f32) - ti.cast(x_res, ti.f32) * 0.5) / focal_x
        y = (ti.cast(j, ti.f32) - ti.cast(y_res, ti.f32) * 0.5) / focal_y
        ray = at.vec3(x, y, 1.0).normalized()
        ray = camera_to_world @ ray * -1

        surface = at.vec3(0.0)
        if textured:
            surface = textured_earth(pos, ray, sun_dir)
        else:
            surface = at._earth(pos, ray, sun_dir)
        color = at.vec3(0.0)
        surface_transmittance = at.vec3(1.0)
        debug = at.vec3(0.0)
        if atmosphere:
            color, surface_transmittance, debug = at._atmos(pos, ray, sun_dir)
        else:
            debug = at._debug_color(pos, ray, sun_dir)

        linear_color = color + surface * surface_transmittance
        if textured:
            linear_color = add_clouds(pos, ray, sun_dir, linear_color, atmosphere)
        linear_color = ti.math.max(linear_color, 0.0)
        # PNG viewers expect sRGB, while light transport is computed in linear RGB.
        srgb = ti.select(linear_color <= 0.0031308, 12.92 * linear_color,
                         1.055 * linear_color ** (1.0 / 2.4) - 0.055)
        clamped_color = ti.math.clamp(srgb * 255, 0.0, 255.0)
        clamped_debug = ti.math.clamp(debug * 255, 0.0, 255.0)
        out_color[j, i, 0] = clamped_color.x
        out_color[j, i, 1] = clamped_color.y
        out_color[j, i, 2] = clamped_color.z
        out_debug[j, i, 0] = clamped_debug.x
        out_debug[j, i, 1] = clamped_debug.y
        out_debug[j, i, 2] = clamped_debug.z


def render_gpu(pos, sun_dir, *, textured=True, atmosphere=True):
    """Render with optional surface/cloud textures and atmospheric light transport."""
    color = np.zeros((Y_RESOLUTION, X_RESOLUTION, 3), dtype=np.float32)
    debug = np.zeros((Y_RESOLUTION, X_RESOLUTION, 3), dtype=np.float32)
    render_kernel(
        at.vec3(*pos), at.vec3(*sun_dir),
        ti.math.mat3(CAMERA_TO_WORLD.tolist()),
        X_RESOLUTION, Y_RESOLUTION, FOCAL_LENGTH_PIXELS_X, FOCAL_LENGTH_PIXELS_Y,
        textured, atmosphere,
        color, debug,
    )
    return color.astype(np.uint8), debug.astype(np.uint8)

if __name__ == "__main__":
    variants = [
        ("output.png", CAMERA_POSITION_KM, True, True),
        ("output_no_atmosphere.png", CAMERA_POSITION_KM, True, False),
        ("output_plain_with_atmosphere.png", CAMERA_POSITION_KM, False, True),
        ("output_plain_no_atmosphere.png", CAMERA_POSITION_KM, False, False),
    ]
    for filename, pos, textured, atmosphere in variants:
        output, debug = render_gpu(pos, SUN_DIRECTION,
                                   textured=textured, atmosphere=atmosphere)
        # Apply the same display smoothing to every comparison.
        output = cv.GaussianBlur(output, (3, 3), 0)
        Image.fromarray(output).save(filename)
        if filename == "output.png":
            Image.fromarray(debug).save("debug.png")
        print(f"generated {filename}")
    print("generated debug.png")
