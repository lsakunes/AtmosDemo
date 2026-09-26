import math

import taichi as ti

vec3 = ti.math.vec3

EARTH_RADIUS_KM = 6371.0
ATMOSPHERE_THICKNESS_KM = 100.0
ATMOSPHERE_RADIUS_KM = EARTH_RADIUS_KM + ATMOSPHERE_THICKNESS_KM

# Earth-like optical properties, in inverse kilometres (RGB at 680, 550, 440 nm).
# Model and references: see README.md. Aerosol extinction includes absorption.
RAYLEIGH_SCATTERING = (0.005802, 0.013558, 0.033100)
MIE_SCATTERING = 0.003996
MIE_EXTINCTION = 0.004440
OZONE_ABSORPTION = (0.000650, 0.001881, 0.000085)
RAYLEIGH_SCALE_HEIGHT_KM = 8.0
MIE_SCALE_HEIGHT_KM = 1.2
MIE_G = 0.8
VIEW_SAMPLES = 128
LIGHT_SAMPLES = 64

@ti.func
def _convert_ray_to_sphere_space(origin: vec3, direction: vec3, width: ti.f32, height: ti.f32):
    w_over_h = width / height
    sphere_origin = vec3(origin.x, origin.y, origin.z * w_over_h)
    sphere_dir = vec3(direction.x, direction.y, direction.z * w_over_h)
    return sphere_origin, sphere_dir.normalized()


@ti.func
def _convert_pos_to_spheroid_space(pos: vec3, width: ti.f32, height: ti.f32) -> vec3:
    return vec3(pos.x, pos.y, pos.z / (width / height))


@ti.func
def cast_ray_against_oblate_spheroid(origin: vec3, direction: vec3, width: ti.f32, height: ti.f32):
    """Return forward intersections with a spheroid whose equatorial radius is width."""
    sphere_origin, sphere_dir = _convert_ray_to_sphere_space(origin, direction, width, height)

    a = sphere_dir.dot(sphere_dir)
    b = 2.0 * sphere_origin.dot(sphere_dir)
    c = sphere_origin.dot(sphere_origin) - width * width
    determinant = (-4.0 * c * a) + (b * b)

    first_pos = vec3(9.0, 9.0, 9.0)
    collides_first = False
    second_pos = vec3(0.0, 0.0, 0.0)
    collides_second = False

    if determinant >= 0.0:
        sqrt_det = ti.sqrt(determinant)
        two_a = 2.0 * a
        small_t = (-b - sqrt_det) / two_a
        large_t = (-b + sqrt_det) / two_a

        first_pos = vec3(0.0, 0.0, 0.0)
        second_pos = vec3(0.0, 0.0, 0.0)

        if small_t >= 0.0:
            hit = sphere_origin + sphere_dir * small_t
            first_pos = _convert_pos_to_spheroid_space(hit, width, height)
            collides_first = True
        if large_t >= 0.0:
            hit = sphere_origin + sphere_dir * large_t
            second_pos = _convert_pos_to_spheroid_space(hit, width, height)
            collides_second = True

    return first_pos, collides_first, second_pos, collides_second


@ti.func
def _debug_color(pos, ray, sun_dir):
    """Preserve the demo's original artistic debug image."""
    hit_pos, collided, _, _ = cast_ray_against_oblate_spheroid(
        pos, ray, ATMOSPHERE_RADIUS_KM, ATMOSPHERE_RADIUS_KM)
    color = vec3(0, 0, 0)
    if collided:
        hit_dir = hit_pos.normalized()
        color = ((hit_dir * 0.5 + vec3(0.5, 0.5, 0.8))
                 * max(hit_dir.dot(sun_dir), 0.05)
                 + _earth(pos, ray, sun_dir) * vec3(0.5, 0.5, 0.5)
                 + vec3(0.05, 0.02, 0.05))
    else:
        color = (ray - vec3(0.5, 0, 0)) * 0.5 + vec3(0.5, 0.5, 0.5)
    return color


@ti.func
def _sphere_interval(pos, direction, radius):
    """Signed entry/exit distances for a unit ray; entry > exit means a miss."""
    projection = pos.dot(direction)
    # Closest approach avoids subtracting two large squared distances at the limb.
    closest = pos - projection * direction
    discriminant = radius * radius - closest.dot(closest)
    entry, exit = 1.0, -1.0
    if discriminant >= 0.0:
        half_chord = ti.sqrt(discriminant)
        entry, exit = -projection - half_chord, -projection + half_chord
    return entry, exit


@ti.func
def _density(pos):
    """Relative Rayleigh, aerosol, and absorber densities at pos."""
    altitude = ti.max(pos.norm() - EARTH_RADIUS_KM, 0.0)
    rayleigh_height = RAYLEIGH_SCALE_HEIGHT_KM
    mie_height = MIE_SCALE_HEIGHT_KM
    # Ozone peaks at 25 km and tapers to zero at 10 and 40 km.
    absorber = ti.max(0.0, 1.0 - ti.abs(altitude - 25.0) / 15.0)
    return vec3(ti.exp(-altitude / rayleigh_height),
                ti.exp(-altitude / mie_height),
                absorber)


@ti.func
def _transmittance(density_integral):
    rayleigh = vec3(*RAYLEIGH_SCATTERING)
    absorber_coeff = vec3(*OZONE_ABSORPTION)
    mie_extinction = MIE_EXTINCTION
    optical_depth = (rayleigh * density_integral.x
                     + mie_extinction * density_integral.y
                     + absorber_coeff * density_integral.z)
    return ti.exp(-optical_depth)


@ti.func
def _sun_transmittance(pos, sun_dir):
    """Beer-Lambert attenuation toward the Sun, including the planet's shadow."""
    radius = EARTH_RADIUS_KM
    ground_entry, ground_exit = _sphere_interval(pos, sun_dir, radius)
    transmission = vec3(0.0)
    if not (ground_entry <= ground_exit and ground_exit > 0.0
            and pos.dot(sun_dir) < 0.0):
        _, distance = _sphere_interval(pos, sun_dir, radius + ATMOSPHERE_THICKNESS_KM)
        density_integral = vec3(0.0)
        # Quadratic spacing resolves the dense air near the starting point.
        for i in range(LIGHT_SAMPLES):
            start = distance * (i / LIGHT_SAMPLES) ** 2
            end = distance * ((i + 1) / LIGHT_SAMPLES) ** 2
            density_integral += _density(pos + sun_dir * (start + end) * 0.5) * (end - start)
        transmission = _transmittance(density_integral)
    return transmission


@ti.func
def _phase_functions(cos_theta):
    """Normalized Rayleigh and Cornette-Shanks aerosol phase functions."""
    g = MIE_G
    rayleigh = 3.0 * (1.0 + cos_theta * cos_theta) / (16.0 * math.pi)
    g2 = g * g
    mie = (3.0 * (1.0 - g2) * (1.0 + cos_theta * cos_theta)
           / (8.0 * math.pi * (2.0 + g2)
              * (1.0 + g2 - 2.0 * g * cos_theta) ** 1.5))
    return rayleigh, mie


@ti.func
def _atmos(pos, ray, sun_dir, max_distance=1e30):
    """Return single-scattered radiance, surface attenuation, and the art debug view.

    Coordinates are centred on Earth, in kilometres. Directions must be unit vectors;
    sun_dir points toward the Sun. Incident solar irradiance is normalized to one.
    A finite max_distance stops at an object and returns camera-path transmission
    alone, unless the ground is reached first.
    """
    radius = EARTH_RADIUS_KM
    entry, exit = _sphere_interval(pos, ray, radius + ATMOSPHERE_THICKNESS_KM)
    start = ti.max(entry, 0.0)
    ground_entry, ground_exit = _sphere_interval(pos, ray, radius)
    hits_ground = ground_entry <= ground_exit and 0.0 <= ground_entry <= max_distance
    exit = ti.min(exit, max_distance)
    if hits_ground:
        exit = ti.min(exit, ground_entry)

    radiance = vec3(0.0)
    surface_transmittance = vec3(1.0)
    if exit > start:
        rayleigh_phase, mie_phase = _phase_functions(
            ti.math.clamp(ray.dot(sun_dir), -1.0, 1.0))
        rayleigh_scattering = vec3(*RAYLEIGH_SCATTERING)
        mie_scattering = MIE_SCATTERING
        step = (exit - start) / VIEW_SAMPLES
        density_integral = vec3(0.0)
        for i in range(VIEW_SAMPLES):
            sample = pos + ray * (start + (i + 0.5) * step)
            density = _density(sample)
            # Evaluate camera attenuation at the midpoint, not the cell's end.
            view_transmittance = _transmittance(density_integral + density * (0.5 * step))
            scattering = (rayleigh_scattering * density.x * rayleigh_phase
                          + mie_scattering * density.y * mie_phase)
            radiance += (view_transmittance * _sun_transmittance(sample, sun_dir)
                         * scattering * step)
            density_integral += density * step
        surface_transmittance = _transmittance(density_integral)

    if hits_ground:
        surface = pos + ray * ground_entry
        surface_transmittance *= _sun_transmittance(surface, sun_dir)

    return radiance, surface_transmittance, _debug_color(pos, ray, sun_dir)


@ti.func
def _earth(pos, direction, sun_dir):
    """Simple surface shading for the artistic debug image."""
    surface_km, hit, _, _ = cast_ray_against_oblate_spheroid(
        pos, direction, EARTH_RADIUS_KM, EARTH_RADIUS_KM)
    color = vec3(0, 0, 0)
    if hit:
        surface_dir = surface_km.normalized()
        ndotl = max(surface_dir.dot(sun_dir), 0)
        color = vec3(0.7, 0.7, 1) * ndotl * 0.5
    return color
