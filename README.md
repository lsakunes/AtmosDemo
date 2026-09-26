# AtmosDemo

Template code for the 2026 HSL raytracing Dawg Daze event.

# Setup

From this directory, create a virtual environment and install the dependencies:

```sh
python -m venv .venv
.venv/bin/python -m pip install 'numpy>=2' taichi pillow opencv-python
```

# Running

Run `.venv/bin/python demo.py`. Each run writes five images to the current working
directory, using the same camera, sunlight, and display settings:

| File | Contents |
| --- | --- |
| `output.png` | Textured Earth and clouds with atmosphere |
| `output_no_atmosphere.png` | Textured Earth and clouds without atmosphere |
| `output_plain_with_atmosphere.png` | Plain surface, no clouds, with atmosphere |
| `output_plain_no_atmosphere.png` | Plain surface, no clouds, without atmosphere |
| `debug.png` | Original artistic debug view |

Without atmosphere means both scattering and atmospheric attenuation are disabled.

# Atmospheric model

`atmos.py` numerically integrates single scattering through a spherical atmosphere.
It uses Rayleigh scattering for molecules, the normalized Cornette–Shanks phase
function for aerosols, and Beer–Lambert extinction from molecules, aerosols, and
ozone. Both the camera and sunlight paths are attenuated; the planet blocks light
and terminates view rays at the ground. Cameras on or above the surface are supported,
including cameras inside the atmosphere. All distances are kilometres, and ray and
Sun directions must be normalized. The Sun direction points toward the Sun.

The transport equations follow [Bruneton's atmospheric scattering reference](https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html).
Optical coefficients and density profiles follow the Earth preset in
[Hillaire's reference implementation](https://github.com/sebh/UnrealEngineSkyAtmosphere/blob/master/Application/SkyAtmosphereCommon.cpp):
8 km molecular and 1.2 km aerosol scale heights, with a triangular ozone layer
between 10 and 40 km. The planet uses Earth's approximate mean radius, 6371 km,
and a 100 km atmosphere cutoff. These are representative clear-air parameters,
not a weather-specific atmosphere.

`VIEW_SAMPLES` (128) and `LIGHT_SAMPLES` (64) control integration quality and cost.
The implementation uses midpoint quadrature, with quadratic spacing along sunlight
paths to resolve dense air. `_atmos` returns linear scattered radiance, transmission,
and the original artistic debug view. For a ground hit, transmission includes both
Sun-to-ground and ground-to-camera attenuation; for a finite endpoint before the
ground, it includes only camera-path attenuation. The demo converts the combined
linear color to sRGB.

This is an RGB, single-scattering approximation, not the full multiple-scattering
algorithms from those references. It omits diffuse sky illumination of the ground,
multiple scattering (especially relevant at twilight), volumetric clouds, refraction, and the
finite solar disk. Solar irradiance is normalized to white with unit intensity;
the surface texture is treated as approximate diffuse albedo. Output is not calibrated
photometry or a full spectral simulation.

# Earth texture

`assets/earth.jpg` is NASA's December 2004 Blue Marble Next Generation map,
credited to Reto Stöckli, NASA Earth Observatory:
[collection](https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/),
[original JPEG](https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74218/world.200412.3x5400x2700.jpg).
The image is bundled locally; rendering needs no internet connection.

The demo maps longitude and latitude onto the sphere, filters the texture in linear
RGB, and applies diffuse sunlight and atmospheric attenuation. North points up.
Change `EARTH_ROTATION_DEG` in `demo.py` to choose the longitude facing the camera.
This satellite composite provides visual surface detail, not calibrated albedo or
ocean reflections. The artistic debug view retains the plain surface.

# Clouds

`assets/clouds.jpg` is the [NASA Blue Marble cloud composite](https://science.nasa.gov/earth/earth-observatory/the-blue-marble-true-color-global-imagery-at-1km-resolution/),
credited to NASA Goddard Space Flight Center and Reto Stöckli
([original JPEG](https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57747/cloud_combined_2048.jpg)).
It is used as an opacity mask on a spherical layer 6 km above the ground.
Sunlight and the atmosphere between the cloud and camera affect its color;
clouds obscure the surface and the atmosphere behind them.

Adjust `CLOUD_ALTITUDE_KM`, `CLOUD_OPACITY`, and `CLOUD_ALBEDO` in `demo.py`.
Set opacity to zero to disable clouds. This is a thin-layer visualization for
orbital views, without cloud volumes, ground shadows, or multiple scattering.
The mask's brightness approximates coverage, not measured optical thickness.
