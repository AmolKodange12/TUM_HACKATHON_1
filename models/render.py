"""Render one drone model in the sky, as a camera would see it.

  blender -b -P models/render.py -- models/delta_wing.glb out.png --heading 250 --roll 15 --distance 40

The camera sits --distance metres from the drone, on bearing --cam-az from it, looking
up at it by --cam-el degrees. Heading/pitch/roll use the truth.csv conventions (out/truth.csv).
"""
import argparse
import math
import sys

import bpy
from mathutils import Matrix, Vector

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--heading", type=float, default=0, help="nose direction, deg clockwise from north")
ap.add_argument("--pitch", type=float, default=0, help="deg, positive nose up")
ap.add_argument("--roll", type=float, default=0, help="deg, positive right side down")
ap.add_argument("--distance", type=float, default=10, help="camera to drone, m")
ap.add_argument("--cam-az", type=float, default=200, help="bearing from the drone to the camera, deg")
ap.add_argument("--cam-el", type=float, default=15, help="how far the camera looks up to the drone, deg")
ap.add_argument("--fov", type=float, default=30, help="horizontal field of view, deg")
ap.add_argument("--sun-el", type=float, default=30, help="sun elevation, deg (negative = night)")
ap.add_argument("--sun-az", type=float, default=150, help="sun bearing, deg")
ap.add_argument("--size", default="1280x720")
ap.add_argument("--samples", type=int, default=64)
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=args.model)

# Blender world: +X east, +Y north, +Z up; the model's nose is +Y and right side +X
drone = bpy.data.objects.new("drone", None)
scene.collection.objects.link(drone)
for ob in list(scene.objects):
    if ob.parent is None and ob is not drone:
        ob.parent = drone
drone.matrix_world = (Matrix.Rotation(math.radians(-args.heading), 4, "Z")
                      @ Matrix.Rotation(math.radians(args.pitch), 4, "X")
                      @ Matrix.Rotation(math.radians(args.roll), 4, "Y"))


def direction(az, el):
    az, el = math.radians(az), math.radians(el)
    return Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))


cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
scene.collection.objects.link(cam)
cam.location = args.distance * direction(args.cam_az, -args.cam_el)
cam.rotation_euler = (-cam.location).to_track_quat("-Z", "Y").to_euler()
cam.data.sensor_fit, cam.data.angle = "HORIZONTAL", math.radians(args.fov)
cam.data.clip_end = max(1000.0, args.distance * 2)
scene.camera = cam

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
scene.collection.objects.link(sun)
sun.rotation_euler = (-direction(args.sun_az, args.sun_el)).to_track_quat("-Z", "Y").to_euler()
sun.data.energy = 25.0 if args.sun_el > 0 else 0.02  # clear day: direct sun ~25x the sky radiance; night: moon-ish
sun.data.angle = math.radians(0.53)

world = bpy.data.worlds.new("sky")
scene.world = world
nodes = world.node_tree.nodes
sky = nodes.new("ShaderNodeTexSky")
sky.sky_type = "MULTIPLE_SCATTERING"
sky.sun_disc = False  # the sun lamp lights the drone; the sky only adds colour and fill
sky.sun_elevation = math.radians(max(args.sun_el, -10))
# ponytail: sky sun_rotation is matched by eye (0 = east, counter-clockwise), check it if the glow direction matters
sky.sun_rotation = math.radians(90 - args.sun_az)
bg = nodes["Background"]
bg.inputs["Strength"].default_value = 1.0 if args.sun_el > 0 else 0.05
world.node_tree.links.new(sky.outputs["Color"], bg.inputs["Color"])

scene.view_settings.exposure = -1.5 if args.sun_el > 0 else 0.0  # expose for the sky, like a real camera
scene.render.engine = "CYCLES"
scene.cycles.samples = args.samples
scene.cycles.use_denoising = True
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    scene.cycles.device = "GPU"
except Exception:  # no GPU: Cycles falls back to the CPU
    pass
w, h = map(int, args.size.split("x"))
scene.render.resolution_x, scene.render.resolution_y = w, h
scene.render.filepath = args.out
bpy.ops.render.render(write_still=True)
print("RENDERED", args.out)
