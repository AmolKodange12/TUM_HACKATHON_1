"""Build the drone models as GLB files:  blender -b -P models/build.py

Frame: metres, nose +Y, right +X, up +Z, origin near the centre of mass.
Propellers are separate objects named prop_* with their origin on the hub, so a
renderer can spin them. Sizes match drones.json; the build prints each bounding box.
"""
import math
import os

import bmesh
import bpy
from mathutils import Matrix, Vector

OUT = os.path.dirname(os.path.abspath(__file__))
MATERIALS = {  # name: (rgb, roughness, metallic)
    "dark_grey": ((0.10, 0.105, 0.11), 0.45, 0.0),
    "black": ((0.015, 0.015, 0.015), 0.5, 0.0),
    "glass": ((0.01, 0.015, 0.03), 0.05, 0.0),
    "metal": ((0.25, 0.25, 0.26), 0.35, 0.9),
    "light_grey": ((0.42, 0.45, 0.47), 0.55, 0.0),
    "night_grey": ((0.035, 0.037, 0.04), 0.7, 0.0),
    "foam": ((0.62, 0.62, 0.58), 0.85, 0.0),
}
MAT = {name: i for i, name in enumerate(MATERIALS)}


def T(x=0.0, y=0.0, z=0.0):
    return Matrix.Translation((x, y, z))


def R(deg, axis):
    return Matrix.Rotation(math.radians(deg), 4, axis)


# ---------- geometry builders: each returns a closed bmesh ----------

def _bridge(bm, a, b):
    if len(a) == 1:
        a, b = b, a
        for j in range(len(a)):
            bm.faces.new((a[j], b[0], a[(j + 1) % len(a)]))
    elif len(b) == 1:
        for j in range(len(a)):
            bm.faces.new((a[j], a[(j + 1) % len(a)], b[0]))
    else:
        for j in range(len(a)):
            k = (j + 1) % len(a)
            bm.faces.new((a[j], a[k], b[k], b[j]))


def _loft(bm, rings):
    for a, b in zip(rings, rings[1:]):
        _bridge(bm, a, b)
    for ring in (rings[0], rings[-1]):
        if len(ring) > 2:
            bm.faces.new(ring)


def body(stations, n=32, square=2.0):
    """Body along Y. stations: (y, half_width, half_height, z_centre) from nose to tail.
    square > 2 gives a rounded-square section (2 = ellipse)."""
    bm = bmesh.new()
    rings = []
    for y, rx, rz, zc in stations:
        if rx < 1e-6:
            rings.append([bm.verts.new((0, y, zc))])
            continue
        ring = []
        for i in range(n):
            a = 2 * math.pi * i / n
            c, s = math.cos(a), math.sin(a)
            ring.append(bm.verts.new((rx * math.copysign(abs(c) ** (2 / square), c), y,
                                      zc + rz * math.copysign(abs(s) ** (2 / square), s))))
        rings.append(ring)
    _loft(bm, rings)
    return bm


def _naca(t, n=20):
    """Symmetric NACA 00xx section, (x along chord 0..1, z) loop: upper LE->TE, lower TE->LE."""
    xs = [(1 - math.cos(math.pi * i / n)) / 2 for i in range(n + 1)]
    zs = [5 * t * (0.2969 * math.sqrt(x) - 0.126 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4) for x in xs]
    return list(zip(xs, zs)) + [(x, -z) for x, z in zip(xs[:0:-1], zs[:0:-1])]


def surface(stations, t=0.12, axis="x", mirror=False):
    """Wing, fin or blade lofted through stations (span, le_y, chord, offset, twist_deg).
    axis 'x': span along +X, offset moves it up (dihedral); axis 'z': span along +Z, offset moves it in X."""
    bm = bmesh.new()
    sec = _naca(t)
    for side in (1, -1) if mirror else (1,):
        rings = []
        for s, le, c, off, tw in stations:
            ca, sa = math.cos(math.radians(tw)), math.sin(math.radians(tw))
            ring = []
            for x, z in sec:
                dy, dz = -(x - 0.25) * c, z * c  # about the quarter chord, +dy toward the nose
                dy, dz = dy * ca - dz * sa, dy * sa + dz * ca  # twist: positive = leading edge up
                y = le - 0.25 * c + dy
                co = (side * s, y, off + dz) if axis == "x" else (off + dz, y, s)
                ring.append(bm.verts.new(co))
            rings.append(ring)
        _loft(bm, rings)
    return bm


def box(sx, sy, sz, bevel=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
    if bevel:
        bmesh.ops.bevel(bm, geom=list(bm.edges) + list(bm.verts), offset=bevel, segments=4,
                        profile=0.5, affect="EDGES")
    return bm


def cylinder(r, h, n=24):  # along Z, centred
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=n, radius1=r, radius2=r, depth=h)
    return bm


def sphere(r):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=r)
    return bm


def blades(radius, chord, pitch_root, pitch_tip, n=2, hub=0.1):
    """n blades turning about +Z; chord and pitch taper from root to tip."""
    st = [(radius * hub, 0.25 * chord * 0.6, chord * 0.6, 0, pitch_root),
          (radius * 0.35, 0.25 * chord, chord, 0, (pitch_root + pitch_tip) / 2),
          (radius, 0.25 * chord * 0.45, chord * 0.45, 0, pitch_tip)]
    bm = bmesh.new()
    for k in range(n):
        merge(bm, surface(st, t=0.09), 0, R(360 * k / n, "Z"))
    return bm


# ---------- assembly ----------

def merge(dst, src, mat, matrix=None):
    if matrix is not None:
        bmesh.ops.transform(src, matrix=matrix, verts=src.verts)
    bmesh.ops.recalc_face_normals(src, faces=src.faces)
    src.verts.index_update()
    vs = [dst.verts.new(v.co) for v in src.verts]
    for f in src.faces:
        nf = dst.faces.new([vs[v.index] for v in f.verts])
        nf.material_index = mat if mat is not None else f.material_index
        nf.smooth = True
    src.free()


class Model:
    def __init__(self, name):
        self.name, self.frame, self.props = name, bmesh.new(), []

    def add(self, bm, mat, matrix=None):
        merge(self.frame, bm, MAT[mat], matrix)

    def prop(self, name, bm, mat, hub, axis_matrix=Matrix()):
        merge_bm = bmesh.new()
        merge(merge_bm, bm, MAT[mat], axis_matrix)
        self.props.append((name, merge_bm, hub))

    def export(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        mats = []
        for name, (rgb, rough, metal) in MATERIALS.items():
            m = bpy.data.materials.new(name)
            bsdf = m.node_tree.nodes["Principled BSDF"]
            bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
            bsdf.inputs["Roughness"].default_value = rough
            bsdf.inputs["Metallic"].default_value = metal
            mats.append(m)
        objs = [("airframe", self.frame, (0, 0, 0))] + self.props
        lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
        for name, bm, loc in objs:
            mesh = bpy.data.meshes.new(name)
            bm.to_mesh(mesh)
            bm.free()
            mesh.set_sharp_from_angle(angle=math.radians(40))
            for m in mats:
                mesh.materials.append(m)
            ob = bpy.data.objects.new(name, mesh)
            ob.location = loc
            bpy.context.scene.collection.objects.link(ob)
            for v in mesh.vertices:
                w = Vector(loc) + v.co
                lo, hi = Vector(map(min, lo, w)), Vector(map(max, hi, w))
        path = os.path.join(OUT, self.name + ".glb")
        bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", export_apply=True)
        d = hi - lo
        print(f"MODEL {self.name}: width (x) {d.x:.2f} m, length (y) {d.y:.2f} m, height (z) {d.z:.2f} m -> {path}")


# ---------- the four drones ----------

def quadcopter():
    m = Model("quadcopter")
    m.add(box(0.096, 0.205, 0.072, bevel=0.022), "dark_grey", T(0, 0.0, 0.005))
    m.add(box(0.07, 0.06, 0.02, bevel=0.008), "dark_grey", T(0, -0.04, 0.045))  # battery top
    m.add(box(0.05, 0.034, 0.044, bevel=0.01), "black", T(0, 0.118, -0.022))  # gimbal
    m.add(cylinder(0.014, 0.014), "glass", T(0, 0.137, -0.022) @ R(90, "X"))  # lens
    motors = {"FL": (-0.13, 0.09, 0.012), "FR": (0.13, 0.09, 0.012),
              "RL": (-0.115, -0.205, -0.008), "RR": (0.115, -0.205, -0.008)}
    for name, (x, y, z) in motors.items():
        root = Vector((math.copysign(0.042, x), 0.05 if y > 0 else -0.06, z))
        tip = Vector((x, y, z))
        d = tip - root
        m.add(box(0.017, d.length, 0.012, bevel=0.004), "dark_grey",
              T(*((root + tip) / 2)) @ R(math.degrees(math.atan2(-d.x, d.y)), "Z"))
        m.add(cylinder(0.016, 0.022), "metal", T(x, y, z + 0.016))
        m.add(cylinder(0.004, 0.03), "black", T(x, y, z - 0.02))  # leg
        spin = 1 if name in ("FL", "RR") else -1  # neighbours turn opposite ways
        m.prop(f"prop_{name}", blades(0.12, 0.026, 20 * spin, 8 * spin), "black", (x, y, z + 0.03))
    m.export()


def fixed_wing_recon():
    m = Model("fixed_wing_recon")
    m.add(body([(0.85, 0.06, 0.066, 0), (0.75, 0.1, 0.11, 0), (0.5, 0.11, 0.12, 0), (0.0, 0.11, 0.12, 0),
                (-0.4, 0.08, 0.09, 0.01), (-0.8, 0.035, 0.04, 0.02), (-0.88, 0.02, 0.022, 0.02)]),
          "light_grey")
    m.add(body([(0.92, 0, 0, 0), (0.85, 0.05, 0.05, 0)]), "black")  # spinner
    m.add(surface([(0, 0.25, 0.30, 0.115, 2), (1.55, 0.22, 0.20, 0.165, 0)], t=0.14, mirror=True), "light_grey")
    m.add(surface([(0, -0.62, 0.18, 0.03, 0), (0.42, -0.70, 0.13, 0.03, 0)], t=0.10, mirror=True), "light_grey")
    m.add(surface([(0.03, -0.60, 0.24, 0, 0), (0.32, -0.74, 0.14, 0, 0)], t=0.10, axis="z"), "light_grey")
    m.add(sphere(0.07), "black", T(0, 0.35, -0.13))  # camera ball
    m.add(cylinder(0.025, 0.012), "glass", T(0, 0.415, -0.14) @ R(90, "X"))
    m.prop("prop", blades(0.25, 0.045, 35, 15), "black", (0, 0.88, 0), R(-90, "X"))
    m.export()


def delta_wing():
    m = Model("delta_wing")
    m.add(body([(1.75, 0, 0, 0), (1.6, 0.08, 0.08, 0), (1.3, 0.16, 0.16, 0), (0.9, 0.2, 0.2, 0),
                (-1.0, 0.2, 0.2, 0), (-1.35, 0.17, 0.17, 0), (-1.55, 0.11, 0.11, 0), (-1.62, 0.06, 0.06, 0)]),
          "night_grey")
    m.add(body([(-1.62, 0.05, 0.05, 0), (-1.72, 0, 0, 0)]), "black")  # hub
    m.add(surface([(0, 0.75, 2.25, 0, 0), (0.18, 0.62, 2.10, 0, 0), (1.25, -0.85, 0.55, 0, 0)],
                  t=0.08, mirror=True), "night_grey")
    for x in (-1.25, 1.25):  # wingtip fins, mostly above the wing
        m.add(surface([(-0.18, -0.80, 0.50, x, 0), (0.0, -0.85, 0.55, x, 0), (0.42, -1.12, 0.32, x, 0)],
                      t=0.08, axis="z"), "night_grey")
    m.prop("prop", blades(0.36, 0.07, 30, 14), "black", (0, -1.66, 0), R(-90, "X"))
    m.export()


def decoy():
    m = Model("decoy")
    m.add(body([(1.1, 0, 0, 0), (1.0, 0.05, 0.05, 0), (0.8, 0.09, 0.09, 0), (-0.8, 0.09, 0.09, 0),
                (-0.95, 0.06, 0.06, 0), (-1.0, 0.04, 0.04, 0)], square=4), "foam")
    m.add(body([(-1.0, 0.03, 0.03, 0), (-1.08, 0, 0, 0)]), "black")
    m.add(surface([(0, 0.45, 1.40, 0, 0), (0.10, 0.38, 1.30, 0, 0), (1.2, -0.65, 0.35, 0, 0)],
                  t=0.06, mirror=True), "foam")
    for x in (-1.2, 1.2):
        m.add(surface([(-0.1, -0.65, 0.33, x, 0), (0.3, -0.85, 0.2, x, 0)], t=0.06, axis="z"), "foam")
    m.prop("prop", blades(0.25, 0.05, 30, 14), "black", (0, -1.04, 0), R(-90, "X"))
    m.export()


for build in (quadcopter, fixed_wing_recon, delta_wing, decoy):
    build()
