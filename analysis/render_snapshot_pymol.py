"""Headless PyMOL ray renderer; called with a JSON scene from the plotter.

All coordinates are trajectory beads in nm, transformed rigidly into the
existing snapshot camera. Every backbone bond is retained. DNA strands are
separate; the two strands are never joined at their ends.
"""

import json
import math
import os
import struct
from pathlib import Path

from pymol import cmd
from pymol.cgo import COLOR, CYLINDER, SPHERE


def rgb(hex_color):
    return [int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]


def render():
    scene_path = Path(os.environ["PUBLICATION_PYMOL_SCENE"])
    scene = json.loads(scene_path.read_text())
    points = list(struct.iter_unpack("<fff", scene_path.with_suffix(".bin").read_bytes()))
    arrays, offset = {}, 0
    for item in scene["chains"]:
        arrays[item["key"]] = points[offset:offset + item["n_beads"]]
        offset += item["n_beads"]
    cmd.reinitialize()
    cmd.bg_color("white")
    cmd.set("orthoscopic", 1)
    cmd.set("ray_opaque_background", 0 if scene.get("transparent_background", False) else 1)
    cmd.set("antialias", 2)
    cmd.set("ray_shadows", 1)
    cmd.set("ambient", 0.48)
    cmd.set("direct", 0.55)
    cmd.set("specular", 0.12)
    cmd.set("shininess", 25)
    cmd.set("depth_cue", 0)
    cmd.set("ray_trace_mode", 0)
    cmd.set("ray_trace_fog", 0)
    cmd.set("max_threads", 4)
    # Opaque light-gray context avoids transparency overlap and colored fog.
    # Ray-traced spheres/cylinders provide depth and local shading.
    objects = {}
    for item in scene["chains"]:
        xyz = arrays[item["key"]]
        color = rgb(item["color"])
        bucket = item["bucket"]
        obj = objects.setdefault(bucket, [])
        obj.extend([COLOR] + color)
        radius = item["radius"]
        for point in xyz:
            obj.extend([SPHERE] + list(point) + [radius])
        strands = [xyz[:len(xyz) // 2], xyz[len(xyz) // 2:]] if item["dna"] else [xyz]
        for strand in strands:
            for a, b in zip(strand[:-1], strand[1:]):
                obj.extend([CYLINDER] + list(a) + list(b)
                           + [radius * 0.55] + color + color)
    for name, obj in objects.items():
        cmd.load_cgo(obj, name)

    width, height = scene["pixels"]
    half_y = scene["half_span"][1]
    fov = 20.0
    distance = half_y / math.tan(math.radians(fov / 2))
    depth = max(abs(point[2]) for point in points) + 3.0
    cmd.viewport(width, height)
    cmd.set("field_of_view", fov)
    cmd.set_view([1., 0., 0., 0., 1., 0., 0., 0., 1.,
                  0., 0., -float(distance), 0., 0., 0.,
                  float(distance - depth), float(distance + depth), fov])
    cmd.ray(width, height)
    cmd.png(scene["output"], dpi=600)
    Path(scene["output"]).with_suffix(".view.json").write_text(
        json.dumps({"view": list(cmd.get_view()), "orthoscopic": True,
                    "half_span_nm": scene["half_span"], "pixels": [width, height],
                    "n_beads": sum(len(arrays[item["key"]]) for item in scene["chains"]),
                    "n_chains": len(scene["chains"]),
                    "source": {key: scene[key] for key in ("system", "replicate", "frame", "mode")}},
                   indent=2) + "\n")


render()
