python
import json
import os
from pathlib import Path
from pymol import cmd

output = Path(os.environ.get("PUBLICATION_STRUCTURE_OUTPUT", "generated/structures"))
output.mkdir(parents=True, exist_ok=True)
width, height = 1000, 700
cmd.reinitialize()
cmd.viewport(width, height)
cmd.set("ray_opaque_background", 1)
cmd.set("orthoscopic", 1)
cmd.set("cartoon_smooth_loops", 1)
cmd.set("cartoon_sampling", 10)
cmd.set("antialias", 2)
cmd.bg_color("white")
proteins = [
    ("oct4", "OCT4.pdb", [
        (136, 217, [0.105882, 0.286275, 0.396078]),
        (234, 286, [0.211765, 0.498039, 0.647059]),
    ]),
    ("med1", "MED1-alphafold.pdb", [
        (77, 232, [0.549020, 0.211765, 0.058824]),
        (252, 274, [0.701961, 0.301961, 0.094118]),
        (287, 349, [0.803922, 0.423529, 0.172549]),
        (361, 518, [0.886275, 0.580392, 0.313725]),
    ]),
]
views = {}
for name, filename, regions in proteins:
    cmd.load(str(Path(os.environ["PUBLICATION_INPUT_ROOT"]) / filename), name)
    cmd.hide("everything", name)
    cmd.show("cartoon", name)
    cmd.color("grey75", name)
    for index, (start, end, rgb) in enumerate(regions, 1):
        color = name + "_region_" + str(index)
        cmd.set_color(color, rgb)
        cmd.color(color, name + " and resi " + str(start) + "-" + str(end))
    cmd.orient(name)
    cmd.zoom(name, buffer=3, complete=1)
    views[name] = list(cmd.get_view())

# Preserve each protein's orientation and center, while fixing the same
# orthographic camera distance, field of view, clipping planes and viewport.
camera_distance = max(-view[11] for view in views.values())
half_depth = max(max(-view[11] - view[15], view[16] + view[11]) for view in views.values())
field_of_view = views["med1"][17]
metadata = {"orthoscopic": True, "image_size_px": [width, height], "views": {}}
for name, _, _ in proteins:
    cmd.disable("all")
    cmd.enable(name)
    view = views[name][:]
    view[9:12] = [0.0, 0.0, -camera_distance]
    view[15:18] = [camera_distance - half_depth, camera_distance + half_depth, field_of_view]
    cmd.set_view(view)
    cmd.set("orthoscopic", 1)
    metadata["views"][name] = list(cmd.get_view())
    cmd.ray(width, height)
    cmd.png(str(output / ("fig1_panel_b_" + name + "_pymol.png")), dpi=300)
metadata["orthoscopic"] = bool(cmd.get_setting_int("orthoscopic"))
(output / "fig1_panel_b_scale.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("Fig.1B shared camera distance (angstrom):", camera_distance)
python end
