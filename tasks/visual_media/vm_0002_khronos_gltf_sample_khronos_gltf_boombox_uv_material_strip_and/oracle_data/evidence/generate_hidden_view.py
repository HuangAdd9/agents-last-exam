import bpy
from mathutils import Vector

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath="/work/reference/boombox_intact.glb")
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
points = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
lo = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
hi = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
center = (lo + hi) * 0.5
extent = max((hi - lo).length, 1e-6)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = 640, 480
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
if scene.world is None:
    scene.world = bpy.data.worlds.new("HiddenViewWorld")
scene.world.color = (0.025, 0.025, 0.025)
cam_data = bpy.data.cameras.new("HiddenViewCamera")
cam = bpy.data.objects.new("HiddenViewCamera", cam_data)
scene.collection.objects.link(cam)
cam.location = center + Vector((-2.15 * extent, -1.25 * extent, 1.55 * extent))
cam_data.type = "ORTHO"
cam_data.ortho_scale = extent * 1.35
cam_data.clip_start = max(extent / 1000.0, 1e-5)
cam_data.clip_end = max(extent * 1000.0, 10.0)
cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
for name, loc, energy in (("Key", center + Vector((-1.5, -1.0, 2.5)) * extent, 2.0),
                          ("Fill", center + Vector((2.0, -0.5, 1.0)) * extent, 1.2),
                          ("Rim", center + Vector((0.5, 2.0, 1.5)) * extent, 1.6)):
    data = bpy.data.lights.new(name, type="AREA")
    data.energy, data.shape, data.size = energy, "DISK", max(extent * 2.0, 1e-4)
    light = bpy.data.objects.new(name, data)
    scene.collection.objects.link(light)
    light.location = loc
    light.rotation_euler = (center - loc).to_track_quat("-Z", "Y").to_euler()
scene.camera = cam
scene.render.filepath = "/work/reference/hidden_reference_view.png"
bpy.ops.render.render(write_still=True)
