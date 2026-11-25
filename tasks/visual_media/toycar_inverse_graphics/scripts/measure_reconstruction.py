"""Blender-side geometry measurement and held-out silhouette renderer."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--points", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--render-dir", required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for store in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for block in list(store):
            if block.users == 0:
                store.remove(block)


def import_model(path: str) -> list[bpy.types.Object]:
    suffix = Path(path).suffix.lower()
    if suffix in {".glb", ".gltf"}:
        result = bpy.ops.import_scene.gltf(filepath=path)
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            result = bpy.ops.wm.obj_import(filepath=path)
        else:
            result = bpy.ops.import_scene.obj(filepath=path)
    elif suffix == ".ply":
        result = bpy.ops.wm.ply_import(filepath=path)
    else:
        raise RuntimeError(f"unsupported model extension: {suffix}")
    if "FINISHED" not in result:
        raise RuntimeError(f"model import failed: {result}")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("import produced no mesh objects")
    return meshes


def evaluated_triangles(meshes: list[bpy.types.Object]) -> tuple[np.ndarray, dict]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    triangles: list[list[list[float]]] = []
    vertex_count = 0
    polygon_count = 0
    object_names = []
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        matrix = evaluated.matrix_world
        vertex_count += len(mesh.vertices)
        polygon_count += len(mesh.polygons)
        object_names.append(obj.name)
        for tri in mesh.loop_triangles:
            triangles.append([
                list(matrix @ mesh.vertices[index].co) for index in tri.vertices
            ])
        evaluated.to_mesh_clear()
    array = np.asarray(triangles, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] == 0:
        raise RuntimeError("scene has no evaluated triangles")
    flat = array.reshape((-1, 3))
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    metadata = {
        "mesh_object_count": len(meshes),
        "object_names": sorted(object_names),
        "vertex_count": int(vertex_count),
        "polygon_count": int(polygon_count),
        "triangle_count": int(len(array)),
        "bbox_min": mins.tolist(),
        "bbox_max": maxs.tolist(),
        "bbox_center": ((mins + maxs) * 0.5).tolist(),
        "bbox_extent": (maxs - mins).tolist(),
        "bbox_diagonal": float(np.linalg.norm(maxs - mins)),
    }
    return array, metadata


def sample_surface(triangles: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, float]:
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    valid = areas > 1e-16
    triangles = triangles[valid]
    areas = areas[valid]
    total_area = float(areas.sum())
    if not math.isfinite(total_area) or total_area <= 0.0:
        raise RuntimeError("scene has no non-degenerate triangle area")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(triangles), size=count, replace=True, p=areas / total_area)
    u = rng.random(count)
    v = rng.random(count)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    selected = triangles[chosen]
    points = selected[:, 0] + u[:, None] * (selected[:, 1] - selected[:, 0]) + v[:, None] * (selected[:, 2] - selected[:, 0])
    return points.astype(np.float32), total_area


def setup_render(contract: dict) -> bpy.types.Material:
    render = contract["render"]
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 16
    scene.render.resolution_x = int(render["resolution"][0])
    scene.render.resolution_y = int(render["resolution"][1])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    material = bpy.data.materials.new("EvaluationWhiteEmission")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs[0], output.inputs[0])
    bpy.context.view_layer.material_override = material
    # Material override alone can preserve the imported GLB's transparency
    # mode. Make silhouettes depend on geometry, never on candidate materials.
    for obj in scene.objects:
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(material)
    return material


def render_hidden_views(contract: dict, render_dir: Path) -> None:
    setup_render(contract)
    render_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    for spec in contract["hidden_views"]:
        data = bpy.data.cameras.new(spec["name"])
        data.type = "PERSP"
        data.lens = float(spec["lens_mm"])
        data.sensor_width = float(spec["sensor_width_mm"])
        data.sensor_height = float(spec["sensor_height_mm"])
        data.clip_start = float(spec["clip_start"])
        data.clip_end = float(spec["clip_end"])
        camera = bpy.data.objects.new(spec["name"], data)
        camera.matrix_world = Matrix(spec["matrix_world"])
        scene.collection.objects.link(camera)
        scene.camera = camera
        scene.render.filepath = str(render_dir / spec["mask_file"])
        bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    clear_scene()
    meshes = import_model(args.model)
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    triangles, metrics = evaluated_triangles(meshes)
    points, surface_area = sample_surface(
        triangles, int(contract["surface_sample_count"]), int(contract["surface_sample_seed"])
    )
    metrics["surface_area"] = surface_area
    np.savez_compressed(args.points, points=points)
    render_hidden_views(contract, Path(args.render_dir))
    Path(args.metrics).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
