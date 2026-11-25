"""Blender-side source audit and live render in an evaluator-owned directory.

No submission code is executed. Saved settings are recorded before bounding
samples. Oversized exports are skipped; an equivalent percentage representation
of 480 x 270 is never accidentally shrunk by the evaluator.
"""
import argparse
import hashlib
import json
import sys
import math
from pathlib import Path

import bpy
import numpy as np

CONTROL_DEFAULTS = {'Gain': 1.4, 'Near': 20, 'Far': 60, 'Epsilon': .08}
CONTROL_STRESS = {'Gain': .65, 'Near': 24, 'Far': 48, 'Epsilon': .015}


def delivery_interface(scene, root):
    """Audit the public cross-file/control interface, not its internal topology."""
    node = scene.node_tree.nodes.get('Delivery') if scene.node_tree else None
    valid = node is not None and node.type == 'GROUP' and node.node_tree is not None
    values = {}
    if valid:
        for name in CONTROL_DEFAULTS:
            item = node.inputs.get(name)
            if item is not None and item.type in {'VALUE', 'INT'} and not item.is_linked:
                values[name] = float(item.default_value)
    library = node.node_tree.library if valid else None
    linked = bool(library and library.filepath.startswith('//') and
                  Path(bpy.path.abspath(library.filepath)).resolve() == root / 'finish.blend')
    controls = sum(name in values and math.isclose(values[name], target, abs_tol=1e-5)
                   for name, target in CONTROL_DEFAULTS.items()) / len(CONTROL_DEFAULTS)
    return {'linked_library': linked, 'control_defaults_score': controls,
            'controllable': set(values) == set(CONTROL_DEFAULTS), 'values': values}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def rna_values(owner):
    result = {}
    for prop in owner.bl_rna.properties:
        if prop.identifier == 'rna_type' or prop.type not in {'BOOLEAN', 'INT', 'FLOAT', 'ENUM', 'STRING'}:
            continue
        value = getattr(owner, prop.identifier)
        if isinstance(value, set):
            value = sorted(value)
        result[prop.identifier] = list(value) if getattr(prop, 'is_array', False) else value
    return result


def node_signature(tree, ignore_names=False):
    if not tree:
        return None
    nodes, aliases = [], {}
    for node in tree.nodes:
        inputs = []
        for socket in node.inputs:
            value = getattr(socket, 'default_value', None)
            if value is not None and not isinstance(value, (int, float, bool, str)):
                value = list(value) if hasattr(value, '__iter__') else None
            inputs.append((socket.identifier, value))
        properties = rna_values(node)
        for key in ('location', 'width', 'height', 'dimensions', 'select', 'show_options',
                    'show_preview', 'hide', 'label', 'use_custom_color', 'color'):
            properties.pop(key, None)
        image = getattr(getattr(node, 'image', None), 'name', None)
        if ignore_names:
            properties.pop('name', None)
            # The mother shaders have distinct functional nodes except equivalent
            # UV coordinate sources. Content identities ignore presentation names
            # while retaining every node, socket and connection.
            alias = digest((node.bl_idname, properties, inputs, image))
            aliases[node] = alias
            nodes.append(alias)
        else:
            aliases[node] = node.name
            nodes.append((node.name, node.bl_idname, properties, inputs, image))
    links = sorted((aliases[link.from_node], link.from_socket.identifier,
                    aliases[link.to_node], link.to_socket.identifier) for link in tree.links)
    return (sorted(nodes), links)


def mesh_signature(mesh, include_surface=True):
    hasher = hashlib.sha256()
    for collection, attribute, width, dtype in [
        (mesh.vertices, 'co', 3, np.float32),
        (mesh.loops, 'vertex_index', 1, np.int32),
        (mesh.polygons, 'loop_total', 1, np.int32),
        (mesh.polygons, 'material_index', 1, np.int32),
        (mesh.polygons, 'use_smooth', 1, np.bool_),
    ]:
        if not include_surface and attribute == 'material_index':
            continue
        array = np.empty(len(collection) * width, dtype=dtype)
        collection.foreach_get(attribute, array)
        hasher.update(array.tobytes())
    for uv in mesh.uv_layers if include_surface else ():
        array = np.empty(len(uv.data) * 2, dtype=np.float32)
        uv.data.foreach_get('uv', array)
        hasher.update(uv.name.encode())
        hasher.update(array.tobytes())
    hasher.update(json.dumps([m.name if m else None for m in mesh.materials]).encode())
    return hasher.hexdigest()


def surface_snapshot(names, destination):
    """Measure native UV corners and face assignments before any rendering."""
    values = {}
    for name in names:
        mesh = bpy.data.meshes.get(name)
        if mesh is None or len(mesh.loops) > 2_000_000:
            continue
        indices = np.empty(len(mesh.polygons), dtype=np.int32)
        mesh.polygons.foreach_get('material_index', indices)
        values[name + '/material_index'] = indices
        values[name + '/material_palette'] = np.array(
            [material.name if material else '' for material in mesh.materials])
        values[name + '/geometry_digest'] = np.array(mesh_signature(mesh, include_surface=False))
        for layer in mesh.uv_layers:
            data = np.empty(len(mesh.loops) * 2, dtype=np.float32)
            layer.data.foreach_get('uv', data)
            values[name + '/' + layer.name] = data.reshape(-1, 2)
    np.savez_compressed(destination, **values)


def collection_state(collection):
    return {'name': collection.name, 'exclude': collection.exclude,
            'holdout': collection.holdout, 'indirect_only': collection.indirect_only,
            'hide_viewport': collection.hide_viewport,
            'children': [collection_state(child) for child in collection.children]}


def source_signature(scene):
    """Semantic scene signatures omit compositor layout and export settings."""
    scene.frame_set(1)
    objects = {}
    for obj in bpy.data.objects:
        properties = {
            'type': obj.type, 'data': getattr(obj.data, 'name', None),
            'matrix': [list(row) for row in obj.matrix_world],
            'parent': getattr(obj.parent, 'name', None),
            'hide_render': obj.hide_render,
            'collections': sorted(c.name for c in obj.users_collection),
            'modifiers': [rna_values(mod) for mod in obj.modifiers],
            'constraints': [rna_values(con) for con in obj.constraints],
        }
        objects[obj.name] = digest(properties)
    actions = {}
    for action in bpy.data.actions:
        curves = []
        for curve in action.fcurves:
            curves.append((curve.data_path, curve.array_index, [
                (list(key.co), list(key.handle_left), list(key.handle_right), key.interpolation)
                for key in curve.keyframe_points]))
        actions[action.name] = digest(curves)
    images = {}
    for image in bpy.data.images:
        if image.source == 'FILE':
            images[image.name] = digest({
                'packed': hashlib.sha256(bytes(image.packed_file.data)).hexdigest()
                if image.packed_file else None,
                'colorspace': image.colorspace_settings.name,
            })
    return {
        'objects': objects,
        'meshes': {m.name: mesh_signature(m) for m in bpy.data.meshes},
        'materials': {m.name: digest(node_signature(m.node_tree, ignore_names=True)) for m in bpy.data.materials},
        'actions': actions, 'images': images,
        'lights': {light.name: digest(rna_values(light)) for light in bpy.data.lights},
        'cameras': {camera.name: digest(rna_values(camera)) for camera in bpy.data.cameras},
        'worlds': {world.name: digest(node_signature(world.node_tree)) for world in bpy.data.worlds},
        'layer_collections': {layer.name: digest(collection_state(layer.layer_collection))
                              for layer in scene.view_layers},
        'timeline': digest([scene.frame_start, scene.frame_end, scene.render.fps,
                            scene.render.fps_base, getattr(scene.camera, 'name', None)]),
    }


def settings(scene):
    return {
        'version': bpy.app.version_string,
        'engine': scene.render.engine,
        'frame': scene.frame_current,
        'resolution': [scene.render.resolution_x, scene.render.resolution_y,
                       scene.render.resolution_percentage],
        'samples': scene.eevee.taa_render_samples,
        'motion_blur': scene.eevee.use_motion_blur,
        'motion_blur_shutter': scene.eevee.motion_blur_shutter,
        'motion_blur_parameters': {
            name: getattr(scene.eevee, name) for name in (
                'motion_blur_position', 'motion_blur_depth_scale',
                'motion_blur_max', 'motion_blur_steps')},
        'view_transform': scene.view_settings.view_transform,
        'look': scene.view_settings.look,
        'exposure': scene.view_settings.exposure,
        'gamma': scene.view_settings.gamma,
        'display': scene.display_settings.display_device,
        'compositing': scene.render.use_compositing and scene.use_nodes,
        'sequencer': scene.render.use_sequencer,
        'transparent_film': scene.render.film_transparent,
        'pixel_aspect': [scene.render.pixel_aspect_x, scene.render.pixel_aspect_y],
        'layers': {layer.name: [layer.use, layer.use_pass_combined, layer.use_pass_z]
                   for layer in scene.view_layers},
    }


def compositor_instances(tree, parents=(), ancestors=frozenset()):
    """Walk instantiated native groups; retain context for Group Input tracing."""
    if tree is None or tree.as_pointer() in ancestors:
        return
    ancestors = ancestors | {tree.as_pointer()}
    for node in tree.nodes:
        yield node, parents
        if node.type == 'GROUP' and node.node_tree and not node.mute:
            yield from compositor_instances(node.node_tree, parents + (node,), ancestors)


def live_native_graph(scene):
    """Trace active export branches, including nested groups, without naming nodes."""
    tree = scene.node_tree
    if not tree:
        return False

    def sources(node, socket, parents=(), visited=frozenset()):
        key = (node.as_pointer(), socket.identifier, tuple(n.as_pointer() for n in parents))
        if key in visited:
            return set()
        visited = visited | {key}
        if node.type == 'R_LAYERS':
            valid = (node.scene is None or node.scene == scene) and node.layer in scene.view_layers
            return {'render'} if valid else {'invalid_layer'}
        if node.type == 'IMAGE':
            return {'baked_image'}
        if node.type == 'GROUP' and node.node_tree:
            output = next((n for n in node.node_tree.nodes
                           if n.type == 'GROUP_OUTPUT' and n.is_active_output), None)
            inputs = [s for s in output.inputs if s.identifier == socket.identifier] if output else []
            context = parents + (node,)
        elif node.type == 'GROUP_INPUT' and parents:
            inputs = [s for s in parents[-1].inputs if s.identifier == socket.identifier]
            context = parents[:-1]
        else:
            inputs = node.inputs
            context = parents
        result = set()
        for input_socket in inputs:
            for link in input_socket.links:
                result |= sources(link.from_node, link.from_socket, context, visited)
        return result

    sinks = [(node, parents) for node, parents in compositor_instances(tree)
             if node.type in {'COMPOSITE', 'OUTPUT_FILE'} and not node.mute]
    if not any(node.type == 'COMPOSITE' for node, _ in sinks) or not any(node.type == 'OUTPUT_FILE' for node, _ in sinks):
        return False
    for node, parents in sinks:
        for socket in node.inputs:
            if not socket.is_linked:
                continue
            dependencies = set()
            for link in socket.links:
                dependencies |= sources(link.from_node, link.from_socket, parents)
            if dependencies != {'render'}:
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--stress', action='store_true')
    parser.add_argument('--surfaces', nargs='*', default=[])
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    scene = bpy.context.scene
    saved = settings(scene)
    signature = source_signature(scene)
    # All outputs must stay inside the relocated submission folder. Absolute
    # obsolete destinations receive a portability penalty and are redirected.
    root = Path(bpy.data.filepath).parent.resolve()
    portable = True
    interface = delivery_interface(scene, root)
    for library in bpy.data.libraries:
        path = Path(bpy.path.abspath(library.filepath)).resolve()
        if not library.filepath.startswith('//') or not path.is_relative_to(root) or not path.is_file():
            portable = False
    main_destination = Path(bpy.path.abspath(scene.render.filepath)).resolve()
    if not scene.render.filepath.startswith('//') or not main_destination.is_relative_to(root):
        portable = False
        scene.render.filepath = str(root / (Path(scene.render.filepath).name or 'render_result'))
    for node, _ in compositor_instances(scene.node_tree):
        if node.type != 'OUTPUT_FILE':
            continue
        destination = Path(bpy.path.abspath(node.base_path)).resolve()
        if not node.base_path.startswith('//') or not destination.is_relative_to(root):
            portable = False
            node.base_path = str(root / Path(node.base_path).name)
        for slot in node.file_slots:
            if Path(slot.path).is_absolute() or '..' in Path(slot.path).parts:
                portable = False
                slot.path = Path(slot.path).name
    report = {'saved_settings': saved, 'signature': signature, 'portable': portable,
              'delivery_interface': interface,
              'live_native_graph': live_native_graph(scene),
              'rendered': False, 'stress': args.stress}
    report_path = Path(args.report)
    if args.surfaces:
        surface_snapshot(args.surfaces, root / '__surfaces.npz')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    if args.render:
        if scene.render.engine != 'BLENDER_EEVEE':
            report['error'] = 'Live checks require the specified EEVEE engine'
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            return
        # Bound resource use by the effective pixel size, not the nominal size.
        # Do not repair dimensions, enable missing passes, or change compositing.
        percentage = scene.render.resolution_percentage
        width = scene.render.resolution_x * percentage // 100
        height = scene.render.resolution_y * percentage // 100
        if width > 480 or height > 270:
            report['error'] = 'Saved output dimensions exceed the frame-1 contract'
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            return
        scene.eevee.taa_render_samples = min(scene.eevee.taa_render_samples, 64)
        if args.stress:
            scene.view_settings.exposure = -0.75
            for light in bpy.data.lights:
                light.energy *= 1.3
            if interface['controllable']:
                for name, value in CONTROL_STRESS.items():
                    scene.node_tree.nodes['Delivery'].inputs[name].default_value = value
                report['controls_changed'] = CONTROL_STRESS
        # Support both a native File Output archive and the main multilayer
        # still output. Their channel suffixes can legitimately be .V or .Z.
        bpy.ops.render.render(write_still=True)
        # Verify the actual Composite result independently of File Output nodes.
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.image_settings.color_depth = '16'
        bpy.data.images['Render Result'].save_render(str(root / '__composite.png'), scene=scene)
        report['rendered'] = True
        report_path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
