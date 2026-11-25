"""Strict, appearance-aware scorer for the Khronos Boom Box repair task."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

COMPONENTS = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
TYPE_ARITY = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
TEXTURE_SLOTS = ("emissiveTexture", "normalTexture", "occlusionTexture", "baseColorTexture", "metallicRoughnessTexture")
MATERIAL_PARAM_FIELDS = (
    "baseColorFactor", "metallicFactor", "roughnessFactor", "emissiveFactor",
    "alphaMode", "alphaCutoff", "doubleSided",
)


class GLBError(ValueError):
    pass


def load_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise GLBError("not a GLB file")
    version, total = struct.unpack_from("<II", raw, 4)
    if version != 2 or total != len(raw):
        raise GLBError("invalid GLB header length/version")
    pos, js, blob = 12, None, b""
    while pos < total:
        if pos + 8 > total:
            raise GLBError("truncated GLB chunk header")
        length, kind = struct.unpack_from("<II", raw, pos)
        pos += 8
        if length % 4 or pos + length > total:
            raise GLBError("invalid GLB chunk length")
        chunk = raw[pos:pos + length]
        pos += length
        if kind == 0x4E4F534A:
            if js is not None:
                raise GLBError("multiple JSON chunks")
            js = json.loads(chunk.rstrip(b" \t\r\n\0").decode("utf-8"))
        elif kind == 0x004E4942:
            if blob:
                raise GLBError("multiple BIN chunks")
            blob = chunk
    if js is None or js.get("asset", {}).get("version") != "2.0":
        raise GLBError("missing glTF 2.0 JSON")
    return js, blob


def accessor_values(doc: dict, blob: bytes, index: int):
    acc = doc["accessors"][index]
    if "bufferView" not in acc:
        return []
    view = doc["bufferViews"][acc["bufferView"]]
    fmt, width = COMPONENTS[acc["componentType"]]
    arity = TYPE_ARITY[acc["type"]]
    elem = width * arity
    stride = int(view.get("byteStride", elem))
    base = int(view.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
    out = []
    for n in range(int(acc["count"])):
        off = base + n * stride
        vals = struct.unpack_from("<" + fmt * arity, blob, off)
        out.append(vals[0] if arity == 1 else vals)
    return out


def _png_info(payload: bytes):
    """Return dimensions only for a CRC-valid, complete embedded PNG."""
    if len(payload) < 33 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    import zlib
    pos, ihdr, idat, iend, width, height = 8, False, False, False, 0, 0
    while pos + 12 <= len(payload):
        size = struct.unpack_from(">I", payload, pos)[0]
        end = pos + 12 + size
        if end > len(payload):
            return None
        kind, body = payload[pos + 4:pos + 8], payload[pos + 8:pos + 8 + size]
        crc = struct.unpack_from(">I", payload, pos + 8 + size)[0]
        if zlib.crc32(kind + body) & 0xffffffff != crc:
            return None
        if kind == b"IHDR":
            if ihdr or size != 13:
                return None
            width, height = struct.unpack_from(">II", body, 0)
            ihdr = bool(width and height)
        elif kind == b"IDAT":
            idat = True
        elif kind == b"IEND":
            iend = True
            break
        pos = end
    return (width, height) if ihdr and idat and iend else None


def _validate_doc(doc: dict, blob: bytes) -> list[str]:
    errors = []
    buffers, views, accessors = doc.get("buffers") or [], doc.get("bufferViews") or [], doc.get("accessors") or []
    if len(buffers) > 1 or (buffers and int(buffers[0].get("byteLength", 0)) > len(blob)):
        errors.append("buffer byteLength exceeds BIN")
    for i, view in enumerate(views):
        start, size = int(view.get("byteOffset", 0)), int(view.get("byteLength", 0))
        if start < 0 or size < 0 or start + size > len(blob):
            errors.append(f"bufferView {i} outside BIN")
    for i, acc in enumerate(accessors):
        if acc.get("componentType") not in COMPONENTS or acc.get("type") not in TYPE_ARITY:
            errors.append(f"accessor {i} has unsupported type")
            continue
        try:
            view = views[int(acc["bufferView"])]
            _, width = COMPONENTS[acc["componentType"]]
            elem, stride = width * TYPE_ARITY[acc["type"]], int(view.get("byteStride", width * TYPE_ARITY[acc["type"]]))
            need = int(acc.get("byteOffset", 0)) + max(0, int(acc.get("count", 0)) - 1) * stride + elem
            if need > int(view.get("byteLength", 0)):
                errors.append(f"accessor {i} exceeds bufferView")
        except (KeyError, IndexError, TypeError, ValueError):
            errors.append(f"accessor {i} has invalid bufferView")
    for i, image in enumerate(doc.get("images") or []):
        try:
            view = views[int(image["bufferView"])]
            start, size = int(view.get("byteOffset", 0)), int(view.get("byteLength", 0))
            if _png_info(blob[start:start + size]) is None:
                errors.append(f"image {i} is not a valid embedded PNG")
        except (KeyError, IndexError, TypeError, ValueError):
            errors.append(f"image {i} is not embedded")
    textures = doc.get("textures") or []
    for i, tex in enumerate(textures):
        if not isinstance(tex.get("source"), int) or not 0 <= tex["source"] < len(doc.get("images") or []):
            errors.append(f"texture {i} has invalid image source")
        if "sampler" in tex and not 0 <= int(tex["sampler"]) < len(doc.get("samplers") or []):
            errors.append(f"texture {i} has invalid sampler")
    for i, mat in enumerate(doc.get("materials") or []):
        for slot in TEXTURE_SLOTS:
            obj = mat.get("pbrMetallicRoughness", {}).get(slot) if slot in ("baseColorTexture", "metallicRoughnessTexture") else mat.get(slot)
            if obj is not None and (not isinstance(obj.get("index"), int) or not 0 <= obj["index"] < len(textures)):
                errors.append(f"material {i} has invalid {slot}")
    for i, node in enumerate(doc.get("nodes") or []):
        for key in ("translation", "rotation", "scale", "matrix"):
            if key in node and (not isinstance(node[key], list) or any(not math.isfinite(float(x)) for x in node[key])):
                errors.append(f"node {i} has non-finite transform")
        if "mesh" in node and (not isinstance(node["mesh"], int) or not 0 <= node["mesh"] < len(doc.get("meshes") or [])):
            errors.append(f"node {i} has invalid mesh")
    for mi, mesh in enumerate(doc.get("meshes") or []):
        for pi, p in enumerate(mesh.get("primitives") or []):
            for attr, ai in p.get("attributes", {}).items():
                if not isinstance(ai, int) or not 0 <= ai < len(accessors):
                    errors.append(f"mesh {mi} primitive {pi} has invalid {attr}")
            if "indices" in p and (not isinstance(p["indices"], int) or not 0 <= p["indices"] < len(accessors)):
                errors.append(f"mesh {mi} primitive {pi} has invalid indices")
            if "material" in p and (not isinstance(p["material"], int) or not 0 <= p["material"] < len(doc.get("materials") or [])):
                errors.append(f"mesh {mi} primitive {pi} has invalid material")
    return errors


def primitive(doc: dict):
    meshes = doc.get("meshes") or []
    if not meshes or not meshes[0].get("primitives"):
        raise GLBError("no mesh primitive")
    return meshes[0]["primitives"][0]


def inspect(path: Path):
    doc, blob = load_glb(path)
    errors, prim = _validate_doc(doc, blob), primitive(doc)
    attrs = prim.get("attributes", {})
    values = {key: accessor_values(doc, blob, idx) for key, idx in attrs.items()}
    images = []
    for image in doc.get("images") or []:
        view = doc["bufferViews"][image["bufferView"]]
        start, size = int(view.get("byteOffset", 0)), int(view.get("byteLength", 0))
        payload = blob[start:start + size]
        images.append({"valid": _png_info(payload) is not None and len(payload) == size, "size": size,
                       "dims": _png_info(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    return {"doc": doc, "blob": blob, "prim": prim, "attrs": attrs, "values": values, "uv": values.get("TEXCOORD_0", []),
            "materials": doc.get("materials") or [], "textures": doc.get("textures") or [], "images": images, "errors": errors}


def finite_normalized(values):
    if not values:
        return False
    flat = []
    for item in values:
        vals = item if isinstance(item, tuple) else (item,)
        if any(not math.isfinite(float(x)) or float(x) < -1e-4 or float(x) > 1.0001 for x in vals):
            return False
        flat.extend(float(x) for x in vals)
    return len(set(round(x, 7) for x in flat)) > 1 and max(flat) - min(flat) > 1e-5


def compare_arrays(a, b, tol=1e-6):
    if len(a) != len(b):
        return 0.0
    if not a:
        return 1.0
    good = 0
    for x, y in zip(a, b):
        xv, yv = (x if isinstance(x, tuple) else (x,)), (y if isinstance(y, tuple) else (y,))
        if len(xv) == len(yv) and all(abs(float(p) - float(q)) <= tol for p, q in zip(xv, yv)):
            good += 1
    return good / len(a)


def _geometry_score(c, r):
    scores, cms, rms = [], c["doc"].get("meshes") or [], r["doc"].get("meshes") or []
    if len(cms) != len(rms):
        return 0.0
    for cm, rm in zip(cms, rms):
        if len(cm.get("primitives") or []) != len(rm.get("primitives") or []):
            return 0.0
        for cp, rp in zip(cm["primitives"], rm["primitives"]):
            for name in ("POSITION", "NORMAL"):
                if name not in cp.get("attributes", {}) or name not in rp.get("attributes", {}):
                    scores.append(0.0)
                else:
                    scores.append(compare_arrays(accessor_values(c["doc"], c["blob"], cp["attributes"][name]), accessor_values(r["doc"], r["blob"], rp["attributes"][name])))
            if "indices" in cp and "indices" in rp:
                scores.append(compare_arrays(accessor_values(c["doc"], c["blob"], cp["indices"]), accessor_values(r["doc"], r["blob"], rp["indices"]), 0.0))
            else:
                scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _transform_score(cdoc, rdoc):
    cn, rn = cdoc.get("nodes") or [], rdoc.get("nodes") or []
    if len(cn) != len(rn):
        return 0.0
    vals = []
    for a, b in zip(cn, rn):
        keys = (set(a) | set(b)) & {"matrix", "translation", "rotation", "scale"}
        def same(k):
            av, bv = a.get(k), b.get(k)
            if isinstance(av, list) and isinstance(bv, list) and len(av) == len(bv):
                return all(abs(float(x) - float(y)) <= 1e-5 for x, y in zip(av, bv))
            return av == bv
        vals.append(1.0 if a.get("mesh") == b.get("mesh") and all(same(k) for k in keys) else 0.0)
    return sum(vals) / len(vals) if vals else 1.0


def _slot_refs(info, material_index=None):
    if material_index is None:
        material_index = info["prim"].get("material")
    mats = info["materials"]
    if not isinstance(material_index, int) or not 0 <= material_index < len(mats):
        return {}
    refs = {}
    mat = mats[material_index]
    for slot in TEXTURE_SLOTS:
        obj = mat.get("pbrMetallicRoughness", {}).get(slot) if slot in ("baseColorTexture", "metallicRoughnessTexture") else mat.get(slot)
        if isinstance(obj, dict) and isinstance(obj.get("index"), int) and obj["index"] < len(info["textures"]):
            ti = obj["index"]
            refs[slot] = (ti, info["textures"][ti].get("source"))
    return refs


def _material_param_value(material: dict, field: str):
    """Return a material parameter with its glTF 2.0 default applied."""
    pbr = material.get("pbrMetallicRoughness") or {}
    defaults = {
        "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
        "metallicFactor": 1.0,
        "roughnessFactor": 1.0,
        "emissiveFactor": [0.0, 0.0, 0.0],
        "alphaMode": "OPAQUE",
        "alphaCutoff": 0.5,
        "doubleSided": False,
    }
    if field in ("baseColorFactor", "metallicFactor", "roughnessFactor"):
        return pbr.get(field, defaults[field])
    return material.get(field, defaults[field])


def _value_similarity(candidate, reference):
    """Compare scalar/vector/string material values with a small float tolerance."""
    if isinstance(reference, (list, tuple)):
        if not isinstance(candidate, (list, tuple)) or len(candidate) != len(reference):
            return 0.0
        try:
            delta = max(abs(float(a) - float(b)) for a, b in zip(candidate, reference))
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, 1.0 - delta / 0.05))
    if isinstance(reference, (int, float)) and not isinstance(reference, bool):
        try:
            return max(0.0, min(1.0, 1.0 - abs(float(candidate) - float(reference)) / 0.05))
        except (TypeError, ValueError):
            return 0.0
    return 1.0 if candidate == reference else 0.0


def _material_param_score(c, r):
    """Score effective core material parameters for corresponding primitives."""
    cms, rms = c["doc"].get("meshes") or [], r["doc"].get("meshes") or []
    if len(cms) != len(rms):
        return 0.0
    scores = []
    for cm, rm in zip(cms, rms):
        cps, rps = cm.get("primitives") or [], rm.get("primitives") or []
        if len(cps) != len(rps):
            return 0.0
        for cp, rp in zip(cps, rps):
            rmi, cmi = rp.get("material"), cp.get("material")
            if not isinstance(rmi, int) or not 0 <= rmi < len(r["materials"]):
                scores.append(1.0 if not isinstance(cmi, int) else 0.0)
                continue
            if not isinstance(cmi, int) or not 0 <= cmi < len(c["materials"]):
                scores.append(0.0)
                continue
            ref_mat, cand_mat = r["materials"][rmi], c["materials"][cmi]
            for field in MATERIAL_PARAM_FIELDS:
                # glTF ignores alphaCutoff unless the effective alpha mode is
                # MASK.  Do not penalize an equivalent asset for carrying a
                # different unused cutoff value.
                if field == "alphaCutoff":
                    cand_mode = _material_param_value(cand_mat, "alphaMode")
                    ref_mode = _material_param_value(ref_mat, "alphaMode")
                    if cand_mode != "MASK" or ref_mode != "MASK":
                        scores.append(1.0)
                        continue
                scores.append(_value_similarity(_material_param_value(cand_mat, field),
                                                _material_param_value(ref_mat, field)))
    return sum(scores) / len(scores) if scores else 0.0


def _material_score(c, r):
    if not c["materials"] or not r["materials"]:
        return 0.0
    cmi = c["prim"].get("material")
    assignment = 1.0 if isinstance(cmi, int) and 0 <= cmi < len(c["materials"]) else 0.0
    rr, cr = _slot_refs(r), _slot_refs(c)
    param_score = _material_param_score(c, r)
    if not rr:
        return assignment * param_score
    linked = sum(1 for k in rr if k in cr and isinstance(cr[k][1], int) and 0 <= cr[k][1] < len(c["images"]) and c["images"][cr[k][1]]["valid"])
    link_score = assignment * (0.35 + 0.65 * linked / len(rr))
    return 0.65 * link_score + 0.35 * param_score


def _resource_score(c, r):
    rr, cr = _slot_refs(r), _slot_refs(c)
    if not rr:
        return 0.0
    linked = hashes = 0
    for slot, (_, rsrc) in rr.items():
        if slot not in cr:
            continue
        _, csrc = cr[slot]
        if isinstance(csrc, int) and 0 <= csrc < len(c["images"]) and c["images"][csrc]["valid"]:
            linked += 1
            if 0 <= rsrc < len(r["images"]) and c["images"][csrc]["sha256"] == r["images"][rsrc]["sha256"]:
                hashes += 1
    return 0.65 * linked / len(rr) + 0.35 * hashes / len(rr)


_BLENDER_SCRIPT = r'''
import bpy, json, math, os, sys
from mathutils import Vector
def clear(): bpy.ops.wm.read_factory_settings(use_empty=True)
def look(obj, target): obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()
def setup(path, outdir):
    clear(); bpy.ops.import_scene.gltf(filepath=path)
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
    if not meshes: raise RuntimeError('asset has no mesh')
    corners=[o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo=Vector((min(v.x for v in corners),min(v.y for v in corners),min(v.z for v in corners))); hi=Vector((max(v.x for v in corners),max(v.y for v in corners),max(v.z for v in corners)))
    target=(lo+hi)*.5; radius=max((hi-lo).length*.5,.01); scene=bpy.context.scene
    try: scene.render.engine='BLENDER_EEVEE_NEXT'
    except: scene.render.engine='BLENDER_EEVEE'
    scene.render.resolution_x=256; scene.render.resolution_y=192; scene.render.resolution_percentage=100; scene.render.image_settings.file_format='PNG'
    if scene.world is None: scene.world=bpy.data.worlds.new('grader_world')
    scene.world.color=(.025,.025,.025)
    cam=bpy.data.objects.new('grader_camera',bpy.data.cameras.new('grader_camera')); scene.collection.objects.link(cam); scene.camera=cam; cam.data.lens=52
    for i,(az,el) in enumerate(((25,12),(145,18),(265,10))):
        a,e=math.radians(az),math.radians(el); cam.location=target+Vector((math.cos(a)*math.cos(e),math.sin(a)*math.cos(e),math.sin(e)))*(radius*3); look(cam,target)
        ld=bpy.data.lights.new('key', 'AREA'); ld.energy=900; ld.size=radius*2.5; key=bpy.data.objects.new('key',ld); scene.collection.objects.link(key); key.location=target+Vector((1.5*radius,-2*radius,3*radius)); look(key,target)
        fd=bpy.data.lights.new('fill', 'AREA'); fd.energy=350; fd.size=radius*3; fill=bpy.data.objects.new('fill',fd); scene.collection.objects.link(fill); fill.location=target+Vector((-2*radius,radius,radius)); look(fill,target)
        scene.render.filepath=os.path.join(outdir,'view_%d.png'%i); bpy.ops.render.render(write_still=True); bpy.data.objects.remove(key,do_unlink=True); bpy.data.objects.remove(fill,do_unlink=True)
def pixels(path):
    im=bpy.data.images.load(path,check_existing=False); p=list(im.pixels[:]); bpy.data.images.remove(im); return p
args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
cand,ref,out=args; cd,rd=os.path.join(out,'c'),os.path.join(out,'r'); os.makedirs(cd); os.makedirs(rd); setup(cand,cd); setup(ref,rd); scores=[]; channel_details=[]
for i in range(3):
    a,b=pixels(os.path.join(cd,'view_%d.png'%i)),pixels(os.path.join(rd,'view_%d.png'%i)); n=min(len(a),len(b))//4
    maes=[sum(abs(float(a[4*j+k])-float(b[4*j+k])) for j in range(n))/max(1,n) for k in range(4)]
    channel_scores=[max(0,min(1,1-m/.22)) for m in maes]
    scores.append(sum(channel_scores)/4); channel_details.append(channel_scores)
print(json.dumps({'views':scores,'appearance':sum(scores)/len(scores),
                  'channel_scores':channel_details}))
'''


def render_appearance(candidate: Path, reference: Path):
    blender = shutil.which("blender")
    if not blender:
        return None, "blender-unavailable"
    with tempfile.TemporaryDirectory(prefix="boombox-render-") as td:
        script = Path(td) / "render.py"
        script.write_text(_BLENDER_SCRIPT, encoding="utf-8")
        try:
            proc = subprocess.run([blender, "-b", "--factory-startup", "--python", str(script), "--", str(candidate), str(reference), td], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=150)
            if proc.returncode != 0:
                return None, "blender-failed: " + proc.stdout[-500:]
            line = next(x for x in reversed(proc.stdout.splitlines()) if x.startswith("{"))
            data = json.loads(line)
            return float(data["appearance"]), data
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, StopIteration) as exc:
            return None, "blender-error: " + str(exc)


def score(candidate: Path, reference: Path):
    try:
        c, r = inspect(candidate), inspect(reference)
    except Exception as exc:
        return {"score": 0.0, "error": str(exc)}
    if c["errors"]:
        return {"score": 0.0, "error": "invalid glTF: " + "; ".join(c["errors"][:4]), "validation_errors": len(c["errors"])}
    geometry = _geometry_score(c, r)
    uv, pos_count = c["uv"], len(c["values"].get("POSITION", []))
    uv_score = 1.0 if finite_normalized(uv) and len(uv) == pos_count else (0.5 if finite_normalized(uv) else 0.0)
    material, resources = _material_score(c, r), _resource_score(c, r)
    material_params = _material_param_score(c, r)
    scene = _transform_score(c["doc"], r["doc"])
    if len(c["doc"].get("meshes", [])) != len(r["doc"].get("meshes", [])): scene *= 0.5
    appearance, detail = render_appearance(candidate, reference)
    if appearance is None:
        appearance = resources
    else:
        # A blank/untextured asset can render identically to the reference
        # only because both renders contain the same background.  Gate render
        # credit on the material actually reaching valid embedded resources.
        appearance *= 0.5 * (material + resources)
    total = 0.25 * geometry + 0.15 * uv_score + 0.15 * material + 0.15 * resources + 0.10 * scene + 0.20 * appearance
    return {"score": max(0.0, min(1.0, total)), "geometry": geometry, "uv": uv_score, "material": material, "material_params": material_params, "resources": resources, "texture": resources, "scene": scene, "appearance": appearance, "appearance_detail": detail, "candidate_attributes": sorted(c["attrs"]), "candidate_images": len(c["images"]), "reference_images": len(r["images"]), "validation_errors": len(c["errors"])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--candidate", type=Path, required=True); ap.add_argument("--reference", type=Path, required=True); ap.add_argument("--result", type=Path, required=True); args = ap.parse_args()
    result = score(args.candidate, args.reference); args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True)); return 0 if "error" not in result else 1


if __name__ == "__main__": raise SystemExit(main())
