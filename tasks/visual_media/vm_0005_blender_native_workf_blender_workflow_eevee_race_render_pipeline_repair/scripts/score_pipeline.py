"""Measure deliveries and two fresh Blender renders, with independent partial credit.

Invoked only by main.py.evaluate. The bundled OpenEXR wheel is extracted into
the evaluation scratch directory; no online install or system change is needed.
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

PNG_NAMES = ('beauty_0001.png', 'ship_matte_0001.png', 'depth_0001.png',
             'finish_0001.png', 'utility_0001.png')
PNG_COLORS = {'beauty_0001.png': 6, 'ship_matte_0001.png': 0,
              'depth_0001.png': 0, 'finish_0001.png': 6, 'utility_0001.png': 2}
WEIGHTS = {'scene_preservation': .05, 'surface_recovery': .10, 'material_recovery': .05,
           'saved_settings': .05, 'portability': .05,
           'linked_controls': .10, 'live_nominal': .15, 'live_stress': .25,
           'delivered_pngs': .10, 'delivered_exr': .10}


def png_array(path):
    with path.open('rb') as handle:
        header = handle.read(29)
    if header[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('Not a PNG')
    width, height, bits, color = struct.unpack('>IIBB', header[16:26])
    if (width, height) != (480, 270):
        raise ValueError('Expected 480 x 270')
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ValueError('Unreadable pixels')
    if array.ndim == 3:
        array = array[:, :, [2, 1, 0, 3]] if array.shape[2] == 4 else array[:, :, ::-1]
    maximum = 65535 if array.dtype == np.uint16 else 255
    return array.astype(np.float32) / maximum, bits, color


def png_score(actual, expected, grayscale=False, color_type=None, focus=None):
    try:
        a, bits, color = png_array(actual)
        b, _, _ = png_array(expected)
        expected_color = color_type if color_type is not None else (0 if grayscale else 6)
        schema = float(bits == 16 and color == expected_color)
        if a.shape != b.shape:
            return 0.0, {'error': 'Channel layout differs'}
        error = float(np.mean(np.abs(a - b)))
        if focus is not None and np.any(focus):
            error = .5 * error + .5 * float(np.mean(np.abs(a[focus] - b[focus])))
        pixels = max(0.0, 1 - max(0.0, error - 3 / 65535) / .025)
        return .25 * schema + .75 * pixels, {'mean_error': error, 'format_ok': bool(schema)}
    except Exception as exc:
        return 0.0, {'error': str(exc)}


def exr_arrays(path):
    import OpenEXR
    import Imath
    image = OpenEXR.InputFile(str(path))
    try:
        header = image.header()
        window = header['dataWindow']
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        if (width, height) != (480, 270):
            raise ValueError('Expected 480 x 270 EXR')
        channels = header['channels']
        # PXR24 and DWA can reduce FLOAT precision even when the channel's
        # declared type remains FLOAT. Check storage as well as decoded pixels.
        lossless = str(header['compression']) in {
            'NO_COMPRESSION', 'RLE_COMPRESSION', 'ZIPS_COMPRESSION',
            'ZIP_COMPRESSION', 'PIZ_COMPRESSION',
            'B44_COMPRESSION', 'B44A_COMPRESSION'}
        result, float_types = {}, {'_lossless_compression': lossless}
        for layer in ('All', 'Ship', 'Set', 'Finish'):
            suffixes = ['Combined.' + c for c in 'RGBA']
            if layer != 'Finish':
                suffixes.append('Depth')
            for suffix in suffixes:
                canonical = f'{layer}.{suffix}'
                candidates = [canonical] if suffix != 'Depth' else [canonical + '.V', canonical + '.Z']
                if layer == 'Finish':
                    candidates.append('Composite.' + suffix)
                name = next((name for name in candidates if name in channels), None)
                if name is None:
                    continue
                float_types[canonical] = channels[name].type == Imath.PixelType(Imath.PixelType.FLOAT)
                data = image.channel(name, Imath.PixelType(Imath.PixelType.FLOAT))
                values = np.frombuffer(data, dtype=np.float32).reshape(height, width)
                if not np.all(np.isfinite(values)):
                    raise ValueError('Non-finite pass values')
                result[canonical] = values
        return result, float_types
    finally:
        image.close()


def exr_score(actual, expected):
    try:
        a, types = exr_arrays(actual)
        b, _ = exr_arrays(expected)
        scores, details = {'raw': [], 'finish': []}, {}
        for name, target in b.items():
            group = scores['finish' if name.startswith('Finish.') else 'raw']
            if name not in a:
                group.append(0.0)
                details[name] = {'error': 'Missing channel'}
                continue
            # Relative error preserves HDR and large depth sentinels. It does
            # not clip highlights to display RGB as an 8-bit comparison would.
            difference = np.minimum(np.abs(a[name] - target) / (1 + np.abs(target)), 1)
            error = float(np.mean(difference))
            if name.startswith('Finish.Combined.') and not name.endswith('.A'):
                alpha = b['Ship.Combined.A']
                ship = alpha > 0
                edge = (alpha > 1e-5) & (alpha < .99999)
                error = (.25 * error + .50 * float(np.mean(difference[ship])) +
                         .25 * float(np.mean(difference[edge])))
            content = max(0.0, 1 - max(0.0, error - 2e-5) / .01)
            value = .10 * float(types[name]) + .10 * float(types['_lossless_compression']) + .80 * content
            group.append(value)
            details[name] = {'relative_mean_error': error, 'float32': types[name],
                             'lossless_compression': types['_lossless_compression'], 'score': value}
        return float(np.mean([np.mean(group) for group in scores.values() if group])), details
    except Exception as exc:
        return 0.0, {'error': str(exc)}


def exported_pngs(directory, reference):
    values, details = [], {}
    utility, _, _ = png_array(reference / 'utility_0001.png')
    focus = utility[:, :, 1] > 0
    occluded = utility[:, :, 1] - utility[:, :, 0] > 1 / 65535
    for name in PNG_NAMES:
        region = occluded if name == 'utility_0001.png' else focus if name == 'finish_0001.png' else None
        value, detail = png_score(directory / name, reference / name,
                                  color_type=PNG_COLORS[name],
                                  focus=region)
        values.append(value)
        details[name] = detail
    return float(np.mean(values)), details


def signature_score(actual, expected, equivalent_meshes=()):
    groups = []
    for name, target in expected.items():
        value = actual.get(name)
        if isinstance(target, dict):
            value = value if isinstance(value, dict) else {}
            union = set(value) | set(target)
            groups.append(sum(value.get(key) == target.get(key) or
                              (name == 'meshes' and key in equivalent_meshes)
                              for key in union) / max(1, len(union)))
        else:
            groups.append(float(value == target))
    return float(np.mean(groups))


def material_score(actual, expected, targets):
    """Independent credit for each recovered original shader definition."""
    details = {name: float(actual.get(name) == expected[name]) for name in targets}
    return float(np.mean(list(details.values()))) if details else 0.0, details


def settings_score(actual, expected):
    # Runtime version is reported as evidence, not used to award solver points.
    pairs = [(key, value) for key, value in expected.items() if key != 'version']
    return sum(actual.get(key) == value for key, value in pairs) / len(pairs)


def surface_score(actual, expected, targets):
    """Compare corner-domain data on the retained shot, with native-transfer tolerance."""
    try:
        with np.load(actual, allow_pickle=False) as submitted, np.load(expected, allow_pickle=False) as reference:
            scores, details = [], {}
            for mesh, uv_names in targets.items():
                geometry_key = mesh + '/geometry_digest'
                geometry_ok = geometry_key in submitted and bool(submitted[geometry_key] == reference[geometry_key])
                details[mesh + '/geometry_preserved'] = geometry_ok
                fractions = []
                for name in (*uv_names, 'material_index'):
                    key = mesh + '/' + name
                    target = reference[key]
                    fraction = 0.0
                    if key in submitted and submitted[key].shape == target.shape:
                        data = submitted[key]
                        if name == 'material_index':
                            # Several genuine palettes contain duplicate slots.
                            # Grade assigned materials, rather than slot numbers.
                            palette_key = mesh + '/material_palette'
                            palette = submitted[palette_key]
                            target_names = reference[palette_key][target]
                            valid = (data >= 0) & (data < len(palette))
                            assigned = np.full(data.shape, '', dtype=np.result_type(target_names.dtype, palette.dtype))
                            assigned[valid] = palette[data[valid]]
                            fraction = float(np.mean(valid & (assigned == target_names)))
                        else:
                            correct = np.all(np.isfinite(data) & (np.abs(data - target) <= 1e-4), axis=1)
                            fraction = float(np.mean(correct))
                    fractions.append(fraction)
                    details[key] = fraction
                scores.append((.75 * float(np.mean(fractions[:-1])) + .25 * fractions[-1]) if geometry_ok else 0)
            return float(np.mean(scores)), details
    except Exception as exc:
        return 0.0, {'error': str(exc)}


def render_submission(blend, directory, probe, stress, surfaces):
    directory.mkdir()
    relocated = directory / 'repaired.blend'
    shutil.copyfile(blend, relocated)
    library = blend.with_name('finish.blend')
    if library.is_file() and library.stat().st_size <= 128 * 1024 * 1024:
        shutil.copyfile(library, directory / 'finish.blend')
    report = directory / 'source_audit.json'
    command = ['blender', '-b', '--factory-startup', '--disable-autoexec', '-t', '8',
               str(relocated), '--python-exit-code', '1', '--python', str(probe),
               '--', '--report', str(report), '--render']
    if surfaces:
        command.extend(['--surfaces', *surfaces])
    if stress:
        command.append('--stress')
    env = os.environ.copy()
    env.update(EGL_PLATFORM='surfaceless', LIBGL_ALWAYS_SOFTWARE='1',
               LP_NUM_THREADS='8', OMP_NUM_THREADS='8', DISPLAY='')
    try:
        with (directory / 'blender.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=600, env=env)
        status = {'exit_code': result.returncode}
    except subprocess.TimeoutExpired:
        status = {'exit_code': 124, 'error': 'Render exceeded 600 seconds'}
    if report.exists():
        status.update(json.loads(report.read_text()))
    return status


def live_score(directory, reference):
    pngs, png_details = exported_pngs(directory, reference)
    exr, exr_details = exr_score(directory / 'passes_0001.exr', reference / 'passes_0001.exr')
    composite, composite_details = png_score(directory / '__composite.png', reference / 'finish_0001.png')
    return .30 * pngs + .35 * exr + .35 * composite, {
        'pngs': png_details, 'passes': exr_details, 'composite': composite_details}


def evaluate(submission, reference, scratch):
    with zipfile.ZipFile(next(reference.glob('openexr-*.whl'))) as archive:
        archive.extractall(scratch / 'dependencies')
    sys.path.insert(0, str(scratch / 'dependencies'))
    scores = dict.fromkeys(WEIGHTS, 0.0)
    details = {}
    scores['delivered_pngs'], details['delivered_pngs'] = exported_pngs(submission, reference / 'nominal')
    scores['delivered_exr'], details['delivered_exr'] = exr_score(
        submission / 'passes_0001.exr', reference / 'nominal/passes_0001.exr')
    blend = submission / 'repaired.blend'
    if blend.is_file() and blend.stat().st_size <= 128 * 1024 * 1024:
        expected = json.loads((reference / 'source_contract.json').read_text())
        probe = Path(__file__).with_name('probe_pipeline.py')
        for variant in ('nominal', 'stress'):
            directory = scratch / variant
            status = render_submission(blend, directory, probe, variant == 'stress', expected['surface_targets'])
            details[variant + '_audit'] = status
            if variant == 'nominal':
                scores['surface_recovery'], details['surface_recovery'] = surface_score(
                    directory / '__surfaces.npz', reference / 'surface_contract.npz', expected['surface_targets'])
                surface_details = details['surface_recovery']
                equivalents = [mesh for mesh, names in expected['surface_targets'].items()
                               if surface_details.get(mesh + '/geometry_preserved') and
                               all(surface_details.get(mesh + '/' + name) == 1
                                   for name in (*names, 'material_index'))]
                scores['scene_preservation'] = signature_score(
                    status.get('signature', {}), expected['signature'], equivalents)
                scores['material_recovery'], details['material_recovery'] = material_score(
                    status.get('signature', {}).get('materials', {}),
                    expected['signature']['materials'], expected['material_targets'])
                scores['saved_settings'] = settings_score(status.get('saved_settings', {}), expected['saved_settings'])
                scores['portability'] = float(status.get('portable', False))
                interface = status.get('delivery_interface', {})
                scores['linked_controls'] = .5 * float(interface.get('linked_library', False)) + \
                    .5 * interface.get('control_defaults_score', 0)
            controlled = variant == 'nominal' or bool(status.get('controls_changed'))
            if status.get('rendered') and status['exit_code'] == 0 and status.get('live_native_graph') and controlled:
                scores['live_' + variant], details['live_' + variant] = live_score(directory, reference / variant)
            else:
                details['live_' + variant] = {'error': 'No successful live native render', 'status': status.get('exit_code')}
    else:
        details['source_error'] = 'Missing repaired.blend or larger than 128 MiB'
    total = sum(scores[key] * weight for key, weight in WEIGHTS.items())
    return {'score': round(total, 6), 'components': scores, 'weights': WEIGHTS, 'details': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='render_checks_', dir=args.report.parent) as temporary:
        result = evaluate(args.submission.resolve(), args.reference.resolve(), Path(temporary))
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'score': result['score'], 'components': result['components']}))


if __name__ == '__main__':
    main()
