"""Decode both streams and measure every frame of the submitted handoff."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

DEFAULT_PATCH = (82, 150, 112, 176)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stream_info(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)],
                            capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout)['streams']
    videos = [s for s in streams if s.get('codec_type') == 'video']
    return {'video_streams': len(videos), 'audio_streams': sum(s.get('codec_type') == 'audio' for s in streams),
            'codec_name': videos[0].get('codec_name') if videos else None,
            'pix_fmt': videos[0].get('pix_fmt') if videos else None}


def _run(source: Path, output: Path, patch: tuple[int, int, int, int]) -> dict:
    import cv2
    import numpy as np
    x1, y1, x2, y2 = patch
    if not (0 <= x1 < x2 <= 640 and 47 <= y1 < y2 <= 325 and
            8 <= x2 - x1 <= 100 and 8 <= y2 - y1 <= 100):
        return {'ok': False, 'error': 'invalid scene measurement patch'}
    src = cv2.VideoCapture(str(source)); dst = cv2.VideoCapture(str(output))
    if not src.isOpened() or not dst.isOpened():
        return {'ok': False, 'error': 'cannot decode source or submitted video'}
    info = stream_info(output)
    rows = []
    baseline = None
    i = 0
    while True:
        a, source_frame = src.read(); b, out_frame = dst.read()
        if not a and not b:
            break
        if a != b:
            return {'ok': False, 'error': 'source/output decoded frame count mismatch', 'frames_read': i}
        if source_frame.shape != out_frame.shape:
            return {'ok': False, 'error': 'source/output resolution mismatch', 'frames_read': i}
        source_patch = float(np.mean(source_frame[y1:y2, x1:x2]))
        corrected_patch = float(np.mean(out_frame[y1:y2, x1:x2]))
        if baseline is None:
            baseline = source_patch
        gain = baseline / max(source_patch, 1e-6)
        expected = np.clip(source_frame[:325].astype(np.float32) * gain, 0, 255)
        error = np.mean(np.abs(out_frame[:325].astype(np.float32) - expected), axis=2)
        # Every tile participates: retaining only a chip, header, or strip cannot pass.
        tile_means = [float(np.mean(error[yy:min(yy + 25, 325), xx:min(xx + 40, 640)]))
                      for yy in range(0, 325, 25) for xx in range(0, 640, 40)]
        strip_mae = float(np.mean(np.abs(out_frame[325:].astype(np.float32) - source_frame[325:].astype(np.float32))))
        rows.append({'frame': i, 'time_s': i / 24.0, 'source_patch_mean': source_patch,
                     'corrected_patch_mean': corrected_patch,
                     'relative_ev': math.log2(max(source_patch / baseline, 1e-9)),
                     'corrected_residual_ev': math.log2(max(corrected_patch / baseline, 1e-9)),
                     'source_frame_luma': float(np.mean(source_frame)),
                     'corrected_frame_luma': float(np.mean(out_frame)),
                     'scene_mae': float(np.mean(error)),
                     'scene_tile_fraction': sum(v <= 20.0 for v in tile_means) / len(tile_means),
                     'strip_mae': strip_mae})
        i += 1
    width = int(dst.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(dst.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(dst.get(cv2.CAP_PROP_FPS))
    src.release(); dst.release()
    if not rows:
        return {'ok': False, 'error': 'empty decoded video'}
    source_levels = [row['source_patch_mean'] for row in rows]
    if not (8.0 <= baseline <= 100.0 and max(source_levels) <= 245.0 and
            max(source_levels) / max(min(source_levels), 1e-6) >= 3.5):
        return {'ok': False, 'error': 'measurement patch does not track the source exposure'}
    return {'ok': True, 'width': width, 'height': height,
            'fps': fps, 'decoded_frames': i,
            'source_sha256': digest(source), 'output_sha256': digest(output),
            'patch_xyxy': list(patch), 'baseline_patch_mean': baseline,
            'stream': info, 'frames': rows}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True); p.add_argument('--output', required=True)
    p.add_argument('--result', required=True); p.add_argument('--patch', default=','.join(map(str, DEFAULT_PATCH)))
    args = p.parse_args()
    src = Path(args.source); out = Path(args.output); result = Path(args.result)
    try:
        patch = tuple(int(x) for x in args.patch.split(','))
        if len(patch) != 4:
            raise ValueError('patch needs four coordinates')
        import cv2  # noqa: F401
        payload = _run(src, out, patch)
    except ImportError:
        # Run local verification in the same locked image as ALE.
        common = Path(os.path.commonpath([str(src.resolve()), str(out.resolve()), str(result.resolve()), str(Path(__file__).resolve())]))
        def inside(path):
            return '/work/' + str(path.resolve().relative_to(common))
        command = ['docker', 'run', '--rm', '--user', f'{os.getuid()}:{os.getgid()}',
                   '-v', f'{common}:/work', '-w', '/work', '--entrypoint', 'python3',
                   'ale/base-visual-media:20260922-2', inside(Path(__file__)),
                   '--source', inside(src), '--output', inside(out), '--result', inside(result), '--patch', args.patch]
        return subprocess.run(command, check=False).returncode
    except Exception as exc:
        payload = {'ok': False, 'error': str(exc)}
    result.write_text(json.dumps(payload), encoding='utf-8')
    print(json.dumps({'ok': payload.get('ok'), 'decoded_frames': payload.get('decoded_frames'),
                      'error': payload.get('error')}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
