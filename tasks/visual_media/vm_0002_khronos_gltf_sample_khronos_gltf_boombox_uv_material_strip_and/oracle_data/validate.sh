#!/usr/bin/env bash
set -u
TASK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${ALE_VISUAL_RUNTIME_IMAGE:-ale/base-visual-media:20260918-1}"
LOG="$TASK_DIR/oracle_data/evidence/runtime_validation.log"
mkdir -p "$TASK_DIR/oracle_data/evidence"
set +e
docker run --rm --user 0:0 -e ALE_VISUAL_RUNTIME_IMAGE="$IMAGE" -v "$TASK_DIR:/work" "$IMAGE" bash -lc '
  set -o pipefail
  echo "runtime_image=$ALE_VISUAL_RUNTIME_IMAGE"
  blender --version | head -n 1
  python3 --version
  python3 -B /work/oracle_data/professional_validation.py
' 2>&1 | tee "$LOG"
code=${PIPESTATUS[0]}
set -e
cat > "$TASK_DIR/oracle_data/VALIDATION.md" <<EOF
# Runtime validation

The approved runtime was executed before publication.

- Image: $IMAGE
- Command: docker run --rm --user 0:0 -v task:/work $IMAGE bash -lc 'blender --version; python3 --version; python3 -B /work/oracle_data/professional_validation.py'
- Exit code: $code
- Log: oracle_data/evidence/runtime_validation.log

The check parses both the canonical repaired GLB and the deliberately stripped
starter with the same scorer used by main.py.evaluate(). It verifies GLB
chunk/accessor validity, geometry and node-transform preservation, finite
non-degenerate UVs, material-to-texture-to-image reachability, CRC-valid
embedded PNG resources, effective core material parameters using glTF default
semantics, and three fixed-view Blender renders compared across all RGBA
channels. Blender and Python versions are recorded in the log.
EOF
exit "$code"
