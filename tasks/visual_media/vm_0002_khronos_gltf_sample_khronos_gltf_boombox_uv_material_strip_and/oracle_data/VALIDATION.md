# Runtime validation

The approved runtime was executed before publication.

- Image: ale/base-visual-media:20260918-1
- Command: docker run --rm --user 0:0 -v task:/work ale/base-visual-media:20260918-1 bash -lc 'blender --version; python3 --version; python3 -B /work/oracle_data/professional_validation.py'
- Exit code: 0
- Log: oracle_data/evidence/runtime_validation.log

The check parses both the canonical repaired GLB and the deliberately stripped
starter with the same scorer used by main.py.evaluate(). It verifies GLB
chunk/accessor validity, geometry and node-transform preservation, finite
non-degenerate UVs, material-to-texture-to-image reachability, CRC-valid
embedded PNG resources, effective core material parameters using glTF default
semantics, and three fixed-view Blender renders compared across all RGBA
channels. Blender and Python versions are recorded in the log.
