# Boom Box repair task

The file `boombox_stripped.glb` is the real Khronos Boom Box mesh with its UV
layer and material links removed. Restore a textured glTF 2.0 asset that keeps
the supplied geometry and shows the Boom Box appearance in
`appearance_target.png`.

Save one file named `boombox_repaired.glb` in the output directory. Keep the
model's transforms and geometry (including vertex positions, normals, and
indices). A valid, non-empty UV layer is required; the exact unwrap topology is
not prescribed. Recreate the material and all appearance maps that are needed
for the visible result, with each material texture slot pointing to the
embedded image it actually uses. The grader validates glTF buffer/image
references and renders the candidate and canonical asset from three fixed
views, so do not paint only one camera view or leave valid-but-unreferenced
PNGs in the GLB. Node TRS/matrix transforms must remain consistent with the
input. Do not modify the input files.
