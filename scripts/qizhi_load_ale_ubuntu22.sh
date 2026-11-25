#!/usr/bin/env bash
set -euo pipefail

project_root=${QZ_PROJECT_ROOT:-/inspire/hdd/project/long-working-agent/ky26299}
cache_dir=${1:-$project_root/cache/docker/ale-ubuntu22-oci}

config_digest=bcaaf30031c87d9a29ef73c700795fea8ff20ed5d91a4311d412e8f2575738e1
layer_digest=4e4b084c809e110b80a67e4d198ebadeec915e280b95568dc60c5ef2bb5c869c
image_ref=agentslastexam/ale-ubuntu22-docker:latest

config_path=$cache_dir/config.upstream.json
layer_path=$cache_dir/blobs/sha256/$layer_digest
complete_path=$cache_dir/download.complete

for required in "$config_path" "$layer_path" "$complete_path"; do
  if [[ ! -f $required ]]; then
    echo "missing Qizhi ALE image cache file: $required" >&2
    exit 2
  fi
done

actual_config=$(sha256sum "$config_path" | awk '{print $1}')
if [[ $actual_config != "$config_digest" ]]; then
  echo "ALE image config digest mismatch: $actual_config" >&2
  exit 3
fi

# Docker's classic image loader accepts a gzip-compressed layer.tar. Build only
# the tiny archive metadata here and dereference a symlink while streaming the
# already-compressed registry layer. This avoids a second persistent 39 GB
# image archive on the shared project disk.
stage=$(mktemp -d)
cleanup() {
  rm -rf -- "$stage"
}
trap cleanup EXIT

mkdir -p "$stage/$layer_digest"
cp "$config_path" "$stage/$config_digest.json"
ln -s "$layer_path" "$stage/$layer_digest/layer.tar"
printf '[{"Config":"%s.json","RepoTags":["%s"],"Layers":["%s/layer.tar"]}]\n' \
  "$config_digest" "$image_ref" "$layer_digest" >"$stage/manifest.json"

tar --dereference -C "$stage" -cf - \
  manifest.json "$config_digest.json" "$layer_digest/layer.tar" |
  docker image load

loaded_id=$(docker image inspect "$image_ref" --format '{{.Id}}')
if [[ $loaded_id != "sha256:$config_digest" ]]; then
  echo "loaded ALE image ID mismatch: $loaded_id" >&2
  exit 4
fi

docker image inspect "$image_ref" --format 'ALE_IMAGE_READY {{.Id}} {{.Size}}'
