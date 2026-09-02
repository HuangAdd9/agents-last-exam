#!/usr/bin/env bash
set -Eeuo pipefail

storage_dir="${STORAGE:-/storage}"
disk_name="${DISK_NAME:-data}"
disk_path="$storage_dir/$disk_name.qcow2"

if [[ ! -c /dev/kvm || ! -r /dev/kvm || ! -w /dev/kvm ]]; then
  echo "ALE QEMU runner requires readable and writable /dev/kvm" >&2
  exit 69
fi

if [[ ! -s "$disk_path" ]]; then
  echo "ALE QEMU runner requires a non-empty guest disk at $disk_path" >&2
  exit 64
fi

if ! qemu-img info "$disk_path" >/dev/null; then
  echo "ALE QEMU runner could not read the guest disk chain at $disk_path" >&2
  exit 65
fi

# The upstream Dockur startup treats this marker as an already-installed guest.
# ALE always supplies a pre-baked qcow2, for both Ubuntu and Windows.
touch "$storage_dir/windows.boot"

if [[ "${INSTALL_WINARENA_APPS,,}" == "true" ]]; then
  install_winarena_apps=true
else
  install_winarena_apps=false
fi
printf '{"INSTALL_WINARENA_APPS": %s}\n' "$install_winarena_apps" \
  > /oem/install_config.json

echo "Starting ALE QEMU guest from $disk_path"
echo "noVNC is exposed on container port 8006"
echo "CUA is expected on guest ${VM_NET_IP:-172.30.0.2}:${ALE_GUEST_CUA_PORT:-5000}"

# Docker's default AppArmor profile blocks nginx socketpair creation. The
# runner already includes websocketd, which can serve noVNC and bridge its
# binary WebSocket directly to a VNC server without extra privileges.
sed -i \
  "s|^  nginx -e stderr$|  /usr/local/bin/ale-qemu-novnc > /var/log/ale-novnc.log 2>\&1 \&|" \
  /run/server.sh
if ! grep -Fq "/usr/local/bin/ale-qemu-novnc" /run/server.sh; then
  echo "ALE QEMU runner could not configure noVNC server" >&2
  exit 66
fi

echo "noVNC bridges to ${ALE_VNC_TARGET:-127.0.0.1:5900}"

# tini becomes PID 1 and forwards signals to the Dockur startup process.
# Unlike the inherited entrypoint, no detached readiness loop can keep the
# container alive after QEMU exits.
exec /usr/bin/tini -s -- /run/entry.sh
