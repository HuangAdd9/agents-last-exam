# Local Proxmox VE provider

The `proxmox` provider creates one disposable PVE VM clone per ALE run. It is an additional backend. The Docker-based `qemu` provider is unchanged and remains the portable choice for Linux hosts that are not managed by Proxmox.

## Lifecycle

For each run, ALE performs the following operations through the Proxmox REST API:

1. Validate the configured PVE template.
2. Allocate a free VMID and clone the template.
3. Apply the task CPU and memory requirements.
4. Start the clone.
5. Read its DHCP address from QEMU Guest Agent.
6. Connect directly to the CUA server in the guest.
7. Optionally publish the guest's x11vnc desktop through a per-VM noVNC bridge.
8. Keep, stop, or delete the clone according to `cleanup_mode`.

Linked clones are used by default. The large ALE base disk is therefore imported into PVE once instead of copied for every task.

## Prepare templates

Create one PVE template for every ALE snapshot you intend to run. On seed3, VMID `9002` provides `ale-ubuntu22` and VMID `9001` provides `ale-win10`.

For each template:

1. Create a VM with the correct firmware and a VirtIO network adapter.
2. Import the corresponding qcow2 from `/ale-cache-runtime/qemu/images/` into PVE storage.
3. Attach the imported disk as the boot disk.
4. Enable QEMU Guest Agent in the PVE VM options.
5. Configure the guest NIC for DHCP and install/start the QEMU Guest Agent service.
6. Boot it once and verify that the CUA server is reachable on port `5000`.
7. For PVE console viewing, verify that Xorg uses the PVE virtual display and that
   `xrandr --display :0` reports a connected output such as `Virtual-1`.
8. Shut it down and convert it to a PVE template.

For Windows, use OVMF with q35, a VirtIO SCSI boot disk, a VirtIO network adapter, and a PVE virtual display such as `virtio`. Install both QEMU Guest Agent and the VirtIO Serial driver before converting the VM. The interactive user must log in automatically because CUA and its watchdog run in that desktop session.

The imported Windows image on seed3 originated on Google Cloud. Its dedicated PVE copy has the unused GCE Agent, OSConfig, GCE VSS services, and GCE startup tasks disabled. The portable Docker/QEMU qcow2 remains unchanged. Windows may still leave `Group Policy Client (gpsvc)` in `Stop Pending` during a guest-requested restart. This does not block normal ALE cleanup because the provider stops disposable clones through PVE before deleting them.

The template network must place clones on a network reachable from the ALE process. Set `guest_cidr` to that network so ALE ignores unrelated addresses reported by the guest agent.

The template storage must support linked clones. Set `full_clone: true` and `storage: <storage-id>` if linked clones are unavailable, with the understanding that full clones consume much more time and disk space.

## View the desktop in the PVE console

The PVE KVM console displays the VM's emulated display adapter. To make it show ALE's working desktop, the template must let Xorg use that adapter. Some ALE Ubuntu images contain `/etc/X11/xorg.conf` with `Driver "dummy"`; that configuration renders only to an in-guest framebuffer, so the PVE console remains black.

For a dedicated PVE template, preserve the file as a reversible backup and restart the guest:

```bash
sudo mv /etc/X11/xorg.conf /etc/X11/xorg.conf.ale-dummy
sudo reboot
```

Then verify:

```bash
xrandr --display :0
lspci -nnk | sed -n '/VGA compatible controller/,+3p'
```

A working seed3 template reports `Virtual-1 connected` and uses the `bochs-drm` kernel driver. ALE continues to control the desktop through CUA, while PVE's built-in noVNC console displays that same Xorg desktop. Open the disposable `ale-pve-...` clone in the PVE management interface and select **Console**. Do not start or modify the template just to watch a run.

Keep the original dummy-display template or qcow2 for the Docker QEMU provider. A dedicated PVE-console template avoids changing the portable Docker/QEMU path.

## Optional standalone desktop bridge

If a template must continue using the dummy Xorg driver, the PVE console cannot display that framebuffer. The optional bridge publishes its x11vnc desktop in a separate browser page:

```text
guest x11vnc :5900 -> ALE host websockify/noVNC :6080-6180 -> browser
```

Install the standard noVNC browser library and websockify on the machine running ALE:

```bash
sudo apt install novnc websockify
```

The Proxmox-specific `novnc-pve` package is not sufficient by itself. It contains a PVE-bundled frontend, not the standard reusable `/usr/share/novnc/core/rfb.js` module.

Enable the bridge on the snapshot only when this fallback is needed:

```yaml
novnc_enabled: true
novnc_bind_address: 0.0.0.0
novnc_public_host: 192.168.0.14
novnc_port_min: 6080
novnc_port_max: 6180
novnc_guest_port: 5900
novnc_ready_timeout_s: 60
novnc_websockify: /usr/bin/websockify
novnc_web_root: /usr/share/novnc
novnc_runtime_root: /tmp/ale-proxmox-novnc
novnc_tls_cert: /path/to/seed3.crt
novnc_tls_key: /path/to/seed3.key
novnc_manage_guest_password: true
```

When a clone is ready, ALE logs a URL such as:

```text
Proxmox live desktop: https://192.168.0.14:6080/#token=...
```

Open that URL from a machine that can reach the ALE host. Modern noVNC needs a browser secure context, so configure both TLS fields when the bridge is not bound to localhost. With a self-signed certificate, accept the browser's certificate warning once. A certificate trusted by client machines is preferable.

Each bridge gets a random WebSocket token and its own runtime directory. `cleanup_mode: keep` leaves both the VM and bridge running. `stop`, `delete`, and cancellation stop the bridge and remove its runtime directory.

For the ALE Ubuntu image, `novnc_manage_guest_password: true` rotates x11vnc's password when each clone starts. The password is derived from the per-run token and supplied by the page automatically. The template's original VNC password is neither read nor exposed. Leave this disabled for images that do not use the ALE Ubuntu x11vnc user service. Without managed authentication, the page asks the viewer for the guest's existing VNC password.

The token protects the WebSocket route, and the configured certificate encrypts the browser side with HTTPS/WSS. The connection from seed3 to the guest's raw x11vnc port is still unencrypted and remains reachable according to the guest firewall and LAN routing. Keep it on a trusted network and restrict guest port `5900` at the firewall.

## Create a dedicated API token

Use a dedicated PVE user instead of a root token. The provider needs these privileges:

```text
Datastore.AllocateSpace
SDN.Use
VM.Allocate
VM.Audit
VM.Clone
VM.Config.CPU
VM.Config.Memory
VM.Config.Options
VM.GuestAgent.Audit
VM.PowerMgmt
```

An administrator can create a role, user, and token with commands similar to:

```bash
pveum role add ALEProvider --privs "Datastore.AllocateSpace SDN.Use VM.Allocate VM.Audit VM.Clone VM.Config.CPU VM.Config.Memory VM.Config.Options VM.GuestAgent.Audit VM.PowerMgmt"
pveum user add ale@pve
pveum acl modify / --user ale@pve --role ALEProvider
pveum user token add ale@pve ale-run --privsep 0
```

For a stricter deployment, scope ACLs to the ALE pool, template VMIDs, storage, and bridge instead of `/`.

Store the returned values in `secret/.env`:

```dotenv
PROXMOX_API_TOKEN_ID=ale@pve!ale-run
PROXMOX_API_TOKEN_SECRET=replace-with-the-returned-secret
```

The token secret is shown only once by Proxmox.

## Configure an experiment

Start from [`configs/environments/proxmox.yaml`](../configs/environments/proxmox.yaml). Update:

- `node`: PVE node name.
- `template_vmid`: persistent template VMID.
- `guest_cidr`: network used by cloned guests.
- `verify_ssl`: use a CA bundle path in production. `false` is appropriate only for a trusted local PVE endpoint with its default self-signed certificate.
- `pool`: optional PVE pool for cloned VMs.
- `full_clone` and `storage`: optional full-clone settings.
- `novnc_*`: optional real-desktop bridge settings described above.

Then point an experiment at that environment file. `cleanup_mode` behaves consistently across the PVE and Docker QEMU providers:

- `keep`: leave the VM running.
- `stop`: stop the VM but retain its disks and configuration.
- `delete`: stop and destroy the disposable clone and its owned disks.

`output_path: local` uses the provider-independent CUA file download path. It does not require a host directory mounted into the PVE guest.

On the current seed3 host, configure both snapshots:

```yaml
snapshots:
  cpu-free-ubuntu:
    provider: proxmox
    image: ale-ubuntu22
    proxmox:
      node: seed3
      template_vmid: 9002
      full_clone: false
      guest_cidr: 192.168.0.0/24

  cpu-free:
    provider: proxmox
    image: ale-win10
    proxmox:
      node: seed3
      template_vmid: 9001
      full_clone: false
      guest_cidr: 192.168.0.0/24
```

Both templates use PVE's built-in console. The Ubuntu template exposes its Xorg `Virtual-1` desktop, while the Windows template uses the PVE `virtio` display. Each linked clone receives a new virtual MAC address and obtains its own DHCP address.

## Current host prerequisites

The PVE API must be accessible from the ALE process, the token must have the listed privileges, and a usable template must already exist. A normal unprivileged shell account does not inherit Proxmox administrator permissions merely because it belongs to the `kvm` or `docker` group.
