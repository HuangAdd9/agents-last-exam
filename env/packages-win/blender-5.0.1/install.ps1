# Blender 5.0.1 @ C:\Program Files\Blender Foundation\Blender 5.0
# Sole baked Blender on the free images. Several task evals resolve Blender by
# "newest installed wins" (uv_reproduction, skeletal_animation_reproduction) and
# blender_character_reconstruction is calibrated on 5.0.1 (rejects 4.5), so any
# stray registered 5.1 / 4.5 installs must be removed or those tasks pick the
# wrong version. The task-pinned portable C:\Softwares\Blender-4.5.1 is kept.
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$blender = 'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe'

# --- 1. Remove any OTHER Blender so "newest-installed-wins" always lands on 5.0.x ---
# choco-managed install
choco uninstall blender -y --no-progress 2>$null | Out-Null
# MSI-installed Blender whose version is not 5.0.x (e.g. a prior 5.1.2)
Get-CimInstance Win32_Product -Filter "Name LIKE 'Blender%'" -ErrorAction SilentlyContinue | Where-Object {
    $_.Version -notlike '5.0*'
} | ForEach-Object {
    Write-Host "Removing stray Blender MSI: $($_.Name) $($_.Version)"
    Start-Process msiexec.exe -ArgumentList '/x', $_.IdentifyingNumber, '/quiet', '/norestart' -Wait
}
# leftover / portable Blender trees that the folder-glob resolvers would still see
foreach ($p in @(
    'C:\Tools\Blender',
    'C:\Program Files\Blender Foundation\Blender 5.1',
    'C:\Program Files\Blender Foundation\Blender 4.5'
)) { if (Test-Path $p) { Remove-Item $p -Recurse -Force -ErrorAction SilentlyContinue } }
Get-ChildItem 'C:\Softwares' -Directory -Filter 'Blender-*' -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notlike 'Blender-5.0*' -and $_.Name -ne 'Blender-4.5.1' } |
    ForEach-Object { Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }

# --- 2. Install 5.0.1 if not already present ---
if ((Test-Path $blender) -and ((Get-Item $blender).VersionInfo.ProductVersion -like '5.0*')) {
    Write-Host "OK blender-5.0.1 already present ($blender)"
    return
}

$msi = Join-Path $env:TEMP 'blender-5.0.1-windows-x64.msi'
curl.exe --fail --location --silent --show-error -o $msi 'https://download.blender.org/release/Blender5.0/blender-5.0.1-windows-x64.msi'
Start-Process msiexec.exe -ArgumentList '/i', $msi, '/quiet', '/norestart' -Wait

if (-not (Test-Path $blender) -or ((Get-Item $blender).VersionInfo.ProductVersion -notlike '5.0*')) {
    throw 'verify failed for blender-5.0.1'
}
Write-Host "OK blender-5.0.1 ($blender)"
