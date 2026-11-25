# install.ps1 -- Ruffle 0.2.0 (Flash/SWF emulator) at C:\Program Files\ruffle\bin — matches dev-win10
# idempotent, version-pinned to dev-win10. Run as admin.
$ErrorActionPreference='Stop'
[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12

$zip="$env:TEMP\ruffle.zip"
curl.exe --fail --location --silent --show-error -o $zip 'https://github.com/ruffle-rs/ruffle/releases/download/v0.2.0/ruffle-0.2.0-windows-x86_64.zip'
if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ceb0374588c1006a3ff6d9dc5282ae21d8f6216f35bf8e7a294abbeb7b8e9792') {
    throw 'SHA-256 verification failed for Ruffle archive'
}
New-Item -ItemType Directory -Force -Path 'C:\Program Files\ruffle\bin' | Out-Null
Expand-Archive -LiteralPath $zip -DestinationPath 'C:\Program Files\ruffle\bin' -Force
# NOTE: dev-win10 reference SHA256 of ruffle.exe = 5A9D583271E703D7836DCE65FA20C1582BE5C831A2AAC5251DEA64DAFEADDE14

$ruffleBin = 'C:\Program Files\ruffle\bin'
$machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
$pathEntries = $machinePath -split ';' | Where-Object { $_ }
if ($pathEntries -notcontains $ruffleBin) {
    [Environment]::SetEnvironmentVariable('Path', ((@($pathEntries) + $ruffleBin) -join ';'), 'Machine')
}
if (($env:Path -split ';') -notcontains $ruffleBin) {
    $env:Path = "$ruffleBin;$env:Path"
}

# self-check
if(-not ((Get-Item 'C:\Program Files\ruffle\bin\ruffle.exe').VersionInfo.ProductVersion -like '0.2.0*')){ throw 'verify failed for ruffle-0.2.0' }
Write-Host 'OK ruffle-0.2.0 @ C:\Program Files\ruffle'
