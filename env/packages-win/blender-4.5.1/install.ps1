# Blender 4.5.1 LTS portable runtime at C:\Softwares\Blender-4.5.1.
# This task-pinned copy coexists with the canonical Blender 5.0 MSI install.
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$installDir = 'C:\Softwares\Blender-4.5.1'
$blender = Join-Path $installDir 'blender.exe'
$archiveSha256 = 'ae2eadb2656d710ffd6ca74a899519fccb800c7a18e6c05b6f39627e48d17aed'

if (Test-Path $blender) {
    $installedVersion = (& $blender --version 2>&1 | Select-Object -First 1)
    if ($installedVersion -like 'Blender 4.5.1*') {
        Write-Host "OK blender-4.5.1 already present ($blender)"
        return
    }
}

$tempDir = Join-Path $env:TEMP ("blender-4.5.1-" + [guid]::NewGuid().ToString('N'))
$zip = "$tempDir.zip"
try {
    curl.exe --fail --location --silent --show-error -o $zip 'https://mirrors.aliyun.com/blender/release/Blender4.5/blender-4.5.1-windows-x64.zip'
    if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $archiveSha256) {
        throw 'SHA-256 verification failed for Blender 4.5.1 archive'
    }
    New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
    Expand-Archive -LiteralPath $zip -DestinationPath $tempDir -Force
    $extractedDir = Join-Path $tempDir 'blender-4.5.1-windows-x64'
    if (-not (Test-Path (Join-Path $extractedDir 'blender.exe'))) {
        throw 'Blender archive did not contain the expected directory'
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $installDir) | Out-Null
    if (Test-Path $installDir) {
        Remove-Item -LiteralPath $installDir -Recurse -Force
    }
    Move-Item -LiteralPath $extractedDir -Destination $installDir
} finally {
    Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $tempDir -Recurse -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path $blender)) {
    throw 'verify failed for blender-4.5.1: executable missing'
}
$installedVersion = (& $blender --version 2>&1 | Select-Object -First 1)
if ($installedVersion -notlike 'Blender 4.5.1*') {
    throw "verify failed for blender-4.5.1 (got: $installedVersion)"
}
Write-Host "OK blender-4.5.1 ($blender)"
