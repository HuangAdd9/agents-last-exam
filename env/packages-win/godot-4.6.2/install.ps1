# Godot Engine 4.6.2 stable portable runtime at C:\Softwares\Godot-4.6.2.
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$installDir = 'C:\Softwares\Godot-4.6.2'
$godot = Join-Path $installDir 'godot.exe'
$archiveSha256 = '14293422efb54b24a51f79d4cb55ab4001ef3d936e064a6c8af32e1f984024be'
$archiveUrl = if ($env:GODOT_4_6_2_ARCHIVE_URL) {
    $env:GODOT_4_6_2_ARCHIVE_URL
} else {
    'https://github.com/godotengine/godot/releases/download/4.6.2-stable/Godot_v4.6.2-stable_win64.exe.zip'
}

if (Test-Path $godot) {
    $installedVersion = (& $godot --headless --version 2>&1 | Select-Object -First 1)
    if ($installedVersion -like '4.6.2.stable*') {
        Write-Host "OK godot-4.6.2 already present ($godot)"
        return
    }
}

$tempRoot = Join-Path $env:TEMP ("godot-4.6.2-" + [guid]::NewGuid().ToString('N'))
$stageRoot = Join-Path $tempRoot 'runtime'
$zip = Join-Path $tempRoot 'Godot_v4.6.2-stable_win64.exe.zip'
try {
    New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null
    curl.exe --fail --location --retry 5 --silent --show-error -o $zip $archiveUrl
    if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $archiveSha256) {
        throw 'SHA-256 verification failed for Godot 4.6.2 archive'
    }
    Expand-Archive -LiteralPath $zip -DestinationPath $stageRoot -Force
    $stagedGodot = Join-Path $stageRoot 'Godot_v4.6.2-stable_win64.exe'
    $stagedConsole = Join-Path $stageRoot 'Godot_v4.6.2-stable_win64_console.exe'
    if (-not (Test-Path $stagedGodot)) {
        throw 'Godot archive did not contain the expected executable'
    }
    Rename-Item -LiteralPath $stagedGodot -NewName 'godot.exe'
    if (Test-Path $stagedConsole) {
        Rename-Item -LiteralPath $stagedConsole -NewName 'godot_console.exe'
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $installDir) | Out-Null
    if (Test-Path $installDir) {
        Remove-Item -LiteralPath $installDir -Recurse -Force
    }
    Move-Item -LiteralPath $stageRoot -Destination $installDir
} finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

$installedVersion = (& $godot --headless --version 2>&1 | Select-Object -First 1)
if ($installedVersion -notlike '4.6.2.stable*') {
    throw "verify failed for godot-4.6.2 (got: $installedVersion)"
}
Write-Host "OK godot-4.6.2 ($godot)"
