# Ghidra 11.3 + Eclipse Temurin JDK 21.0.10+7 portable runtime.
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$installRoot = 'C:\Softwares\Ghidra-11.3'
$ghidraDir = Join-Path $installRoot 'ghidra_11.3_PUBLIC'
$jdkDir = Join-Path $installRoot 'jdk-21.0.10+7'
$ghidraRun = Join-Path $ghidraDir 'ghidraRun.bat'
$ghidraProperties = Join-Path $ghidraDir 'Ghidra\application.properties'
$java = Join-Path $jdkDir 'bin\java.exe'
$jdkRelease = Join-Path $jdkDir 'release'
$ghidraSha256 = '73b42cec587f370bfef49809ae55d2c7a3db689e8f164a1e7d7b8de295b6359f'
$jdkSha256 = '08cae782814f027f8b159d6b68823f0f87422eb475c1a0ea1abc7a4357aaf11f'
$ghidraUrl = if ($env:GHIDRA_11_3_ARCHIVE_URL) {
    $env:GHIDRA_11_3_ARCHIVE_URL
} else {
    'https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.3_build/ghidra_11.3_PUBLIC_20250205.zip'
}
$jdkUrl = if ($env:TEMURIN_JDK_21_0_10_ARCHIVE_URL) {
    $env:TEMURIN_JDK_21_0_10_ARCHIVE_URL
} else {
    'https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.10%2B7/OpenJDK21U-jdk_x64_windows_hotspot_21.0.10_7.zip'
}

if (
    (Test-Path $ghidraRun) -and
    (Test-Path $java) -and
    (Test-Path $ghidraProperties) -and
    (Test-Path $jdkRelease) -and
    (Select-String -LiteralPath $ghidraProperties -Pattern '^application.version=11\.3$' -Quiet) -and
    (Select-String -LiteralPath $jdkRelease -Pattern '^JAVA_VERSION=.*21\.0\.10.*$' -Quiet)
) {
    Write-Host "OK ghidra-11.3 already present ($installRoot)"
    return
}

$tempRoot = Join-Path $env:TEMP ("ghidra-11.3-" + [guid]::NewGuid().ToString('N'))
$stageRoot = Join-Path $tempRoot 'runtime'
$ghidraZip = Join-Path $tempRoot 'ghidra_11.3_PUBLIC_20250205.zip'
$jdkZip = Join-Path $tempRoot 'OpenJDK21U-jdk_x64_windows_hotspot_21.0.10_7.zip'
try {
    New-Item -ItemType Directory -Force -Path $stageRoot | Out-Null
    curl.exe --fail --location --retry 5 --silent --show-error -o $ghidraZip $ghidraUrl
    curl.exe --fail --location --retry 5 --silent --show-error -o $jdkZip $jdkUrl
    if ((Get-FileHash -LiteralPath $ghidraZip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ghidraSha256) {
        throw 'SHA-256 verification failed for Ghidra 11.3 archive'
    }
    if ((Get-FileHash -LiteralPath $jdkZip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $jdkSha256) {
        throw 'SHA-256 verification failed for Temurin JDK archive'
    }
    Expand-Archive -LiteralPath $ghidraZip -DestinationPath $stageRoot -Force
    Expand-Archive -LiteralPath $jdkZip -DestinationPath $stageRoot -Force

    $stagedGhidraRun = Join-Path $stageRoot 'ghidra_11.3_PUBLIC\ghidraRun.bat'
    $stagedProperties = Join-Path $stageRoot 'ghidra_11.3_PUBLIC\Ghidra\application.properties'
    $stagedJava = Join-Path $stageRoot 'jdk-21.0.10+7\bin\java.exe'
    $stagedJdkRelease = Join-Path $stageRoot 'jdk-21.0.10+7\release'
    if (
        (-not (Test-Path $stagedGhidraRun)) -or
        (-not (Test-Path $stagedJava)) -or
        (-not (Test-Path $stagedProperties)) -or
        (-not (Test-Path $stagedJdkRelease)) -or
        (-not (Select-String -LiteralPath $stagedProperties -Pattern '^application.version=11\.3$' -Quiet)) -or
        (-not (Select-String -LiteralPath $stagedJdkRelease -Pattern '^JAVA_VERSION=.*21\.0\.10.*$' -Quiet))
    ) {
        throw 'expanded Ghidra/JDK runtime failed layout or version verification'
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $installRoot) | Out-Null
    if (Test-Path $installRoot) {
        Remove-Item -LiteralPath $installRoot -Recurse -Force
    }
    Move-Item -LiteralPath $stageRoot -Destination $installRoot
} finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

if (
    (-not (Test-Path $ghidraRun)) -or
    (-not (Test-Path $java)) -or
    (-not (Test-Path $ghidraProperties)) -or
    (-not (Test-Path $jdkRelease)) -or
    (-not (Select-String -LiteralPath $ghidraProperties -Pattern '^application.version=11\.3$' -Quiet)) -or
    (-not (Select-String -LiteralPath $jdkRelease -Pattern '^JAVA_VERSION=.*21\.0\.10.*$' -Quiet))
) {
    throw 'verify failed for ghidra-11.3'
}
Write-Host "OK ghidra-11.3 ($installRoot)"
