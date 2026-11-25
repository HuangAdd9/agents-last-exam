# install.ps1 -- DaVinci Resolve 21.0 (free edition) — current version on the target.
# idempotent, pins the CURRENT version (NOT a downgrade). Requires admin. ~3-4 GB.
#
# Download is automatable via Blackmagic's register API — the only trick is sending a
# browser User-Agent (without it the API returns HTTP 400).
$ErrorActionPreference='Stop'
[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12

$resolve='C:\Program Files\Blackmagic Design\DaVinci Resolve\Resolve.exe'
if((Test-Path $resolve) -and ((Get-Item $resolve).VersionInfo.ProductVersion -like '21.0*')){
  Write-Host 'OK davinci-resolve-21.0 already present'; return
}

# 1) signed official download URL for DaVinci Resolve 21.0 (Windows)
$downloadId='e2096c01ac344e33926f9da9b5097c4f'
$archiveSize=3563531244
$archiveSha256='8aef53861117c60a6a3ecfe15a2e43bf7fc17ac8b79231736cbb4c5d050a1d29'
$ua='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0 Safari/537.36'
$reg='{"firstname":"Eval","lastname":"Lab","email":"agenthle.sv@gmail.com","phone":"4155550100","country":"us","street":"1 Market St","city":"San Francisco","state":"California","product":"DaVinci Resolve"}'
$url = if ($env:DAVINCI_RESOLVE_21_ARCHIVE_URL) {
  $env:DAVINCI_RESOLVE_21_ARCHIVE_URL
} else {
  # The registration endpoint currently redirects HTTPS to HTTP with a 301. Web
  # clients turn that redirected POST into a GET and receive the support HTML
  # instead of a signed archive URL, so post to the vendor's canonical endpoint.
  Invoke-RestMethod -Method Post -Uri "http://www.blackmagicdesign.com/api/register/us/download/$downloadId" `
    -ContentType 'application/json;charset=UTF-8' `
    -Headers @{'User-Agent'=$ua;'Origin'='https://www.blackmagicdesign.com';'Referer'='https://www.blackmagicdesign.com/support/download/'} `
    -Body $reg
}
$zip="$env:TEMP\DaVinci_Resolve_21.0_Windows.zip"
try {
  $signedUrl = ([string]$url).Trim()
  $parsedUrl = $null
  if(
    (-not [Uri]::TryCreate($signedUrl, [UriKind]::Absolute, [ref]$parsedUrl)) -or
    $parsedUrl.Scheme -notin @('http', 'https')
  ){ throw 'Blackmagic registration API did not return an absolute download URL' }
  curl.exe --fail --location --silent --show-error -o $zip $signedUrl
  if((Get-Item $zip).Length -ne $archiveSize){ throw 'DaVinci archive size mismatch' }
  if((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $archiveSha256){
    throw 'SHA-256 verification failed for DaVinci Resolve 21.0 archive'
  }

  # 2) extract installer .exe -> embedded MSI. Delete each prior layer as soon
  # as it is no longer needed to keep peak temporary disk use bounded.
  & 'C:\Program Files\7-Zip\7z.exe' x -y "-o$env:TEMP\dv210" $zip | Out-Null
  if($LASTEXITCODE -ne 0){ throw "outer archive extraction failed ($LASTEXITCODE)" }
  Remove-Item -LiteralPath $zip -Force
  $exe = Get-ChildItem "$env:TEMP\dv210" -Filter *.exe -Recurse | Select-Object -First 1
  if(-not $exe){ throw 'DaVinci installer executable missing after extraction' }
  & 'C:\Program Files\7-Zip\7z.exe' x -y "-o$env:TEMP\dv210msi" $exe.FullName | Out-Null
  if($LASTEXITCODE -ne 0){ throw "embedded MSI extraction failed ($LASTEXITCODE)" }
  Remove-Item -LiteralPath "$env:TEMP\dv210" -Recurse -Force
  $msi = Get-ChildItem "$env:TEMP\dv210msi" -Filter *.msi -Recurse | Select-Object -First 1
  if(-not $msi){ throw 'DaVinci MSI missing after extraction' }

  # 3) silent install
  $install = Start-Process msiexec.exe -ArgumentList '/i',('"'+$msi.FullName+'"'),'/qn','/norestart' -Wait -PassThru
  if($install.ExitCode -notin @(0,3010)){ throw "DaVinci MSI failed ($($install.ExitCode))" }
} finally {
  Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath "$env:TEMP\dv210" -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath "$env:TEMP\dv210msi" -Recurse -Force -ErrorAction SilentlyContinue
}

# 4) self-check
if(-not ((Get-Item $resolve).VersionInfo.ProductVersion -like '21.0*')){ throw 'verify failed for davinci-resolve-21.0' }
Write-Host 'OK davinci-resolve-21.0 @ C:\Program Files\Blackmagic Design\DaVinci Resolve'
