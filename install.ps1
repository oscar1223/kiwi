# Instalador de kiwi para Windows.
#   irm https://raw.githubusercontent.com/oscar1223/kiwi/main/install.ps1 | iex
$ErrorActionPreference = 'Stop'

$repo = 'oscar1223/kiwi'
$installDir = if ($env:KIWI_INSTALL_DIR) { $env:KIWI_INSTALL_DIR }
              else { "$env:LOCALAPPDATA\Programs\kiwi" }

$arch = switch ($env:PROCESSOR_ARCHITECTURE) {
  'AMD64' { 'amd64' }
  'ARM64' { 'arm64' }
  default { throw "Arquitectura no soportada: $env:PROCESSOR_ARCHITECTURE" }
}

$version = if ($env:KIWI_VERSION) { $env:KIWI_VERSION }
           else { (Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest").tag_name }

$archive = "kiwi_windows_$arch.zip"
$base    = "https://github.com/$repo/releases/download/$version"
$tmp     = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null

try {
  Write-Host "Instalando kiwi $version (windows/$arch)..."
  Invoke-WebRequest "$base/$archive"       -OutFile "$tmp\$archive"
  Invoke-WebRequest "$base/checksums.txt"  -OutFile "$tmp\checksums.txt"

  # Verificar SHA256 antes de tocar nada.
  $expected = Get-Content "$tmp\checksums.txt" |
    Where-Object { $_ -match "\s$([regex]::Escape($archive))$" } |
    ForEach-Object { ($_ -split '\s+')[0] } |
    Select-Object -First 1
  if (-not $expected) { throw "$archive no aparece en checksums.txt" }

  $actual = (Get-FileHash "$tmp\$archive" -Algorithm SHA256).Hash.ToLower()
  if ($actual -ne $expected.ToLower()) {
    throw "Checksum incorrecto (esperado $expected, obtenido $actual)"
  }

  Expand-Archive "$tmp\$archive" -DestinationPath $tmp -Force
  New-Item -ItemType Directory -Path $installDir -Force | Out-Null
  Copy-Item "$tmp\kiwi.exe" $installDir -Force

  # PATH de usuario, no de máquina: no pedimos admin.
  $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
  if ($userPath -notlike "*$installDir*") {
    [Environment]::SetEnvironmentVariable('Path', "$userPath;$installDir", 'User')
    Write-Host "Anadido $installDir al PATH de usuario. Abre una terminal nueva."
  }
  $env:Path = "$env:Path;$installDir"

  Write-Host "kiwi instalado en $installDir"
  & "$installDir\kiwi.exe" --version
}
finally {
  Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
