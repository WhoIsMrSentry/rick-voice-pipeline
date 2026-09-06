$ErrorActionPreference = "Stop"

$venv = Resolve-Path ".\.venv"
$nvidiaRoot = Join-Path $venv "Lib\site-packages\nvidia"

Write-Host "NVIDIA root:" $nvidiaRoot

if (-not (Test-Path $nvidiaRoot)) {
    Write-Host "NVIDIA package root bulunamadı."
    exit 1
}

# Tüm NVIDIA DLL klasörlerini bul.
$dlls = Get-ChildItem $nvidiaRoot -Recurse -File -Filter "*.dll" -ErrorAction SilentlyContinue

if (-not $dlls) {
    Write-Host "NVIDIA DLL bulunamadı."
    exit 1
}

$dirs = $dlls | Select-Object -ExpandProperty DirectoryName -Unique

foreach ($d in $dirs) {
    if ($env:Path -notlike "*$d*") {
        $env:Path = "$d;$env:Path"
    }
    Write-Host "PATH +" $d
}

Write-Host ""
Write-Host "Toplam DLL:" $dlls.Count
Write-Host "Toplam DLL klasörü:" $dirs.Count
Write-Host ""
Write-Host "Önemli DLL kontrolleri:"

$dlls | Where-Object {
    $_.Name -eq "cublas64_12.dll" -or
    $_.Name -eq "cublasLt64_12.dll" -or
    $_.Name -eq "cudart64_12.dll" -or
    $_.Name -like "cudnn*.dll" -or
    $_.Name -like "nvrtc*.dll"
} | Select-Object Name, DirectoryName | Sort-Object Name | Format-Table -AutoSize
