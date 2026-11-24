param(
    [string]$InnoCompiler = "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
)

$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & .\.venv\Scripts\python.exe -m PyInstaller --log-level WARN --noconfirm --onedir --windowed --name MemStack --add-data 'web;web' --icon 'packaging\icon.ico' app.py
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }
    $notices = New-Item -ItemType Directory -Path 'dist\notices' -Force
    $pythonRoot = & .\.venv\Scripts\python.exe -c 'import sys; print(sys.base_prefix)'
    Copy-Item -LiteralPath (Join-Path $pythonRoot 'LICENSE.txt') -Destination (Join-Path $notices.FullName 'Python.txt')
    foreach ($package in Get-ChildItem -Path '.venv\Lib\site-packages' -Filter '*.dist-info' -Directory) {
        $licenses = Get-ChildItem -LiteralPath $package.FullName -File -Recurse |
            Where-Object Name -Match '^(LICENSE|COPYING|NOTICE)'
        foreach ($license in $licenses) {
            $name = ($package.Name -replace '\.dist-info$', '') + '-' + $license.Name
            Copy-Item -LiteralPath $license.FullName -Destination (Join-Path $notices.FullName $name)
        }
    }
    & $InnoCompiler /Qp setup.iss
    if ($LASTEXITCODE -ne 0) { throw 'Inno Setup failed.' }
} finally {
    Pop-Location
}

