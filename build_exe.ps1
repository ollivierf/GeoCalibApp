$ErrorActionPreference = "Stop"

$scriptPath = "GeoCalibApp.py"
$appName = "GeoCalibApp"
$pyinstallerPath = ".\venv\Scripts\pyinstaller.exe"

# Cleanup previous builds
$distDir = "dist"
$buildDir = "build_temp_new"
$specFile = "$appName.spec"

Write-Host "Cleaning up previous build artifacts..."
if (Test-Path $distDir) { Remove-Item -Recurse -Force $distDir }
if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }
if (Test-Path $specFile) { Remove-Item -Force $specFile }

if (-not (Test-Path $pyinstallerPath)) {
    Write-Error "PyInstaller not found at $pyinstallerPath. Please ensure venv is set up correctly."
    exit 1
}

Write-Host "Building $appName (Folder Mode)..."

# Using --onedir to avoid AV locking issues with large single files
# including rcbox modules explicitly just in case, though hidden imports should handle it
& $pyinstallerPath --noconsole --onedir --name $appName --clean --workpath $buildDir `
    --hidden-import="scipy" `
    --hidden-import="scipy.signal" `
    --hidden-import="scipy.linalg" `
    --hidden-import="scipy.spatial.transform._rotation_groups" `
    --hidden-import="scipy.special.cython_special" `
    --hidden-import="sklearn.utils._cython_blas" `
    --hidden-import="sklearn.neighbors.typedefs" `
    --hidden-import="sklearn.neighbors.quad_tree" `
    --hidden-import="sklearn.tree" `
    --hidden-import="sklearn.tree._utils" `
    --collect-all "rcbox" `
    --collect-all "scipy" `
    $scriptPath

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build Successful!"
    Write-Host "Executable is located at: dist\$appName\$appName.exe"
} else {
    Write-Error "Build failed. Check the logs."
}
