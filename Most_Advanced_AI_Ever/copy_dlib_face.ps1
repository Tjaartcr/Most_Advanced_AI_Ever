# copy_dlib_face.ps1
# Copies dlib and face_recognition from a global site-packages to a virtualenv site-packages.
# WARNING: Only safe when both Python interpreters are the same minor version and same architecture.

# ------------- CONFIG -------------
$globalSite = "C:\Users\ITF\AppData\Local\Programs\Python\Python311\Lib\site-packages"
$venvSite   = "D:\Python_Env\New_Virtual_Env\Alfred_Offline_GUI_Env\Lib\site-packages"

# Optional: paths to python executables to compare versions (adjust if different)
$globalPythonExe = "C:\Users\ITF\AppData\Local\Programs\Python\Python311\python.exe"
$venvPythonExe   = "D:\Python_Env\New_Virtual_Env\Alfred_Offline_GUI_Env\Scripts\python.exe"

# Packages to copy
$packages = @("dlib","face_recognition")

# If you set to $true the script will proceed even when Python versions differ.
$Force = $false

# ------------- FUNCTIONS -------------
function Get-PythonVersion($pyExe) {
    if (-not (Test-Path $pyExe)) { return $null }
    try {
        $ver = & $pyExe -c "import sys; print(sys.version.split()[0])" 2>$null
        return $ver.Trim()
    } catch {
        return $null
    }
}

function Copy-ItemSafe($src, $dst) {
    try {
        if ((Test-Path $src) -and (Get-Item $src).PSIsContainer) {
            Copy-Item -Path $src -Destination $dst -Recurse -Force -ErrorAction Stop
        } elseif (Test-Path $src) {
            $dstDir = Split-Path -Path $dst -Parent
            if (-not (Test-Path $dstDir)) { New-Item -Path $dstDir -ItemType Directory | Out-Null }
            Copy-Item -Path $src -Destination $dst -Force -ErrorAction Stop
        } else {
            Write-Host "  [!] Source not found: $src"
        }
    } catch {
        Write-Host "  [ERROR] Failed to copy $src -> $dst : $($_.Exception.Message)"
    }
}

# ------------- MAIN -------------
Write-Host ""
Write-Host "===== dlib & face_recognition copier ====="
Write-Host "Global site-packages: $globalSite"
Write-Host "Venv site-packages:   $venvSite"
Write-Host ""

# Check Python versions (best-effort)
$globalVer = Get-PythonVersion $globalPythonExe
$venvVer   = Get-PythonVersion $venvPythonExe

if ($globalVer -and $venvVer) {
    Write-Host "Global python version: $globalVer"
    Write-Host "Venv   python version: $venvVer"
    if ($globalVer -ne $venvVer) {
        Write-Host ""
        Write-Host "WARNING: Python versions differ. Copying compiled extension modules (.pyd) between different Python minor versions often fails."
        if (-not $Force) {
            Write-Host "Aborting because $Force is false. If you understand the risk, set `$Force = $true` at the top of this script and re-run."
            exit 1
        } else {
            Write-Host "Force is true: continuing despite version mismatch."
        }
    }
} else {
    Write-Host "Could not determine one or both python exes. Continuing without version check."
}

# Ensure paths exist
if (-not (Test-Path $globalSite)) {
    Write-Host "ERROR: Global site-packages not found at: $globalSite"
    exit 1
}
if (-not (Test-Path $venvSite)) {
    Write-Host "ERROR: Venv site-packages not found at: $venvSite"
    exit 1
}

foreach ($pkg in $packages) {
    Write-Host ""
    Write-Host "Processing package: $pkg"

    # Candidate items: directories, .pyd/.dll files, .py files, and dist-info
    $candidates = @()

    # directories or files beginning with package name
    $candidates += Get-ChildItem -Path $globalSite -Force -ErrorAction SilentlyContinue |
                   Where-Object { $_.Name -like "$pkg*" } 

    # Also look for possible compiled binaries that may not start exactly same:
    $binaryPatterns = @("$pkg*.pyd","*$pkg*.pyd","*$pkg*.dll")
    foreach ($pat in $binaryPatterns) {
        $candidates += Get-ChildItem -Path $globalSite -Filter $pat -File -Force -ErrorAction SilentlyContinue
    }

    # Remove duplicates
    $candidates = $candidates | Select-Object -Unique

    if (-not $candidates -or $candidates.Count -eq 0) {
        Write-Host "  [!] No items found for package '$pkg' in global site-packages."
        continue
    }

    foreach ($item in $candidates) {
        $src = $item.FullName
        $dst = Join-Path $venvSite $item.Name
        Write-Host "  Copying: $item.Name"
        Copy-ItemSafe -src $src -dst $dst

        # If it's a package folder, also try to copy nested compiled files (safety)
        if ($item.PSIsContainer) {
            $nestedBinaries = Get-ChildItem -Path $item.FullName -Include *.pyd,*.dll -Recurse -Force -ErrorAction SilentlyContinue
            foreach ($nb in $nestedBinaries) {
                $relative = $nb.FullName.Substring($globalSite.Length).TrimStart('\')
                $dstNested = Join-Path $venvSite $relative
                Copy-ItemSafe -src $nb.FullName -dst $dstNested
            }
        }
    }

    Write-Host "  Done with $pkg."
}

Write-Host ""
Write-Host "All done. Activate your venv and test:"
Write-Host '  <venv>\Scripts\activate'
Write-Host '  python -c "import dlib; import face_recognition; print(\"import ok\")"'
Write-Host ""
Write-Host "If import fails with binary/architecture errors, reinstall packages inside the venv using pip (recommended)."
Write-Host ""
