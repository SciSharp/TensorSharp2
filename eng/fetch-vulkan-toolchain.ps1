# Provision the Vulkan build toolchain needed by the ggml-vulkan backend
# (TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON) into ExternalProjects/vulkan-toolchain.
#
# ggml-vulkan needs three things at build time that a plain Windows box does not
# have (the Vulkan *runtime* ships with the GPU driver, the *SDK* does not):
#   1. Vulkan headers (vulkan_core.h / vulkan.hpp)         -> KhronosGroup/Vulkan-Headers
#   2. A vulkan-1 import library to link against            -> generated from the
#      system C:\Windows\System32\vulkan-1.dll with dumpbin/lib (MSVC)
#   3. The glslc GLSL->SPIR-V compiler                      -> Google shaderc CI prebuilt
# plus the SPIRV-Headers CMake package (find_package(SPIRV-Headers) in ggml).
#
# When a LunarG Vulkan SDK is installed (VULKAN_SDK env var) all of the above are
# already available and this script exits without doing anything;
# build-windows.ps1 then lets CMake's FindVulkan discover the SDK on its own.
#
# Environment overrides:
#   TENSORSHARP_GGML_NO_UPDATE  if set to 1/ON/true and the toolchain is already
#                               populated, skip all network access and reuse it
#                               (same contract as eng/fetch-ggml.ps1).
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$ToolchainDir = Join-Path $RepoRoot "ExternalProjects\vulkan-toolchain"

# Get-VisualStudioInstallation / Import-VcVarsEnvironment. A bare
# `vswhere -latest` cannot be trusted (it silently skips installer instances
# flagged incomplete); see eng/vs-locate.ps1.
. (Join-Path $ScriptDir "vs-locate.ps1")

function Test-Truthy([string] $Value) {
    return $Value -match '^(1|ON|on|On|TRUE|true|True|YES|yes|Yes)$'
}

function Invoke-CMakeDefanged([string] $FailureMessage, [string[]] $Arguments) {
    # This script usually runs inside an MSBuild Exec task, whose logger scans
    # every output line for the canonical MSBuild error/warning format and fails
    # the managed build when one matches - even when the caller catches the
    # failure and deliberately degrades to a CPU/CUDA-only build. CMake's
    # "CMake Error: ..." diagnostics match that format, so capture the output
    # and re-emit it with the "<keyword> :" pattern broken.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & cmake @Arguments 2>&1
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }

    foreach ($line in @($output)) {
        Write-Host ("$line" -replace '(?i)\b(error|warning)(\s+[A-Z0-9]+)?\s*:', '$1$2 -')
    }

    if ($LASTEXITCODE -ne 0) { throw $FailureMessage }
}

function Test-VulkanSdkComplete {
    $sdk = $env:VULKAN_SDK
    if ([string]::IsNullOrWhiteSpace($sdk)) { return $false }
    return (Test-Path (Join-Path $sdk "Include\vulkan\vulkan.h")) -and
           (Test-Path (Join-Path $sdk "Lib\vulkan-1.lib")) -and
           (Test-Path (Join-Path $sdk "Bin\glslc.exe"))
}

function Test-ToolchainComplete {
    return (Test-Path (Join-Path $ToolchainDir "Vulkan-Headers\include\vulkan\vulkan.h")) -and
           (Test-Path (Join-Path $ToolchainDir "loader\vulkan-1.lib")) -and
           (Test-Path (Join-Path $ToolchainDir "shaderc\bin\glslc.exe")) -and
           (Test-Path (Join-Path $ToolchainDir "spirv-headers-install\share\cmake\SPIRV-Headers\SPIRV-HeadersConfig.cmake"))
}

if (Test-VulkanSdkComplete) {
    Write-Host "vulkan-toolchain: using installed Vulkan SDK at $env:VULKAN_SDK"
    exit 0
}

if (Test-ToolchainComplete) {
    if (Test-Truthy $env:TENSORSHARP_GGML_NO_UPDATE) {
        Write-Host "vulkan-toolchain: TENSORSHARP_GGML_NO_UPDATE set; using existing toolchain at $ToolchainDir"
    }
    else {
        Write-Host "vulkan-toolchain: already populated at $ToolchainDir"
    }
    exit 0
}

Write-Host "vulkan-toolchain: provisioning portable Vulkan toolchain in $ToolchainDir"
New-Item -ItemType Directory -Force -Path $ToolchainDir | Out-Null

# Steps 2 and 4 need the MSVC toolset (a C/C++ compiler for the SPIRV-Headers
# configure, dumpbin/lib.exe for the vulkan-1 import library). Locate it up
# front so a box without Visual Studio fails fast instead of after the downloads.
$VisualStudio = Get-VisualStudioInstallation
if ($null -eq $VisualStudio) {
    throw ("No Visual Studio installation with the MSVC x64 C++ toolset was found; it is required to build the " +
        "SPIRV-Headers package and the vulkan-1 import library. Set TENSORSHARP_VS_INSTALL_DIR to a VS installation root.")
}

# --- 1. Vulkan headers -------------------------------------------------------
$HeadersDir = Join-Path $ToolchainDir "Vulkan-Headers"
if (-not (Test-Path (Join-Path $HeadersDir "include\vulkan\vulkan.h"))) {
    if (Test-Path $HeadersDir) { Remove-Item -Recurse -Force $HeadersDir }
    git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Headers.git $HeadersDir
    if ($LASTEXITCODE -ne 0) { throw "git clone Vulkan-Headers failed" }
}

# --- 2. SPIRV-Headers CMake package -----------------------------------------
$SpirvSrcDir = Join-Path $ToolchainDir "SPIRV-Headers"
$SpirvInstallDir = Join-Path $ToolchainDir "spirv-headers-install"
if (-not (Test-Path (Join-Path $SpirvInstallDir "share\cmake\SPIRV-Headers\SPIRV-HeadersConfig.cmake"))) {
    if (-not (Test-Path (Join-Path $SpirvSrcDir "CMakeLists.txt"))) {
        if (Test-Path $SpirvSrcDir) { Remove-Item -Recurse -Force $SpirvSrcDir }
        git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers.git $SpirvSrcDir
        if ($LASTEXITCODE -ne 0) { throw "git clone SPIRV-Headers failed" }
    }
    # This script runs in its own powershell process, so a plain `dotnet build`
    # reaches this point without a compiler on PATH; CMake then cannot enable a
    # language ("CMAKE_C_COMPILER not set, after EnableLanguage"), because its
    # own Visual Studio discovery trusts the installer state the way
    # `vswhere -latest` does. Import vcvars and pin the NMake generator (cl.exe
    # from PATH); SPIRV-Headers is header-only, so nothing is actually compiled.
    Import-VcVarsEnvironment $VisualStudio.VcVars64
    $SpirvBuildDir = Join-Path $SpirvSrcDir "build"
    if (Test-Path $SpirvBuildDir) { Remove-Item -Recurse -Force $SpirvBuildDir }
    Invoke-CMakeDefanged "SPIRV-Headers cmake configure failed" @(
        "-S", $SpirvSrcDir, "-B", $SpirvBuildDir, "-G", "NMake Makefiles",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_INSTALL_PREFIX=$SpirvInstallDir")
    Invoke-CMakeDefanged "SPIRV-Headers cmake install failed" @("--install", $SpirvBuildDir, "--config", "Release")
}

# --- 3. glslc (Google shaderc CI prebuilt) -----------------------------------
$ShadercDir = Join-Path $ToolchainDir "shaderc"
if (-not (Test-Path (Join-Path $ShadercDir "bin\glslc.exe"))) {
    # The badge HTML is a meta-refresh redirect to the latest continuous build's
    # install archive; extract the target URL from it. The advertised file name
    # has changed over time (install.zip, then install.tgz) and has been seen
    # naming an extension that does not actually exist in the bucket while a
    # sibling archive does, so try the advertised URL first and then the known
    # sibling names.
    $badgeUrl = "https://storage.googleapis.com/shaderc/badges/build_link_windows_vs2022_amd64_release.html"
    $badge = (Invoke-WebRequest -UseBasicParsing -Uri $badgeUrl).Content
    if ($badge -notmatch 'url=(https://[^"'']+/install\.[a-z0-9.]+)') {
        throw "Could not resolve the shaderc prebuilt download URL from $badgeUrl"
    }
    $advertisedUrl = $Matches[1]
    $buildDirUrl = $advertisedUrl.Substring(0, $advertisedUrl.LastIndexOf('/'))
    $candidateUrls = @($advertisedUrl, "$buildDirUrl/install.zip", "$buildDirUrl/install.tgz") | Select-Object -Unique
    $archivePath = $null
    foreach ($candidateUrl in $candidateUrls) {
        $ext = if ($candidateUrl.EndsWith(".zip")) { "zip" } else { "tgz" }
        $tryPath = Join-Path $ToolchainDir "shaderc-install.$ext"
        try {
            Write-Host "vulkan-toolchain: downloading glslc from $candidateUrl"
            Invoke-WebRequest -UseBasicParsing -Uri $candidateUrl -OutFile $tryPath
            $archivePath = $tryPath
            $archiveExt = $ext
            break
        }
        catch {
            Write-Host "vulkan-toolchain: $candidateUrl not available: $($_.Exception.Message)"
        }
    }
    if ($null -eq $archivePath) {
        throw "Could not download the shaderc prebuilt from any of: $($candidateUrls -join ', ')"
    }
    $extractDir = Join-Path $ToolchainDir "shaderc-extract"
    if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
    if ($archiveExt -eq "zip") {
        Expand-Archive -Path $archivePath -DestinationPath $extractDir
    }
    else {
        New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
        tar -xzf $archivePath -C $extractDir
        if ($LASTEXITCODE -ne 0) { throw "Extracting $archivePath failed with exit code $LASTEXITCODE" }
    }
    if (Test-Path $ShadercDir) { Remove-Item -Recurse -Force $ShadercDir }
    # The archive contains a single top-level "install" folder with bin/lib/include.
    Move-Item (Join-Path $extractDir "install") $ShadercDir
    Remove-Item -Recurse -Force $extractDir
    Remove-Item -Force $archivePath
}

# --- 4. vulkan-1.lib import library ------------------------------------------
$LoaderDir = Join-Path $ToolchainDir "loader"
if (-not (Test-Path (Join-Path $LoaderDir "vulkan-1.lib"))) {
    $systemDll = Join-Path $env:SystemRoot "System32\vulkan-1.dll"
    if (-not (Test-Path $systemDll)) {
        throw "No Vulkan runtime found at $systemDll - install a GPU driver with Vulkan support (or the LunarG Vulkan SDK)."
    }

    $vcVersion = (Get-Content (Join-Path $VisualStudio.Path "VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt")).Trim()
    $vcBin = Join-Path $VisualStudio.Path "VC\Tools\MSVC\$vcVersion\bin\Hostx64\x64"
    if (-not (Test-Path (Join-Path $vcBin "lib.exe"))) {
        throw "MSVC binaries not found under $vcBin - the Visual Studio installation at '$($VisualStudio.Path)' has no x64 C++ toolset."
    }

    New-Item -ItemType Directory -Force -Path $LoaderDir | Out-Null
    $exports = & (Join-Path $vcBin "dumpbin.exe") /exports $systemDll
    $names = $exports | Where-Object { $_ -match '^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+(\S+)' } | ForEach-Object { $Matches[1] }
    if ($names.Count -lt 10) { throw "dumpbin found only $($names.Count) exports in vulkan-1.dll; refusing to build a broken import library." }
    $defPath = Join-Path $LoaderDir "vulkan-1.def"
    Set-Content -Path $defPath -Value (@("LIBRARY vulkan-1.dll", "EXPORTS") + $names) -Encoding ascii
    & (Join-Path $vcBin "lib.exe") /nologo "/def:$defPath" /machine:x64 "/out:$(Join-Path $LoaderDir 'vulkan-1.lib')"
    if ($LASTEXITCODE -ne 0) { throw "lib.exe failed to generate vulkan-1.lib" }
}

if (-not (Test-ToolchainComplete)) {
    throw "vulkan-toolchain: provisioning finished but the toolchain is still incomplete at $ToolchainDir"
}
Write-Host "vulkan-toolchain: ready at $ToolchainDir"
