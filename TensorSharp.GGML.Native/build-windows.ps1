param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $RemainingArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $ScriptDir "build-windows"

# Get-VisualStudioInstallation / Import-VcVarsEnvironment. See eng/vs-locate.ps1
# for why a bare `vswhere -latest` cannot be trusted to find the local MSVC.
. (Join-Path $ScriptDir "..\eng\vs-locate.ps1")

$EnableCuda = $env:TENSORSHARP_GGML_NATIVE_ENABLE_CUDA
$EnableVulkan = $env:TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN
$BuildTests = if ([string]::IsNullOrWhiteSpace($env:TENSORSHARP_GGML_NATIVE_BUILD_TESTS)) { "OFF" } else { $env:TENSORSHARP_GGML_NATIVE_BUILD_TESTS }
$CudaArchitectures = $env:TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES
$BuildParallelLevel = if ([string]::IsNullOrWhiteSpace($env:TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL)) { $env:CMAKE_BUILD_PARALLEL_LEVEL } else { $env:TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL }
$ExtraCMakeArgs = New-Object System.Collections.Generic.List[string]
$UserSetCudaArchitectures = $false
$UserSetGenerator = $false
$UserGenerator = ""
$UserSetPlatform = $false

function Normalize-Bool([string] $Value) {
    switch -Regex ($Value) {
        '^(ON|on|On|TRUE|true|True|YES|yes|Yes|1)$' { return "ON" }
        '^(OFF|off|Off|FALSE|false|False|NO|no|No|0)$' { return "OFF" }
        default { return "" }
    }
}

function Test-CudaToolkit {
    if (Get-Command nvcc.exe -ErrorAction SilentlyContinue) {
        return $true
    }

    foreach ($variableName in @("CUDA_PATH", "CUDA_HOME")) {
        $root = [Environment]::GetEnvironmentVariable($variableName)
        if ([string]::IsNullOrWhiteSpace($root)) {
            continue
        }

        if (Test-Path (Join-Path $root "bin\nvcc.exe")) {
            return $true
        }
    }

    return $false
}

function Read-CachedBackendSetting([string] $CacheVariable) {
    $cacheFile = Join-Path $BuildDir "CMakeCache.txt"
    if (-not (Test-Path $cacheFile)) {
        return ""
    }

    $line = Select-String -Path $cacheFile -Pattern "^$([regex]::Escape($CacheVariable)):BOOL=" | Select-Object -First 1
    if ($null -eq $line) {
        return ""
    }

    return Normalize-Bool (($line.Line -split '=', 2)[1])
}

function Detect-LocalCudaArchitectures {
    $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        return ""
    }

    $caps = & $nvidiaSmi.Source --query-gpu=compute_cap --format=csv,noheader 2>$null
    if ($LASTEXITCODE -ne 0 -or $null -eq $caps) {
        return ""
    }

    $architectures = New-Object System.Collections.Generic.List[string]
    foreach ($cap in $caps) {
        $clean = ($cap -as [string]).Trim()
        if ($clean -notmatch '^([0-9]+)\.([0-9]+)$') {
            continue
        }

        $arch = "$($Matches[1])$($Matches[2])-real"
        if (-not $architectures.Contains($arch)) {
            $architectures.Add($arch)
        }
    }

    return ($architectures -join ';')
}

function Detect-DefaultBuildParallelLevel([bool] $CudaEnabled) {
    # Use all CPUs, bounded only by RAM. nvcc is memory-hungry, so allow ~3 GB per
    # parallel job (matches llama.cpp/ggml's own heuristic). The previous hard cap
    # of 4 jobs throttled the CUDA compile to a fraction of available cores even on
    # large machines with plenty of RAM; the memory bound below already prevents
    # OOM. Override with TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL when needed.
    #
    # Only a CUDA build needs the 3 GB bound: cl.exe peaks around a few hundred MB
    # per translation unit, so applying nvcc's budget to a CPU-only build left
    # cores idle for no reason (a 32 GB / 16-core box was capped at 10 jobs).
    $jobs = [Math]::Max(1, [Environment]::ProcessorCount)
    $bytesPerJob = if ($CudaEnabled) { 3GB } else { 1GB }

    try {
        $computer = Get-CimInstance -ClassName Win32_ComputerSystem
        if ($null -ne $computer.TotalPhysicalMemory) {
            $memoryLimitedJobs = [Math]::Max(1, [int][Math]::Floor([double]$computer.TotalPhysicalMemory / $bytesPerJob))
            $jobs = [Math]::Min($jobs, $memoryLimitedJobs)
        }
    }
    catch {
        # Keep the CPU-based default if CIM is unavailable.
    }

    return $jobs
}

function Find-NinjaProgram([object] $VisualStudio) {
    # A user-installed ninja on PATH wins.
    $onPath = Get-Command ninja.exe -ErrorAction SilentlyContinue
    if ($null -ne $onPath) {
        return $onPath.Source
    }

    # Every Visual Studio install carrying the "C++ CMake tools for Windows"
    # component ships a private ninja; CMake's own installer does not. Using it
    # means no download and no extra prerequisite for the common VS setup.
    if ($null -ne $VisualStudio) {
        $bundled = Join-Path $VisualStudio.Path "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
        if (Test-Path $bundled) {
            return $bundled
        }
    }

    return ""
}

function Read-CachedGenerator {
    $cacheFile = Join-Path $BuildDir "CMakeCache.txt"
    if (-not (Test-Path $cacheFile)) {
        return ""
    }

    $line = Select-String -Path $cacheFile -Pattern "^CMAKE_GENERATOR:INTERNAL=" | Select-Object -First 1
    if ($null -eq $line) {
        return ""
    }

    return (($line.Line -split '=', 2)[1]).Trim()
}

for ($i = 0; $i -lt $RemainingArgs.Length; $i++) {
    $arg = $RemainingArgs[$i]

    switch -Regex ($arg) {
        '^--cuda$' {
            $EnableCuda = "ON"
            continue
        }
        '^--no-cuda$' {
            $EnableCuda = "OFF"
            continue
        }
        '^--vulkan$' {
            $EnableVulkan = "ON"
            continue
        }
        '^--no-vulkan$' {
            $EnableVulkan = "OFF"
            continue
        }
        '^--tests$' {
            $BuildTests = "ON"
            continue
        }
        '^--cuda-arch=(.+)$' {
            $CudaArchitectures = $Matches[1]
            continue
        }
        '^--cuda-arch$' {
            $i++
            if ($i -ge $RemainingArgs.Length) {
                throw "Missing value for --cuda-arch"
            }
            $CudaArchitectures = $RemainingArgs[$i]
            continue
        }
        '^-DCMAKE_CUDA_ARCHITECTURES=.+$' {
            $UserSetCudaArchitectures = $true
            $ExtraCMakeArgs.Add($arg)
            continue
        }
        '^-G$' {
            $UserSetGenerator = $true
            $ExtraCMakeArgs.Add($arg)
            $i++
            if ($i -ge $RemainingArgs.Length) {
                throw "Missing value for -G"
            }
            # Remember the name too, not just that one was given: the generator
            # decides whether this script has to import the MSVC environment.
            $UserGenerator = $RemainingArgs[$i]
            $ExtraCMakeArgs.Add($RemainingArgs[$i])
            continue
        }
        '^-G.+$' {
            $UserSetGenerator = $true
            $UserGenerator = $arg.Substring(2)
            $ExtraCMakeArgs.Add($arg)
            continue
        }
        '^-A$' {
            $UserSetPlatform = $true
            $ExtraCMakeArgs.Add($arg)
            $i++
            if ($i -ge $RemainingArgs.Length) {
                throw "Missing value for -A"
            }
            $ExtraCMakeArgs.Add($RemainingArgs[$i])
            continue
        }
        '^-A.+$' {
            $UserSetPlatform = $true
            $ExtraCMakeArgs.Add($arg)
            continue
        }
        default {
            $ExtraCMakeArgs.Add($arg)
        }
    }
}

$EnableCuda = Normalize-Bool $EnableCuda
if ([string]::IsNullOrWhiteSpace($EnableCuda)) {
    $EnableCuda = Read-CachedBackendSetting "TENSORSHARP_GGML_NATIVE_ENABLE_CUDA"
}
if ([string]::IsNullOrWhiteSpace($EnableCuda) -and (Test-CudaToolkit)) {
    $EnableCuda = "ON"
}
if ([string]::IsNullOrWhiteSpace($EnableCuda)) {
    $EnableCuda = "OFF"
}

# Vulkan: an explicit choice (--vulkan/--no-vulkan or
# TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN) wins and sticks via the CMake cache;
# TENSORSHARP_GGML_NATIVE_VULKAN_EXPLICIT records that the cached value was
# explicit, so an auto-detected default never becomes sticky. Otherwise the
# backend is auto-enabled when the machine supports Vulkan (the runtime
# vulkan-1.dll that ships with every recent GPU driver is present) and the
# build toolchain (headers, loader import lib, glslc, SPIRV-Headers) is
# downloaded automatically by eng/fetch-vulkan-toolchain.ps1 when missing. An
# auto-enable whose toolchain cannot be provisioned (e.g. no network) degrades
# to a warning and builds without Vulkan; an explicit --vulkan makes it fatal.
$VulkanExplicit = "OFF"
$EnableVulkan = Normalize-Bool $EnableVulkan
if (-not [string]::IsNullOrWhiteSpace($EnableVulkan)) {
    $VulkanExplicit = "ON"
}
elseif ((Read-CachedBackendSetting "TENSORSHARP_GGML_NATIVE_VULKAN_EXPLICIT") -eq "ON") {
    $EnableVulkan = Read-CachedBackendSetting "TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN"
    if (-not [string]::IsNullOrWhiteSpace($EnableVulkan)) {
        $VulkanExplicit = "ON"
    }
}
if ([string]::IsNullOrWhiteSpace($EnableVulkan)) {
    $EnableVulkan = if (Test-Path (Join-Path $env:SystemRoot "System32\vulkan-1.dll")) { "ON" } else { "OFF" }
}

$VulkanCMakeArgs = New-Object System.Collections.Generic.List[string]
if ($EnableVulkan -eq "ON") {
    try {
        # Ensure the Vulkan build toolchain (headers, loader import lib, glslc,
        # SPIRV-Headers) is available. With a LunarG SDK installed the script is a
        # no-op and FindVulkan discovers everything through VULKAN_SDK; otherwise it
        # provisions a portable toolchain that is passed to CMake explicitly below.
        & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $ScriptDir "..\eng\fetch-vulkan-toolchain.ps1")
        if ($LASTEXITCODE -ne 0) { throw "fetch-vulkan-toolchain.ps1 failed with exit code $LASTEXITCODE" }

        $VulkanSdk = $env:VULKAN_SDK
        $HasFullSdk = -not [string]::IsNullOrWhiteSpace($VulkanSdk) -and
            (Test-Path (Join-Path $VulkanSdk "Include\vulkan\vulkan.h")) -and
            (Test-Path (Join-Path $VulkanSdk "Lib\vulkan-1.lib")) -and
            (Test-Path (Join-Path $VulkanSdk "Bin\glslc.exe"))
        if (-not $HasFullSdk) {
            $ToolchainDir = (Resolve-Path (Join-Path $ScriptDir "..\ExternalProjects\vulkan-toolchain")).Path
            $VulkanCMakeArgs.Add("-DVulkan_INCLUDE_DIR=$(Join-Path $ToolchainDir 'Vulkan-Headers\include')")
            $VulkanCMakeArgs.Add("-DVulkan_LIBRARY=$(Join-Path $ToolchainDir 'loader\vulkan-1.lib')")
            $VulkanCMakeArgs.Add("-DVulkan_GLSLC_EXECUTABLE=$(Join-Path $ToolchainDir 'shaderc\bin\glslc.exe')")
            $VulkanCMakeArgs.Add("-DSPIRV-Headers_DIR=$(Join-Path $ToolchainDir 'spirv-headers-install\share\cmake\SPIRV-Headers')")
        }
    }
    catch {
        if ($VulkanExplicit -eq "ON") { throw }
        Write-Warning ("This machine supports Vulkan but the ggml-vulkan build toolchain could not be provisioned: " +
            "$($_.Exception.Message) Building without the ggml-vulkan backend; pass --vulkan to make this an error.")
        $EnableVulkan = "OFF"
        $VulkanCMakeArgs.Clear()
    }
}

if ($EnableCuda -eq "ON" -and [string]::IsNullOrWhiteSpace($CudaArchitectures) -and -not $UserSetCudaArchitectures) {
    $detectedArchitectures = Detect-LocalCudaArchitectures
    if (-not [string]::IsNullOrWhiteSpace($detectedArchitectures)) {
        $CudaArchitectures = $detectedArchitectures
    }
}

$CudaArchSummary = "n/a"
if ($EnableCuda -eq "ON") {
    if ($UserSetCudaArchitectures) {
        $CudaArchSummary = "custom via CMAKE_CUDA_ARCHITECTURES"
    }
    elseif (-not [string]::IsNullOrWhiteSpace($CudaArchitectures)) {
        $CudaArchSummary = $CudaArchitectures
    }
    else {
        $CudaArchSummary = "ggml default"
    }
}

if ([string]::IsNullOrWhiteSpace($BuildParallelLevel)) {
    $BuildParallelLevel = Detect-DefaultBuildParallelLevel ($EnableCuda -eq "ON")
}
if (($BuildParallelLevel -as [int]) -lt 1) {
    throw "Invalid build parallel level: $BuildParallelLevel"
}

# Generator choice, in order of preference:
#
#   1. Ninja - one dependency graph over every translation unit, so all ~190 CUDA
#      TUs plus the ggml/GgmlOps C++ sources saturate the requested job count
#      regardless of which CMake target they belong to. It also sidesteps the
#      MSBuild file-tracker path-length limit that CMakeLists.txt guards against,
#      and needs no CUDA MSBuild integration for the installed VS version.
#   2. The "Visual Studio NN" generator - MSBuild parallelises across projects
#      only, so ggml-cuda's TUs still compile serially, but it works without a
#      vcvars environment.
#   3. Nothing, letting CMake decide - warned about below, because CMake's own
#      fallback when it cannot see a VS instance is the strictly serial
#      NMake Makefiles generator, where `--parallel` is silently ignored.
$VisualStudio = Get-VisualStudioInstallation
$NinjaProgram = Find-NinjaProgram $VisualStudio

$GeneratorArgs = New-Object System.Collections.Generic.List[string]
if (-not $UserSetGenerator -and [string]::IsNullOrWhiteSpace($env:CMAKE_GENERATOR)) {
    if (-not [string]::IsNullOrWhiteSpace($NinjaProgram)) {
        $GeneratorArgs.Add("-G")
        $GeneratorArgs.Add("Ninja")
    }
    elseif ($null -ne $VisualStudio -and -not [string]::IsNullOrWhiteSpace($VisualStudio.Generator)) {
        $GeneratorArgs.Add("-G")
        $GeneratorArgs.Add($VisualStudio.Generator)
    }
}

# An explicit -G on the command line beats $env:CMAKE_GENERATOR, which beats the
# generator chosen above. A caller-supplied -G was appended to $ExtraCMakeArgs by
# the argument loop, so it is tracked separately in $UserGenerator.
$effectiveGenerator =
    if (-not [string]::IsNullOrWhiteSpace($UserGenerator)) { $UserGenerator }
    elseif (-not [string]::IsNullOrWhiteSpace($env:CMAKE_GENERATOR)) { $env:CMAKE_GENERATOR }
    else { ($GeneratorArgs | Select-Object -Last 1) }
if (-not $UserSetPlatform -and [string]::IsNullOrWhiteSpace($env:CMAKE_GENERATOR_PLATFORM) -and $effectiveGenerator -like "Visual Studio*") {
    $GeneratorArgs.Add("-A")
    $GeneratorArgs.Add("x64")
}

$GeneratorCMakeArgs = New-Object System.Collections.Generic.List[string]
if ($effectiveGenerator -like "*Ninja*") {
    # Ninja drives cl.exe/link.exe/nvcc itself, so this process needs the MSVC
    # environment. Importing it here means a plain `dotnet build` works from any
    # shell, not just from an "x64 Native Tools" prompt.
    if ($null -ne $VisualStudio) {
        Import-VcVarsEnvironment $VisualStudio.VcVars64
    }
    elseif ([string]::IsNullOrWhiteSpace($env:VCToolsInstallDir)) {
        Write-Warning ("Generator '$effectiveGenerator' needs the MSVC command-line environment but no Visual Studio " +
            "installation was found; run from an 'x64 Native Tools' prompt if the configure step cannot find a compiler.")
    }

    # Pin the ninja we found: when it comes from the Visual Studio install it is
    # not on PATH, so CMake would not otherwise locate it.
    if (-not [string]::IsNullOrWhiteSpace($NinjaProgram)) {
        $GeneratorCMakeArgs.Add("-DCMAKE_MAKE_PROGRAM=$NinjaProgram")
    }
}
elseif ([string]::IsNullOrWhiteSpace($effectiveGenerator) -or $effectiveGenerator -like "*NMake*") {
    Write-Warning ("No Ninja and no usable Visual Studio generator were found, so this build may fall back to the " +
        "serial NMake Makefiles generator, which ignores --parallel and compiles one file at a time. Install the " +
        "'C++ CMake tools for Windows' VS component (it provides ninja), put ninja.exe on PATH, or set " +
        "TENSORSHARP_VS_INSTALL_DIR to a working Visual Studio installation.")
}

# The generator is sticky in the CMake cache and CMake refuses to reconfigure an
# existing build tree with a different one, so switching (e.g. off an older
# NMake-configured tree onto Ninja) means starting the tree over.
$CachedGenerator = Read-CachedGenerator
if (-not [string]::IsNullOrWhiteSpace($CachedGenerator) -and
    -not [string]::IsNullOrWhiteSpace($effectiveGenerator) -and
    $CachedGenerator -ne $effectiveGenerator) {
    Write-Host "Generator changed ('$CachedGenerator' -> '$effectiveGenerator'); removing $BuildDir to reconfigure from scratch."
    Remove-Item -Recurse -Force $BuildDir
}

$GeneratorSummary = if ([string]::IsNullOrWhiteSpace($effectiveGenerator)) { "CMake default" } else { $effectiveGenerator }
Write-Host "Configuring TensorSharp.GGML.Native (CUDA=$EnableCuda, CUDA_ARCHITECTURES=$CudaArchSummary, VULKAN=$EnableVulkan, TESTS=$BuildTests, GENERATOR=$GeneratorSummary, PARALLEL=$BuildParallelLevel)"

$CMakeArgs = @(
    "-DCMAKE_BUILD_TYPE=Release",
    "-DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=$EnableCuda",
    "-DTENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=$EnableVulkan",
    "-DTENSORSHARP_GGML_NATIVE_VULKAN_EXPLICIT=$VulkanExplicit",
    "-DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=$BuildTests"
)
$CMakeArgs += $VulkanCMakeArgs

if ($EnableCuda -eq "ON" -and -not [string]::IsNullOrWhiteSpace($CudaArchitectures) -and -not $UserSetCudaArchitectures) {
    $CMakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
}

# Ensure the ggml sources are present (cloned from ggml-org/ggml at build time).
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $ScriptDir "..\eng\fetch-ggml.ps1")
if ($LASTEXITCODE -ne 0) { throw "fetch-ggml.ps1 failed with exit code $LASTEXITCODE" }

cmake -S $ScriptDir -B $BuildDir @GeneratorArgs @GeneratorCMakeArgs @CMakeArgs @ExtraCMakeArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed with exit code $LASTEXITCODE" }

$BuildArgs = @("--build", $BuildDir, "--config", "Release", "--parallel", "$BuildParallelLevel")
if ((Normalize-Bool $BuildTests) -eq "ON") {
    cmake @BuildArgs
}
else {
    cmake @BuildArgs --target GgmlOps
}
