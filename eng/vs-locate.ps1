# Locate the Visual Studio installation that carries the MSVC x64 C++ toolset.
#
# Why this exists instead of a bare `vswhere -latest`: the VS installer flags an
# instance "incomplete" after an interrupted update or an unfinished repair, and
# `vswhere -latest` silently skips such instances - it prints nothing and still
# exits 0 - even though cl.exe, vcvars64.bat and the CUDA MSBuild integration on
# disk all work fine. Callers that trusted `-latest` therefore concluded "no
# Visual Studio here" and degraded badly rather than visibly: CMake could not use
# a "Visual Studio NN" generator and fell back to the *serial* NMake Makefiles
# generator, which ignores `--parallel` and compiles ggml's ~190 CUDA
# translation units one at a time.
#
# Resolution order:
#   1. TENSORSHARP_VS_INSTALL_DIR - explicit override, wins outright
#   2. vswhere -all -prerelease   - relaxed query that does see incomplete and
#                                   preview instances (newest version first)
#   3. filesystem probe           - the standard install roots, newest year
#                                   first, for VC\Auxiliary\Build\vcvars64.bat
#
# Dot-source this file to get Get-VisualStudioInstallation, or run it directly
# to print what was found. Returns $null when no MSVC toolset can be located.

$ErrorActionPreference = "Stop"

function Get-VisualStudioGeneratorName([string] $Version) {
    # Map the installed VS major version to its CMake generator name; a
    # hardcoded generator breaks on machines that only carry a different VS
    # (e.g. GitHub's windows-latest image now ships Visual Studio 2026 only).
    switch (("$Version".Trim() -split '\.')[0]) {
        "16" { return "Visual Studio 16 2019" }
        "17" { return "Visual Studio 17 2022" }
        "18" { return "Visual Studio 18 2026" }
        # Unknown/newer VS: return nothing and let CMake pick its own default.
        default { return "" }
    }
}

function Get-VisualStudioVersionFromPath([string] $Path) {
    # The filesystem probe has no version metadata, only the release-year
    # directory ("...\Microsoft Visual Studio\2022\Community").
    if ("$Path" -match '\\Microsoft Visual Studio\\(\d{4})\\') {
        switch ($Matches[1]) {
            "2019" { return "16.0" }
            "2022" { return "17.0" }
            "2026" { return "18.0" }
        }
    }

    return ""
}

function New-VisualStudioInstallation([string] $Path, [string] $Version) {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }

    # vcvars64.bat is the thing callers actually need (and the proof that the x64
    # C++ toolset is really on disk, whatever the installer's state flag says).
    $vcvars = Join-Path $Path "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) {
        return $null
    }

    if ([string]::IsNullOrWhiteSpace($Version)) {
        $Version = Get-VisualStudioVersionFromPath $Path
    }

    return [pscustomobject]@{
        Path      = $Path
        Version   = $Version
        Generator = Get-VisualStudioGeneratorName $Version
        VcVars64  = $vcvars
    }
}

function Find-VisualStudioViaVswhere {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        return $null
    }

    # -all -prerelease lifts the default filters that hide incomplete and
    # preview instances. -format json keeps installationPath paired with
    # installationVersion on machines carrying several instances, which separate
    # -property queries would not.
    $output = & $vswhere -all -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json
    if ($LASTEXITCODE -ne 0) {
        return $null
    }

    $text = ($output | Out-String).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $null
    }

    try {
        $instances = @($text | ConvertFrom-Json)
    }
    catch {
        return $null
    }

    $sorted = $instances | Sort-Object -Property @{ Expression = {
        try { [version] $_.installationVersion } catch { [version] "0.0" }
    } } -Descending

    foreach ($instance in $sorted) {
        $found = New-VisualStudioInstallation $instance.installationPath $instance.installationVersion
        if ($null -ne $found) {
            return $found
        }
    }

    return $null
}

function Find-VisualStudioViaFilesystem {
    $roots = @($env:ProgramFiles, ${env:ProgramFiles(x86)}) |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { Join-Path $_ "Microsoft Visual Studio" }

    foreach ($root in $roots) {
        if (-not (Test-Path $root)) {
            continue
        }

        $yearDirs = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^\d{4}$' } |
            Sort-Object -Property Name -Descending
        foreach ($yearDir in $yearDirs) {
            # Any edition will do (Community/Professional/Enterprise/BuildTools);
            # they all ship the same vcvars64.bat and MSVC toolset.
            $editionDirs = Get-ChildItem -Path $yearDir.FullName -Directory -ErrorAction SilentlyContinue
            foreach ($editionDir in $editionDirs) {
                $found = New-VisualStudioInstallation $editionDir.FullName ""
                if ($null -ne $found) {
                    return $found
                }
            }
        }
    }

    return $null
}

function Get-VisualStudioInstallation {
    if (-not [string]::IsNullOrWhiteSpace($env:TENSORSHARP_VS_INSTALL_DIR)) {
        $override = New-VisualStudioInstallation $env:TENSORSHARP_VS_INSTALL_DIR ""
        if ($null -ne $override) {
            return $override
        }

        Write-Warning ("TENSORSHARP_VS_INSTALL_DIR='$env:TENSORSHARP_VS_INSTALL_DIR' does not contain " +
            "VC\Auxiliary\Build\vcvars64.bat; ignoring it and probing for Visual Studio normally.")
    }

    $found = Find-VisualStudioViaVswhere
    if ($null -ne $found) {
        return $found
    }

    return Find-VisualStudioViaFilesystem
}

# Import the MSVC command-line environment (PATH/INCLUDE/LIB/LIBPATH) into the
# current process, the way an "x64 Native Tools" prompt does. Generators that
# invoke the compilers directly (Ninja, NMake) need it; the Visual Studio
# generator does not, because MSBuild sets it up per project.
#
# The variables land in this PowerShell process only, so an MSBuild that shelled
# out to this script never sees them.
function Import-VcVarsEnvironment([string] $VcVarsPath) {
    if (-not [string]::IsNullOrWhiteSpace($env:VCToolsInstallDir)) {
        Write-Host "MSVC environment already active (VCToolsInstallDir=$env:VCToolsInstallDir); not importing vcvars."
        return
    }

    if ([string]::IsNullOrWhiteSpace($VcVarsPath) -or -not (Test-Path $VcVarsPath)) {
        throw "vcvars64.bat not found at '$VcVarsPath'."
    }

    Write-Host "Importing MSVC environment from $VcVarsPath"
    # vcvars' sub-scripts shell out to a bare `vswhere.exe`, which only resolves
    # when the VS Installer directory is on PATH (it is inside a VS developer
    # prompt, but not in a plain shell). Without it vcvars prints "'vswhere.exe'
    # is not recognized ..." and skips part of its detection, so put it on PATH
    # for the duration of the import.
    $installerDir = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"
    $originalPath = $env:PATH
    if ((Test-Path $installerDir) -and $env:PATH -notlike "*$installerDir*") {
        $env:PATH = "$installerDir;$env:PATH"
    }

    try {
        # cmd /s /c "<quoted bat> && set": /s makes cmd strip only the outermost
        # quote pair, which keeps the quoted path (spaces) intact.
        $lines = & cmd.exe /s /c "`"$VcVarsPath`" && set"
    }
    finally {
        # vcvars' own PATH (captured in $lines) is applied below; drop the
        # temporary prepend so it cannot end up duplicated.
        $env:PATH = $originalPath
    }

    if ($LASTEXITCODE -ne 0) {
        throw "vcvars64.bat failed with exit code $LASTEXITCODE"
    }

    $applied = 0
    foreach ($line in $lines) {
        if ("$line" -notmatch '^([^=]+)=(.*)$') {
            continue
        }

        $name = $Matches[1]
        $value = $Matches[2]
        # Platform=x64 is a vcvars artifact that nothing in a compiler-driven
        # build reads, but any nested MSBuild would - it overrides the project's
        # own platform and breaks managed builds ("Any CPU"). Drop it.
        if ($name -eq "Platform") {
            continue
        }

        Set-Item -Path "Env:$name" -Value $value
        $applied++
    }

    if ($applied -eq 0) {
        throw "vcvars64.bat produced no environment variables."
    }
}

# Direct invocation (not dot-sourced): report what was found.
if ($MyInvocation.InvocationName -ne '.') {
    $installation = Get-VisualStudioInstallation
    if ($null -eq $installation) {
        Write-Warning "No Visual Studio installation with the MSVC x64 C++ toolset was found."
        exit 1
    }

    $installation | Format-List | Out-String | Write-Host
}
