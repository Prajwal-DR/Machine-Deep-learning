<#
.SYNOPSIS
    Switches the active Claude Code provider by dynamically discovering all
    variable names used across providers\*.env, clearing every one of them
    (current process + persisted Windows user environment), then loading
    only the selected provider's variables.

.PARAMETER Provider
    Name of the provider (matches providers\<Provider>.env, without extension).

.PARAMETER SessionVarsFile
    Optional path to a file that will receive "NAME=VALUE" lines for the
    variables that were applied, so the calling cmd session can import them
    into its own (parent) environment.
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Provider,

    [string]$SessionVarsFile,

    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

if ($Provider -notmatch '^[A-Za-z0-9_-]+$') {
    Write-Host "ERROR: Invalid provider name '$Provider'." -ForegroundColor Red
    exit 1
}

$scriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$providersDir = Join-Path $scriptDir 'providers'
$envFile      = Join-Path $providersDir "$Provider.env"

if (!(Test-Path $providersDir)) {
    Write-Host "ERROR: providers folder not found at $providersDir" -ForegroundColor Red
    exit 1
}

if (!(Test-Path $envFile)) {
    Write-Host "ERROR: $envFile not found!" -ForegroundColor Red
    exit 1
}

function Read-EnvFile {
    param([string]$Path)

    $result = [ordered]@{}
    foreach ($rawLine in Get-Content -Path $Path) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith('#')) { continue }

        $idx = $line.IndexOf('=')
        if ($idx -le 0) { continue }

        $key   = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1).Trim()

        # Strip a single pair of matching outer quotes (KEY="value" / KEY='value').
        # Values without surrounding quotes (e.g. "header: 01") are left untouched.
        if ($value.Length -ge 2) {
            $first = $value[0]
            $last  = $value[$value.Length - 1]
            if (($first -eq '"' -or $first -eq "'") -and $first -eq $last) {
                $value = $value.Substring(1, $value.Length - 2)
            }
        }

        $result[$key] = $value
    }
    return $result
}

function Get-DisplayValue {
    param([string]$Key, [string]$Value)

    if ($Key -match '(TOKEN|KEY|SECRET|AUTH|PASSWORD)') {
        if ($Value.Length -le 4) { return '****' }
        return '****' + $Value.Substring($Value.Length - 4)
    }
    return $Value
}

# ----------------------------------------------------------------------
# 1. Discover every variable name used by ANY provider .env file
# ----------------------------------------------------------------------
$allVarNames = New-Object System.Collections.Generic.HashSet[string]
$envFiles = Get-ChildItem -Path $providersDir -Filter '*.env'

foreach ($file in $envFiles) {
    foreach ($key in (Read-EnvFile -Path $file.FullName).Keys) {
        [void]$allVarNames.Add($key)
    }
}

Write-Host "Discovered $($allVarNames.Count) provider-related variable(s) across $($envFiles.Count) .env file(s)."

# ----------------------------------------------------------------------
# 2. Remove ALL of those variables from the process AND the persisted
#    Windows user environment (true delete, not just blanking them out)
# ----------------------------------------------------------------------
foreach ($name in $allVarNames) {
    if ($DryRun) {
        Write-Host "  [dry-run] would clear $name"
        continue
    }
    Remove-Item -Path "Env:\$name" -ErrorAction SilentlyContinue
    [Environment]::SetEnvironmentVariable($name, $null, 'User')
}

if ($DryRun) {
    Write-Host "[dry-run] No variables were actually cleared."
} else {
    Write-Host "Cleared stale variables from this session and the Windows user environment."
}
Write-Host ""

# ----------------------------------------------------------------------
# 3. Load the selected provider's variables (process + persisted user env)
# ----------------------------------------------------------------------
$selected = Read-EnvFile -Path $envFile

Write-Host "Loading $Provider.env ..."
foreach ($entry in $selected.GetEnumerator()) {
    if ($DryRun) { continue }
    Set-Item -Path "Env:\$($entry.Key)" -Value $entry.Value
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'User')
}

Write-Host ""
Write-Host "===== Variables Loaded ====="
foreach ($entry in $selected.GetEnumerator()) {
    Write-Host "$($entry.Key)=$(Get-DisplayValue -Key $entry.Key -Value $entry.Value)"
}
Write-Host "============================"
Write-Host ""

# ----------------------------------------------------------------------
# 4. Update Claude's settings.json to reflect ONLY this provider
# ----------------------------------------------------------------------
$settingsPath = Join-Path $env:USERPROFILE ".claude\settings.json"

if (Test-Path $settingsPath) {
    $json = Get-Content $settingsPath -Raw | ConvertFrom-Json

    # Build the variable list from ONLY the selected provider's .env, so any
    # entries left over from a previously selected provider are fully removed.
    $variables = $selected.GetEnumerator() |
        Where-Object { $_.Key -match '^(ANTHROPIC|CLAUDE|CLOUD)' } |
        Sort-Object Key |
        ForEach-Object { [PSCustomObject]@{ name = $_.Key; value = $_.Value } }

    $json.'claudeCode.environmentVariables' = @($variables)
    $jsonText = $json | ConvertTo-Json -Depth 100

    if ($DryRun) {
        Write-Host "[dry-run] would write the following claudeCode.environmentVariables:"
        Write-Host ($jsonText)
    } else {
        # Write UTF-8 WITHOUT BOM (a BOM can trip up JSON consumers).
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($settingsPath, $jsonText, $utf8NoBom)
        Write-Host "settings.json updated successfully."
    }
} else {
    Write-Host "WARNING: $settingsPath not found - skipped settings.json update." -ForegroundColor Yellow
}
Write-Host ""

# ----------------------------------------------------------------------
# 5. Hand the new variables back to the calling cmd session
# ----------------------------------------------------------------------
if ($SessionVarsFile -and -not $DryRun) {
    $lines = $selected.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }
    Set-Content -Path $SessionVarsFile -Value $lines -Encoding ASCII
}

if ($DryRun) {
    Write-Host "Dry run complete - no changes were made. Provider : $Provider" -ForegroundColor Cyan
} else {
    Write-Host "Active Provider : $Provider" -ForegroundColor Green
}
