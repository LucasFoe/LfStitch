<#
.SYNOPSIS
    Convenience script to activate the project virtual environment (.venv).
.DESCRIPTION
    Dot-sources the virtual environment activation script located in .venv\Scripts\Activate.ps1.
#>

$venvActivate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"

if (Test-Path $venvActivate) {
    . $venvActivate @args
} else {
    Write-Error "Virtual environment activation script not found at: $venvActivate"
}
