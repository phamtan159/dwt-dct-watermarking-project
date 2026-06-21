param(
    [string]$CondaEnv = "mfa",
    [string]$CorpusDir = "data/audio",
    [string]$Dictionary = "custom_mfa.dict",
    [string]$AcousticModel = "english_mfa",
    [string]$OutputDir = "data/aligned",
    [switch]$SkipValidate,
    [switch]$NoClean
)

$ErrorActionPreference = "Stop"

function Find-Conda {
    $candidates = @()
    if ($env:CONDA_EXE) {
        $candidates += $env:CONDA_EXE
    }
    $candidates += @(
        "D:\Miniconda3\Scripts\conda.exe",
        "D:\Anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    $command = Get-Command conda -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    throw "Could not find conda.exe. Set `$env:CONDA_EXE to your conda executable path."
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

$Conda = Find-Conda
$cleanFlag = if ($NoClean) { @() } else { @("--clean") }

Write-Host "Project: $ProjectRoot"
Write-Host "Conda:   $Conda"
Write-Host "Env:     $CondaEnv"
Write-Host "Corpus:  $CorpusDir"
Write-Host "Dict:    $Dictionary"
Write-Host "Model:   $AcousticModel"
Write-Host "Output:  $OutputDir"

if (-not (Test-Path -LiteralPath $CorpusDir)) {
    throw "Corpus directory not found: $CorpusDir. Run tools/03_prepare_mfa.py first."
}
if (-not (Test-Path -LiteralPath $Dictionary)) {
    throw "Dictionary not found: $Dictionary. Run tools/03_prepare_mfa.py first."
}

if (-not $SkipValidate) {
    Write-Host ""
    Write-Host "== MFA validate =="
    & $Conda run -n $CondaEnv mfa validate $CorpusDir $Dictionary $AcousticModel
}

Write-Host ""
Write-Host "== MFA align =="
& $Conda run -n $CondaEnv mfa align @cleanFlag $CorpusDir $Dictionary $AcousticModel $OutputDir

Write-Host ""
Write-Host "MFA done. Next:"
Write-Host "  python tools/04_textgrid_to_json.py"
