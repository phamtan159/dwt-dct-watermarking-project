$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$OutputDir = "D:\A Project YTB\L2artic\regenerated_cache_v1"

& $Python tools\30_regenerate_l2arctic_feature_cache.py `
  --output-dir $OutputDir `
  --resume `
  --save-every 500

& $Python tools\31_train_l2arctic_frozen_protocol.py `
  --protocol data\protocols\l2arctic_frozen_source_vietnamese_raw_cache_v1.json `
  --summary (Join-Path $OutputDir "source_only_summary.csv")
