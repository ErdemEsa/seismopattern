param(
    [switch]$NoRecord
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $root "venv\Scripts\python.exe"
$logDir = Join-Path $root "logs"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "prospective_weekly_$ts.log"

if (-not (Test-Path $pythonPath)) {
    throw "Python bulunamadi: $pythonPath"
}

function Invoke-Step {
    param(
        [string]$Label,
        [string[]]$ScriptArgs
    )

    "=== $Label ===" | Tee-Object -FilePath $logFile -Append
    & $pythonPath @ScriptArgs *>&1 | Tee-Object -FilePath $logFile -Append

    if ($LASTEXITCODE -ne 0) {
        throw "Adim basarisiz: $Label (exit=$LASTEXITCODE)"
    }

    "" | Tee-Object -FilePath $logFile -Append | Out-Null
}

"SeismoPattern weekly prospective run" | Out-File $logFile -Encoding utf8
"Timestamp: $(Get-Date -Format s)" | Out-File $logFile -Encoding utf8 -Append
"Python: $pythonPath" | Out-File $logFile -Encoding utf8 -Append
"" | Out-File $logFile -Encoding utf8 -Append

Set-Location $root

if (-not $NoRecord) {
    Invoke-Step -Label "record" -ScriptArgs @("scripts/prospective_tracker.py", "--record")
}

Invoke-Step -Label "verify" -ScriptArgs @("scripts/prospective_tracker.py", "--verify")
Invoke-Step -Label "check-events" -ScriptArgs @("scripts/prospective_tracker.py", "--check-events")

# 90 gun sonra acilabilir:
# Invoke-Step -Label "evaluate" -ScriptArgs @("scripts/prospective_tracker.py", "--evaluate")

"Tamamlandi: $logFile" | Tee-Object -FilePath $logFile -Append
