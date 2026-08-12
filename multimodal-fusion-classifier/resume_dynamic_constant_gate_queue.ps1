param(
    [int]$WaitForPid = 21408,
    [string]$Python = "E:\Anaconda\envs\yolov11\python.exe"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$reportRoot = Join-Path $root "reports\qwen-material\submission-revision\dynamic-vs-constant-gate"

Wait-Process -Id $WaitForPid -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

$stdout = Join-Path $reportRoot "run.resumed.stdout.log"
$stderr = Join-Path $reportRoot "run.resumed.stderr.log"
$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $root "run_dynamic_constant_gate_experiment.ps1"),
    "-Python", $Python,
    "-Device", "cuda"
)
$process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $arguments `
    -WorkingDirectory $root `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

Set-Content `
    -LiteralPath (Join-Path $reportRoot "run.pid") `
    -Value $process.Id `
    -Encoding ascii
Set-Content `
    -LiteralPath (Join-Path $reportRoot "run.resumed.started.txt") `
    -Value ("PID={0}`nSTARTED={1:o}" -f $process.Id, (Get-Date)) `
    -Encoding ascii
