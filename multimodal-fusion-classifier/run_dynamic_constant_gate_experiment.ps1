param(
    [string]$Python = "E:\Anaconda\envs\yolov11\python.exe",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ArtifactRoot = "C:\mmfc-reports\submission-revision\dynamic-vs-constant-gate"
$TrainingPidRoot = "reports/qwen-material/submission-revision/dynamic-vs-constant-gate/training-pids"
New-Item -ItemType Directory -Path $ArtifactRoot -Force | Out-Null
New-Item -ItemType Directory -Path $TrainingPidRoot -Force | Out-Null
& $Python prepare_constant_gate_experiment.py
if ($LASTEXITCODE -ne 0) { throw "config preparation failed" }

$runs = @(
    [pscustomobject]@{
        Seed = 42
        DynamicConfig = "configs/ablations/full_cross_modal_quality.json"
        DynamicCheckpoint = "checkpoints/ablations/full_cross_modal_quality/best.pt"
    },
    [pscustomobject]@{
        Seed = 43
        DynamicConfig = "configs/replicates/full_cross_modal_quality_seed43.json"
        DynamicCheckpoint = "checkpoints/ablations/v3_seed_replicates/full_cross_modal_quality_seed43/best.pt"
    },
    [pscustomobject]@{
        Seed = 44
        DynamicConfig = "configs/replicates/full_cross_modal_quality_seed44.json"
        DynamicCheckpoint = "checkpoints/ablations/v3_seed_replicates/full_cross_modal_quality_seed44/best.pt"
    }
)

function Test-ConstantTrainingComplete {
    param([int]$Seed)

    $root = "C:\mmfc-checkpoints\submission-revision\constant-gate-workers4\seed$Seed"
    $checkpoint = "$root/best.pt"
    $historyPath = "$root/history.json"
    if (-not (Test-Path -LiteralPath $checkpoint) -or -not (Test-Path -LiteralPath $historyPath)) {
        return $false
    }
    $history = Get-Content -LiteralPath $historyPath -Raw | ConvertFrom-Json
    return @($history.history).Count -eq 100
}

function Get-ActiveConstantTrainingProcess {
    param([int]$Seed)

    $pidPath = "$TrainingPidRoot/seed$Seed.pid"
    if (-not (Test-Path -LiteralPath $pidPath)) {
        return $null
    }
    $trainingPid = [int](Get-Content -LiteralPath $pidPath -Raw).Trim()
    $cimProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $trainingPid" -ErrorAction SilentlyContinue
    if ($null -eq $cimProcess -or $cimProcess.CommandLine -notlike "*constant_gate_seed$Seed.json*") {
        Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
        return $null
    }
    return Get-Process -Id $trainingPid -ErrorAction SilentlyContinue
}

function Start-ConstantTrainingProcess {
    param(
        [int]$Seed,
        [string]$Config
    )

    $stdout = "$TrainingPidRoot/seed$Seed.stdout.log"
    $stderr = "$TrainingPidRoot/seed$Seed.stderr.log"
    $process = Start-Process `
        -FilePath $Python `
        -ArgumentList @("train.py", "--config", $Config) `
        -WorkingDirectory (Get-Location).Path `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
    Set-Content -LiteralPath "$TrainingPidRoot/seed$Seed.pid" -Value $process.Id -Encoding ascii
    return $process
}

# Training seeds are independent. Running incomplete seeds concurrently changes
# neither their configuration nor sample order, while using otherwise idle GPU
# capacity. PID files make this block restart-safe and prevent duplicate runs.
$trainingProcesses = @()
foreach ($run in $runs) {
    $seed = $run.Seed
    if (Test-ConstantTrainingComplete -Seed $seed) {
        continue
    }
    $process = Get-ActiveConstantTrainingProcess -Seed $seed
    if ($null -eq $process) {
        $constantConfig = "configs/submission-revision/constant-gate/constant_gate_seed$seed.json"
        $process = Start-ConstantTrainingProcess -Seed $seed -Config $constantConfig
    }
    $trainingProcesses += $process
}

foreach ($process in $trainingProcesses) {
    $process.WaitForExit()
}

foreach ($run in $runs) {
    if (-not (Test-ConstantTrainingComplete -Seed $run.Seed)) {
        throw "constant-gate training did not complete for seed $($run.Seed)"
    }
    Remove-Item -LiteralPath "$TrainingPidRoot/seed$($run.Seed).pid" -Force -ErrorAction SilentlyContinue
}

foreach ($run in $runs) {
    $seed = $run.Seed
    $constantConfig = "configs/submission-revision/constant-gate/constant_gate_seed$seed.json"
    $constantRoot = "C:\mmfc-checkpoints\submission-revision\constant-gate-workers4\seed$seed"
    $constantCheckpoint = "$constantRoot/best.pt"
    $historyPath = "$constantRoot/history.json"
    $trainingComplete = $false
    if ((Test-Path -LiteralPath $constantCheckpoint) -and (Test-Path -LiteralPath $historyPath)) {
        $history = Get-Content -LiteralPath $historyPath -Raw | ConvertFrom-Json
        $trainingComplete = @($history.history).Count -eq 100
    }
    if (-not $trainingComplete) {
        & $Python train.py --config $constantConfig
        if ($LASTEXITCODE -ne 0) { throw "constant-gate training failed for seed $seed" }
    }
    $history = Get-Content -LiteralPath $historyPath -Raw | ConvertFrom-Json
    if (-not (Test-Path -LiteralPath $constantCheckpoint) -or @($history.history).Count -ne 100) {
        throw "constant-gate run is incomplete for seed $seed"
    }
    $lastCheckpoint = "$constantRoot/last.pt"
    if (Test-Path -LiteralPath $lastCheckpoint) {
        Remove-Item -LiteralPath $lastCheckpoint -Force
    }
    $finalRoot = "checkpoints/submission-revision/constant-gate-matched/seed$seed"
    New-Item -ItemType Directory -Path $finalRoot -Force | Out-Null
    Copy-Item -LiteralPath $constantCheckpoint -Destination "$finalRoot/best.pt" -Force
    Copy-Item -LiteralPath $historyPath -Destination "$finalRoot/history.json" -Force
    $variants = @(
        [pscustomobject]@{
            Name = "dynamic"
            Config = $run.DynamicConfig
            Checkpoint = $run.DynamicCheckpoint
        },
        [pscustomobject]@{
            Name = "constant"
            Config = $constantConfig
            Checkpoint = $constantCheckpoint
        }
    )
    foreach ($variant in $variants) {
        $testOutput = "$ArtifactRoot/seed$seed/$($variant.Name)/test"
        & $Python evaluate.py `
            --config $variant.Config `
            --checkpoint $variant.Checkpoint `
            --split test `
            --output $testOutput `
            --batch-size 16 `
            --num-workers 0 `
            --device $Device `
            --bootstrap-repeats 2000 `
            --bootstrap-seed 20260727
        if ($LASTEXITCODE -ne 0) { throw "$($variant.Name) clean evaluation failed for seed $seed" }

        $output = "$ArtifactRoot/seed$seed/$($variant.Name)/robustness"
        & $Python evaluate_robustness.py `
            --config $variant.Config `
            --checkpoint $variant.Checkpoint `
            --split test `
            --output $output `
            --scenarios missing_100 noise_015 bias_008 stuck mixed `
            --seeds 2026 2027 2028 2029 2030 `
            --batch-size 16 `
            --num-workers 0 `
            --device $Device
        if ($LASTEXITCODE -ne 0) { throw "$($variant.Name) robustness failed for seed $seed" }
        & $Python analyze_reliability_gate.py `
            --input $output `
            --output "$ArtifactRoot/seed$seed/$($variant.Name)/gate-analysis" `
            --seeds 2026 2027 2028 2029 2030 `
            --scenarios clean missing_100 noise_015 bias_008 stuck mixed
        if ($LASTEXITCODE -ne 0) { throw "$($variant.Name) gate analysis failed for seed $seed" }
    }
}

& $Python summarize_dynamic_constant_gate.py `
    --input-root $ArtifactRoot `
    --output-root "reports/qwen-material/submission-revision/dynamic-vs-constant-gate"
if ($LASTEXITCODE -ne 0) { throw "dynamic/constant summary failed" }

# The required CLIP assets are already cached locally. Disable remote HEAD
# probes so a transient Hugging Face TLS failure cannot invalidate this step.
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
& $Python evaluate_low_light_sensor_counterfactual.py `
    --output "reports/qwen-material/submission-revision/low-light-sensor-counterfactual-aligned" `
    --device $Device `
    --batch-size 16 `
    --num-workers 0 `
    --low-light-scale 0.35 `
    --permutation-seed 20260811
if ($LASTEXITCODE -ne 0) { throw "aligned low-light counterfactual failed" }
