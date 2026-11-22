param(
    [string]$Python = 'C:\Users\Lenovo\anaconda3\envs\aic26\python.exe',
    [string]$ModelDir = 'D:\STUDY\BIN\CS\PROJECT\ZAI\asr_tv3\weights\A5_zipvi_aug_s1200\onnx_s1200',
    [int]$BatchSize = 16,
    [string]$Output = 'indexes/asr_segments.parquet'
)
$repoDir = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$torchDllDir = Join-Path (Split-Path $Python) 'Lib\site-packages\torch\lib'
Push-Location $repoDir
try {
    & $Python scripts/asr_local_gpu.py `
        --runtime-dir scratch/asr-gpu-deps `
        --ssh-runtime-dir scratch/ssh-deps `
        --dll-dir $torchDllDir `
        --model-dir $ModelDir `
        --batch-size $BatchSize --output $Output
    $jobExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $jobExitCode
