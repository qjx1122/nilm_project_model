$ErrorActionPreference = "Stop"
conda activate test_gpu

if (-not $env:UKDALE_PREPARED_NPZ) {
    Write-Host "请先设置 UKDALE_PREPARED_NPZ，例如："
    Write-Host '$env:UKDALE_PREPARED_NPZ="D:\Work\testPython\datasets\ukdale_prepared.npz"'
    exit 1
}

python scripts\train.py --config configs\baseline.yaml --data-path $env:UKDALE_PREPARED_NPZ --out reports\baseline
python scripts\evaluate.py --run-dir reports\baseline

Write-Host "Baseline 完成。接下来应根据 Validation 指标决定调参，而不是直接使用 Test 指标。"
