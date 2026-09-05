# Transformer-NILM：UK-DALE + PyTorch 完整训练与调优项目

本项目面向 Windows + Conda + PyTorch，采用公开 UK-DALE 数据集构建一个可复现的 Transformer Encoder Seq2Point NILM 实验。

> 重要：本项目不会伪造实验结果。仓库已内置处理好的真实数据 `data/ukdale_prepared.npz`（UK-DALE House1 kettle，6s，aggregate+target），真实指标直接跑 §6 即可复现；已产出第一组真实指标（缩短版，见 `reports/ukdale_baseline_cpu_short/` 与 `REPORT_TEST.md`），全量 baseline 仍待运行。合成 smoke test 仅用于验证代码链路。

## 1. 实验目标

输入：
- 家庭总负荷 Aggregate

输出：
- 指定电器 Kettle 的功率

模型：
- Input Projection
- Sinusoidal Positional Encoding
- Transformer Encoder
- Center-point Regression Head

评价：
- MAE
- RMSE
- R²
- SAE
- Precision / Recall / F1（基于 ON/OFF 阈值）
- Energy Error

调优：
- learning rate
- d_model
- nhead
- num_layers
- dim_feedforward
- dropout
- weight_decay
- batch_size
- window_size

## 2. 数据来源

本项目优先支持 NILMbench 公开的 UK-DALE 处理数据，以及标准 UK-DALE HDF5/CSV 数据。

UK-DALE 官方：
https://jack-kelly.com/data/

UK-DALE 论文：
https://www.nature.com/articles/sdata20157

NILMbench：
https://github.com/nilmtk/nilmbench

处理后的 UK-DALE：
https://huggingface.co/datasets/Pybunny/nilmbench-ukdale

### 推荐路径

对于本项目的低频 Seq2Point 教学实验，推荐使用 UK-DALE 6-second 数据，而不是 16 kHz 波形。原始 UK-DALE 的 6 秒数据包含 whole-home 与 appliance-level readings。

如果已经有 `ukdale.h5`，可以先执行：

```powershell
conda activate transformer_nilm
python scripts\inspect_h5.py --path D:\datasets\ukdale.h5
```

**本仓库已内置处理数据**：`data/ukdale_prepared.npz`（键：`aggregate`、`target`，float32，n=10,344,744，≈718 天@6s；由 6 秒版 UK-DALE House1 kettle 生成）。使用该 npz 时无需再下载数据，`--data-path data/ukdale_prepared.npz` 直接训练。

## 3. Windows + Conda

```powershell
conda create -n transformer_nilm python=3.11 -y
conda activate transformer_nilm
pip install -r requirements.txt
```

GPU 用户请先按自己的 CUDA 版本安装对应 PyTorch，再安装其余依赖。

### 3.1 Linux / CPU（无 conda 的沙箱环境）

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt   # 若 PyTorch 官方 index 被网络拦截，直接走 PyPI 默认源亦可（CPU 可用）
./.venv/bin/python -m pytest tests/ -q        # 环境自检
```

## 4. 项目结构

```text
transformer_nilm_project/
├── configs/
│   ├── baseline.yaml
│   └── tuning.yaml
├── data/
│   └── ukdale_prepared.npz      # 内置真实数据：aggregate+target（kettle, 6s, House1）
├── checkpoints/
├── logs/
├── reports/
│   ├── smoke/                   # 合成数据冒烟产物（非真实结果）
│   └── ukdale_baseline_cpu_short/  # 第一组真实指标（缩短版）
├── STATUS.md                    # 续接文件（见 BOOTSTRAP.md）
├── session/                     # 会话纪要（追加式）
├── REPORT_TEST.md               # 专题报告（追加式）
├── BOOTSTRAP.md                 # 会话与任务协议
├── src/
│   ├── data.py
│   ├── model.py
│   ├── metrics.py
│   ├── trainer.py
│   └── experiment.py
├── scripts/
│   ├── inspect_h5.py
│   ├── train.py
│   ├── evaluate.py
│   ├── eval_ckpt.py     # checkpoint 按原配置换口径复评（可加密 test 集做跨 run 公平对比）
│   ├── tune.py
│   └── run_smoke.py
├── tests/
├── requirements.txt
├── run_baseline.ps1
├── run_tuning.ps1
├── run_real.ps1
└── README.md
```

## 5. 先验证代码

```powershell
python scripts\run_smoke.py
```

Smoke test 使用可控的合成信号，只用于验证代码链路：
数据 -> Dataset -> Transformer -> Train -> Validation -> Test -> Metrics -> Checkpoint。

它不是 UK-DALE 科学实验结果。

## 6. Baseline

配置见 `configs/baseline.yaml`。

典型参数：

```yaml
window_size: 128
d_model: 64
nhead: 4
num_layers: 2
dim_feedforward: 128
dropout: 0.1     # 真实数据 A/B 结论：kettle 上 dropout=0 更优（见 REPORT.md）
lr: 0.0005
lr_schedule: cosine   # 可选 none|cosine；cosine 可平滑 val 震荡、改善早停选点
weight_decay: 0.0001
batch_size: 128
epochs: 30
```

运行：

```powershell
python scripts\train.py --config configs\baseline.yaml --data-path D:\datasets\ukdale.h5
python scripts\evaluate.py --run-dir reports\baseline
```

使用仓库内置真实数据（Linux/CPU 同理，路径换斜杠）：

```bash
./.venv/bin/python scripts/train.py --config configs/baseline.yaml --data-path data/ukdale_prepared.npz --out reports/baseline
```

CPU（2 核）参考耗时：全量配置 ≈ 80–90 分钟；缩短版（train 10k × 10 epochs）≈ 10.6 分钟，产物见 `reports/ukdale_baseline_cpu_short/`（result.json / history.json / best.pt / config.yaml / train.log，已入库）。

已有 checkpoint 换口径复评（不重训；`--max-samples-test` 可加密测试集做跨 run 公平对比）：

```bash
./.venv/bin/python scripts/eval_ckpt.py --config reports/<run>/config.yaml --ckpt reports/<run>/best.pt \
    --data-path data/ukdale_prepared.npz --max-samples-test 30000 --out reports/<run>/dense_test_eval.json
```

## 7. 调优

不要直接拿 Test 指标调参。

流程：

```text
Train
  ↓
Validation
  ↓
调整超参数
  ↓
重新训练
  ↓
Validation 最优
  ↓
锁定配置
  ↓
Test 一次
```

运行：

```powershell
python scripts\tune.py --config configs\tuning.yaml --data-path D:\datasets\ukdale.h5
```

调优结果会保存到：

```text
reports/tuning_summary.csv
reports/best_config.yaml
```

## 8. 指标驱动的调优规则

### Train 差 + Val 差
优先检查：
1. 数据对齐
2. normalization
3. learning rate
4. window
5. 模型容量

### Train 好 + Val 差
说明过拟合：
1. dropout
2. weight decay
3. 减少层数
4. 数据增强
5. early stopping

### Train/Val 好 + Test 差
优先检查：
1. 时间分布变化
2. household/domain shift
3. 预处理一致性
4. 测试家庭是否与训练家庭不同

### Loss 震荡
优先降低 LR。

### Loss 下降非常慢
尝试提高 LR，但必须观察 Validation。

### MAE 好但 RMSE 很差
说明存在少量大误差/异常事件，应重点检查事件启动、关闭和大功率区间。

## 9. 真实 UK-DALE 实验

建议第一阶段：

```text
Train: House 1 前 70% 时间
Val:   House 1 中间 15%
Test:  House 1 后 15%
Appliance: kettle
```

第二阶段再做真正更有意义的跨家庭：

```text
Train: House 1
Test:  House 2
```

不要把跨家庭 Test 用于反复调参。

## 10. 实验记录

每次实验保存：

- experiment_id
- dataset
- appliance
- split
- window_size
- d_model
- nhead
- num_layers
- dropout
- lr
- weight_decay
- batch_size
- best_epoch
- train/val/test metrics
- checkpoint
- runtime
- git/config hash

## 11. 下一步

当真实 UK-DALE 实验跑通后，可以继续扩展：

```text
Transformer
├── Kettle
├── Microwave
├── Fridge
├── Dishwasher
└── Washing Machine
```

再扩展：

```text
LSTM
GRU
TCN
CNN
CNN-LSTM
Informer
PatchTST
```

最后接入 AI Agent：

```text
Training Agent
   ↓
Log Agent
   ↓
Metric Agent
   ↓
Diagnosis Agent
   ↓
Hyperparameter Agent
   ↓
Retraining Agent
   ↓
Experiment Manager
   ↓
Report Agent
```
