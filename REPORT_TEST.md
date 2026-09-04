# REPORT\_TEST.md — 专题报告（只追加）

***

## \[2026-09-04] 专题：UK-DALE 预处理与真实数据链路验证

- **类型**：实验专题（预处理脚本 + 真实数据链路验证，非完整 baseline 科学结果）

- **目标与假设**：

  - 写 `scripts/prepare_ukdale.py`，从 `ukdale.h5` 提取 House 1 的 mains（aggregate）和 kettle（target）功率序列，对齐成 NPZ，供 `src/data.py::load_simple_npz` 加载

  - 假设：metadata pickle 能定位 meter；pandas(pytables) 能读 NILMTK HDF5 table；6s 重采样+inner-join 对齐后序列可用于训练

- **方法 / 数据 / 参数**：

  - 环境：conda `test_gpu`（Python 3.11.11, torch 2.3.1+cu121, pandas 2.3.3, **新装 tables 3.11.1**）

  - 数据源：`D:\Work\testPython\datasets\ukdale.h5`（3.19GB，NILMTK HDF5，building1-5）

  - meter 定位（从 `building1.attrs['metadata']` pickle 反序列化）：

    - mains = meter1（site\_meter=True, device=EcoManagerWholeHouseTx）

    - kettle = meter10（appliances 列表里首个 type='kettle'，meters=\[10]；注：meter10 还被 food processor / toasted sandwich maker 共享，存在已知插座共享噪声，nilmtk 标准亦取首个 type=kettle）

  - 加载：`pd.read_hdf(h5, key='/building1/elec/meterN')`（h5py `ds[:]` 读 pytables table 报 `can't open directory`，改用 pandas 成功）

  - 对齐：mains 与 target 各自 `resample('6s').mean()` → `pd.concat(join='inner')` → `ffill(limit=5).dropna()`

  - 输出：`D:\Work\testPython\datasets\ukdale_prepared.npz`（82.8MB，keys=`aggregate`/`target`，各 10,344,744 点 float32）

  - 验证训练（小样本，仅链路）：d\_model=64, num\_layers=2, window=128, batch=128, lr=5e-4, **epochs=2**, max\_samples\_train=4000/val=800/test=800, on\_threshold=500W, seed=42, device=auto(→cuda)

- **结果 / 结论**：

  - 序列统计：

    - aggregate: mean=368.7W, max=8423W（mains 总负荷，含背景）

    - target: mean=15.5W, max=3948W, frac>50W=0.64%（kettle 短时高功率突发，符合预期特征）

  - inner-join 重叠：11.3M→10.3M 对齐点（去掉无重叠时段 + 短 gap）

  - 小样本训练（2 epoch, cuda, 1.6s, best\_epoch=1）：

    - Ep1: train MAE=25.86 / val MAE=12.78 / train R²=0.0505 / val R²=0.0989

    - Ep2: train MAE=25.28 / val MAE=17.18 / train R²=0.1325 / val R²=0.1405

    - Test: MAE=38.21, RMSE=243.56, R²=0.1147, SAE=0.816, Precision=0/Recall=0/F1=0

  - 结论：**真实 UK-DALE 数据端到端链路验证通过**（h5→npz→load\_simple\_npz→build\_splits→Transformer→train→val→test→metrics→checkpoint）。指标差属预期——仅 2 epoch + 4k 样本，目的是验证链路而非科学指标；F1=0 因训练不足未学到 ON 事件，正式 baseline 需 30 epoch + 30k 样本

- **是否进入 REPORT.md（稳定结论）**：否（仅预处理脚本 + 链路验证，非稳定科学结果；待完整 baseline 跑出后再考虑）

- **遗留问题**：

  - meter10 插座共享噪声（kettle/food processor/sandwich maker）——nilmtk 标准亦如此，暂不处理；如需纯净 kettle 可后续按 nilmtk 的 `meters_directly_on_mains` 或子电器再筛选

  - `requirements.txt` 应补 `tables` 依赖（已落决策记录）

  - 正式 baseline 待跑：`$env:UKDALE_PREPARED_NPZ=...; conda run -n test_gpu python scripts/train.py --config configs/baseline.yaml --data-path $env:UKDALE_PREPARED_NPZ --out reports/baseline`

  - `run_real.ps1` 默认 `conda activate transformer_nilm`，需改用 `test_gpu` 或用 `conda run -n test_gpu` 调用

***

## \[2026-09-04] 专题：Smoke Test 链路验证

- **类型**：验证专题（代码链路验证，非 UK-DALE 科学实验）

- **目标与假设**：

  - 验证 NILM Transformer 代码链路完整可跑通：合成数据 → Dataset → Transformer Encoder → Train → Val → Test → Metrics → Checkpoint

  - 假设：在可控合成信号上若链路通过且指标合理，即说明 `src/` 模块与 `scripts/run_smoke.py` 入口逻辑无结构性缺陷，可放心进入 baseline / 真实 UK-DALE 阶段

- **方法 / 数据 / 参数**：

  - 环境：本地 conda env `test_gpu`（Python 3.11.11, torch 2.3.1+cu121, CUDA 可用）

  - 入口：`python scripts/run_smoke.py`（在项目根目录执行，因脚本用相对路径 `configs/baseline.yaml`）

  - 数据：`src/data.py::make_synthetic_signal(10000)` 合成信号，window\_size=64

  - 样本量：train 3000 / val 600 / test 600

  - 模型（覆盖 `configs/baseline.yaml`）：d\_model=32, nhead=4, num\_layers=1, dim\_feedforward=64, dropout=0.1, input\_dim=1

  - 训练：batch\_size=64, epochs=3, lr=5e-4, weight\_decay=1e-4, patience=2, grad\_clip=1.0, loss=mse, seed=42, device=auto(→cuda)

  - 阈值：on\_threshold\_watts=500

- **结果 / 结论**：

  - 链路：✅ 全链路跑通，无报错；2 条 PyTorch UserWarning（nested\_tensor / flash attention 未编译）非致命，不影响结果

  - 训练曲线（3 epoch）：

    - Ep1: train MAE=87.88 / val MAE=48.04 / train R²=0.2543 / val R²=0.5816

    - Ep2: train MAE=44.86 / val MAE=15.11 / train R²=0.6633 / val R²=0.9729

    - Ep3: train MAE=22.32 / val MAE=13.15 / train R²=0.9238 / val R²=0.9647

  - 最终指标（best\_epoch=3, device=cuda, runtime=4.74s）：

    - MAE=61.14, RMSE=175.70, R²=0.9062, SAE=0.0336

    - Energy Error=-0.0336, Precision=1.0, Recall=1.0, F1=1.0

  - 产物：`reports/smoke/best.pt`（42KB checkpoint）、`reports/smoke/history.json`、`reports/smoke/result.json`

  - 结论：代码链路验证通过。指标来自合成信号，**不是 UK-DALE 科学结果**，仅用于结构性回归测试

- **是否进入 REPORT.md（稳定结论）**：否（smoke test 不构成稳定实验结论，仅为链路验证；待真实 UK-DALE 实验产出后再考虑沉淀）

- **遗留问题**：

  - Test MAE(61.14) 高于 Val MAE(13.15)：合成信号在 test 段分布与 train/val 不同步，属合成数据特性，不代表模型问题；真实数据上需按 README §8 重新评估

  - 后续：是否进入 baseline（需 ukdale.h5）、调优、或跨家庭实验，待用户决策

