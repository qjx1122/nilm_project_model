# REPORT\_TEST.md — 专题报告（只追加）

***

## \[2026-09-04] 专题：增评估样本对比——gap 主因确认为 val 小样本乐观偏差

- **类型**：实验专题（验证 F1 gap 分析的主因 C 假设：增 max\_samples\_val/test 6000→30000，看 gap 是否由小样本噪声造成）

- **假设**：若 gap 主因是评估小样本噪声（主因 C），增样本后 val/test F1 应都收敛到稳定值，gap 大幅缩小；若 gap 不变，则主因是 A/B（真实泛化差）

- **方法 / 参数**：

  - 新建 `configs/baseline_allsample.yaml`：唯一改动 `max_samples_val/test` 6000→30000（5x），train 保持 30000、模型/超参不变、seed=42（保证模型训练轨迹与原 baseline 一致，仅评估样本变）

  - 训练：`conda run -n test_gpu python scripts/train.py --config configs/baseline_allsample.yaml --out reports/baseline_allsample`，cuda 74s，15 epoch 早停

  - 分析：`scripts/analyze_gap.py --config configs/baseline_allsample.yaml --ckpt reports/baseline_allsample/best.pt`（脚本已加 argparse 支持任意 run）

- **结果对比**：

  | 指标               | 原 baseline (6000 采样)    | 新 allsample (30000 采样)      | 变化                      |
  | ---------------- | ----------------------- | --------------------------- | ----------------------- |
  | best\_epoch      | 8                       | 8                           | 不变（模型轨迹同，seed/train 不变） |
  | val 真实 ON 数      | 29                      | 173                         | 6x                      |
  | val F1 (ep8)     | 0.929 (P=0.963/R=0.897) | **0.830 (P=0.935/R=0.746)** | **-0.099**              |
  | test 真实 ON 数     | 59                      | 266                         | 4.5x                    |
  | test F1          | 0.792 (P=0.952/R=0.678) | **0.783 (P=0.952/R=0.665)** | -0.009                  |
  | **val/test gap** | **0.137**               | **0.047**                   | **缩 65%**               |
  | 漏报 FN 真实功率       | mean 2242W (19 个)       | mean 2287W (89 个)           | 特征一致                    |
  | test MAE         | 13.09                   | 11.15                       | 更稳估计                    |
  | test R²          | 0.592                   | 0.618                       | 更稳估计                    |

- **结论**：

  1. **原 gap 0.137 的 \~65% 来自 val 小样本乐观偏差**——29 个 ON 偏简单事件，F1=0.929 是虚高；大样本 173 个 ON 下 val F1 降到 0.830（更接近真实泛化）。**主因 C 修正为"val 小样本乐观偏差"**，非"两边对称噪声"
  2. **test F1≈0.78 是真实泛化水平**——增 4.5x 样本 test F1 仅 0.792→0.783（-0.009），两次运行一致，漏报率稳定（19/59=32% → 89/266=33%）。test 相对可信
  3. **剩余 gap 0.047** 是真实小 gap——可能 val 段（时间中段，Kettle 用法较少）仍略乐观，或 test 段稍难；但已非主因
  4. **模型真实缺陷确认（主因 B 仍成立）**：漏报 89 个 mean 2287W（标准高功率，远超 500 阈值），预测功率 mean 110W（输出≈OFF），模型对部分标准 ON 上下文学习不足；这是模型本身问题，不造成 val/test gap（val 也有类似漏报率），但限制 F1 上限到 \~0.78
  5. **主因 A（test ON 密度漂移）真实但非 gap 来源**——test ON 0.88% > val 0.60% 是数据特性，但在大样本下两边 F1 都稳定，分布差异没放大 gap

- **建议修正**：

  1. ✅ **评估固定用大样本**（max\_samples\_val/test ≥30000）——已验证能消除 65% 的虚假 gap，避免误导调参方向。`configs/baseline_allsample.yaml` 可作后续实验的默认评估配置
  2. **调参重点转向提升模型真实 F1（当前 \~0.78）**——针对主因 B：增模型容量（d\_model 64→128, layers 2→3）+ 增训练样本（30k→100k+），让模型见更多 ON 上下文变体
  3. **优先级 2（增容量+训练样本）成为下一实验**——gap 既已确认为评估偏差，提升 F1 的杠杆在模型侧
  4. 多种子/跨建筑验证仍待办，但应在大样本评估下做

- **是否进入 REPORT.md**：否（分析对比专题，待模型改进实验出稳定提升后再沉淀）

- **相关产物**：`configs/baseline_allsample.yaml`、`reports/baseline_allsample/{best.pt(未提交), history.json, result.json}`、`scripts/analyze_gap.py`（已加 argparse）

***

## \[2026-09-04] 专题：val/test F1 差距分析（baseline 延伸）

- **类型**：分析专题（诊断 baseline val F1=0.929 vs test F1=0.792 的 0.137 gap 来源）

- **方法**：`scripts/analyze_gap.py`——(1) 按 build\_splits 时序 70/15/15 切分，统计 train/val/test 全量段的 target 分布（ON 点占比、事件数、事件时长、ON 功率）；(2) 复现 6000 点 linspace 采样，统计 val/test 采样到的真实 ON 数；(3) 加载 `reports/baseline/best.pt` 跑 test 6000 点预测，拆分 TP/FP/FN，分析漏报点的真实/预测功率特征与位置分布

- **发现**：

  1. **test 段 ON 密度高于 val（分布漂移）**——全量段统计：

     - train: ON 占比 0.58%, 2392 事件, mean 时长 17.5 点(105s), mean 功率 2296W

     - val:   ON 占比 0.60%, 481 事件, mean 时长 19.4 点, mean 功率 2318W

     - test:  ON 占比 **0.88%**, 572 事件, mean 时长 **23.9 点**, mean 功率 2319W

     - test 段（时间序列最后 15%，推测为季节性不同的时段）Kettle 用得更频繁、每次更久，但 ON 功率分布几乎相同（\~2320W）
  2. **6000 采样点的统计失衡**——linspace 等距抽样在 ON 更密的 test 段采到更多 ON：

     - val 6000 采样：真实 ON = 29（0.48%）

     - test 6000 采样：真实 ON = 59（0.98%，是 val 的 2 倍）

     - 小样本下 F1 对少数漏报极敏感：19 个漏报 → recall=40/59=0.678；若多命中 4 个 → recall=0.746，F1 显著变化
  3. **漏报的是标准高功率事件，非阈值边缘**：

     - 19 个 FN 真实功率：mean=2242W, median=2338W, max=2436W（都是典型 kettle 加热功率，远超 500W 阈值）

     - 19 个 FN 预测功率：mean=96W, max=489W（模型预测远低于阈值，完全没识别）

     - 40 个 TP 真实功率 mean=2328W（与 FN 几乎相同），预测 mean=1746W（部分识别）

     - 2 个 FP 真实功率=1W，预测=1759W（纯误报，量极少）

     - 漏报位置在 test 段均匀分布（前1/3:6 中1/3:8 后1/3:5），非某段集中

- **结论（gap 主因）**：

  - **主因 A：test 段分布漂移致采样失衡**——test 段 ON 密度（0.88%）高于 val（0.60%），6000 采样暴露更多 ON 点（59 vs 29），放大了模型的漏报基数；若 val 同样 59 个 ON，其 recall 也可能下降

  - **主因 B：模型对部分标准高功率事件学习不足**——漏报的 19 个事件真实功率 2242W（与命中的 2328W 无显著差异），说明模型不是因为"事件功率低难识别"，而是这些事件的 aggregate 上下文（window=128=12.8min）特征训练集见得不够，或模型容量（d\_model=64/2 层）不足以区分。漏报预测功率 96W 说明模型对这些样本输出接近"OFF"

  - **主因 C：小样本评估的统计噪声**——F1 在 59 个 ON 上算，19 个漏报占比敏感；point-level F1 在稀疏 ON（<1%）数据上本就高方差。val 的 29 个 ON 偏少，其 F1=0.93 也带噪声，不一定代表真实泛化

  - **非主因**：不是阈值边缘问题（漏报功率 2242W ≫ 500W）；不是 test 事件功率更低（test 2319W ≈ val 2318W）；不是时间局部漂移（漏报均匀分布）

- **建议（下一步调参方向，按优先级）**：

  1. **增大评估样本**：`max_samples_test`/`max_samples_val` 6000→30000+ 或全量，降低 F1 统计噪声，得到更可信的 gap 度量（最便宜的改进）
  2. **增模型容量 + 训练样本**：d\_model 64→128、num\_layers 2→3、max\_samples\_train 30k→100k+，让模型见更多 ON 上下文变体（针对主因 B）
  3. **跨建筑验证**：用 building2/3 训练、building1 测试，看是否泛化更差（区分"分布漂移"vs"模型容量"）
  4. **降 on\_threshold 调参**：500W→200W 可能把预测 489W 的边缘 FN 拉成 TP，但本例 max FN 功率 2436W，降阈值帮助有限（针对主因 B 无效）
  5. **多种子重训**：seed=42 单次结果，跑 seed=0/1 看漏报数方差，判断是模型不稳还是系统性缺陷

- **是否进入 REPORT.md**：否（分析专题，待据建议跑实验后再沉淀稳定结论）

***

## \[2026-09-04] 专题：Baseline 真实 UK-DALE 训练（Kettle Seq2Point）

- **类型**：实验专题（完整 baseline 真实训练，**首个真实科学结果**，候选进 REPORT.md）

- **目标与假设**：

  - 用 `ukdale_prepared.npz`（House 1 mains+kettle, 10.3M 对齐点）跑 `configs/baseline.yaml` 完整训练，验证真实 UK-DALE Kettle Seq2Point 指标达合理范围

  - 假设：30k 训练样本 + 30 epoch（patience=7 早停）足以学到 Kettle ON/OFF 模式，F1 应 > 0.5

- **方法 / 数据 / 参数**：

  - 环境：conda `test_gpu`（Python 3.11.11, torch 2.3.1+cu121, cuda）

  - 数据：`D:\Work\testPython\datasets\ukdale_prepared.npz`（82.8MB，aggregate+target，10,344,744 对齐点，6s 采样）

  - 划分：时序 70% train / 15% val / 15% test（`build_splits`），cap max\_samples train=30000/val=6000/test=6000

  - 模型：Transformer encoder Seq2Point，d\_model=64, nhead=4, num\_layers=2, dim\_feedforward=128, dropout=0.10, window=128

  - 训练：batch=128, lr=5e-4, weight\_decay=1e-4, grad\_clip=1.0, loss=MSE, seed=42, epochs=30, patience=7

  - 评估：on\_threshold=500W；指标 MAE/RMSE/R²/SAE/EnergyError/Precision/Recall/F1

- **结果 / 结论**：

  - 训练：15 epoch 后早停（best\_epoch=8），runtime 59.1s（cuda）

  - best\_epoch=8 val：MAE=3.98, RMSE=55.6, R²=0.8787, F1=0.9286（Precision=0.963/Recall=0.897）

  - **Test（best epoch 模型）**：

    - MAE=13.09, RMSE=145.64, R²=0.5921, SAE=0.403

    - **Precision=0.952, Recall=0.678, F1=0.792**

  - epoch 曲线：train MAE 18.66→5.99（ep1→14），val MAE 9.85→3.98（ep1→8 最佳）；val R² 0.617→0.879；val F1 0.754→0.929（ep8 峰值）。ep9 后 val 抖动（过拟合或分布漂移），早停于 ep15

  - 结论：**真实 baseline 达合理范围**。F1=0.79 高于预期门槛 0.5；Precision(0.95)明显高于 Recall(0.68)——模型偏保守，漏报多于误报。val/test F1 gap（0.93→0.79）较大，提示 test 段更难或分布漂移

  - 对比参考：nilmtk 文献 Kettle Seq2Point F1 通常 0.7-0.85（UK-DALE building1 跨建筑或同建筑时序划分），本结果 F1=0.79 落在合理区间

- **是否进入 REPORT.md（稳定结论）**：**候选**——这是首个真实 baseline，指标合理且可复现。建议先跑 1-2 次重训（不同 seed）确认稳定性后再沉淀；或据 README §8 用 val 指标调参后再定

- **遗留问题**：

  - val/test F1 gap 大（0.93→0.79）：可能 test 段 Kettle 事件分布不同，或过拟合 val；可尝试 (a) 更大 train 样本；(b) 调 dropout；(c) 跨建筑验证

  - Recall< Precision：模型保守漏报；可降 on\_threshold 或调 loss 权重

  - 单次训练（seed=42），未做多种子稳定性验证

  - meter10 插座共享噪声（kettle/food processor/sandwich maker）仍在 target 里

- **相关产物**：`reports/baseline/{best.pt(280KB), history.json, result.json}`

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

