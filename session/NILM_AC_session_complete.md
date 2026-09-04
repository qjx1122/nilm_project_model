# NILM\_AC 会话纪要（只追加）

## \[2026-09-04] 会话纪要

- 目标：执行 BOOTSTRAP.md 开局仪式，恢复 NILM Transformer 项目现场

- 完成项：

  - 拉取 git 现状：分支 `nilm-project-model-ritual-4zSHFv`，工作树干净，最近 2 次提交（协议文档、原始代码）

  - 验证远端可访问（`git ls-remote origin`，HTTPS，无需 gh CLI）

  - 读取 README.md：确认实验目标（Kettle Seq2Point）、依赖（conda env `transformer_nilm` + `requirements.txt`）、入口（`scripts/run_smoke.py`）

  - 按 BOOTSTRAP.md 模板创建 `STATUS.md` 续接骨架

  - 创建 `session/` 目录与本纪要文件

- 关键决策：

  - 续接文件此前不存在，按 BOOTSTRAP.md v2.0 模板从零创建

  - gh CLI 未安装，远端验证改用 `git ls-remote origin`，HTTPS 凭据可用

  - `checkpoints/`、`logs/`、`data/` 暂不创建，待实际训练/下载数据时建立

- 未决问题：

  - 本会话具体目标待用户确认（smoke test / baseline / 真实 UK-DALE / 调优）

  - conda env `transformer_nilm` 是否已创建待确认；若未创建需 `pip install -r requirements.txt`

  - UK-DALE 数据尚未下载，真实实验指标无法立即产出

- 相关文件/分支：

  - 分支：`nilm-project-model-ritual-4zSHFv`（worktree session 分支）

  - 新建：`STATUS.md`、`session/NILM_AC_session_complete.md`

  - 参考：`BOOTSTRAP.md`、`README.md`

## \[2026-09-04] 会话纪要（续：Smoke test 任务）

- 目标：在本地 conda `test_gpu` 环境运行 `scripts/run_smoke.py`，验证 NILM Transformer 代码链路

- 完成项：

  - 任务立项：登记到 STATUS.md，落盘「用 conda test\_gpu 而非 README 默认 transformer\_nilm」决策

  - 环境恢复：`test_gpu`（Python 3.11.11, torch 2.3.1+cu121, numpy/pandas/sklearn/matplotlib/yaml/tqdm/h5py 全部满足 `requirements.txt` 下限）

  - 入口核对：`run_smoke.py` 覆盖 `configs/baseline.yaml` 为小尺寸（window=64, d\_model=32, 1 layer, 3 epochs），用相对路径→须在项目根目录执行

  - 执行：`conda run -n test_gpu python scripts/run_smoke.py` 全链路跑通（device=cuda, runtime=4.74s），best\_epoch=3

  - 指标（合成信号，非 UK-DALE）：Test MAE=61.14, RMSE=175.70, R²=0.9062, SAE=0.0336, P/R/F1=1.0

  - 产物：`reports/smoke/{best.pt, history.json, result.json}` 已生成

  - 专题报告：首次创建 `REPORT_TEST.md`，追加本专题完整记录

- 关键决策：

  - 用 conda `test_gpu`（用户指定，本机已有 GPU 环境）替代 README §3 的 `transformer_nilm`

  - 两条 PyTorch UserWarning（nested\_tensor / flash attention）非致命，不阻塞验证

  - Smoke test 指标不沉淀进 `REPORT.md`（仅链路验证，非稳定科学结论）

  - `README.md` 不改：环境差异属用户本机偏好，非项目安装方式变更

- 未决问题：

  - 下一阶段方向待用户选择：baseline / 真实 UK-DALE / 调优

  - ukdale.h5 数据未下载，真实实验需先准备数据

- 相关文件/分支：

  - 分支：`nilm-project-model-ritual-4zSHFv`

  - 新建：`STATUS.md`、`session/NILM_AC_session_complete.md`、`REPORT_TEST.md`

  - 产物：`reports/smoke/best.pt`、`reports/smoke/history.json`、`reports/smoke/result.json`

  - 入口：`scripts/run_smoke.py`、`configs/baseline.yaml`、`src/data.py`、`src/experiment.py`

## \[2026-09-04] 会话纪要（续：UK-DALE 数据下载）

- 目标：下载 UK-DALE 数据集，为真实 Kettle Seq2Point 实验准备 `ukdale.h5`

- 完成项：

  - 调研下载源：UKERC EDC 官方（需注册）/ HF `Pybunny/nilmbench-ukdale` / CSV 版

  - 探测 HF 可达性：huggingface.co 系统层超时（GFW）；hf-mirror.com 返回 429 限流；conda run 不传递 `HF_ENDPOINT` 给 Python 子进程

  - 与用户确认路径：用户选**手动下载** + 存 `D:\Work\testPython\datasets`（sandbox 可写区内）

  - 创建 `D:\Work\testPython\datasets` 目录，提供下载指引（hf-mirror / UKERC / Jack Kelly 个人页）

  - 用户完成下载：`ukdale.h5`（3.19GB）+ `ukdale.h5.tgz`（2.84GB 压缩备份）

  - 验证：`python scripts/inspect_h5.py --path D:\Work\testPython\datasets\ukdale.h5` 确认合法 NILMTK HDF5，root keys = building1-5

  - 数据流分析：读 `src/data.py`、`scripts/train.py`、`run_real.ps1`，确认项目期望 `ukdale_prepared.npz`（aggregate+target 数组），**项目缺** **`ukdale.h5 → npz`** **预处理脚本**

- 关键决策：

  - 数据存 `D:\Work\testPython\datasets`（用户选定，绕开 sandbox 写入限制；README 默认 `D:\datasets` 需在实验时调整）

  - 不改 README.md：路径差异属本机偏好，待预处理脚本落地后统一在 README 补说明

  - 不删 `ukdale.h5.tgz`：留作备份

  - 真实实验前置阻塞 = 缺预处理脚本，登记到下一步 TODO

- 未决问题：

  - 预处理脚本用 nilmtk（未装，需 `pip install nilmtk`）还是 h5py 直接读（需探 building1 内部 key 结构）

  - 真实训练是否继续用 `test_gpu` env（待用户确认）

  - README 默认路径 `D:\datasets` 与实际 `D:\Work\testPython\datasets` 不符，是否更新 README

- 相关文件/分支：

  - 分支：`nilm-project-model-ritual-4zSHFv`

  - 更新：`STATUS.md`（当前目标/已完成/下一步/决策记录）

  - 数据：`D:\Work\testPython\datasets\ukdale.h5`（3.19GB）、`ukdale.h5.tgz`（2.84GB）

  - 验证脚本：`scripts/inspect_h5.py`

  - 数据流参考：`src/data.py`、`scripts/train.py`、`run_real.ps1`、`configs/baseline.yaml`

## \[2026-09-04] 会话纪要（续：UK-DALE 预处理脚本）

- 目标：写 `scripts/prepare_ukdale.py`，把 `ukdale.h5` 转成 `ukdale_prepared.npz`，打通真实数据训练链路

- 完成项：

  - 探查 `ukdale.h5` 结构：`building1/elec/meter1-54`，每个 meter 是 pandas pytables table（`meterN/table` shape=(N,) dtype=\[('index','\<i8'),('values\_block\_0','\<f4',(1,))]）

  - 定位 meter：从 `building1.attrs['metadata']`（pickle dict）反序列化，`elec_meters` 含 site\_meter 标志、`appliances` 含 type→meters 映射。**mains=meter1, kettle=meter10**（meter10 还被 food processor/sandwich maker 共享，nilmtk 标准取首个 type=kettle）

  - 尝试装 nilmtk：清华镜像无、GitHub SSL 被墙。改纯 h5py+pandas+metadata pickle 方案，无 nilmtk 依赖

  - 装 `tables 3.11.1`（pytables）到 `test_gpu`（pandas.read\_hdf 需它；h5py `ds[:]` 读 pytables table 报 `can't open directory`）

  - 写 `scripts/prepare_ukdale.py`：读 metadata→定位 meter→`pd.read_hdf` 读 mains+target→`resample('6s').mean()`→`concat(inner)`→`ffill(5).dropna()`→存 npz(aggregate,target)

  - 跑脚本：输出 `ukdale_prepared.npz`（82.8MB，10.3M 对齐点）。aggregate mean=368.7W/max=8423W，target mean=15.5W/max=3948W/frac>50W=0.64%（kettle 特征合理）

  - 验证：`load_simple_npz` 加载 + `train_experiment`（2 epoch, 4k 样本, cuda, 1.6s）端到端跑通，best\_epoch=1, Test MAE=38.21/R²=0.1147/F1=0（指标差属预期，仅链路验证）

  - 删除临时探查脚本 `_probe_h5.py`/`_verify_npz.py`

- 关键决策：

  - 纯 h5py+metadata pickle 方案（nilmtk 不可达），无新 nilmtk 依赖，脚本自包含

  - meter10 接受插座共享噪声（与 nilmtk 标准一致），暂不子筛选

  - 对齐用 inner-join + ffill(5) + dropna，去掉无重叠段和长 gap

  - `reports/verify_real/` 产物不提交（仅验证非正式实验）

- 未决问题：

  - `requirements.txt` 应补 `tables` 依赖（落决策记录，待统一更新）

  - `run_real.ps1` 默认 `conda activate transformer_nilm`，需改 `test_gpu`（或用 `conda run -n test_gpu`）

  - 正式 baseline（30 epoch, 30k 样本）待下一会话跑

- 相关文件/分支：

  - 分支：`nilm-project-model-ritual-4zSHFv`

  - 新建：`scripts/prepare_ukdale.py`

  - 更新：`STATUS.md`、`REPORT_TEST.md`（追加预处理专题）、`session/NILM_AC_session_complete.md`

  - 产物：`D:\Work\testPython\datasets\ukdale_prepared.npz`（82.8MB）、`reports/verify_real/{best.pt,history.json,result.json}`（未提交）

## \[2026-09-04] 会话纪要（续：补依赖）

- 目标：补 `requirements.txt` 的 `tables` 依赖 + 改 `run_real.ps1` 用 `test_gpu` env

- 完成项：

  - `requirements.txt` 加 `tables>=3.9`（pandas.read\_hdf 读 pytables table 必需，test\_gpu 已装 3.11.1）

  - `run_real.ps1`：`conda activate transformer_nilm` → `conda activate test_gpu`；示例 NPZ 路径 `D:\datasets\ukdale_prepared.npz` → `D:\Work\testPython\datasets\ukdale_prepared.npz`

  - 验证：tables 已装确认；run\_real.ps1 逻辑读回确认无误

  - 发现 PS 5.1 编码问题：UTF-8 无 BOM 文件在 PS 5.1 用 GBK 解码导致中文 Write-Host 乱码 + Parser 报字符串终止符错误；PS 7 正常。脚本逻辑正确，落决策记录供用户参考

- 关键决策：

  - 只做最小改动（env 名 + 示例路径），保留原有 `$env:UKDALE_PREPARED_NPZ` 安全检查逻辑，不设默认值（强制用户显式设路径，避免误用）

  - 不修 PS 编码（原文件就有的问题，非本次引入；PS 7 用户不受影响），仅落记录

- 未决问题：

  - 正式 baseline（30 epoch, 30k 样本）待下一会话跑

- 相关文件/分支：

  - 分支：`nilm-project-model-ritual-4zSHFv`

  - 更新：`requirements.txt`、`run_real.ps1`、`STATUS.md`、`session/NILM_AC_session_complete.md`

## [2026-09-04] 会话纪要（续：Baseline 真实训练）
- 目标：跑完整 baseline（`configs/baseline.yaml` + `ukdale_prepared.npz`），产出真实 Kettle Seq2Point 指标
- 完成项：
  - 启动训练：`conda run -n test_gpu python scripts/train.py --config configs/baseline.yaml --data-path D:\Work\testPython\datasets\ukdale_prepared.npz --out reports/baseline`（后台跑，59.1s cuda）
  - 训练过程：15 epoch 后早停（patience=7，best_epoch=8）。train MAE 18.66→5.99（ep1→14），val MAE 9.85→3.98（ep8 最佳），val R² 0.617→0.879，val F1 0.754→0.929（ep8 峰值）。ep9 后 val 抖动
  - 跑 `evaluate.py --run-dir reports/baseline` 确认 result.json
  - **Test 指标（best epoch 8 模型）**：MAE=13.09, RMSE=145.64, R²=0.5921, SAE=0.403, **Precision=0.952, Recall=0.678, F1=0.792**
  - 产物：`reports/baseline/{best.pt(280KB), history.json(10.6KB), result.json(1.2KB)}`
- 关键决策：
  - 用 `conda run -n test_gpu` 而非 `run_real.ps1`（conda activate 在非交互 shell 可能失败；run_real.ps1 留给交互式 PS 用）
  - baseline 结果放 `REPORT_TEST.md`（候选进 REPORT.md，待稳定性验证/调参后再沉淀）
  - 提交 `reports/baseline/{history.json, result.json}` 作训练证据；`best.pt` 不提交（可重训，避免 git 膨胀）
- 未决问题：
  - val/test F1 gap 大（0.93→0.79）：test 段更难或分布漂移，需进一步分析
  - Precision>Recall（0.95>0.68）：模型保守漏报，可降 on_threshold 或调 loss 权重
  - 单 seed=42，未做多种子稳定性
  - 据README §8 调参方向待用户定（增样本/调 dropout/ lr 调度/降 threshold）
- 相关文件/分支：
  - 分支：`nilm-project-model-ritual-4zSHFv`
  - 更新：`STATUS.md`、`REPORT_TEST.md`（追加 baseline 专题）、`session/NILM_AC_session_complete.md`
  - 产物：`reports/baseline/{history.json, result.json}`（提交）；`best.pt`（未提交）

## [2026-09-04] 会话纪要（续：val/test F1 差距分析）
- 目标：诊断 baseline val F1=0.929 vs test F1=0.792 的 0.137 gap 来源
- 完成项：
  - 读 `src/data.py::build_splits`（时序 70/15/15 + linspace 等距抽样）、`metrics.py`（F1 是 point-level，逐点 ≥on_threshold）、`trainer.py`（早停基于 val MAE 非 F1）
  - 写 `scripts/analyze_gap.py`：(1) 全量段 target 分布统计；(2) 6000 采样点 ON 数对比；(3) 加载 best.pt 跑 test 预测，拆 TP/FP/FN 分析漏报功率与位置
  - 跑分析，关键数据：
    - 全量段 ON 占比：train 0.58% / val 0.60% / **test 0.88%**（test 更密集，季节性漂移）
    - 6000 采样：val 真实 ON=29 / test=59（test 2x）
    - 漏报 19 个：真实功率 mean=2242W（远超 500 阈值，标准高功率事件），预测功率 mean=96W（模型输出接近 OFF）
    - 命中 40 个：真实 mean=2328W ≈ 漏报 2242W（功率无差异）
    - 漏报位置均匀分布（前1/3:6 中:8 后:5）
  - 结论：gap 三主因——(A) test ON 密度高致采样失衡；(B) 模型对部分标准 ON 上下文学习不足（容量/样本受限）；(C) 6000 采样小样本 F1 高方差
- 关键决策：
  - 保留 `analyze_gap.py` 为正式脚本（可复用诊断工具，未来调参可重跑）
  - 分析结论放 REPORT_TEST.md（非稳定科学结果，待实验验证建议后沉淀）
- 未决问题：
  - 据建议调参方向待用户定（增样本/容量/跨建筑/多种子）
- 相关文件/分支：
  - 分支：`nilm-project-model-ritual-4zSHFv`
  - 新建：`scripts/analyze_gap.py`
  - 更新：`STATUS.md`、`REPORT_TEST.md`（追加分析专题）、`session/NILM_AC_session_complete.md`

## [2026-09-04] 会话纪要（续：增评估样本对比）
- 目标：验证 F1 gap 主因 C（小样本噪声）——增 max_samples_val/test 6000→30000 重训对比
- 完成项：
  - 建 `configs/baseline_allsample.yaml`（唯一改动 val/test 6000→30000，train/模型/seed=42 不变，保证模型轨迹同）
  - 改 `scripts/analyze_gap.py` 加 argparse（`--npz --config --ckpt --threshold`），可分析任意 run
  - 训练 `baseline_allsample`：cuda 74s，15 epoch 早停，best_epoch=8（与原一致，确认模型同）
  - 跑 `analyze_gap.py --config baseline_allsample --ckpt reports/baseline_allsample/best.pt`
  - 读 `history.json` 拿 ep8 val F1
- 关键数据对比：
  - val F1：0.929（29 ON，小样本）→ **0.830（173 ON，大样本）**，降 0.099
  - test F1：0.792（59 ON）→ 0.783（266 ON），仅降 0.009
  - gap：0.137 → **0.047（缩 65%）**
  - 漏报特征一致：19 个 mean 2242W → 89 个 mean 2287W（标准高功率，非边缘）
  - 漏报率稳定：19/59=32% → 89/266=33%
- 结论：
  - **原 gap 65% 来自 val 小样本乐观偏差**（29 ON 偏简单事件致 F1 虚高 0.929），非 test 异常难
  - **test F1≈0.78 是真实泛化水平**（两次运行一致，漏报率稳定）
  - 模型真实缺陷（主因 B）确认：漏报标准高功率事件 mean 2287W，预测功率 110W（≈OFF），但这是模型本身问题不造成 gap（val 同样漏报率）
  - **评估应固定用大样本（≥30000）**避免小样本误导
- 关键决策：
  - 提交 `configs/baseline_allsample.yaml` + `reports/baseline_allsample/{history.json, result.json}`（best.pt 不提交）
  - `baseline_allsample.yaml` 作后续实验默认评估配置
- 未决问题：
  - 优先级 2（增模型容量+训练样本）待下一会话
- 相关文件/分支：
  - 分支：`nilm-project-model-ritual-4zSHFv`
  - 新建：`configs/baseline_allsample.yaml`
  - 更新：`scripts/analyze_gap.py`（argparse）、`STATUS.md`、`REPORT_TEST.md`（增样本对比专题）、`session/NILM_AC_session_complete.md`
  - 产物：`reports/baseline_allsample/{history.json, result.json}`（提交）；`best.pt`（未提交）

