# STATUS.md

## 当前目标

- **本会话任务（已完成）**：优先级 2——增模型容量+训练样本。建 `configs/baseline_big.yaml`（d_model 128/layers 3/FFN 512/train 100k/val-test 30k），重训 + analyze_gap 对比。**Test F1 0.783→0.850（+8.6%），Recall 0.665→0.820（漏报减半）**，主因 B 缓解。但 best_epoch=2 早停暴露 val 抖动（lr 偏高/早停基于 MAE 非 F1）。详见 `REPORT_TEST.md` 增容量专题
- **下一会话待定**：解决早停问题——lr 调度（5e-4→1e-4 或 ReduceLROnPlateau）/ 早停改基于 val F1 / 增 epoch 30→50；然后多种子验证 F1=0.85 稳定性；视结果沉淀进 REPORT.md

## 已完成

- [x] 仓库初始化与原始代码上传（commit `ff2a340`）
- [x] 添加 Agent 会话与任务协议文档 BOOTSTRAP.md（commit `7a91470`）
- [x] 开局仪式执行：拉取 git 现状、确认远端可访问、创建 STATUS.md 续接骨架
- [x] **Smoke test 验证**：conda `test_gpu` 下 `run_smoke.py` 全链路跑通（cuda, 4.74s），产物落 `reports/smoke/`
- [x] **UK-DALE 数据下载与验证**：用户手动下载 `ukdale.h5`（3.19GB）；`inspect_h5.py` 确认合法 NILMTK HDF5，5 buildings
- [x] **预处理脚本**：`scripts/prepare_ukdale.py`（纯 h5py+pandas，无 nilmtk），mains=meter1/kettle=meter10，6s 重采样对齐，输出 `ukdale_prepared.npz`（82.8MB，10.3M 对齐点）
- [x] **真实数据链路验证**：`load_simple_npz` + `train_experiment`（2 epoch 小样本）端到端跑通（cuda, 1.6s）
- [x] **补依赖**：`requirements.txt` 加 `tables>=3.9`；`run_real.ps1` 改用 `conda activate test_gpu` + 示例路径更新
- [x] **Baseline 真实训练**：`configs/baseline.yaml` + npz，15ep 早停(best=8)，cuda 59s。Test: MAE=13.09/R²=0.592/**F1=0.792**(P=0.952/R=0.678)
- [x] **val/test F1 差距分析**：`scripts/analyze_gap.py` 诊断 0.929→0.792 gap。主因：test ON 密度高(0.88% vs 0.60%) + 漏报标准高功率事件(mean 2242W) + 6000 采样小样本噪声
- [x] **增评估样本对比**：`configs/baseline_allsample.yaml`（val/test 30000）重训 + analyze_gap。**gap 0.137→0.047（缩 65%）**，主因确认为 val 小样本乐观偏差（val F1 0.929→0.830）；test F1 0.792→0.783 稳定，模型真实 F1≈0.78
- [x] **增容量+训练样本**：`configs/baseline_big.yaml`（d_model 128/layers 3/FFN 512/train 100k），9ep 早停(best=2)，cuda 180s。**Test F1 0.783→0.850(+8.6%), Recall 0.665→0.820, 漏报 89→48(减46%), MAE 11.15→9.16, R² 0.618→0.729**。主因 B 缓解。但 best_epoch=2 早停暴露 val 抖动

## 进行中

- 无（本会话任务收尾中）

## 下一步（TODO）

1. **解决早停问题**：lr 调度（5e-4→1e-4 或加 ReduceLROnPlateau）让大模型稳定收敛；早停改基于 val F1（ep2 MAE 最低但 ep4 F1=0.881 更高）；增 epoch 30→50 让模型充分训练（train F1 还在升）
2. **多种子验证**：跑 seed=0/1 确认 F1=0.85 稳定（当前单 seed=42）
3. 跨建筑验证：building2/3 训练、building1 测试（用大样本评估）
4. 据稳定性+早停优化结果决定是否沉淀进 `REPORT.md`（当前 F1=0.85 候选）

## 决策记录 / 踩坑

- gh CLI 未安装，用 `git ls-remote origin` 验证远端（HTTPS 凭据可用）
- 工作目录在 worktree 分支 `nilm-project-model-ritual-4zSHFv`，远端 main 为 `7a91470`
- 用户指定用本地 conda `test_gpu` 环境运行（非 README 默认 `transformer_nilm`）；baseline/allsample/big 均沿用
- **[2026-09-04]** Smoke test 两条 PyTorch UserWarning（nested_tensor / flash attention 未编译）非致命
- **[2026-09-04]** UK-DALE 下载阻塞：huggingface.co 超时（GFW）、hf-mirror.com 429 限流。用户手动下载存 `D:\Work\testPython\datasets`
- **[2026-09-04]** conda run 不传递父 shell 的 `HF_ENDPOINT` 给 Python 子进程
- **[2026-09-04]** **关键数据流**：`src/data.py` 不直接读 `ukdale.h5`，只支持 npz；`train.py` 用 `load_simple_npz`。项目原缺 `h5→npz` 预处理——已由 `prepare_ukdale.py` 补齐
- **[2026-09-04]** 数据存 `D:\Work\testPython\datasets\ukdale.h5`；README 默认 `D:\datasets` 与实际不符
- **[2026-09-04]** nilmtk 装不上（不在清华镜像、GitHub SSL 墙）；改纯 h5py+pandas+metadata pickle，无 nilmtk 依赖
- **[2026-09-04]** **meter 定位**：building1 metadata pickle 含 `elec_meters`+`appliances`。mains=meter1(site)，kettle=meter10（meter10 还被 food processor/sandwich maker 共享，存在已知噪声）
- **[2026-09-04]** pandas.read_hdf 需 `tables`——已装 `tables 3.11.1`，`requirements.txt` 已补 `tables>=3.9`
- **[2026-09-04]** **补依赖完成**：`run_real.ps1` 已改 `conda activate test_gpu` + 示例 NPZ 路径更新
- **[2026-09-04]** **PS 5.1 编码**：`run_real.ps1` UTF-8 无 BOM，PS 5.1 用 GBK 解码致中文乱码 + Parser 报字符串终止符错误；PS 7 正常。脚本逻辑正确
- **[2026-09-04]** h5py `ds[:]` 读 pytables table 报错，改 `pd.read_hdf` 成功
- **[2026-09-04]** 对齐策略：`resample('6s').mean()` + `concat(inner)` + `ffill(5).dropna()`；11.3M→10.3M 对齐点
- **[2026-09-04]** baseline 用 `conda run -n test_gpu python scripts/train.py`（非 run_real.ps1，因 conda activate 在非交互 shell 可能失败）
- **[2026-09-04]** **F1 gap 分析**：val/test F1 差距（0.929→0.792）主因三：(A) test 段 ON 密度 0.88% > val 0.60%（季节性漂移），6000 采样暴露 59 vs 29 个 ON；(B) 漏报 19 个标准高功率事件（mean 2242W），模型对部分 aggregate 上下文学习不足（d_model=64/2 层容量受限）；(C) point-level F1 在 <1% ON 密度的 6000 采样上高方差
- **[2026-09-04]** build_splits 用 `np.linspace` 等距抽样（非随机），max_samples 控数量但保留时序稀疏结构；评估在 6000 点上算 F1，稀疏 ON（<1%）下小样本噪声大
- **[2026-09-04]** F1 是 point-level（逐点 ON/OFF ≥ on_threshold），非 event-level；早停基于 val MAE（非 F1），best.pt 是 val MAE 最低的模型
- **[2026-09-04]** **增样本对比结论**：原 gap 0.137 的 65% 来自 **val 小样本乐观偏差**（29 ON 偏简单→F1 虚高 0.929；大样本 173 ON→F1 0.830 真实）。test F1 0.792→0.783 稳定（59→266 ON），真实泛化 F1≈0.78。gap 缩到 0.047。**主因 B 是真实缺陷但不造成 gap**（val/test 漏报率同 33%）。**评估固定用大样本（≥30000）**
- **[2026-09-04]** `analyze_gap.py` 已加 argparse（`--npz --config --ckpt --threshold`），可分析任意 run；`baseline_allsample.yaml` 作后续实验默认评估配置
- **[2026-09-04]** **增容量+样本结论**：d_model 64→128/layers 2→3/FFN 128→512/train 30k→100k，**Test F1 0.783→0.850(+8.6%)**，主因 B 缓解——Recall 0.665→0.820，漏报 89→48 个（减 46%）。Precision 略降 0.952→0.883（误报+20，可接受 trade-off）。MAE 11.15→9.16，R² 0.618→0.729。达 nilmtk 文献 Kettle Seq2Point 上限（~0.85）
- **[2026-09-04]** **早停问题暴露**：big run best_epoch=2（val MAE=5.90 最低），之后 val MAE 剧烈抖动（ep3=8.45/ep4=12.84/ep5=7.55/ep6=5.96），9ep 早停。train F1 仍升（ep1=0.756→ep9=0.858），模型未充分训练。原因：(a) lr 5e-4 对 d_model=128 偏高致 val 震荡；(b) 早停基于 val MAE 非 F1，ep4 val F1=0.881 更高但 MAE=12.84 被错过。下一步调 lr/改 F1 早停/增 epoch
- **[2026-09-04]** val/test gap 反转：allsample +0.047(val 高) → big -0.022(test 高)。ep2 模型在 test 段表现更好，或 val 段 Kettle 用法少（ON 0.60%<test 0.88%）致 val 偏难。非关键，提示早停选择可优化
- **[2026-09-04]** STATUS.md 被 markdown 渲染器反复重排（条目间空行+转义反斜杠）致 Edit 字符串失配，改用 Write 整体重写维护

## 关键文件路径

- 协议：`BOOTSTRAP.md`
- 续接：`STATUS.md`、`session/NILM_AC_session_complete.md`、`REPORT_TEST.md`、`REPORT.md`
- 入口：`README.md`、`requirements.txt`
- 脚本：`run_baseline.ps1`、`run_tuning.ps1`、`run_real.ps1`、`scripts/run_smoke.py`、`scripts/train.py`、`scripts/evaluate.py`、`scripts/inspect_h5.py`、`scripts/prepare_ukdale.py`、`scripts/analyze_gap.py`
- 配置：`configs/baseline.yaml`、`configs/baseline_allsample.yaml`、`configs/baseline_big.yaml`、`configs/tuning.yaml`
- 模块：`src/data.py`、`src/model.py`、`src/metrics.py`、`src/trainer.py`、`src/experiment.py`
- 数据：`D:\Work\testPython\datasets\ukdale.h5`（3.19GB，NILMTK HDF5，5 buildings）
- 预处理：`D:\Work\testPython\datasets\ukdale_prepared.npz`（82.8MB，10.3M 对齐点，aggregate+target）
- 预处理脚本：`scripts/prepare_ukdale.py`
- Smoke 产物：`reports/smoke/{best.pt, history.json, result.json}`
- Baseline 产物：`reports/baseline/{best.pt(280KB, 未提交), history.json, result.json}`
- Baseline allsample 产物：`reports/baseline_allsample/{best.pt(未提交), history.json, result.json}`
- Baseline big 产物：`reports/baseline_big/{best.pt(未提交), history.json, result.json}`
- 验证产物：`reports/verify_real/{best.pt, history.json, result.json}`（未提交，仅磁盘留证）
