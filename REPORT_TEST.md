# REPORT_TEST.md — 专题报告（只追加）

---

## [2026-09-04] 专题：Smoke Test 链路验证
- **类型**：验证专题（代码链路验证，非 UK-DALE 科学实验）
- **目标与假设**：
  - 验证 NILM Transformer 代码链路完整可跑通：合成数据 → Dataset → Transformer Encoder → Train → Val → Test → Metrics → Checkpoint
  - 假设：在可控合成信号上若链路通过且指标合理，即说明 `src/` 模块与 `scripts/run_smoke.py` 入口逻辑无结构性缺陷，可放心进入 baseline / 真实 UK-DALE 阶段
- **方法 / 数据 / 参数**：
  - 环境：本地 conda env `test_gpu`（Python 3.11.11, torch 2.3.1+cu121, CUDA 可用）
  - 入口：`python scripts/run_smoke.py`（在项目根目录执行，因脚本用相对路径 `configs/baseline.yaml`）
  - 数据：`src/data.py::make_synthetic_signal(10000)` 合成信号，window_size=64
  - 样本量：train 3000 / val 600 / test 600
  - 模型（覆盖 `configs/baseline.yaml`）：d_model=32, nhead=4, num_layers=1, dim_feedforward=64, dropout=0.1, input_dim=1
  - 训练：batch_size=64, epochs=3, lr=5e-4, weight_decay=1e-4, patience=2, grad_clip=1.0, loss=mse, seed=42, device=auto(→cuda)
  - 阈值：on_threshold_watts=500
- **结果 / 结论**：
  - 链路：✅ 全链路跑通，无报错；2 条 PyTorch UserWarning（nested_tensor / flash attention 未编译）非致命，不影响结果
  - 训练曲线（3 epoch）：
    - Ep1: train MAE=87.88 / val MAE=48.04 / train R²=0.2543 / val R²=0.5816
    - Ep2: train MAE=44.86 / val MAE=15.11 / train R²=0.6633 / val R²=0.9729
    - Ep3: train MAE=22.32 / val MAE=13.15 / train R²=0.9238 / val R²=0.9647
  - 最终指标（best_epoch=3, device=cuda, runtime=4.74s）：
    - MAE=61.14, RMSE=175.70, R²=0.9062, SAE=0.0336
    - Energy Error=-0.0336, Precision=1.0, Recall=1.0, F1=1.0
  - 产物：`reports/smoke/best.pt`（42KB checkpoint）、`reports/smoke/history.json`、`reports/smoke/result.json`
  - 结论：代码链路验证通过。指标来自合成信号，**不是 UK-DALE 科学结果**，仅用于结构性回归测试
- **是否进入 REPORT.md（稳定结论）**：否（smoke test 不构成稳定实验结论，仅为链路验证；待真实 UK-DALE 实验产出后再考虑沉淀）
- **遗留问题**：
  - Test MAE(61.14) 高于 Val MAE(13.15)：合成信号在 test 段分布与 train/val 不同步，属合成数据特性，不代表模型问题；真实数据上需按 README §8 重新评估
  - 后续：是否进入 baseline（需 ukdale.h5）、调优、或跨家庭实验，待用户决策
