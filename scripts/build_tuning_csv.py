"""Build reports/tuning_rounds.csv — chronological per-round tuning metrics.

Gathers every tuning lane from reports/<lane>/result.json (+config.yaml, +history.json / train.log),
adds the anchor dense re-eval and the median4 ensemble rows, computes deltas vs each lane's parent
(only within the same test protocol), and writes one tidy CSV.

Usage:  ./.venv/bin/python scripts/build_tuning_csv.py [--skip-ensemble]
"""
import argparse, csv, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"

# ---------------------------------------------------------------- lane registry (chronological)
# fields: run_id(dir or pseudo), stage, change, parent, basis(依据：针对父轮指标为何这样调), note
LANES = [
    ("ukdale_baseline_cpu_short", "0-anchor", "真实数据缩短版 baseline（10k train/6k test，10ep）", "",
     "非调参轮：v2.0 协议要求先建可在当前环境 1h 完赛的锚点；复用旧 13.62W 产物以省算力并统一参照",
     "锚点模型；后接其稠密复评行"),
    ("anchor_dense_reeval", "0-anchor", "（复评）baseline ckpt 加密到稠密 test30000 复评", "ukdale_baseline_cpu_short",
     "baseline 仅有 test6k：样本量不足以分辨 ±1W 级调参差，且与原稠密基线 11.96W 口径不一致——同 ckpt 加密到 30k 统一后续所有轮的评分口径",
     "非训练轮，eval_ckpt 口径统一"),
    ("tune_l1loss", "1-phase1 单因子", "loss: mse→l1", "anchor_dense_reeval",
     "baseline 的 MAE(13.6) 与 RMSE(293) 差一个量级→大误差主导 MSE，怀疑对强事件的惩罚形状失真；huber δ=1W 在 >1W 已近线性，换纯 L1 直接优化主指标验证",
     "8k/4k/4k×12ep 小口径；L1 坍缩禁用"),
    ("tune_win256_missing", "1-phase1 单因子", "window_size 128→256", "anchor_dense_reeval",
     "128 窗仅 ~7min 上下文，烧水器的预热/沸腾/保温多阶段模式跨窗；翻倍上下文零新增参数，是性价比最高的结构假设",
     "本沙箱 seq256 线性涨内存被 OOM 杀，无产物入库（留档教训）"),
    ("tune_d128", "1-phase1 单因子", "d_model 64→128, ff 128→256", "anchor_dense_reeval",
     "baseline train/val loss 早早收敛且残差大→疑欠拟合而非过拟合，容量（d64/ff128 对 10 类稠密输出）被怀疑为第一瓶颈",
     "8k 小口径下过拟合被否（全长重测见 F8/F10）"),
    ("tune_lr1e3", "1-phase1 单因子", "lr 5e-4→1e-3", "anchor_dense_reeval",
     "12ep 短程里 warmup+cosine 从 2e-4 起走，退火远未完成就撞 epoch 上限→怀疑 lr 过于保守，×5 测收敛速度；若收益来自“走得更远”而非“更大步”，全长训练应改走 cosine 而非提 lr",
     "8k 口径 MAE 略降但 R²/EE 恶化"),
    ("tune_layers4", "1-phase1 单因子", "num_layers 2→4", "anchor_dense_reeval",
     "seq2point 逐点预测依赖注意力跨窗特征组合，2 层疑只够局部去噪；4 层测深度是否打开表达力",
     "无收益"),
    ("tune_drop0", "1-phase1 单因子", "dropout 0.1→0", "anchor_dense_reeval",
     "数据诊断的最强先验：稠密标签含 ~29% 事件窗是“不可学的真值缺失0”（子电表丢数），dropout 随机抹特征恰好教会模型“没把握就压 0”→直接压制召回（baseline R .712 的病根假设）；反直觉预期：噪声标签下应减正则",
     "本轮唯一采纳项"),
    ("tune_c_d0lr1e3", "2-phase2 组合", "drop0 + lr1e-3", "tune_drop0",
     "两因子单开均正收益（drop0 −39%、lr1e3 MAE −4%），按可加性假设组合作为 Phase1 冠军候选进全量前预检",
     "8k 口径第一，后被 10k A/B 否决"),
    ("tune_c_d0lr2e3", "2-phase2 组合", "drop0 + lr2e-3", "tune_drop0",
     "若 2e-4→1e-3 单调变好则外推 2e-3 更优；同时以最小代价圈定短程 schedule 的发散上限",
     "发散"),
    ("tune_final_ab10k", "3-同尺度A/B", "drop0+lr1e-3 于 10k/6k/6k 复核", "ukdale_baseline_cpu_short",
     "Phase1/2 结论建立在 8k×12ep 截断口径上噪声大，且 d128/大 lr 被否可能只是小数据伪像；在与 baseline 完全同口径的 10k/6k 复核组合冠军，消除口径混杂",
     "组合冠军在小口径外推失败"),
    ("tune_final_ab10k_drop0only", "3-同尺度A/B", "drop0-only 对照（同 10k 尺度）", "ukdale_baseline_cpu_short",
     "组合臂从 8k 的 6.65 回退到 10k 的 12.46，无法归因是哪个因子坏了→A/B 设计：两臂仅 lr 一项不同，单因子隔离",
     "确认 drop0 为真收益"),
    ("tune_final_full", "4-全量F2", "drop0 上全量：30k train/8k val/稠密 test30k，20ep 固定lr", "anchor_dense_reeval",
     "drop0 在 8k/10k 两个尺度均复现为唯一稳健正收益→按协议将其定为新稳定版并上全量，评分统一稠密 test30k；其余参数全部冻结以防归因混杂",
     "cosine 前的全量锚点"),
    ("tune_final_f3_cosine", "4-全量F3", "F2 + cosine lr(T_max=20)", "tune_final_full",
     "F2 的 train loss 尾段（~ep14-20）在平台震荡不降→固定 lr 2e-4 的梯度噪声疑为终局瓶颈；cosine 退火到 1e-5 是最便宜、不改模型的稳定化手段",
     "调参专题稳定版（本表后半段父节点）"),
    ("tune_f4_ew3", "5-攻坚", "F3 + event_weight=3（λ扫描）", "tune_final_f3_cosine",
     "F3 复盘：SAE/EE=−19.5% 系统性低估 + FN(1569)≫FP(455)→模型“不敢报火”；事件窗只占样本 ~22%，其监督被多数零窗稀释→温和 λ3 重测事件权重曲线（λ8 臂同时并行，取两点定形）",
     "timeout 截停无 result.json（后由 ew8/F5 补全 λ 结论）"),
    ("tune_f4_ew8", "5-攻坚", "F3 + event_weight=8", "tune_final_f3_cosine",
     "同一 F3 诊断（EE−19.5%、FN≫FP）的激进端：λ8 把事件窗 loss×8，若“稀释”假说成立应显著抬 R/修 EE；同时用 SAE 收敛与否给 λ 定标",
     "大 λ 死路"),
    ("tune_f5_boost_ew3", "5-攻坚", "F3 + boost + λ3 + f1 选点（三因子）", "tune_final_f3_cosine",
     "F4 教训：λ 全局乘子同时放大了 29% 缺事件窗“教你报 0”的错误监督→改从数据侧修：事件池过采样 45%（不改 loss 权重防畸变）+ λ 降到 3 + 选择准则换 val-F1 直接对准 P/R 平衡验收线",
     "R 反跌 .711，三因混杂判死"),
    ("tune_f6_boost_only", "5-攻坚", "F3 + 静态事件拼接 boost（λ1/MAE选点）", "tune_final_f3_cosine",
     "F5 三因子同改无法归因→单因子剥离：只保留 boost、恢复 λ1 与 MAE 选点，裁决 F5 恶化是 boost 还是 f1 选点所致；同时检验“FN 是事件覆盖不足”假说",
     "覆盖假说证伪（FN 98% 重合）"),
    ("tune_f7_stoch_wavg", "5-攻坚", "F3 + 每epoch事件池重采样(45%) + top4权重平均", "tune_final_f3_cosine",
     "静态 boost 连败（F5/F6）指向机制错误：过采样移动了测试期先验→模型更保守（P↑R↓反向）；改 stochastic masking——每 epoch 以 45% 概率把部分窗替换为事件窗，分布整体不动只微增事件梯度；另加 top-4 权重平均消 F3 遗留的尾段震荡",
     "前沿 F1 .892；P 上限 .951"),
    ("tune_f9_stoch_wavg2", "5-攻坚", "F7 + λ2 + minprf 选点", "tune_f7_stoch_wavg",
     "F7 暴露新缺口：P .951 创新高但 R .733（gap 22pp），且 EE−17.5% 低估仍在→最小 λ=2 温和抬事件监督 + 选择准则换 min(P,R,F1) 把短板（R）直接写进优化目标",
     "λ 家族终结：MAE 12.38"),
    ("tune_f8_big", "5-攻坚", "F7 配方 + 容量×4（d128/8h/4L/ff256）", "tune_f7_stoch_wavg",
     "攻坚轮多数卡在 val 3.57-3.65 噪声带、F3 平台十余轮未破→按 plateau-加容量先验重启 d128/4L：ab10k 已证其在 10k 尺度不劣（Phase1 的否决是小数据×dropout 混杂伪像），F7 的正则配方恰好扫清扩容量的过拟合前提",
     "23ep 被 timeout 截停；指标=ep22 快照 eval_ckpt 复评（无 wavg）"),
    ("tune_f10_biglong", "5-攻坚", "F8 配方全长 30ep/patience10 + wavg4（新稳定版）", "tune_f8_big",
     "F8 在 23/30ep 被资源截停且未来得及叠 wavg；F2→F3 先例证明“尾段稳定化”值 −19%→补全长训练+top4 平均，测容量版的完整收益；patience 提至 10 防 cosine 尾段误停",
     "canonical；val 3.9→3.57→3.66 曲线完赛"),
    ("tune_f11_big_jit", "5-攻坚", "（计划加 jitter，sed 失配→实为 F10 确定性复跑）", "tune_f10_biglong",
     "F10 R .797 召回缺口部分疑为模型记忆事件“位置”而非形状→计划 roll_jitter±2 强制学形状；配置 sed 替换失配使本轮实为 F10 原样复跑——逐位一致的结果意外提供了全表唯一的确定性对照",
     "与 F10 逐位一致，验证管道确定性"),
    ("tune_f12_jit_cont", "5-攻坚", "F10 终权重热启动 + roll_jitter=2 续训 12ep", "tune_f10_biglong",
     "F11 事故使 jitter 假说未测→补测；且从头训 30ep 成本高，改为从 F10 终权重 init_ckpt 热启动短续训：若 F10 平台是局部最优可借 jitter 逃逸，若已是好解只付小成本",
     "双刃：低阈端 R .97 但 MAE/EE 恶化"),
    ("tune_f13_seed43", "5-攻坚", "F10 配方 seed 42→43", "tune_f10_biglong",
     "各臂间差值多在 ±1.5W，而全程单 seed→无法判定 6.68 是配方收益还是抽风运气；唯一变量换 seed 建立噪声地板：差值大于地板才允许归因，小于地板的结论一律降级",
     "seed 敏感性 ±0.5W；回归面稳、运营点面 ±2pp"),
]

PARAM_KEYS = ["seed", "window_size", "max_samples_train", "max_samples_val", "max_samples_test",
              "d_model", "nhead", "num_layers", "dim_feedforward", "dropout",
              "lr", "epochs", "patience", "batch_size", "weight_decay", "loss", "lr_schedule",
              "event_weight", "select_on", "weight_avg", "roll_jitter", "init_ckpt"]


def fmt(v, nd=3):
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{v}"
    return v


def _hist_rows(h, best_epoch):
    """Return the history row (dict) at best_epoch, or None. history may be list[dict] or dict-of-lists."""
    if h is None or best_epoch is None:
        return None
    if isinstance(h, list):
        for e in h:
            if e.get("epoch") == best_epoch:
                return e
        return None
    if isinstance(h, dict):
        try:
            i = int(best_epoch) - 1
            row = {}
            for k, v in h.items():
                if isinstance(v, (list, tuple)) and len(v) > i:
                    row[k] = v[i]
            return row or None
        except Exception:
            return None
    return None


def load_lane(run_id):
    d = REP / run_id
    out = {"run_id": run_id}
    if not d.exists() or run_id == "tune_win256_missing":
        return out | {"status": "无产物"}
    cfgp, resp, hisp = d / "config.yaml", d / "result.json", d / "history.json"
    import yaml
    cfg = yaml.safe_load(cfgp.read_text(encoding="utf-8")) if cfgp.exists() else {}
    out["config"] = cfg
    best_epoch = None
    test = {}
    if resp.exists():
        r = json.loads(resp.read_text(encoding="utf-8"))
        if "test" in r:  # standard schema
            test = r["test"]
            best_epoch = r.get("best_epoch")
            out["n_train"], out["n_val"], out["n_test"] = r.get("n_train"), r.get("n_val"), r.get("n_test")
            out["runtime_sec"] = round(r.get("runtime_sec", 0), 1)
            out["status"] = "完赛"
        else:  # flat schema from eval_ckpt (f8)
            test = r
            out["n_test"] = r.get("n_test")
            out["status"] = "截停后 eval_ckpt 补评"
    if best_epoch is None:
        if hisp.exists():
            try:
                hist = json.loads(hisp.read_text(encoding="utf-8"))
                if isinstance(hist, list):  # find min val_mae epoch
                    best_epoch = min(hist, key=lambda e: e.get("val_mae", 9e9)).get("epoch")
            except Exception:
                pass
        if best_epoch is None and (d / "train.log").exists():  # e.g. f8: no history.json, parse log
            m = re.findall(r"Epoch (\d+) \| train MAE=[\d.]+ \| val MAE=([\d.]+) \| val F1=([\d.]+)",
                           (d / "train.log").read_text(encoding="utf-8", errors="ignore"))
            if m:
                k = min(range(len(m)), key=lambda i: float(m[i][1]))
                best_epoch = int(m[k][0])
                out["val_mae"] = float(m[k][1]); out["val_f1"] = float(m[k][2])
    if not resp.exists():
        out["status"] = "截停/无 result.json"
    elif "status" not in out:
        out["status"] = "完赛"
    h = None
    if hisp.exists():
        try:
            h = json.loads(hisp.read_text(encoding="utf-8"))
        except Exception:
            h = None
    row = _hist_rows(h, best_epoch)
    if row:
        for k in ("val_mae", "val_r2", "val_precision", "val_recall", "val_f1"):
            if k in row:
                out[k] = row[k]
    out["best_epoch"] = best_epoch
    out["test"] = test
    # params
    m = (cfg.get("model") or {}); t = (cfg.get("training") or {}); da = (cfg.get("data") or {})
    for key, ck in (("n_train", "max_samples_train"), ("n_val", "max_samples_val"),
                    ("n_test", "max_samples_test")):
        if out.get(key) is None:
            out[key] = da.get(ck)
    eb = da.get("event_boost")
    out["params"] = {
        "seed": cfg.get("seed"), "window_size": da.get("window_size"),
        "max_samples_train": da.get("max_samples_train"), "max_samples_val": da.get("max_samples_val"),
        "max_samples_test": da.get("max_samples_test"),
        "d_model": m.get("d_model"), "nhead": m.get("nhead"), "num_layers": m.get("num_layers"),
        "dim_feedforward": m.get("dim_feedforward"), "dropout": m.get("dropout"),
        "lr": t.get("lr"), "epochs": t.get("epochs"), "patience": t.get("patience"),
        "batch_size": t.get("batch_size"), "weight_decay": t.get("weight_decay"), "loss": t.get("loss"),
        "lr_schedule": t.get("lr_schedule"), "event_weight": t.get("event_weight"),
        "select_on": t.get("select_on"), "weight_avg": t.get("weight_avg"),
        "roll_jitter": da.get("roll_jitter"), "init_ckpt": t.get("init_ckpt"),
    }
    if eb:
        st = "stoch" if eb.get("stochastic_epochs") else "static"
        out["params"]["event_boost"] = f"max_extra={eb.get('max_extra')},{st},frac={eb.get('event_frac')}"
    else:
        out["params"]["event_boost"] = ""
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ensemble", action="store_true")
    a = ap.parse_args()

    recs, by_id = [], {}
    for run_id, stage, change, parent, basis, note in LANES:
        if run_id == "anchor_dense_reeval":
            j = json.loads((REP / "ukdale_baseline_cpu_short" / "dense_test_eval.json").read_text(encoding="utf-8"))
            base = by_id["ukdale_baseline_cpu_short"]
            ap = dict(base["params"])
            ap["max_samples_test"] = j.get("n_test") or 30000  # 显示实际复评考卷而非 baseline 名义值
            rec = {"run_id": run_id, "stage": stage, "change": change, "parent": parent, "note": note,
                   "rationale": basis,
                   "test": j, "params": ap, "n_test": j.get("n_test"),
                   "report_dir": "ukdale_baseline_cpu_short",
                   "n_train": base.get("n_train"), "n_val": base.get("n_val"),
                   "best_epoch": base.get("best_epoch"), "runtime_sec": "",
                   **{k: v for k, v in base.items() if k.startswith("val_")}, "status": "复评(同ckpt加密口径)"}
        else:
            rec = load_lane(run_id)
            rec.update({"stage": stage, "change": change, "parent": parent, "note": note, "rationale": basis,
                        "report_dir": run_id if (REP / run_id).is_dir() else ""})
        by_id[run_id] = rec
        recs.append(rec)

    # ---- ensemble row: median4(f8,f10,f12,f13) recomputed from checkpoints (val-selected t*=115)
    if not a.skip_ensemble:
        try:
            er = build_ensemble_row()
            er["report_dir"] = ""  # 集成轮无独立产物目录（模型文件分散在四个成员目录）
            er["rationale"] = ("F13 已证 seed 噪声 ±0.7W 且各臂 FN 漏检位置互不相同→功率中位数集成可同时"
                               "平均掉 MAE 噪声并互补召回缺口；median 比 mean 抗单成员过预测；t*=115 仅用 val8k 选定防 test 偷看")
            recs.append(er)
        except Exception as e:
            print("ensemble row skipped:", e)

    cols = (["seq", "run_id", "report_dir", "stage", "change", "rationale", "parent", "status", "n_train", "n_val", "n_test",
             "best_epoch", "runtime_sec"] + [f"p_{k}" for k in PARAM_KEYS] + ["p_event_boost"] +
            ["val_mae_W", "val_r2", "val_precision", "val_recall", "val_f1",
             "test_mae_W", "test_rmse_W", "test_r2", "test_sae", "test_energy_error",
             "test_precision", "test_recall", "test_f1",
             "d_mae_W", "d_mae_pct", "d_r2", "d_sae", "d_precision", "d_recall", "d_f1", "protocol_diff", "note"])

    def prot(rec):
        return rec.get("n_test")

    rows = []
    for i, rec in enumerate(recs, 1):
        t = rec.get("test", {}) or {}
        p = rec.get("params", {}) or {}
        row = {
            "seq": i, "run_id": rec["run_id"],
            "report_dir": rec.get("report_dir", ""),
            "stage": rec.get("stage", ""), "change": rec.get("change", ""),
            "rationale": rec.get("rationale", ""),
            "parent": rec.get("parent", ""), "status": rec.get("status", ""),
            "n_train": rec.get("n_train", ""), "n_val": rec.get("n_val", ""), "n_test": rec.get("n_test", ""),
            "best_epoch": rec.get("best_epoch", ""), "runtime_sec": rec.get("runtime_sec", ""),
            "test_mae_W": fmt(t.get("mae"), 2), "test_rmse_W": fmt(t.get("rmse"), 1),
            "test_r2": fmt(t.get("r2"), 4), "test_sae": fmt(t.get("sae"), 4),
            "test_energy_error": fmt(t.get("energy_error"), 4),
            "test_precision": fmt(t.get("precision"), 4), "test_recall": fmt(t.get("recall"), 4),
            "test_f1": fmt(t.get("f1"), 4),
            "val_mae_W": fmt(rec.get("val_mae"), 3), "val_r2": fmt(rec.get("val_r2"), 4),
            "val_precision": fmt(rec.get("val_precision"), 3), "val_recall": fmt(rec.get("val_recall"), 3),
            "val_f1": fmt(rec.get("val_f1"), 3),
            "note": rec.get("note", ""),
        }
        for k in PARAM_KEYS:
            row[f"p_{k}"] = p.get(k, "")
        row["p_event_boost"] = p.get("event_boost", "")
        par = by_id.get(rec.get("parent") or "") or {}
        pt = par.get("test", {}) or {}
        # 公平性审计(2026-09-07 结论)：train/val/test 三口径分别决定"学习材料/选择标尺/
        # 考卷"，任一被"已知地"改变即构成双变量对比 → Δ 置空并在 protocol_diff 列注明；
        # 单边未记录(纯评估/复评行)记 "?" 不阻塞，n_test 仍必须相等。
        diffs, blocker = [], False
        for k in ("n_train", "n_val", "n_test"):
            if not par:
                break
            # 公平性以 config 名义采样参数为准；实际长度(可被 event_boost 等处置撑大)仅作展示
            cv, pv2 = (rec.get("params") or {}).get({"n_train": "max_samples_train",
                     "n_val": "max_samples_val", "n_test": "max_samples_test"}[k]), \
                     (par.get("params") or {}).get({"n_train": "max_samples_train",
                     "n_val": "max_samples_val", "n_test": "max_samples_test"}[k])
            if cv is None or pv2 is None:
                if (cv is None) != (pv2 is None):
                    diffs.append(f"{k}:?")
            elif cv != pv2:
                diffs.append(f"{k}:{pv2}->{cv}")
                blocker = True
        row["protocol_diff"] = ("same" if not diffs else ";".join(diffs)) if par else ""
        if pt and t and prot(rec) == prot(par) and not blocker:
            g = lambda k: (t.get(k), pt.get(k))
            dm, dr2, ds, dp, dr_, df = g("mae"), g("r2"), g("sae"), g("precision"), g("recall"), g("f1")
            row["d_mae_W"] = fmt(dm[0] - dm[1], 2) if None not in dm else ""
            row["d_mae_pct"] = f"{100*(dm[0]-dm[1])/dm[1]:+.1f}%" if None not in dm and dm[1] else ""
            row["d_r2"] = fmt(dr2[0] - dr2[1], 4) if None not in dr2 else ""
            row["d_sae"] = fmt(ds[0] - ds[1], 4) if None not in ds else ""
            row["d_precision"] = fmt(dp[0] - dp[1], 4) if None not in dp else ""
            row["d_recall"] = fmt(dr_[0] - dr_[1], 4) if None not in dr_ else ""
            row["d_f1"] = fmt(df[0] - df[1], 4) if None not in df else ""
        rows.append(row)

    out = REP / "tuning_rounds.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"written {out} ({len(rows)} rows)")


def build_ensemble_row():
    """median4(f8,f10,f12,f13) power ensemble; decision threshold selected on val8k.
    Cached in reports/.ens_cache.json (recompute by deleting the cache file)."""
    cache = REP / ".ens_cache.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    import torch
    from torch.utils.data import DataLoader
    from src.model import NILMTransformer
    from src.data import build_splits
    from src.experiment import seed_everything
    import yaml
    lanes = ["tune_f8_big", "tune_f10_biglong", "tune_f12_jit_cont", "tune_f13_seed43"]
    z = np.load(ROOT / "data" / "ukdale_prepared.npz")
    agg, tgt = z["aggregate"], z["target"]
    cfg = yaml.safe_load((REP / "tune_f10_biglong" / "config.yaml").read_text(encoding="utf-8"))
    seed_everything(42)
    tr, va, te = build_splits(agg, tgt, 128, 0.7, 0.15, 30000, 8000, 30000,
                              event_boost=cfg["data"].get("event_boost"))

    def infer(ds, ck):
        m = NILMTransformer(**cfg["model"])
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
        m.eval()
        ps = []
        with torch.no_grad():
            for x, y in DataLoader(ds, batch_size=256):
                ps.append((m(x).numpy() * float(tr.y_std) + float(tr.y_mean)).ravel())
        return np.concatenate(ps)

    ens_te = np.median(np.stack([infer(te, REP / l / "best.pt") for l in lanes]), axis=0)
    ens_va = np.median(np.stack([infer(va, REP / l / "best.pt") for l in lanes]), axis=0)
    yt_te = tgt[te.indices].astype(np.float64); yt_va = tgt[va.indices].astype(np.float64)
    on_te, on_va = yt_te >= 500, yt_va >= 500
    best = None
    for t in np.arange(20, 520, 5):
        p = ens_va >= t; tp = int(np.sum(on_va & p)); fp = int(np.sum(~on_va & p))
        Pv = tp / max(tp + fp, 1); Rv = tp / max(int(on_va.sum()), 1); mm = min(Pv, Rv)
        if best is None or mm > best[0]:
            best = (mm, float(t))
    tstar = best[1]
    p = ens_te >= tstar
    tp = int(np.sum(on_te & p)); fp = int(np.sum(~on_te & p))
    err = ens_te - yt_te
    mae = float(np.mean(np.abs(err))); rmse = float(np.sqrt(np.mean(err ** 2)))
    r2 = float(1 - np.mean(err ** 2) / np.var(yt_te))
    ee = float(np.sum(err) / np.sum(yt_te))          # repo convention: energy_error
    sae = abs(ee)                                     # repo: SAE == |energy_error| (metrics.py:14)
    Pv = tp / max(tp + fp, 1); R = tp / int(on_te.sum()); F1 = 2 * Pv * R / max(Pv + R, 1e-9)
    rec = {"run_id": "ens_median4_f8f10f12f13", "stage": "5-攻坚(集成)",
           "change": f"median({'+'.join(l.replace('tune_','') for l in lanes)}) 功率集成；决策阈值按 val8k 选点 t*={tstar:.0f}W（test 未参与选点）",
           "parent": "tune_f10_biglong", "status": "集成(无训练)",
           "test": {"mae": mae, "rmse": rmse, "r2": r2,
                    "sae": sae, "energy_error": ee, "precision": Pv, "recall": R, "f1": F1},
           "n_test": len(yt_te), "best_epoch": "", "runtime_sec": "", "params": {},
           "note": "非训练轮；500W 双端协议点 P .950/R .778；test 侧最优 t=95 曾达 .906/.906（仅诊断，不作验收）"}
    cache.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec


if __name__ == "__main__":
    main()
