
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import common as C

C.ensure_dirs()
aligned = pd.read_parquet(C.OUT / "aligned_detection_table.parquet")
manifest = pd.read_csv(C.OUT / "split_manifest.csv")
raw = C.load_raw()
_, feature_cols, value_cols = C.build_aligned(raw)
FS = C.feature_sets(feature_cols, value_cols)

# Document measurement access per setting
fs_doc = pd.DataFrame([
    {"setting": k, "n_features": len(v["features"]), "description": v["description"], "features": ", ".join(v["features"])}
    for k, v in FS.items()
])
print(fs_doc[["setting", "n_features", "description"]].to_string(index=False))

# Check phase composition of validation and test partitions
print("\nPhase distribution per split:")
print(pd.crosstab(aligned["split"], aligned["phase"]))

aligned_log = C.add_log_ber(aligned)
test_events = manifest.loc[manifest["split"].eq("test")].copy()

# %%
# Shared PyTorch compact causal 1D-CNN trainer (identical architecture for all settings)
class CausalCNN1D(nn.Module):
    def __init__(self, in_channels: int, window_points: int):
        super().__init__()
        self.pad1 = nn.ConstantPad1d((2, 0), 0.0)
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3)
        self.pad2 = nn.ConstantPad1d((4, 0), 0.0)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=3, dilation=2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * window_points, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, C) -> (B, C, W)
        h = x.transpose(1, 2)
        h = torch.relu(self.conv1(self.pad1(h)))
        h = torch.relu(self.conv2(self.pad2(h)))
        return self.head(h).squeeze(-1)


def fit_eval_cnn(df_source: pd.DataFrame, setting_key: str, window_points: int = 11, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    fcols, vcols = FS[setting_key]["features"], FS[setting_key]["values"]
    df_proc = C.scale_split_features(df_source, fcols)
    data = {s: C.prepare_temporal_windows(df_proc, fcols, vcols, window_points, s) for s in ["train", "validation", "test"]}

    model = CausalCNN1D(len(fcols), window_points)
    test_prob, thr, val_f1 = C.fit_predict_binary_torch(
        model,
        data["train"]["X_labeled"], data["train"]["y_labeled"],
        data["validation"]["X_labeled"], data["validation"]["y_labeled"],
        data["test"]["X_all"],
        batch_size=128, max_epochs=40, patience=6,
    )
    tl = C.map_window_predictions_to_timeline(df_source, data["test"]["metadata"], test_prob, thr, "test")
    return tl, thr, val_f1


def fit_eval_tabular(df_source: pd.DataFrame, setting_key: str, model_kind: str):
    fcols = FS[setting_key]["features"]
    vcols = FS[setting_key]["values"]

    def valid_rows(split_name: str, labeled_only: bool):
        m = df_source["split"].eq(split_name) & df_source[vcols].notna().any(axis=1)
        if labeled_only:
            m = m & df_source["target"].notna()
        return df_source.loc[m]

    tr = valid_rows("train", True)
    va = valid_rows("validation", True)

    if model_kind == "lr":
        clf = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ])
    elif model_kind == "rf":
        clf = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)),
        ])
    else:
        raise ValueError(model_kind)

    clf.fit(tr[fcols], tr["target"].astype(int))
    va_prob = clf.predict_proba(va[fcols])[:, 1]
    thr, val_f1 = C.select_threshold_f1(va["target"].astype(int).to_numpy(), va_prob)

    tl = df_source.loc[df_source["split"].eq("test"), ["Timestamp", "split", "target"]].copy()
    te_valid = valid_rows("test", False)
    te_prob = clf.predict_proba(te_valid[fcols])[:, 1]
    pred_df = pd.DataFrame({
        "Timestamp": te_valid["Timestamp"].to_numpy(),
        "prob": te_prob,
        "prediction": (te_prob >= thr).astype(int),
    })
    tl = tl.merge(pred_df, on="Timestamp", how="left")
    return tl, thr, val_f1


# %%
results = []
timelines = {}

# 1. Ampli1 Input-Power Threshold
tr_a1 = aligned.loc[aligned["split"].eq("train") & aligned["target"].notna(), "Ampli1_InputPower"].dropna()
va_a1 = aligned.loc[aligned["split"].eq("validation") & aligned["target"].notna() & aligned["Ampli1_InputPower"].notna()]
cands = np.unique(np.round(np.quantile(tr_a1, np.linspace(0.005, 0.995, 300)), 4))
best_a1_thr, best_a1_vf1 = float(cands[0]), -1.0
for c in cands:
    p = (va_a1["Ampli1_InputPower"] <= c).astype(int)
    y = va_a1["target"].astype(int)
    f1 = 2 * ((p == 1) & (y == 1)).sum() / max(1, (p == 1).sum() + (y == 1).sum())
    if f1 > best_a1_vf1:
        best_a1_vf1 = float(f1)
        best_a1_thr = float(c)

tl_a1 = aligned.loc[aligned["split"].eq("test"), ["Timestamp", "split", "target", "Ampli1_InputPower"]].copy()
avail_a1 = tl_a1["Ampli1_InputPower"].notna()
tl_a1["prob"] = np.where(avail_a1, -tl_a1["Ampli1_InputPower"], np.nan)
tl_a1["prediction"] = np.where(avail_a1, (tl_a1["Ampli1_InputPower"] <= best_a1_thr).astype(float), np.nan)
tl_a1 = tl_a1[["Timestamp", "split", "target", "prob", "prediction"]]
C.export_predictions(tl_a1, "ampli1_threshold", "test")
timelines["Ampli1_Threshold"] = tl_a1
r = C.evaluate_predictions(tl_a1, test_events, "Ampli1_Threshold", "ampli1_only")
r["val_threshold"] = best_a1_thr
r["val_f1"] = round(best_a1_vf1, 4)
results.append(r)

# 2. Tabular + Temporal models across matched settings & BER representations
specs = [
    ("Receiver_LR_rawBER", "receiver_only", "lr", aligned, "receiver_lr_rawber"),
    ("Receiver_LR_logBER", "receiver_only", "lr", aligned_log, "receiver_lr_logber"),
    ("Receiver_RF_rawBER", "receiver_only", "rf", aligned, "receiver_rf_rawber"),
    ("Receiver_RF_logBER", "receiver_only", "rf", aligned_log, "receiver_rf_logber"),
    ("Receiver_CNN_W11_logBER", "receiver_only", "cnn", aligned_log, "receiver_cnn_w11_logber"),
    ("NoAmpli1_RF_logBER", "no_ampli1", "rf", aligned_log, "noampli1_rf_logber"),
    ("NoAmpli1_CNN_W11_logBER", "no_ampli1", "cnn", aligned_log, "noampli1_cnn_w11_logber"),
    ("AllDevices_RF_logBER", "all_devices", "rf", aligned_log, "alldevices_rf_logber"),
    ("AllDevices_CNN_W11_logBER", "all_devices", "cnn", aligned_log, "alldevices_cnn_w11_logber"),
]

for name, skey, mkind, df_src, slug in specs:
    if mkind == "cnn":
        tl, thr, vf1 = fit_eval_cnn(df_src, skey, window_points=11, seed=42)
    else:
        tl, thr, vf1 = fit_eval_tabular(df_src, skey, mkind)
    C.export_predictions(tl, slug, "test")
    timelines[name] = tl
    res = C.evaluate_predictions(tl, test_events, name, skey)
    res["val_threshold"] = round(thr, 4)
    res["val_f1"] = round(vf1, 4)
    results.append(res)

res_df = pd.DataFrame(results)
res_df.to_csv(C.OUT / "matched_baselines_results.csv", index=False)
print("\n=== Matched Baselines (Full Valid Test Timeline per Model) ===")
cols_show = [
    "model", "setting", "scored_known_seconds", "prediction_coverage",
    "precision", "recall", "f1", "time_in_alarm_scored_frac",
    "detected_events", "event_recall", "false_alarm_starts",
    "alarm_starts_unknown_target", "median_observed_delay_s", "max_observed_delay_s"
]
print(res_df[cols_show].to_string(index=False))

# %%
# Common-intersection comparison (seconds where ALL models have predictions & known target)
common_ts = None
for tl in timelines.values():
    valid = set(tl.loc[tl["prediction"].notna() & tl["target"].notna(), "Timestamp"].tolist())
    common_ts = valid if common_ts is None else (common_ts & valid)
common_ts = sorted(common_ts)
print(f"\nCommon scored known-target test seconds across all {len(timelines)} models: {len(common_ts)}")

common_rows = []
for name, tl in timelines.items():
    sub = tl.loc[tl["Timestamp"].isin(common_ts)]
    ev_r = C.evaluate_predictions(sub, test_events, name, "common_intersection")
    common_rows.append({
        "model": name,
        "common_seconds": len(sub),
        "precision": ev_r["precision"],
        "recall": ev_r["recall"],
        "f1": ev_r["f1"],
        "time_in_alarm_frac": ev_r["time_in_alarm_scored_frac"],
    })
common_df = pd.DataFrame(common_rows)
print(common_df.to_string(index=False))
