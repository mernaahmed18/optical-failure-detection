

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression
import common as C

C.ensure_dirs()

# Load aligned table and candidate episodes
aligned = pd.read_parquet(C.OUT / "aligned_detection_table.parquet")
raw = C.load_raw()
_, feature_cols, value_cols = C.build_aligned(raw)
FS = C.feature_sets(feature_cols, value_cols)
ep_cand = C.candidate_episodes(raw, max_gap_s=7.0)

# Build a 60s-context / 120s-gap split so a 61s window stays inside one partition
split_h, manifest_h = C.build_split(aligned, ep_cand, context_s=60, partition_gap_s=120)
table_h = C.add_log_ber(aligned.drop(columns=["split"]).copy())
table_h["split"] = split_h

manifest_h.to_csv(C.OUT / "split_manifest_history_experiment.csv", index=False)
test_events_h = manifest_h.loc[manifest_h["split"].eq("test")].copy()
print("Episodes per split:", manifest_h["split"].value_counts().to_dict())

# %% [markdown]
# ### Step 2: Compare History Lengths (`W=1` vs. `W=11` vs. `W=61`)

# %%
# Same compact 1D-CNN for all window sizes (1s, 11s, 61s)
class MatchedTemporalCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pad1 = nn.ConstantPad1d((2, 0), 0.0)
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3)
        self.pad2 = nn.ConstantPad1d((4, 0), 0.0)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=3, dilation=2)
        self.head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        h = x.transpose(1, 2)  # (batch, features, time)
        h = torch.relu(self.conv1(self.pad1(h)))
        h = torch.relu(self.conv2(self.pad2(h)))
        last_step = h[:, :, -1]
        avg_over_time = h.mean(dim=-1)
        features = torch.cat([last_step, avg_over_time], dim=-1)
        return self.head(features).squeeze(-1)


all_rows = []
pred_exports = []

# Test both restricted sensor settings with 3 window lengths and 3 random seeds
for setting_name in ["receiver_only", "no_ampli1"]:
    fcols = FS[setting_name]["features"]
    vcols = FS[setting_name]["values"]
    df_scaled = C.scale_split_features(table_h, fcols)

    for w in [1, 11, 61]:
        data = {
            split_name: C.prepare_temporal_windows(df_scaled, fcols, vcols, w, split_name)
            for split_name in ["train", "validation", "test"]
        }

        for seed in (42, 43, 44):
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = MatchedTemporalCNN(len(fcols))
            prob, thr, val_f1 = C.fit_predict_binary_torch(
                model,
                data["train"]["X_labeled"], data["train"]["y_labeled"],
                data["validation"]["X_labeled"], data["validation"]["y_labeled"],
                data["test"]["X_all"],
                batch_size=512, max_epochs=25, patience=4,
            )

            tl = C.map_window_predictions_to_timeline(table_h, data["test"]["metadata"], prob, thr, "test")
            metrics = C.evaluate_predictions(tl, test_events_h, f"{setting_name}_W{w}", setting_name)
            metrics["seed"] = seed
            metrics["val_f1"] = round(val_f1, 4)
            all_rows.append(metrics)

            if seed == 42:
                tl.insert(0, "model", f"hist_{setting_name}_w{w}_test_exploratory")
                pred_exports.append(tl)

# Average results across the 3 seeds
hist_summary = (
    pd.DataFrame(all_rows)
    .groupby(["setting", "model"], sort=False)
    .agg(
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        event_recall_mean=("event_recall", "mean"),
        fa_starts_mean=("false_alarm_starts", "mean"),
        unknown_starts_mean=("alarm_starts_unknown_target", "mean"),
        coverage_mean=("prediction_coverage", "mean"),
    )
    .reset_index()
    .round(4)
)
hist_summary.to_csv(C.OUT / "history_experiment_summary.csv", index=False)
print(hist_summary.to_string(index=False))

# %% [markdown]
# ### Step 3: Schedule-Only Diagnostic (uses only time since previous episode)

# %%
# Average time between episode starts in training data (~51 s)
period_est = float(manifest_h.loc[manifest_h["split"].eq("train"), "start_to_start_s"].dropna().mean())
cols = ["dt", "sin_p", "cos_p"]


def make_sched_df(split_name):
    sub = table_h.loc[table_h["split"].eq(split_name), ["Timestamp", "split", "target"]].sort_values("Timestamp").copy()
    eps = manifest_h.loc[manifest_h["split"].eq(split_name)].sort_values("first_observed_failure")
    starts, ends = eps["first_observed_failure"].to_numpy(), eps["last_observed_failure"].to_numpy()
    dt = [float(t - starts[np.where(ends < t)[0][-1]]) if np.any(ends < t) else np.nan for t in sub["Timestamp"]]
    sub["dt"] = dt
    sub["sin_p"] = np.sin(2 * np.pi * sub["dt"] / period_est)
    sub["cos_p"] = np.cos(2 * np.pi * sub["dt"] / period_est)
    return sub


tr = make_sched_df("train").dropna(subset=["target", "dt"])
va = make_sched_df("validation").dropna(subset=["target", "dt"])
te = make_sched_df("test")

# Train Logistic Regression on time features only and select threshold on validation
clf = LogisticRegression(class_weight="balanced", random_state=42).fit(tr[cols], tr["target"].astype(int))
thr_s, vf1_s = C.select_threshold_f1(va["target"].astype(int).to_numpy(), clf.predict_proba(va[cols])[:, 1])

# Predict on test timeline
te_ok = te["dt"].notna()
te["prob"], te["prediction"] = np.nan, np.nan
te.loc[te_ok, "prob"] = clf.predict_proba(te.loc[te_ok, cols])[:, 1]
te.loc[te_ok, "prediction"] = (te.loc[te_ok, "prob"] >= thr_s).astype(int)

sched_res = C.evaluate_predictions(te, test_events_h, "Schedule_Only_Diagnostic", "schedule_only_non_deployable")
sched_res["val_f1"] = round(vf1_s, 4)
pd.DataFrame([sched_res]).to_csv(C.OUT / "schedule_only_diagnostic_results.csv", index=False)

te_exp = te[["Timestamp", "split", "target", "prob", "prediction"]].copy()
te_exp.insert(0, "model", "schedule_only_diagnostic_test_exploratory")
pred_exports.append(te_exp)
pd.concat(pred_exports, ignore_index=True).to_csv(C.PRED_DIR / "history_experiment_test_predictions.csv", index=False)
print(pd.DataFrame([sched_res])[["model", "precision", "recall", "f1", "event_recall", "false_alarm_starts"]].to_string(index=False))
