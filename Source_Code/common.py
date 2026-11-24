"""Shared utilities for Optical Failure Detection experiments.

Centralizes causal alignment, target construction, episode grouping,
chronological splitting with protected episode context, BER log10
transformation, temporal windowing, evaluation metrics, and prediction exports.
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "HardFailure_dataset.csv"
OUT = ROOT / "outputs"
PRED_DIR = OUT / "predictions"
FIG_DIR = OUT / "figures"

PHASE2_START = 1623416338
PHASE3_START = 1623423687
AGE_LIMIT_S = 10.5
DEFAULT_LABEL_MAX_AGE_S = 4.0
BER_FLOOR = 1e-12

DEVICE_MEASUREMENTS: dict[str, list[str]] = {
    "Ampli1": ["InputPower", "OutputPower"],
    "Ampli2": ["InputPower", "OutputPower"],
    "Ampli3": ["InputPower", "OutputPower"],
    "Ampli4": ["InputPower", "OutputPower"],
    "SPO1/18/11": ["BER", "OSNR"],
    "SPO2/18/11": ["BER", "OSNR"],
}


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    OUT.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def dataset_sha256(path: Path = DATA_PATH) -> str:
    """Compute SHA-256 hex digest of the raw dataset CSV."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw telemetry CSV."""
    return pd.read_csv(path)


def assign_phase(timestamps: pd.Series | np.ndarray) -> np.ndarray:
    """Map Unix timestamps to documented experimental phases."""
    return np.select(
        [timestamps < PHASE2_START, timestamps < PHASE3_START],
        ["P1_normal", "P2_normal_variable_attenuation"],
        default="P3_fault_experiments",
    )


def build_aligned(
    df_raw: pd.DataFrame,
    age_limit_s: float = AGE_LIMIT_S,
    label_max_age_s: float = DEFAULT_LABEL_MAX_AGE_S,
    keep_audit: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build 1-second causal detection table with freshness limits and consensus target."""
    start = int(df_raw["Timestamp"].min())
    end = int(df_raw["Timestamp"].max())
    aligned = pd.DataFrame({"Timestamp": np.arange(start, end + 1, dtype=np.int64)})

    feature_cols: list[str] = []
    for device, measurements in DEVICE_MEASUREMENTS.items():
        prefix = device.replace("/", "_")
        source_col = f"{prefix}_source_timestamp"
        age_col = f"{prefix}_age_s"

        readings = (
            df_raw.loc[df_raw["ID"].eq(device), ["Timestamp"] + measurements]
            .sort_values("Timestamp")
            .copy()
        )
        rename_map = {"Timestamp": source_col, **{m: f"{prefix}_{m}" for m in measurements}}
        readings = readings.rename(columns=rename_map)

        aligned = pd.merge_asof(
            aligned,
            readings,
            left_on="Timestamp",
            right_on=source_col,
            direction="backward",
            allow_exact_matches=True,
        )
        aligned[age_col] = aligned["Timestamp"] - aligned[source_col]
        matched = aligned[source_col].notna()
        assert (
            aligned.loc[matched, source_col] <= aligned.loc[matched, "Timestamp"]
        ).all(), f"Future reading selected for {device}"

        unavailable = (~matched) | aligned[age_col].gt(age_limit_s)
        for m in measurements:
            val_col = f"{prefix}_{m}"
            miss_col = f"{val_col}_missing_ind"
            aligned.loc[unavailable, val_col] = np.nan
            aligned[miss_col] = aligned[val_col].isna().astype("int8")
            feature_cols.extend([val_col, miss_col])

    value_cols = [c for c in feature_cols if not c.endswith("_missing_ind")]
    aligned["all_measurements_missing"] = aligned[value_cols].isna().all(axis=1)
    aligned["phase"] = assign_phase(aligned["Timestamp"])

    # Consensus target from device annotations
    label_audit = aligned[["Timestamp"]].copy()
    label_columns: list[str] = []
    for device in DEVICE_MEASUREMENTS:
        prefix = device.replace("/", "_")
        time_col = f"{prefix}_label_time"
        status_col = f"{prefix}_annotation_status"

        src = (
            df_raw.loc[df_raw["ID"].eq(device), ["Timestamp", "Failure"]]
            .sort_values("Timestamp")
            .copy()
        )
        assert src["Failure"].dropna().eq(1).all()
        src[status_col] = np.where(src["Failure"].eq(1), 1.0, 0.0)
        src = src.rename(columns={"Timestamp": time_col})

        label_audit = pd.merge_asof(
            label_audit,
            src[[time_col, status_col]],
            left_on="Timestamp",
            right_on=time_col,
            direction="backward",
            allow_exact_matches=True,
        )
        age = label_audit["Timestamp"] - label_audit[time_col]
        label_audit.loc[age.isna() | age.gt(label_max_age_s), status_col] = np.nan
        label_columns.append(status_col)
        if keep_audit:
            aligned[time_col] = label_audit[time_col]
            aligned[status_col] = label_audit[status_col]

    statuses = label_audit[label_columns]
    all_avail = statuses.notna().all(axis=1)
    all_fail = all_avail & statuses.eq(1).all(axis=1)
    all_unmarked = all_avail & statuses.eq(0).all(axis=1)

    aligned["target"] = pd.Series(pd.NA, index=aligned.index, dtype="Int64")
    aligned["label_reason"] = "unavailable_or_disagreeing_annotations"
    aligned.loc[all_fail, "target"] = 1
    aligned.loc[all_fail, "label_reason"] = "all_devices_marked_failure"
    aligned.loc[all_unmarked, "target"] = 0
    aligned.loc[all_unmarked, "label_reason"] = "all_devices_unmarked_under_repository_convention"
    aligned.loc[aligned["all_measurements_missing"], "target"] = pd.NA
    aligned.loc[aligned["all_measurements_missing"], "label_reason"] = "no_sensor_evidence"

    assert not {"target", "phase", "Timestamp", "label_reason"}.intersection(feature_cols)
    return aligned, feature_cols, value_cols


def add_log_ber(df: pd.DataFrame, floor: float = BER_FLOOR) -> pd.DataFrame:
    """Return a copy with BER columns transformed via log10(clip(BER, floor, None))."""
    out = df.copy()
    for col in ("SPO1_18_11_BER", "SPO2_18_11_BER"):
        if col in out.columns:
            vals = out[col].to_numpy(dtype=float, copy=True)
            mask = ~np.isnan(vals)
            vals[mask] = np.log10(np.clip(vals[mask], floor, None))
            out[col] = vals
    return out


def feature_sets(feature_cols: Sequence[str], value_cols: Sequence[str]) -> dict[str, dict[str, list[str]]]:
    """Return documented feature subsets separating Ampli1, No-Ampli1, and Receiver-only."""
    ampli1_cols = {"Ampli1_InputPower", "Ampli1_InputPower_missing_ind", "Ampli1_OutputPower", "Ampli1_OutputPower_missing_ind"}
    receiver_cols = [c for c in feature_cols if c.startswith("SPO1_") or c.startswith("SPO2_")]
    return {
        "all_devices": {
            "features": list(feature_cols),
            "values": list(value_cols),
            "description": "All 6 devices (Ampli1-4 powers + SPO1-2 BER/OSNR + 12 missing indicators = 24)",
        },
        "no_ampli1": {
            "features": [c for c in feature_cols if c not in ampli1_cols],
            "values": [c for c in value_cols if not c.startswith("Ampli1_")],
            "description": "Ampli2-4 + SPO1-2 without Ampli1 (20 features)",
        },
        "receiver_only": {
            "features": receiver_cols,
            "values": [c for c in value_cols if c.startswith("SPO1_") or c.startswith("SPO2_")],
            "description": "SPO1 & SPO2 BER/OSNR + 4 missing indicators only (8 features)",
        },
        "ampli1_only": {
            "features": ["Ampli1_InputPower"],
            "values": ["Ampli1_InputPower"],
            "description": "Single feature: Ampli1_InputPower",
        },
    }


def candidate_episodes(df_raw: pd.DataFrame, max_gap_s: float = 7.0) -> pd.DataFrame:
    """Group explicit failure timestamps into candidate episodes."""
    failure_times = np.sort(df_raw.loc[df_raw["Failure"].eq(1), "Timestamp"].unique())
    points = pd.DataFrame({"Timestamp": failure_times})
    points["episode_id"] = points["Timestamp"].diff().gt(max_gap_s).cumsum() + 1
    ep = (
        points.groupby("episode_id")
        .agg(
            first_observed_failure=("Timestamp", "min"),
            last_observed_failure=("Timestamp", "max"),
            unique_failure_times=("Timestamp", "size"),
        )
        .reset_index()
    )
    ep["observed_span_s"] = ep["last_observed_failure"] - ep["first_observed_failure"]
    ep["start_to_start_s"] = ep["first_observed_failure"].diff()
    return ep


def build_split(
    aligned: pd.DataFrame,
    episodes_candidate: pd.DataFrame,
    context_s: int = 10,
    partition_gap_s: int = 30,
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
) -> tuple[pd.Series, pd.DataFrame]:
    """Chronological split keeping complete episodes + context_s inside one partition.

    Uses fixed-point exclusion so that if an excluded episode's context overlaps a
    neighboring episode's context (which happens when context_s > episode spacing),
    the neighboring episode is also cleanly excluded rather than partially clipped.
    """
    ep = (
        episodes_candidate.sort_values("first_observed_failure")
        .reset_index(drop=True)
        .copy()
    )
    n = len(ep)
    t_idx = int(n * train_fraction)
    v_idx = int(n * (train_fraction + val_fraction))

    def boundary_between(idx: int) -> int:
        return int((ep.loc[idx - 1, "last_observed_failure"] + ep.loc[idx, "first_observed_failure"]) // 2)

    b1 = boundary_between(t_idx)
    b2 = boundary_between(v_idx)
    half_gap = partition_gap_s // 2
    first_t = int(aligned["Timestamp"].min())
    last_t = int(aligned["Timestamp"].max())

    ranges = {
        "train": (first_t, b1 - half_gap - 1),
        "validation": (b1 + half_gap, b2 - half_gap - 1),
        "test": (b2 + half_gap, last_t),
    }

    ts = aligned["Timestamp"]
    split = pd.Series("excluded", index=aligned.index, dtype="object")
    for sname, (s0, s1) in ranges.items():
        split.loc[ts.between(s0, s1)] = sname

    ep["context_start"] = (ep["first_observed_failure"] - context_s).clip(lower=first_t)
    ep["context_end"] = (ep["last_observed_failure"] + context_s).clip(upper=last_t)
    ep["context_truncated_at_recording_edge"] = (
        (ep["first_observed_failure"] - context_s < first_t)
        | (ep["last_observed_failure"] + context_s > last_t)
    )
    ep["split"] = "excluded"
    for sname, (s0, s1) in ranges.items():
        fits = ep["context_start"].ge(s0) & ep["context_end"].le(s1)
        ep.loc[fits, "split"] = sname

    # Mask only the core 10s-padded failure interval of excluded episodes so long contexts (e.g. 60s > 51s spacing)
    # do not cascade-exclude an entire partition.
    core_pad = min(context_s, 10)
    for r in ep.loc[ep["split"].eq("excluded")].itertuples():
        c0 = max(first_t, int(r.first_observed_failure - core_pad))
        c1 = min(last_t, int(r.last_observed_failure + core_pad))
        split.loc[ts.between(c0, c1)] = "excluded"

    for r in ep.loc[ep["split"].ne("excluded")].itertuples():
        c0 = max(first_t, int(r.first_observed_failure - core_pad))
        c1 = min(last_t, int(r.last_observed_failure + core_pad))
        if not split.loc[ts.between(c0, c1)].eq(r.split).all():
            ep.loc[ep["episode_id"] == r.episode_id, "split"] = "excluded"
            split.loc[ts.between(c0, c1)] = "excluded"

    for r in ep.loc[ep["split"].ne("excluded")].itertuples():
        c0 = max(first_t, int(r.first_observed_failure - core_pad))
        c1 = min(last_t, int(r.last_observed_failure + core_pad))
        assert split.loc[ts.between(c0, c1)].eq(r.split).all(), (
            f"episode {r.episode_id} crosses a partition boundary"
        )
    ep.attrs["ranges"] = ranges
    return split, ep


def select_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Select probability threshold maximizing F1 on validation labels."""
    candidates = np.unique(np.round(np.quantile(y_prob, np.linspace(0.01, 0.999, 250)), 6))
    best_thr, best_f1 = 0.5, -1.0
    for thr in candidates:
        pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr, best_f1


def prepare_temporal_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    value_cols: Sequence[str],
    window_points: int,
    split_name: str,
) -> dict[str, object]:
    """Create causal temporal windows strictly within a single split without gaps or total outages."""
    positions = np.flatnonzero(df["split"].eq(split_name).to_numpy()) # Get indices of rows belonging to the specified split
    times = df["Timestamp"].to_numpy()
    has_evidence = df[list(value_cols)].notna().any(axis=1).to_numpy()
    vals = df[list(feature_cols)].to_numpy(dtype=np.float32)

    windows = []
    endpoints = []
    for j in range(window_points - 1, len(positions)):
        wpos = positions[j - window_points + 1 : j + 1]
        if window_points > 1:
            if not np.all(np.diff(wpos) == 1):
                continue
            if not np.all(np.diff(times[wpos]) == 1):
                continue
        if not has_evidence[wpos].all():
            continue
        windows.append(vals[wpos])
        endpoints.append(wpos[-1])

    assert windows, f"No valid windows in {split_name} for W={window_points}"
    X = np.stack(windows)
    metadata = df.iloc[endpoints][["Timestamp", "split", "target"]].reset_index(drop=True)
    known = metadata["target"].notna().to_numpy()
    X_labeled = X[known]
    y_labeled = metadata.loc[known, "target"].astype(int).to_numpy()
    return {
        "X_all": X,
        "metadata": metadata,
        "X_labeled": X_labeled,
        "y_labeled": y_labeled,
    }


def evaluate_predictions(
    timeline_df: pd.DataFrame,
    events_df: pd.DataFrame,
    model_name: str,
    setting_name: str,
    pred_col: str = "prediction",
) -> dict[str, object]:
    """Compute second-level, coverage, time-in-alarm, and event-level metrics."""
    tl = timeline_df.sort_values("Timestamp").reset_index(drop=True)
    pred_avail = tl[pred_col].notna()
    known = tl["target"].notna()
    scored = pred_avail & known

    y_true = tl.loc[scored, "target"].astype(int).to_numpy()
    y_pred = tl.loc[scored, pred_col].astype(int).to_numpy()

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    known_total = int(known.sum())
    n_scored = int(scored.sum())
    pred_coverage_known = float(n_scored / known_total) if known_total > 0 else np.nan
    timeline_coverage = float(pred_avail.mean())

    pos = tl[pred_col].eq(1).fillna(False)
    prev_pos = pos.shift(1, fill_value=False)
    consec = tl["Timestamp"].diff().eq(1)
    alarm_start = pos & ~(prev_pos & consec)

    normal = tl["target"].eq(0).fillna(False)
    unknown = tl["target"].isna()

    fa_starts = int((alarm_start & normal).sum())
    unk_starts = int((alarm_start & unknown).sum())
    normal_eval_s = int((normal & pred_avail).sum())
    normal_hours = normal_eval_s / 3600.0
    fa_per_hour = float(fa_starts / normal_hours) if normal_hours > 0 else np.nan

    time_in_alarm_scored = float(y_pred.mean()) if len(y_pred) > 0 else np.nan
    time_in_alarm_timeline = float((pos & pred_avail).sum() / pred_avail.sum()) if pred_avail.sum() > 0 else np.nan
    time_in_alarm_normal_s = int((pos & normal & pred_avail).sum())
    time_in_alarm_unknown_s = int((pos & unknown & pred_avail).sum())

    ev_detected = []
    ev_delays = []
    for ev in events_df.itertuples():
        win = tl.loc[tl["Timestamp"].between(ev.first_observed_failure, ev.last_observed_failure)]
        dets = win.loc[win[pred_col].eq(1).fillna(False), "Timestamp"]
        if not dets.empty:
            ev_detected.append(True)
            ev_delays.append(float(dets.iloc[0] - ev.first_observed_failure))
        else:
            ev_detected.append(False)

    return {
        "model": model_name,
        "setting": setting_name,
        "scored_known_seconds": n_scored,
        "prediction_coverage": round(pred_coverage_known, 4),
        "timeline_coverage": round(timeline_coverage, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "time_in_alarm_scored_frac": round(time_in_alarm_scored, 4),
        "time_in_alarm_timeline_frac": round(time_in_alarm_timeline, 4),
        "time_in_alarm_normal_s": time_in_alarm_normal_s,
        "time_in_alarm_unknown_s": time_in_alarm_unknown_s,
        "test_events": len(events_df),
        "detected_events": int(sum(ev_detected)),
        "event_recall": round(float(np.mean(ev_detected)) if ev_detected else np.nan, 4),
        "normal_evaluation_hours": round(normal_hours, 4),
        "false_alarm_starts": fa_starts,
        "false_alarms_per_hour": round(fa_per_hour, 4) if not np.isnan(fa_per_hour) else np.nan,
        "alarm_starts_unknown_target": unk_starts,
        "median_observed_delay_s": float(np.median(ev_delays)) if ev_delays else np.nan,
        "max_observed_delay_s": float(np.max(ev_delays)) if ev_delays else np.nan,
    }


def export_predictions(df: pd.DataFrame, model_slug: str, split_name: str = "test") -> Path:
    """Save timestamped prediction export CSV to outputs/predictions/."""
    ensure_dirs()
    out = df.copy()
    out["exported_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = PRED_DIR / f"{model_slug}_{split_name}.csv"
    out.to_csv(path, index=False)
    return path


def scale_split_features(df_source: pd.DataFrame, fcols: Sequence[str]) -> pd.DataFrame:
    """Fit median imputer and StandardScaler on train rows only, then transform all rows."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    tr_mask = df_source["split"].eq("train")
    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    df_proc = df_source.copy()
    scl.fit(imp.fit_transform(df_proc.loc[tr_mask, list(fcols)]))
    df_proc[list(fcols)] = scl.transform(imp.transform(df_proc[list(fcols)]))
    return df_proc


def fit_predict_binary_torch(
    model,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te_all: np.ndarray,
    batch_size: int = 128,
    max_epochs: int = 40,
    patience: int = 6,
):
    """Train a binary PyTorch model with class weighting + early stopping, then predict val/test."""
    import torch
    import torch.nn as nn
    from sklearn.utils.class_weight import compute_class_weight
    from torch.utils.data import DataLoader, TensorDataset

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cw[1] / cw[0]], dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )
    X_va_t, y_va_t = torch.from_numpy(X_va), torch.from_numpy(y_va.astype(np.float32))

    best_loss, best_state, wait = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(crit(model(X_va_t), y_va_t))
        if vloss < best_loss - 1e-4:
            best_loss = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(X_va_t)).numpy()
        test_prob = torch.sigmoid(model(torch.from_numpy(X_te_all))).numpy()
    thr, val_f1 = select_threshold_f1(y_va, val_prob)
    return test_prob, thr, val_f1


def map_window_predictions_to_timeline(
    df_source: pd.DataFrame, metadata: pd.DataFrame, probs: np.ndarray, thr: float, split_name: str = "test"
) -> pd.DataFrame:
    """Align window-level probabilities back onto the full split timeline."""
    tl = df_source.loc[df_source["split"].eq(split_name), ["Timestamp", "split", "target"]].copy()
    pred_df = metadata[["Timestamp"]].copy()
    pred_df["prob"] = probs
    pred_df["prediction"] = (probs >= thr).astype(int)
    return tl.merge(pred_df, on="Timestamp", how="left")
