
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata

import common as C

C.ensure_dirs()

pkgs = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "pyarrow",
    "torch",
    "nbformat",
    "nbclient",
    "ipykernel",
]
versions = {}
for p in pkgs:
    try:
        versions[p] = metadata.version(p)
    except metadata.PackageNotFoundError:
        versions[p] = "not-installed"

sha256 = C.dataset_sha256()
df_raw = C.load_raw()

env_info = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "python_version": sys.version,
    "platform": platform.platform(),
    "dataset_file": C.DATA_PATH.name,
    "dataset_bytes": C.DATA_PATH.stat().st_size,
    "dataset_sha256": sha256,
    "raw_rows": int(len(df_raw)),
    "raw_cols": int(df_raw.shape[1]),
    "timestamp_min": int(df_raw["Timestamp"].min()),
    "timestamp_max": int(df_raw["Timestamp"].max()),
    "duration_hours": round((df_raw["Timestamp"].max() - df_raw["Timestamp"].min()) / 3600.0, 4),
    "packages": versions,
}

with (C.OUT / "environment_and_hash.json").open("w", encoding="utf-8") as f:
    json.dump(env_info, f, indent=2)

print(json.dumps(env_info, indent=2))
