# Schedule-Robust Optical Failure Detection 
**Dataset:** [Optical Failure Dataset](https://github.com/Network-And-Services/optical-failure-dataset) (`HardFailure_dataset.csv`)

---

## 1. Submission Folder Structure

```text

├── README.md                                  # Technical documentation & execution guide
├── requirements.txt                           # Python package dependencies
├── Dataset/
│   ├── DATASET.docx                           # Dataset source, description & column specification
│   ├── HardFailure_dataset.csv                # Raw telemetry (65,733 rows, ~9.93 hours)
│   ├── aligned_detection_table.csv            # Preprocessed 1s causal table (CSV for Excel)
│   └── aligned_detection_table.parquet        # Preprocessed 1s causal table (Parquet)
│
├── Source_Code/
│   ├── common.py                              # Shared causal alignment, log10(BER), splits, windowing & metrics
│   ├── build_nb.py                            # Converts nb_src/*.py -> .ipynb and runs full "Run All"
│   ├── nb_src/                                # Clean Python sources for notebooks 00–04
│   ├── 00_environment_and_data_hash.ipynb     # Package versions & dataset SHA-256 hash
│   ├── 01_data_audit_and_alignment.ipynb      # Data Preprocessing, EDA plots & Stage-2 split
│   ├── 02_target_policy_and_timelines.ipynb   # Annotation-age sensitivity (3–7s) & timelines
│   ├── 03_matched_baselines.ipynb             # Matched-input baselines & log10(BER) comparison
│   └── 04_history_length_experiment.ipynb     # Exploratory history-length (W=1,11,61) & schedule diagnostic
│
└── Experimental_Results/
    ├── environment_and_hash.json              # Exact Python/package versions & SHA-256 digest
    ├── episodes_candidate.csv                 # 131 candidate failure episodes
    ├── split_manifest.csv                     # Stage-2 split manifest (10s context, 30s gap)
    ├── split_manifest_history_experiment.csv  # Exploratory split manifest (60s context, 120s gap)
    ├── annotation_age_sensitivity.csv         # Target policy sensitivity (3, 4, 5, 7 s)
    ├── representative_timelines.csv           # Onset/recovery/unknown timelines
    ├── matched_baselines_results.csv          # Stage-2 matched baseline metrics
    ├── history_experiment_summary.csv         # Exploratory W=1/11/61 summary (3 seeds)
    ├── schedule_only_diagnostic_results.csv   # Non-deployable schedule-only diagnostic
    ├── figures/                               # All PNG visualizations
    └── predictions/                           # Consolidated timestamped test predictions
```

---

## 2. Dependencies

Requires **Python 3.12+** (executed on **Python 3.14.4**). Install dependencies with:

```bash
pip install -r requirements.txt
```

Key packages: `numpy==2.4.4`, `pandas==3.0.2`, `scikit-learn==1.8.0`, `torch==2.15.0`, `matplotlib==3.10.8`, `pyarrow==24.0.0`, `nbformat==5.10.4`, `nbclient==0.11.0`.

---

## 3. Execution Instructions

All notebooks in `Source_Code/` are already executed end-to-end (**Run All**) with saved outputs, inline plots, and execution counters.

To re-run from scratch:
1. Copy `Dataset/HardFailure_dataset.csv` into `Source_Code/` (so `common.py` locates it in the working directory).
2. From inside `Source_Code/`, run all notebooks in order (`00` → `04`):
   ```bash
   python build_nb.py
   ```
   Or run a single notebook:
   ```bash
   python build_nb.py 03_matched_baselines
   ```
3. Generated tables, figures, and timestamped predictions are saved into `Source_Code/outputs/`.
