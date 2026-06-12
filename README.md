# Machine Learning for Band Gap Prediction in Alloy Halide Perovskites

Predicting the electronic band gap and thermodynamic stability of B-site alloy halide perovskites $AA'(B_{1-x}B'_{x})X_{6}$ using features derived from DFT density-of-states calculations.

---

## Motivation

Halide double perovskites are a promising class of lead-free semiconductors for photovoltaic and optoelectronic applications. Alloying on the B-site ($AA'BB'X_{6}$ → $AA'B_{1-x}B'_{x}X_{6}$) enables band gap tuning, but the relationship between composition and electronic structure is non-linear and chemistry-dependent. DFT calculations are accurate but expensive at screening scale.

This project builds an ML pipeline to predict band gap and stability descriptors from composition and DOS-derived features, enabling rapid screening across a large alloy space without additional DFT calculations.

---

## Dataset

~6500 DFT calculations (PBE functional, VASP) across a systematic alloy space, structure optimization with MLIP (DOI: https://doi.org/10.1103/n6dj-hlzm):
- **Alloys:** $AA'B_{1-x}B'_{x}X_{6}$ at x = 0.25, 0.50, 0.75 for all B-site pairs and halides (F, Cl, Br, I)
- **Pure phases:** $AA'B_{2}X_{6}$ and $AA'B'_{2}X_{6}$ endpoints for each alloy series
- **Targets:** band gap Eg (eV), formation enthalpy dHf, decomposition enthalpy dHd
- **Features:** spin-resolved DOS band edge energies and orbital character (s/p/d) per element

Data files are too large for GitHub. The extraction scripts and assembled DataFrames are tracked; raw DFT outputs are local only.

---

## Pipeline

```
1. Data extraction        extract_dos_features.py  (local, not in repo)
   - Raw DFT DOS JSONs → band edge energies, orbital character → dos_features.parquet

2. DataFrame assembly     build_dataframes.py  (local, not in repo)
   - DFT result JSONs → alloy_df.parquet, pure_phases_df.parquet

3. EDA                    2_exploratory_analysis.ipynb
   - Pipeline verification (DOS-derived Eg vs DFT Eg parity)
   - Target distribution (Eg histogram, metallic fraction)
   - Bowing parameter analysis (linear interpolation vs fitted b)
   - Stability EDA (Goldschmidt t and Bartel τ vs dHd)
   - Feature-target correlation (Pearson r, spin channel redundancy check)

4. Feature engineering    3_feature_engineering.ipynb
   - DOS features: band edge orbital character (orb_idx, frac) + Fermi energy per element (225 features)
   - Composition features: full-formula MAGPIE preset, 22 properties × 6 statistics = 132 features
   - Site features: per-site MagpieData lookup for A1, A2, B1, B2, X (7 properties × 5 sites + x_frac = 36 features)
   - Endpoint DOS features (Use Case B): pure phase endpoint DOS (x=0 and x=100) renamed to
     site-specific columns — 4 sites × 8 orbital suffixes × 2 endpoints + 4 band edge energies
     × 2 endpoints = 72 features. Enables screening without alloy-specific DFT.

   Produces 4 feature matrices (see below).

5. Baseline modeling      4_baseline_modeling.ipynb
   - Alloy-group-based train/test split: held-out chemistries (not just held-out mixing fractions)
   - Ablation across 4 feature sets × 2 scopes × 2 approaches (see below)
   - Feature importance analysis for chemical interpretation
```

---

## Use Cases

| Use Case | DFT Required | Feature Set | Purpose |
|---|---|---|---|
| **A** | None | Composition (MAGPIE) / Site (MagpieData) | High-throughput screening, no DFT |
| **B** | Pure phase endpoints only | Endpoint DOS (x=0, x=100 pure phases) | Practical screening with minimal DFT |
| **C** | Full alloy DFT | Alloy DOS | Upper bound — quantifies non-interpolable alloy electronic structure |

The performance gap C → B → A quantifies how much alloy-specific electronic structure cannot be recovered from endpoint properties alone.

---

## Feature Sets

| Matrix | Rows | Feature cols | Description |
|---|---|---|---|
| `composition_feature_matrix` | 6519 | 133 | 132 MAGPIE stats + x_frac |
| `site_feature_matrix` | 6519 | 36 | 7 per-site MagpieData props × 5 sites + x_frac |
| `dos_feature_matrix` | 6519 | 226 | 225 alloy DOS features + x_frac (Use Case C) |
| `dos_endpoint_feature_matrix` | 6519 | 82 | 72 endpoint DOS features + 10 metadata (Use Case B) |

**Dataset:** 6519 rows (5992 alloys + 527 pure phases), ~31% metallic (Eg = 0)

---

## Key EDA Findings

- **DOS-derived band gap** agrees well with DFT values across the full gap range (0–7 eV), validating the extraction pipeline
- **Target distribution:** Eg is skewed towards 0 eV with ~31% metallic systems (Eg=0); F/Cl compositions span higher gaps than Br/I
- **Bowing analysis:** a single bowing parameter b improves over linear interpolation but leaves substantial scatter, particularly for TM-containing B-site alloys, consistent with d-orbital character not following simple interpolation. This motivates ML to learn complex alloy band gap behavior
- **Tolerance factors:** Bartel τ shows a weak positive correlation with dHd (consistent with Bartel et al. 2019); Goldschmidt t shows no correlation, as expected for halide perovskites
- **Feature-target correlation:** Pearson r indicates Ag VBM contribution is the strongest single predictor of Eg (r ≈ 0.85). TM contributions show near-zero r due to step-like gap collapse rather than linear scaling, motivating tree-based models
- **Spin channels:** Up and down spin channel features are uncorrelated for most TM elements (r < 0.2) due to magnetic splitting; both channels are retained as independent features

---

## Baseline Modeling Design

**Train/test split:** Alloy-group-based — unique formulas split at ~80/20 so all compositions of an alloy stay together. Tests generalization to unseen chemistries, not interpolation within known alloys.

**Experiments (3 axes):**
- **Feature sets:** Composition, Site, DOS alloy, DOS endpoint
- **Scopes:** alloys-only, combined (alloys + pure phases)
- **Approaches:** single XGBoost regression, two-stage (XGBClassifier → XGBRegressor for metal/semiconductor)

Linear models (Ridge, Lasso) are run on the Composition feature set only, as DOS and Site features contain high rates of structural NaN that are informative for tree models but unsuitable for median imputation.

---

## Repository Structure

```
ML_alloy_halide_perovskites/
├── README.md
├── DECISIONS.md                          # Design decisions and EDA findings log
├── 2_exploratory_analysis.ipynb          # EDA notebook
├── 3_feature_engineering.ipynb           # Feature engineering notebook
├── 4_baseline_modeling.ipynb             # Baseline modeling notebook
├── alloy_df.parquet/.csv                 # Alloy compositions with DFT targets
├── pure_phases_df.parquet/.csv           # Pure phase compositions with DFT targets
├── dos_features.parquet                  # DOS-derived band edge features (raw)
├── composition_feature_matrix.parquet    # 6519×143  Use Case A
├── site_feature_matrix.parquet           # 6519×46   Use Case A (site-resolved)
├── dos_feature_matrix.parquet            # 6519×236  Use Case C (alloy DOS)
├── dos_endpoint_feature_matrix.parquet   # 6519×82   Use Case B (endpoint DOS)
├── full_feature_matrix.parquet           # 6519×403  All features combined
├── alloy_space_overall_dict_EMPTY.json   # Alloy series groupings
└── src/
    └── ml_alloy_halide_perovskites/
        └── helper_functions.py           # Shared utilities (WIP)
```

---

## Reproducing the Results

**Environment:**
```
Python >=3.8
pymatgen
matminer
pandas, numpy, scipy, matplotlib
scikit-learn
xgboost
```

**Order of execution:**
1. `build_dataframes.py` — generates `alloy_df.parquet`, `pure_phases_df.parquet` (requires local DFT data)
2. `extract_dos_features.py` — generates `dos_features.parquet` (requires local DFT data)
3. `2_exploratory_analysis.ipynb` — EDA
4. `3_feature_engineering.ipynb` — assembles all 4 feature matrices
5. `4_baseline_modeling.ipynb` — baseline experiments and feature importance

---

## References

- Bartel et al. (2019): New tolerance factor to predict the stability of perovskite oxides and halides. *Science Advances* https://doi.org/10.1126/sciadv.aav0693
- Shannon (1976): Revised effective ionic radii and systematic studies of interatomic distances in halides and chalcogenides. *Acta Crystallographica* https://doi.org/10.1107/S0567739476001551
- Ong et al. (2013): Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis. *Computational Materials Science* https://doi.org/10.1016/j.commatsci.2012.10.028
- Ward et al. (2018): Matminer: An open source toolkit for materials data mining. *Computational Materials Science* https://doi.org/10.1016/j.commatsci.2018.05.018
