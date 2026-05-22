# Machine Learning for Band Gap Prediction in Alloy Halide Perovskites

Predicting the electronic band gap and thermodynamic stability of B-site alloy halide perovskites $AA'(B_{1-x}B'_{x})X_{6}$ using features derived from DFT density-of-states calculations.

---

## Motivation

Halide double perovskites are a promising class of lead-free semiconductors for photovoltaic and optoelectronic applications. Alloying on the B-site ($AA'BB'X_{6}$ -> $AA'B_{1-x}B'_{x}X_{6}$) enables band gap tuning, but the relationship between composition and electronic structure is non-linear and chemistry-dependent. DFT calculations are accurate but expensive at screening scale.

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

*Check 4. and make changes, 4 possible combinations of DOS information, composition ElementProperty, individual ElementProperty*

```
1. Data extraction        extract_dos_features.py
   - Raw DFT DOS JSONs -> band edge energies, orbital character -> dos_features.parquet

2. DataFrame assembly     build_dataframes.py  (local, not in repo)
   - DFT result JSONs -> alloy_df.parquet, pure_phases_df.parquet

3. EDA                    2_exploratory_analysis.ipynb
   - Pipeline verification (DOS-derived Eg vs DFT Eg parity)
   - Target distribution (Eg histogram, metallic fraction)
   - Bowing parameter analysis (linear interpolation vs fitted b)
   - Stability EDA (Goldschmidt t and Bartel τ vs dHd)
   - Feature-target correlation (Pearson r, spin channel redundancy check)

4. Feature engineering    3_feature_engineering.ipynb
   - Option A: full-composition ElementProperty MAGPIE preset (22 properties x 6 stats = 132 features)
   - Option B: per-site MagpieData lookup for B1, B2, X sites (7 properties x 3 sites + x_frac = 22 features)
   - DOS features + Option B site features -> feature_matrix.parquet

5. Modelling              [planned]
   - Use case 1 (full element DOS contribution features): upper bound on model performance
   - Use case 2 (endpoint DOS + composition only): practical screening model with pure phase DFT results only
   - Ablation 1 -> 2 quantifies how much alloy-specific electronic structure is non-interpolable from endpoints
   - Baseline regression -> gradient boosting -> neural network
```

---

## Key EDA Findings

- **DOS-derived band gap** agrees well with DFT values across the full gap range (0–7 eV), validating the extraction pipeline
- **Target distribution:** Eg is skewed towards 0 eV with ~7% metallic systems (Eg=0); F/Cl compositions span higher gaps than Br/I
- **Bowing analysis:** a single bowing parameter b improves over linear interpolation but leaves substantial scatter, particularly for TM-containing B-site alloys. TM containing B-site alloys also produce large, scattered bowing parameters consistent with d-orbital character not following simple interpolation of band gaps. This motivates ML to learn complex band gap behavior in alloys
- **Tolerance factors:** Bartel τ shows a weak positive correlation with dHd (consistent with Bartel et al. 2019); Goldschmidt t shows no correlation, as expected for halide perovskites. Neither descriptor is sufficient alone, motivating ML with richer electronic features
- **Feature-target correlation:** Pearson r indicates Ag VBM contribution is the strongest single predictor of Eg (Pearson r ≈ 0.85). However alloy perovskite band gap behavior cannot be mapped linearly, as TM contributions to band edges show near-zero Pearson r due to step-like gap collapse rather than linear scaling (as shown for the scatter for 'Cr_down_cbm_frac'). This motivates tree-based models instead of feature removal, as band edge character is undoubtedly important for learning band gaps.
- **Spin channels:** Up and down spin channel features are uncorrelated for most TM elements (r < 0.2) due to magnetic splitting, while correlated for nonmagnetic elements. Both channels are retained as independent features

---

## Repository Structure

```
ML_alloy_halide_perovskites/
- README.md
- DECISIONS.md                         # Design decisions and EDA findings log
- 2_exploratory_analysis.ipynb         # EDA notebook
- 3_feature_engineering.ipynb          # Feature engineering notebook
- alloy_df.parquet/.csv                # Alloy compositions with DFT targets
- pure_phases_df.parquet/.csv          # Pure phase compositions with DFT targets
- dos_features.parquet/.csv            # DOS-derived band edge features
- feature_matrix.parquet/.csv          # ML-ready feature matrix (DOS + site elemental features)
- alloy_space_overall_dict_EMPTY.json  # Alloy series groupings
- src/
    - ml_alloy_halide_perovskites/
        - helper_functions.py          # Shared utilities (WIP)
```

---

## Reproducing the Results

**Environment:**
```
Python >=3.8
matminer
pandas
numpy
scipy
matplotlib
```

**Order of execution:**
1. `build_dataframes.py` : generates `alloy_df.parquet`, `pure_phases_df.parquet` (requires local DFT data)
2. `extract_dos_features.py` : generates `dos_features.parquet` (requires local DFT data)
3. `2_exploratory_analysis.ipynb` : EDA, reads from parquet files
4. `3_feature_engineering.ipynb` : Feature engineering — assembles DOS features and matminer elemental descriptors into feature_matrix.parquet

---

## References
*WIP add doi*
- Bartel et al. (2019) : New tolerance factor to predict the stability of perovskite oxides and halides. *Science Advances* https://doi.org/10.1126/sciadv.aav0693
- Shannon (1976) : Revised effective ionic radii and systematic studies of interatomic distances in halides and chalcogenides. *Acta Crystallographica* https://doi.org/10.1107/S0567739476001551
- Ong et al. (2013) : Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis. *Computational Materials Science* https://doi.org/10.1016/j.commatsci.2012.10.028
- Ward et al. (2018) : Matminer: An open source toolkit for materials data mining. *Computational Materials Science* https://doi.org/10.1016/j.commatsci.2018.05.018
