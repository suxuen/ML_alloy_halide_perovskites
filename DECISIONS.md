# ML Pipeline — Decision Log

## Data

**Input files**
- `dos_features.parquet` — DOS-derived band edge features for all formulas (alloys + pure phases)
- `alloy_df.csv` / `alloy_df.parquet` — alloy compositions with Eg, dHf, dHd from DFT
- `pure_phases_df.csv` — pure phase compositions

**File format: parquet as primary, CSV as secondary**
- Parquet preserves dtypes on reload (`is_alloy` as bool, `orb_idx` as int, NaN stays NaN)
- CSV kept alongside for human inspection in Excel

---

## Features

**Band edge energies are absolute (unshifted)**
- Raw DOS JSONs store energy as E − E_F (Fermi level zeroed)
- `Efermi` added back so VBM/CBM are in absolute eV, consistent with DFT reference

**Orbital character encoding**
- `_orb_idx`: integer — `0=s, 1=p, 2=d`; `-1` = element present but below contribution threshold; `NaN` = band edge absent for this spin channel
- `-1` chosen over `NaN` to prevent `fillna(0)` silently converting non-contributors into s-orbital contributors during preprocessing

**Pure phase identifiers on alloy rows**
- Each alloy row carries `formula_0`, `formula_100` (end-member formula strings) and `x_frac`
- Enables joining pure phase DOS features in EDA/ML without a separate lookup

---

## Target variable

**Band gap: minimum across all spin combinations**
- Evaluated across up→up, down→down, up→down, down→up channel pairs
- Correct for magnetic/half-metallic systems where gap may be between spin channels

**Metal/semiconductor boundary**
- ~485 of ~6500 materials have Eg = 0 (metallic)
- ~0.1 eV discrepancy near Eg = 0 acceptable — arises from DOS grid discretization
- Practical implication: model may struggle at metal/semiconductor boundary; expected and acceptable for optoelectronics target (Eg > 1 eV)
- Decision deferred: train on all materials, or classify metal/semiconductor first, then regress

---

## Data mismatch

- 163 formulas in `alloy_df`/`pure_phases_df` with no DOS file — no DOS features available for these; cannot use DOS-feature model for prediction
- 385 formulas in `dos_features.parquet` not in `alloy_df`/`pure_phases_df` — have DOS features but no DFT Eg label; these are inference targets after training
- Training set is the ~6100-formula intersection of both sources

---

## Bug fixes

**`parse_formula` halide regex** (`build_dataframes.py`)
- Original: `re.search(r'(Cl|Br|I|F)')` matched `F` inside `Fe` before the actual halide — e.g. `RbCsFe2I6` → `X='F'` instead of `X='I'`
- Fix: `re.findall(r'[A-Z][a-z]?', formula)` on the whole formula string; take the last element as X
- Requires re-running `build_dataframes.py` to regenerate `alloy_df.parquet` / `pure_phases_df.parquet`

---

## EDA findings

**Parity plot (DOS-derived gap vs DFT gap)**
- Strong agreement across full gap range
- Small scatter at Eg ≈ 0 (~0.1 eV) — see metal/semiconductor boundary note above

**Target variable distribution**
- Eg distribution is right-skewed with a spike at Eg=0 (metallic fraction)
- Halide-split distribution shows F/Cl compositions span higher Eg than Br/I

**Bowing parameter analysis**
- Fitted per alloy series using closed-form 1-parameter least squares: `b = dot(w, residual) / dot(w, w)` where `w = x*(1-x)`
- Bowing fit significantly improves over linear interpolation but leaves substantial scatter — motivates ML
- Halide dependence weak (B-site alloying, not X-site); ranking F > Cl > Br > I attributed to TM stabilisation by electronegative halides
- TM B-site compositions show large, scattered b (bimodal distribution) — single bowing parameter insufficient; motivates DOS orbital features in ML
- Negative b (upward bowing) likely a model artefact from the 3-point quadratic fit, not a physical effect

**Stability EDA (tolerance factors vs dHd)**
- No correlation between Goldschmidt t and dHd — expected, t was derived for oxide perovskites
- Weak positive correlation between Bartel τ and dHd — consistent with Bartel et al. 2019
- Neither t nor τ gives tight dHd prediction; motivates ML with richer electronic/elemental features

**Shannon radii for tolerance factors**
- Used `Species.get_shannon_radius(cn, spin, radius_type)` with CN=XII for A-site (+1), CN=VI for B-site (+2) and X-site (-1), High Spin default
- ~2% of formulas skipped due to missing radii or missing dHd
- Supplementary ML-predicted radii added manually for elements absent from pymatgen tables
- Alloy B-site radius: weighted geometric mean `r_B = r_B1^(1-x) * r_B2^x` — reduces to `√(r_B1·r_B2)` at x=0.5, consistent with Bartel et al.

---

## Pending decisions

- Metal/semiconductor split vs single regression model
- Which spin channel features to use (both, min-gap channel only, average)
- Whether to include `x_frac` as a feature (direct physical meaning: B-site mixing ratio)
- Matminer elemental features to add at step 3
- Feature-target correlation: DOS features to be correlated with Eg before step 3; full heatmap after matminer feature engineering
