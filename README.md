Motor Insurance Pure Premium Pricing —
GLM FrameworkMotor Insurance Pricing Framework built using GLMs on the French Motor TPL (freMTPL) dataset. Implements a Negative Binomial model for claim frequency and a Gamma model for claim severity to estimate pure premium. Includes diagnostics, coefficient analysis, visualizations, and portfolio-level pricing insights.

> **Actuarial pricing model using Negative Binomial (frequency) × Gamma (severity) GLMs**  
> Built on the French Motor TPL dataset (freMTPL) from Kaggle

---

📌 Project Overview

This project implements a complete **General Insurance Pricing Framework** using  
Generalised Linear Models (GLMs) — the industry-standard actuarial approach to  
calculating motor insurance premiums.

```
Pure Premium = E[Frequency] × E[Severity]
             = NB2 GLM     × Gamma GLM
```

| Model | Distribution | Target Variable | Why |
|---|---|---|---|
| Frequency | Negative Binomial (NB2) | Claim count per year | Handles overdispersion (VMR > 1) |
| Severity | Gamma | Average cost per claim | Positive, right-skewed amounts |
| Pure Premium | — | Frequency × Severity | Expected annual claim cost |

---

📊 Dataset

**Source:** [freMTPL — Kaggle](https://www.kaggle.com/datasets/karansarpal/fremtpl-french-motor-tpl-insurance-claims)

| File | Rows | Description |
|---|---|---|
| `freMTPL2freq.csv` | ~678,000 | One row per policy — claim counts + features |
| `freMTPL2sev.csv` | ~26,000 | One row per claim — individual claim amounts |

**Columns used:**

| Column | Type | Description |
|---|---|---|
| `PolicyID` | ID | Join key between both files |
| `ClaimNb` | Target (freq) | Number of claims in the period |
| `ClaimAmount` | Target (sev) | Individual claim cost (€) |
| `Exposure` | Offset | Fraction of year policy was active (0–1) |
| `DriverAge` | Feature | Age of the driver |
| `CarAge` | Feature | Age of the vehicle |
| `Power` | Feature | Vehicle power band |
| `Gas` | Feature | Fuel type (Regular / Diesel) |
| `Region` | Feature | French administrative region |
| `Density` | Feature | Population density of policyholder's commune |
| `Brand` | Feature | Vehicle brand |

---

## 🔢 Model Details

### Frequency Model — Negative Binomial GLM

```
log(E[ClaimNb]) = β₀ + β₁·AgeBand + β₂·CarAgeBand 
                + β₃·Gas + β₄·log(Density) + β₅·Power
                + log(Exposure)    ← offset, coefficient fixed at 1
```

**Why Negative Binomial?**  
Claim counts are **overdispersed** — variance exceeds the mean (VMR > 1).  
Poisson assumes Var = Mean exactly, which is violated in real insurance data.  
NB2 adds a dispersion parameter α where `Var(Y) = μ + α·μ²`.

### Severity Model — Gamma GLM

```
log(E[AvgSeverity]) = β₀ + β₁·AgeBand + β₂·CarAgeBand
                    + β₃·Gas + β₄·log(Density) + β₅·Power
```

**Why Gamma?**  
Claim costs are always positive, right-skewed, and variance grows with  
the mean — all properties of the Gamma distribution.

### Pure Premium

```python
PurePremium        = PredictedFrequency × PredictedSeverity
LoadedPremium      = PurePremium × 1.30    # 30% expense + profit loading
```

---

## 📈 Results

| Metric | Value |
|---|---|
| NB2 Frequency AIC | 134,881.82 |
| Gamma Severity AIC | 312,869.29 |
| Portfolio Mean Pure Premium | €1,367.82 / yr |
| Loaded Premium (30%) | €1,778.17 / yr |

### Sample Policyholder Quotes

| Profile | Pred. Freq | Pred. Severity | Pure Premium | Loaded |
|---|---|---|---|---|
| Young driver (22, new car) | High | High | High | High |
| Middle-aged (42) | Medium | Medium | Medium | Medium |
| Experienced (58) | Low | Medium | Low | Low |
| Elderly (70, old car) | Low | Low | Low | Low |

---

## 📉 Visualisations

The script produces an 11-panel chart (`pure_premium_glm.png`) including:

1. Actual vs Predicted frequency by age band
2. Actual vs Predicted severity by age band
3. Pure premium by age band
4. NB2 coefficients with 95% confidence intervals
5. Gamma coefficients with 95% confidence intervals
6. Pure premium distribution across portfolio
7. NB2 Pearson residuals (model diagnostics)
8. Gamma Pearson residuals (model diagnostics)
9. Pure premium by fuel type
10. Pure premium heatmap — Driver Age × Car Age
11. Sample policyholder quote table

---

## 🧮 Mathematical Background

### Log-Link Function
Both models use the **log link**: `log(μ) = Xβ → μ = exp(Xβ)`  
This ensures predictions are always positive and effects are multiplicative.

### Exposure Offset
```
log(E[ClaimNb]) = Xβ + log(Exposure)
```
Equivalent to modelling the **claim rate** (per year) rather than raw count.  
A 6-month policy is automatically expected to have half the claims.

### Overdispersion Test
```
VMR = Var(ClaimNb) / Mean(ClaimNb)
VMR = 1   → Poisson adequate
VMR > 1   → Negative Binomial required
```

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥1.5 | Data loading and manipulation |
| `numpy` | ≥1.23 | Numerical operations |
| `statsmodels` | ≥0.14 | GLM fitting (NB2 + Gamma) |
| `matplotlib` | ≥3.6 | Visualisations |
| `seaborn` | ≥0.12 | Heatmaps |
| `scikit-learn` | ≥1.1 | Utilities |

---

## 📚 References

- Frees, Derrig & Meyers — *Predictive Modeling Applications in Actuarial Science* (2014)
- McCullagh & Nelder — *Generalised Linear Models* (1989)
- freMTPL2 dataset — R `CASdatasets` package

---

## 👤 Author

Built as an actuarial data science portfolio project.  
Feel free to fork, star ⭐, or open an issue.

---
