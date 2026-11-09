# Pre-registration v1 (written BEFORE any weather–fire association was computed)
Date: 2026-11-05. Data: panel_dev only (sealed file untouched; sha256 prefix dcd05900e722c397).
Seen before this document: dataset sizes, per-region base rates of fire_any_forest / large_event_active (descriptive only).

## Sample
- Regions with forest signal: TB, DB, BTB, NTB, TN, DNB. DBSH, DBSCL excluded from forest analyses (contrast only).
- Unit: province-day inside the placeholder region season (in_season==1). Drop seasons that start before 2012-01-20 (incomplete fire record).
- Province eligible if >=150 in-season days with fire_any_forest==1 AND >=6 seasons with >=5 such days.
## Outcomes
- Primary: fire_any_forest (>=1 forest, nominal+high-confidence VIIRS hotspot that local day).
- Secondary: large_event_active.
## Validation
- Leave-one-season-out CV within region (held-out season removed for all provinces of the region at once).
- Baseline M0: province x month climatology of the outcome from training seasons (logit, smoothed).
- M1: logistic regression on [logit(M0), log1p(P)] with P = primary variant P_linear5_Ddew (classical Nesterov, 5 mm reset, dew-point deficit; PROVISIONAL pending Gate 0).
- M2 (ceiling): HistGradientBoosting on weather, lag, anomaly, static features (no calendar-human features).
- Metrics: ROC-AUC and PR-AUC per held-out season, averaged; Brier skill score vs M0.
- Inference: bootstrap over held-out SEASONS (blocks), 2000 resamples; Holm correction within each hypothesis family.
## Hypotheses
- H1: M1 beats M0 in AUC in each eligible region (family: 6 regions). One-sided: delta AUC > 0.
- H2: The best of the 26 P variants in a region beats the primary variant by delta AUC > 0 (family: 6 regions). Selection is done on the same dev OOF predictions -> optimistic; confirmed ONLY on sealed seasons later.
- H3: M2 beats M1 in AUC (family: 6 regions): how much skill is left that the rule ignores.
- H4 (human ignition): mean OOF residual (y - M1 prediction) differs between flagged and control days for weekend, tet_win3, giotuo_win1, thanh_minh_win1, holiday_solar (family: 5 flags, pooled over 6 regions). Control = same flag==0 days.
- H5 (exploratory): delta AUC (M1 - M0) for any-hotspot occurrence (n_hot_all>0) in DBSCL (cropland burning) vs forest outcome in forest regions.
- H6 (exploratory): season-level association between mean oni_lag3 and province-standardised number of fire days (per region, Spearman).
## Not allowed after seeing results
Changing the primary P variant, the eligibility thresholds, the families, or the season definition for confirmatory claims. Any deviation is reported as exploratory.

## Amendment 1 (2026-11-09, written after the analyses in notebook 05 had been run)
- The original primary variant P_linear5_Ddew had no recorded source for its choice (linear K and the 5 mm reset were team choices). It is replaced as the primary variant by P_vn_5mm_Ddew: K = 1 and the sum restarts from 0 when daily rain exceeds 5 mm, P = sum of T13 * (T13 - Td13) (Nguyen Van Them, Le Hong Viet, Nguyen Van Quy, 2024, Tap chi Khoa hoc va Cong nghe Lam nghiep 13(6), 67-74, eq. 3 and Section 3.2; the 5 mm threshold with K in {0,1} also appears in Quyet dinh 07/2024/QD-UBND of Kien Giang, whose formula is an image on the page read and whose saturation-deficit form is kept as P_vn_5mm_Dsat).
- This change was made after the association results were seen, so under the rule above it is a deviation. Notebook 05 reports the ranking of all 29 variants (the new primary ranks 8th, AUC 0.687, versus 0.695 for the 3 mm variant), so the choice is not claimed to be data-optimal.
- Also disclosed: the eligibility thresholds and region seasons were set after dataset summaries had been viewed, not only base rates. The season definition is no longer used by the shipped panel; the analysis sample is defined in notebook 03.
