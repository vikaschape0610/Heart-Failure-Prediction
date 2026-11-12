# Summary of changes (report upload & automatic extraction)

## New files
| File | Purpose |
|---|---|
| `report_extractor.py` | Validation + PDF text extraction (pdfplumber) + OCR (Tesseract via stdin/stdout; scanned PDF pages rendered by pypdfium2) + label/value parser + unit conversion + final validation. No Streamlit dependency. |
| `report_ui.py` | Uploader, extraction summary, editable review form, per-field status highlighting, verify/confirm gating. |
| `sample_reports/` | 4 synthetic reports (+ README): text PDF, scanned PDF, JPG, incomplete/ambiguous PDF. All stamped "SYNTHETIC SAMPLE - FOR SOFTWARE TESTING ONLY - NOT A REAL MEDICAL RECORD". |
| `tools/generate_sample_reports.py` | Regenerates the samples and `tests/expected_values.json`. |
| `tests/` | `test_report_extractor.py` (59 pytest tests), `e2e_playwright.py` (14 browser tests), `baseline_reference.json` (predictions + SHAP produced by the **untouched original** code), `expected_values.json`. |
| `requirements-dev.txt`, `CHANGES.md`, `TEST_REPORT.md` | Dev deps, this summary, the test report. |

## Modified files
* `app.py` - Tab 1 only: added an "Input method" radio. The original manual form is unchanged but now sits inside `if input_mode is manual:` (so it is re-indented one level). In upload mode the confirmed values feed the **same** result / gauge / SHAP code. Verified with `diff -w`: no original line removed or altered, only the radio + upload branch added.
* `requirements.txt` - added `pdfplumber`, `pypdfium2`, `Pillow`; `streamlit>=1.40` (empty-by-default inputs / placeholders). Tesseract is a system program (see README).
* `README.md` - new section on upload / OCR / privacy / tests; file list.

## Byte-identical to the original
`model.py` (model, GA, SMOTE, scaler, `predict_patient`, SHAP), `heart_failure_prediction.py` (CLI), `heart299.csv`, `XGB-GA.ipynb`, `Base paper.pdf`, figures, `model_xgb_ga.pkl`, results CSVs.

## Feature encodings used (from `model.FEATURE_META` / `heart299.csv`)
age (years); anaemia / diabetes / high_blood_pressure / smoking 0=No 1=Yes; sex 0=Female 1=Male; creatinine_phosphokinase (mcg/L; U/L treated as the same number); ejection_fraction (%); platelets (full count, e.g. 265000 = kiloplatelets/mL); serum_creatinine (mg/dL); serum_sodium (mEq/L); time (days).

## Behaviour guarantees
* No defaults: missing fields are empty and required. Ranges, conflicting duplicates, "borderline", "ex-smoker", unit-less platelets < 1000 are left blank.
* Every conversion (µmol/L, ×10³/µL, weeks/months) and every OCR-derived value is flagged and needs an explicit "I verified this value" tick.
* Prediction runs only after: all 12 present, flagged fields verified, confirmation box ticked, validation passed. Hard physiological limits can never be bypassed; the model's training range can be bypassed only by explicit opt-in.
* Report text shown in the UI is HTML-escaped. Error messages never echo file contents.
