# Synthetic sample patient reports  (SOFTWARE TESTING ONLY)

Every file here is **entirely fictional** and is stamped
"SYNTHETIC SAMPLE - FOR SOFTWARE TESTING ONLY - NOT A REAL MEDICAL RECORD". No real person, hospital or record is represented.

| File | Purpose | Expected behaviour in the app |
|---|---|---|
| `sample_report_01_text.pdf` | Clean text-based PDF, all 12 values in model units | 12/12 extracted, no warnings |
| `sample_report_02_scanned.pdf` | Image-only (scanned-looking, slightly rotated + noisy) PDF | OCR; all 12 found but each flagged "verify" |
| `sample_report_03_image.jpg` | JPEG "photo" with unit differences (platelets x10^3/uL, creatinine umol/L, follow-up in weeks) | OCR; conversions applied and flagged |
| `sample_report_04_incomplete_ambiguous.pdf` | Missing CPK, EF as a range, two different sodium values, "borderline" BP, ex-smoker | 5 clearly extracted; the rest left blank / flagged - prediction blocked until you fill them in |

Values in report 01: age 67, anaemia 1, CPK 250, diabetes 1, EF 30, high BP 1, platelets 245000, serum creatinine 1.8, sodium 134, sex 1 (male), smoking 0, follow-up 95 days.
Regenerate everything with `python tools/generate_sample_reports.py`.
