# Heart Failure Survival Prediction System
### XGBoost + Genetic Algorithm (XGB-GA) with SHAP Explainability
**Reference Paper:** *IEEE Access 2025* | Ahmed et al. | DOI: 10.1109/ACCESS.2025.3541069

---

## 📋 Overview
This project classifies and predicts heart failure patient survival outcomes using XGBoost
optimized by a Genetic Algorithm (XGB-GA). It uses all **12 clinical features** with SMOTE
balancing and achieves **95.0% accuracy** on the test set with **100% Recall (zero missed deaths)**.

| Metric         | Value              |
|----------------|--------------------|
| Accuracy       | **95.0%**          |
| Recall         | **1.0000 (100%)**  |
| Precision      | 0.8636             |
| F1 Score       | 0.9268             |
| AUC-ROC        | **0.9859**         |
| FN (Missed Deaths) | **0**          |
| CV Mean Acc    | 85.48% ± 4.05%     |
| AIC / BIC      | 28.28 / 34.56      |

**GA-discovered best parameters:** `n_estimators=107, learning_rate=0.3320, max_depth=4`

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Web Application (Modern Web UI)
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser to access:
- **[ 1 ] 🔮 Predict Patient**: Enter 12 clinical biomarkers for real-time predictions with gauge risk charts and patient-specific SHAP explanations.
- **[ 2 ] 📊 Model Performance**: 95.0% accuracy, confusion matrix (FN=0), ROC curve, and model evaluation metrics table.
- **[ 3 ] 📈 Feature Importance**: All 12 ranked clinical biomarkers with percentage contributions from XGB-GA.
- **[ 4 ] 🧠 SHAP Explainability**: Global mean |SHAP| biomarker influence and interactive test patient waterfall selector.

### 2b. NEW - Upload a patient report (PDF / JPG / JPEG / PNG)
In **[ 1 ] Predict Patient** choose **📄 Upload patient report**. The app extracts the 12 model inputs, shows them in an
editable review form (missing / uncertain values highlighted, nothing guessed, no defaults), and runs the XGB-GA
prediction + SHAP **only after you tick the confirmation box and press the confirm button**. Manual entry is unchanged
(**✍️ Enter values manually**). Try the files in `sample_reports/`.

**OCR needs the Tesseract program** (text-based PDFs work without it; scanned PDFs and images do not).

#### Windows setup (step by step)
1. Install Python 3.10+ from https://www.python.org/downloads/ (tick **"Add python.exe to PATH"** in the installer).
2. Open **PowerShell** in the project folder and create a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Install Tesseract OCR: download the Windows installer from https://github.com/UB-Mannheim/tesseract/wiki and keep the English language data selected.
   The default location `C:\Program Files\Tesseract-OCR\tesseract.exe` is detected automatically. If you installed it elsewhere, either add its folder to `PATH` or set
   `$env:TESSERACT_CMD = "D:\path\to\tesseract.exe"` before starting the app. Check with `tesseract --version`.
4. Start the app: `streamlit run app.py` and open http://localhost:8501.
5. Test: upload the files from `sample_reports\` (see `sample_reports\README.md` for what each should show).

If the app says the OCR engine is not installed, step 3 is the cause; text PDFs (e.g. `sample_report_01_text.pdf`) still work without it.
Linux: `sudo apt install tesseract-ocr` · macOS: `brew install tesseract`.
**The Windows steps have NOT been tested by me** (my testing was on Linux); please report any problems.

Extraction rules (details in `report_extractor.py`): text-layer PDFs are read with pdfplumber; scanned pages / images are
OCR'd. Unit conversions (µmol/L→mg/dL, ×10³/µL→count, weeks→days) are applied **and flagged**. Every OCR-derived value
requires an explicit "I verified this value" tick. Ranges ("35-40 %"), conflicting duplicates, "borderline", "ex-smoker" etc.
are left blank for you to decide. Values outside the model's training range are blocked unless you explicitly allow them;
physiologically impossible values are always blocked.

**Privacy:** files are processed in memory only (Tesseract is fed via stdin/stdout - no temp files), never saved, never sent
to any external service; results are dropped from the session when the file is removed. Patient name/ID/DOB are not
extracted. This is decision-support software, not a medical device - always verify values against the original report.

**Tests:** `pip install -r requirements-dev.txt && pytest tests` (unit/integration) and, with the app running on port 8599,
`python tests/e2e_playwright.py` (browser end-to-end; needs `playwright install chromium`).

### 3. Run the Interactive Terminal System (Console UI)
```bash
python heart_failure_prediction.py
```
Interactive numbered menu with ANSI styling, clinical predictions, SHAP breakdown, and performance statistics directly in the terminal.

---

## 📂 Key Project Files
- `app.py`: Streamlit web dashboard — premium dark theme, in-app tabs, mobile responsive.
- `report_extractor.py`: NEW - PDF/image text extraction, OCR, parsing, validation (no Streamlit dependency).
- `report_ui.py`: NEW - upload + review/confirm form used by Tab 1.
- `sample_reports/`: NEW - synthetic test reports (`tools/generate_sample_reports.py` regenerates them). `tests/`: NEW - test suites.
- `model.py`: ML backend engine — XGB-GA training, SMOTE, SHAP, model export.
- `heart_failure_prediction.py`: Standalone CLI with ANSI design, 12-feature input, SHAP waterfall.
- `heart299.csv`: Heart failure clinical records (299 patients, 12 features + 1 target).
- `XGB-GA.ipynb`: Jupyter notebook with full pipeline, GA optimization, and result analysis.
- `model_xgb_ga.pkl`: Serialized trained XGB-GA model, scaler, and feature names.
- `Base paper.pdf`: Original IEEE Access 2025 research paper.
- `Fig_*.png`: High-resolution figures (correlation heatmap, SHAP, feature importance, ROC).
- `results_*.csv`: Evaluated performance metrics for XGB-GA.

---

## 🧬 Pipeline Summary
1. **Data**: `heart299.csv` — 299 patients, 12 clinical features, target: `DEATH_EVENT`
2. **Split**: Stratified 80/20 (Train=239, Test=60), `random_state=11`
3. **Scaling**: `StandardScaler` — fit on train, applied to test
4. **Balancing**: `SMOTE(random_state=42)` — train only (162→162 balanced)
5. **Optimization**: Genetic Algorithm — 5-fold CV accuracy fitness, 3 generations, pop=10
6. **Model**: `XGBClassifier(n_estimators=107, learning_rate=0.3320, max_depth=4)`
7. **Explainability**: SHAP TreeExplainer for global and per-patient biomarker contributions
