"""Unit / integration tests for report_extractor + model compatibility.  Run: pytest -v tests"""
import io, json, os, sys, zlib, struct
import numpy as np, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import report_extractor as rx
from report_extractor import Line

SAMPLES = os.path.join(ROOT, "sample_reports")
EXP = json.load(open(os.path.join(ROOT, "tests", "expected_values.json")))
def read(n): return open(os.path.join(SAMPLES, n), "rb").read()
def parse(*texts): return rx.parse_features([Line(t) for t in texts])

# ── sample files ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["sample_report_01_text.pdf", "sample_report_02_scanned.pdf"])
def test_clean_samples_all_12_values_correct(name):
    r = rx.extract_from_bytes(read(name), name)
    for k, v in EXP[name]["values"].items():
        assert r.fields[k].value == pytest.approx(v), (k, r.fields[k])
    assert all(f.value is not None for f in r.fields.values())

def test_text_pdf_uses_text_layer_and_no_flags():
    r = rx.extract_from_bytes(read("sample_report_01_text.pdf"))
    assert r.methods_used == ["text"] and r.n_ok == 12

def test_scanned_pdf_uses_ocr_and_every_value_flagged_for_verification():
    r = rx.extract_from_bytes(read("sample_report_02_scanned.pdf"))
    assert r.methods_used == ["ocr"]
    assert all(f.status == "uncertain" for f in r.fields.values())

def test_image_upload_with_unit_conversions_flagged():
    n = "sample_report_03_image.jpg"
    r = rx.extract_from_bytes(read(n), n)
    for k, v in EXP[n]["values"].items():
        assert r.fields[k].value == pytest.approx(v, rel=1e-3), k
    for k in EXP[n]["flagged"]:
        assert r.fields[k].status == "uncertain" and any("converted" in x for x in r.fields[k].notes)

def test_png_upload():
    from PIL import Image
    import pypdfium2 as pdfium
    d = pdfium.PdfDocument(os.path.join(SAMPLES, "sample_report_01_text.pdf"))
    buf = io.BytesIO(); d[0].render(scale=2.5).to_pil().save(buf, "PNG")
    r = rx.extract_from_bytes(buf.getvalue(), "x.png")
    assert r.methods_used == ["ocr"] and r.fields["age"].value == 67 and r.fields["serum_sodium"].value == 134

def test_incomplete_ambiguous_never_guesses():
    n = "sample_report_04_incomplete_ambiguous.pdf"
    r = rx.extract_from_bytes(read(n), n)
    for k in EXP[n]["missing_or_uncertain"]:
        assert r.fields[k].value is None and r.fields[k].status in ("missing", "uncertain"), k
    assert r.fields["creatinine_phosphokinase"].status == "missing"
    assert "conflicting" in " ".join(r.fields["serum_sodium"].notes)
    assert "range" in " ".join(r.fields["ejection_fraction"].notes)
    for k, v in EXP[n]["values"].items():
        assert r.fields[k].value == pytest.approx(v, rel=1e-2)

# ── parser behaviour ────────────────────────────────────────────────────────
def test_reference_range_not_taken_as_result():
    f = parse("Serum Sodium 128 L 135 - 145 mEq/L")["serum_sodium"]
    assert f.value == 128

def test_cpk_vs_creatinine_not_confused():
    f = parse("Creatinine Phosphokinase 582 mcg/L", "Serum Creatinine 1.3 mg/dL", "Creatinine clearance 40 mL/min", "CK-MB 20 U/L")
    assert f["creatinine_phosphokinase"].value == 582 and f["serum_creatinine"].value == 1.3

def test_legend_text_not_parsed_as_value():
    f = parse("Anaemia (0=No, 1=Yes): 1", "Sex (0=F, 1=M): 0")
    assert f["anaemia"].value == 1 and f["sex"].value == 0

@pytest.mark.parametrize("txt,val", [("Anaemia: Yes", 1), ("Anaemia: Absent", 0), ("Anaemic - Positive", 1), ("Anaemia: 0", 0),
                                      ("Anemia: Negative", 0), ("Anaemia (Yes)", 1)])
def test_binary_variants(txt, val): assert parse(txt)["anaemia"].value == val

def test_smoking_variants():
    assert parse("Smoking: Non-smoker")["smoking"].value == 0
    assert parse("Smoker: Yes")["smoking"].value == 1
    f = parse("Smoking status: Former smoker")["smoking"]; assert f.value is None and f.status == "uncertain"

def test_sex_variants_and_combined_line():
    assert parse("Sex: Female")["sex"].value == 0 and parse("Gender: M")["sex"].value == 1
    f = parse("Age / Sex: 61 Y / F"); assert f["age"].value == 61 and f["sex"].value == 0

def test_unit_conversions_flagged():
    f = parse("Serum Creatinine 106 umol/L", "Platelets 250 x10^3/uL", "Follow-up 4 weeks")
    assert f["serum_creatinine"].value == pytest.approx(1.2, abs=0.01) and f["serum_creatinine"].status == "uncertain"
    assert f["platelets"].value == 250000 and f["platelets"].status == "uncertain"
    assert f["time"].value == 28 and f["time"].status == "uncertain"

def test_platelets_without_unit_under_1000_not_guessed():
    f = parse("Platelets 250")["platelets"]; assert f.value is None and f.status == "uncertain"

def test_implausible_values_rejected():
    f = parse("Ejection Fraction 250 %", "Serum Sodium 14", "Age 400")
    assert f["ejection_fraction"].value is None and f["serum_sodium"].value is None and f["age"].value is None

def test_age_in_months_not_accepted():
    assert parse("Age: 14 months")["age"].value is None

def test_conflicting_duplicates_left_blank():
    f = parse("Age: 60", "Age: 66")["age"]; assert f.value is None and f.status == "uncertain"

def test_identical_duplicates_ok():
    assert parse("Age: 60", "Age 60 years")["age"].value == 60 and parse("Age: 60", "Age 60 years")["age"].status == "ok"

def test_missing_is_missing_no_default():
    f = parse("nothing relevant here")
    assert all(x.value is None and x.status == "missing" for x in f.values())

def test_thousands_separator():
    assert parse("Platelets 245,000 /uL")["platelets"].value == 245000

# ── invalid / hostile files ─────────────────────────────────────────────────
def ex(data, name="f.pdf"): return pytest.raises(rx.ExtractionError)

def test_empty_file():
    with ex(b""): rx.extract_from_bytes(b"", "a.pdf")
def test_random_bytes():
    with ex(b""): rx.extract_from_bytes(os.urandom(5000), "a.pdf")
def test_text_file_renamed_pdf():
    with ex(b""): rx.extract_from_bytes(b"Age: 67\nSex: Male\n", "a.pdf")
def test_truncated_pdf():
    with ex(b""): rx.extract_from_bytes(read("sample_report_01_text.pdf")[:300], "a.pdf")
def test_pdf_header_garbage_body():
    with ex(b""): rx.extract_from_bytes(b"%PDF-1.7\n" + os.urandom(3000), "a.pdf")
def test_truncated_png():
    d = read("sample_report_03_image.jpg")
    from PIL import Image; b = io.BytesIO(); Image.open(io.BytesIO(d)).save(b, "PNG")
    with ex(b""): rx.extract_from_bytes(b.getvalue()[:200], "a.png")
def test_truncated_jpeg():
    with ex(b""): rx.extract_from_bytes(read("sample_report_03_image.jpg")[:500], "a.jpg")
def test_oversize_file():
    with ex(b""): rx.extract_from_bytes(b"%PDF-" + b"0" * (rx.MAX_FILE_BYTES + 1), "a.pdf")
def test_pixel_bomb_png():
    from PIL import Image
    b = io.BytesIO(); Image.new("L", (8000, 8000), 255).save(b, "PNG")
    with ex(b""): rx.extract_from_bytes(b.getvalue(), "a.png")
def test_blank_image_no_text():
    from PIL import Image
    b = io.BytesIO(); Image.new("RGB", (800, 600), "white").save(b, "PNG")
    with ex(b""): rx.extract_from_bytes(b.getvalue(), "a.png")
def test_encrypted_pdf():
    from pypdf import PdfReader, PdfWriter
    w = PdfWriter(); [w.add_page(p) for p in PdfReader(os.path.join(SAMPLES, "sample_report_01_text.pdf")).pages]
    w.encrypt("secret"); b = io.BytesIO(); w.write(b)
    with pytest.raises(rx.ExtractionError, match="password|corrupted|could not"): rx.extract_from_bytes(b.getvalue(), "a.pdf")
def test_extension_mismatch_warns_but_reads_real_type():
    r = rx.extract_from_bytes(read("sample_report_01_text.pdf"), "report.png")
    assert any("does not match" in w for w in r.warnings) and r.fields["age"].value == 67
def test_unrelated_valid_pdf_warns():
    from reportlab.pdfgen import canvas
    b = io.BytesIO(); c = canvas.Canvas(b); c.drawString(72, 700, "Quarterly invoice for office supplies, total due 30 days, thank you for your business"); c.save()
    r = rx.extract_from_bytes(b.getvalue(), "a.pdf")
    assert r.n_ok == 0 and any("None of the 12" in w for w in r.warnings)
def test_error_messages_do_not_leak_content():
    try: rx.extract_from_bytes(b"SECRET-PATIENT-NAME" * 10, "a.pdf")
    except rx.ExtractionError as e: assert "SECRET" not in str(e)
def test_no_temp_files_written_during_extraction(tmp_path, monkeypatch):
    import tempfile; before = set(os.listdir(tempfile.gettempdir()))
    rx.extract_from_bytes(read("sample_report_02_scanned.pdf"))
    assert set(os.listdir(tempfile.gettempdir())) - before == set()

# ── confirmation validation ─────────────────────────────────────────────────
from model import FEATURE_META
SOFT = {k: (float(v[2]), float(v[3])) for k, v in FEATURE_META.items()}
GOOD = EXP["sample_report_01_text.pdf"]["values"]
def test_validate_ok(): c, e = rx.validate_confirmed(dict(GOOD), SOFT); assert not e and c == {k: float(v) for k, v in GOOD.items()}
def test_validate_missing_required():
    v = dict(GOOD); v["smoking"] = None; c, e = rx.validate_confirmed(v, SOFT); assert e and "smoking" in e[0]
def test_validate_nan_rejected():
    v = dict(GOOD); v["age"] = float("nan"); assert rx.validate_confirmed(v, SOFT)[1]
def test_validate_outside_training_range_needs_opt_in():
    v = dict(GOOD); v["platelets"] = 20000
    assert rx.validate_confirmed(v, SOFT)[1] and not rx.validate_confirmed(v, SOFT, True)[1]
def test_validate_hard_limits_never_bypassed():
    v = dict(GOOD); v["ejection_fraction"] = 140; assert rx.validate_confirmed(v, SOFT, True)[1]
def test_validate_binary_must_be_01():
    v = dict(GOOD); v["sex"] = 2; assert rx.validate_confirmed(v, SOFT, True)[1]

# ── compatibility with the original model / SHAP ────────────────────────────
@pytest.fixture(scope="module")
def S():
    from model import train_model; return train_model()
BASE = json.load(open(os.path.join(ROOT, "tests", "baseline_reference.json")))

def test_original_model_metrics_unchanged(S):
    assert S["metrics"] == BASE["metrics"] and S["cols"] == BASE["cols"] and S["cm"].tolist() == BASE["cm"]
    assert S["best_params"]["n_estimators"] == BASE["best_params"]["n_estimators"]

def test_feature_keys_match_model_exactly(S):
    assert sorted(rx.FEATURE_KEYS) == sorted(FEATURE_META) == sorted(S["cols"])

def test_extracted_sample_prediction_identical_to_original_baseline(S):
    from model import predict_patient
    r = rx.extract_from_bytes(read("sample_report_01_text.pdf"))
    clean, errs = rx.validate_confirmed(r.values(), SOFT); assert not errs
    out = predict_patient(S["model"], S["scaler"], S["cols"], clean)
    ref = BASE["preds"][0]       # sample-01 values, computed with the pristine original project code
    assert out["prob_death"] == pytest.approx(ref["prob_death"], abs=1e-9) and out["label"] == ref["label"]
    assert np.allclose(out["shap_vals"], ref["shap"], atol=1e-9) and len(out["shap_vals"]) == 12

def test_scanned_sample_prediction_identical_to_original_baseline(S):
    from model import predict_patient
    r = rx.extract_from_bytes(read("sample_report_02_scanned.pdf"))
    clean, errs = rx.validate_confirmed(r.values(), SOFT); assert not errs
    out = predict_patient(S["model"], S["scaler"], S["cols"], clean); ref = BASE["preds"][1]
    assert out["prob_death"] == pytest.approx(ref["prob_death"], abs=1e-9) and np.allclose(out["shap_vals"], ref["shap"], atol=1e-9)

def test_scanned_sample_prediction_equals_manual_dict(S):
    from model import predict_patient
    r = rx.extract_from_bytes(read("sample_report_02_scanned.pdf"))
    clean, errs = rx.validate_confirmed(r.values(), SOFT); assert not errs
    a = predict_patient(S["model"], S["scaler"], S["cols"], clean)
    b = predict_patient(S["model"], S["scaler"], S["cols"], {k: float(v) for k, v in EXP["sample_report_02_scanned.pdf"]["values"].items()})
    assert a["prob_death"] == b["prob_death"] and np.array_equal(a["shap_vals"], b["shap_vals"])

@pytest.mark.parametrize("i", [2, 3, 4])
def test_other_baseline_patients_unchanged(S, i):
    from model import predict_patient
    ref = BASE["preds"][i]; out = predict_patient(S["model"], S["scaler"], S["cols"], ref["p"])
    assert out["prob_death"] == pytest.approx(ref["prob_death"], abs=1e-9) and np.allclose(out["shap_vals"], ref["shap"], atol=1e-9)

# ── environment edge cases ──────────────────────────────────────────────────
def test_missing_tesseract_gives_clear_error_but_text_pdf_still_works(monkeypatch):
    monkeypatch.setattr(rx, "tesseract_available", lambda: False)
    with pytest.raises(rx.ExtractionError, match="Tesseract"):
        rx.extract_from_bytes(read("sample_report_02_scanned.pdf"))
    assert rx.extract_from_bytes(read("sample_report_01_text.pdf")).n_ok == 12

def test_multipage_pdf_page_cap_warns():
    from reportlab.pdfgen import canvas
    b = io.BytesIO(); c = canvas.Canvas(b)
    for i in range(14):
        c.drawString(72, 700, f"Page {i} clinical note with enough text to count as a text layer for extraction.")
        if i == 0: c.drawString(72, 650, "Age: 61 years")
        c.showPage()
    c.save(); r = rx.extract_from_bytes(b.getvalue(), "a.pdf")
    assert r.pages == rx.MAX_PDF_PAGES and any("first" in w for w in r.warnings) and r.fields["age"].value == 61

def test_mixed_pdf_text_page_plus_scanned_page():
    from reportlab.pdfgen import canvas
    from PIL import Image
    import pypdfium2 as pdfium
    img = pdfium.PdfDocument(os.path.join(SAMPLES, "sample_report_02_scanned.pdf"))[0].render(scale=1.5).to_pil().convert("RGB")
    ib = io.BytesIO(); img.save(ib, "PDF", resolution=150)
    from pypdf import PdfReader, PdfWriter
    b = io.BytesIO(); c = canvas.Canvas(b); c.drawString(72, 700, "Cover page: Age: 90 years  - text layer page with plenty of characters to pass the threshold."); c.save()
    w = PdfWriter(); w.add_page(PdfReader(io.BytesIO(b.getvalue())).pages[0]); w.add_page(PdfReader(io.BytesIO(ib.getvalue())).pages[0])
    o = io.BytesIO(); w.write(o)
    r = rx.extract_from_bytes(o.getvalue(), "a.pdf")
    assert r.methods_used == ["text", "ocr"] and r.fields["serum_sodium"].value == 138
    assert r.fields["age"].status == "uncertain" and r.fields["age"].value is None   # 90 (text) vs 58 (OCR) conflict -> not guessed

def test_tesseract_lookup_env_override_and_windows_default(monkeypatch, tmp_path):
    fake = tmp_path / "tesseract.exe"; fake.write_text("x")
    monkeypatch.setenv("TESSERACT_CMD", str(fake)); assert rx.tesseract_cmd() == str(fake)
    monkeypatch.delenv("TESSERACT_CMD"); monkeypatch.setattr(rx.shutil, "which", lambda _: None)
    monkeypatch.setattr(rx, "_WIN_DEFAULTS", (str(fake),)); assert rx.tesseract_cmd() == str(fake)
    monkeypatch.setattr(rx, "_WIN_DEFAULTS", ()); assert rx.tesseract_cmd() is None and not rx.tesseract_available()


# ── out-of-range highlighting (RED only) ────────────────────────────────────
TRAIN = {"serum_creatinine": (0.5, 9.4), "ejection_fraction": (14, 80), "serum_sodium": (113, 148),
         "platelets": (25000, 850000), "creatinine_phosphokinase": (23, 7861), "age": (1, 120), "time": (4, 285)}

@pytest.mark.parametrize("key", list(TRAIN))
def test_range_flag_red_below_and_above_set_range(key):
    lo, hi = TRAIN[key]
    below = rx.range_flag(key, lo - 0.5, TRAIN[key]); above = rx.range_flag(key, hi + 1, TRAIN[key])
    assert below.level == rx.LEVEL_OUTSIDE_TRAINING and below.direction == "low" and below.out_of_range
    assert above.level == rx.LEVEL_OUTSIDE_TRAINING and above.direction == "high" and above.out_of_range
    assert not rx.range_flag(key, (lo + hi) / 2, TRAIN[key]).out_of_range

def test_range_flag_boundaries_are_inclusive():
    for lo_hi in (113, 148):
        assert not rx.range_flag("serum_sodium", lo_hi, TRAIN["serum_sodium"]).out_of_range

def test_no_purple_level_exists_any_more():
    assert not hasattr(rx, "LEVEL_ABNORMAL") and not hasattr(rx, "REFERENCE_RANGES")
    # a value that is clinically abnormal but inside the set range is NOT highlighted
    assert not rx.range_flag("serum_sodium", 130, TRAIN["serum_sodium"]).out_of_range

def test_range_flag_skips_missing_and_never_flags_binary():
    assert rx.range_flag("age", None) is None and rx.range_flag("age", float("nan")) is None
    assert not rx.range_flag("anaemia", 1, (0, 1)).out_of_range

def test_range_flags_only_contains_available_values():
    vals = {k: None for k in rx.FEATURE_KEYS}; vals.update(serum_sodium=105, ejection_fraction=30)
    fl = rx.range_flags(vals, TRAIN)
    assert set(fl) == {"serum_sodium", "ejection_fraction"}
    assert fl["serum_sodium"].out_of_range and not fl["ejection_fraction"].out_of_range

def test_validate_confirmed_out_of_set_range_accepted_when_allowed_but_hard_limits_still_apply():
    vals = {"age": 67, "anaemia": 1, "creatinine_phosphokinase": 250, "diabetes": 0, "ejection_fraction": 85,
            "high_blood_pressure": 1, "platelets": 245000, "serum_creatinine": 1.8, "serum_sodium": 105,
            "sex": 1, "smoking": 0, "time": 95}
    soft = {k: TRAIN[k] for k in TRAIN}; soft.update({k: (0, 1) for k in rx.BINARY_KEYS})
    clean, errs = rx.validate_confirmed(vals, soft, allow_outside_training_range=True)
    assert errs == [] and clean["ejection_fraction"] == 85 and clean["serum_sodium"] == 105
    vals["ejection_fraction"] = 150                                    # physiologically impossible -> always refused
    clean, errs = rx.validate_confirmed(vals, soft, allow_outside_training_range=True)
    assert any("ejection_fraction" in e for e in errs)