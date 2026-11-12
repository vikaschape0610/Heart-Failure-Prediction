"""
Generate the SYNTHETIC sample patient reports used to test the report-upload feature.

All patients are fictional.  Every page carries a banner saying so.  Run:
    python tools/generate_sample_reports.py
Outputs go to ./sample_reports/ and the ground-truth values to tests/expected_values.json
"""
import io, json, os, random
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "sample_reports")
os.makedirs(OUT, exist_ok=True)

BANNER = "SYNTHETIC SAMPLE - FOR SOFTWARE TESTING ONLY - NOT A REAL MEDICAL RECORD"
SS = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=SS["Heading1"], fontSize=16, spaceAfter=2)
H2 = ParagraphStyle("h2", parent=SS["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#1f3a5f"))
BODY = ParagraphStyle("b", parent=SS["BodyText"], fontSize=9.5, leading=12.5)
BAN = ParagraphStyle("ban", parent=BODY, fontSize=10, textColor=colors.white, alignment=1, fontName="Helvetica-Bold")

def page_deco(c, d):
    c.saveState()
    c.setFillColor(colors.HexColor("#b3261e")); c.rect(0, A4[1] - 14 * mm, A4[0], 14 * mm, stroke=0, fill=1)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(A4[0] / 2, A4[1] - 9 * mm, BANNER)
    c.setFillColor(colors.HexColor("#b3261e")); c.setFont("Helvetica", 8)
    c.drawCentredString(A4[0] / 2, 9 * mm, BANNER + "  |  All names, numbers and facilities are fictional.")
    c.restoreState()

def tbl(rows, widths, header=True):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    st = [("FONTSIZE", (0, 0), (-1, -1), 9.5), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6ecf5")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st)); return t

def build(path, p):
    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=22 * mm, bottomMargin=18 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
                            title=p["title"], author="Synthetic data generator")
    s = [Paragraph(p["facility"], H1), Paragraph(p["subtitle"], BODY), Spacer(1, 4)]
    s.append(Paragraph("Patient details (fictional)", H2))
    s.append(tbl(p["demo"], [45 * mm, 125 * mm], header=False))
    s.append(Paragraph("Clinical history", H2))
    s.append(tbl(p["history"], [70 * mm, 100 * mm], header=False))
    s.append(Paragraph("Laboratory results", H2))
    s.append(tbl([["Test", "Result", "Flag", "Reference range", "Unit"]] + p["labs"], [62 * mm, 22 * mm, 12 * mm, 34 * mm, 40 * mm]))
    s.append(Paragraph("Cardiac imaging", H2))
    s.append(tbl(p["echo"], [70 * mm, 100 * mm], header=False))
    s.append(Paragraph("Follow-up", H2))
    s.append(tbl(p["followup"], [70 * mm, 100 * mm], header=False))
    s.append(Spacer(1, 6))
    for n in p.get("notes", []):
        s.append(Paragraph(n, BODY))
    s.append(Spacer(1, 8))
    s.append(Paragraph(BANNER + ". Values were invented to exercise the extraction software; they describe no real person.", BODY))
    doc.build(s, onFirstPage=page_deco, onLaterPages=page_deco)

# ───────────────────────── 1. clean text PDF (all 12 values) ─────────────────
R1 = dict(title="Synthetic sample report 01 (text PDF)", facility="Demo Cardiology Clinic (fictional)",
  subtitle="Heart-failure clinic summary and laboratory report - Report ID SYN-0001",
  demo=[["Patient name", "Test Patient A. Demo (fictional)"], ["Patient ID", "SYN-000101"],
        ["Age", "67 years"], ["Sex", "Male"], ["Report date", "2026-01-15"]],
  history=[["Anaemia", "Yes"], ["Diabetes mellitus", "Yes"], ["High blood pressure (hypertension)", "Yes"], ["Smoking status", "No (non-smoker)"]],
  labs=[["Creatinine Phosphokinase (CPK)", "250", "", "23 - 300", "mcg/L"],
        ["Platelets", "245000", "", "150000 - 450000", "kiloplatelets/mL"],
        ["Serum Creatinine", "1.8", "H", "0.6 - 1.3", "mg/dL"],
        ["Serum Sodium", "134", "L", "135 - 145", "mEq/L"],
        ["Hemoglobin", "10.8", "L", "13.0 - 17.0", "g/dL"],
        ["Creatinine clearance (eGFR)", "38", "L", "> 60", "mL/min"],
        ["CK-MB", "22", "", "0 - 25", "U/L"]],
  echo=[["Ejection Fraction", "30 %"], ["Blood pressure at visit", "148/92 mmHg"], ["Heart rate", "78 bpm"]],
  followup=[["Follow-up period", "95 days"]],
  notes=["Clinical impression: reduced systolic function, on guideline-directed therapy. Review in clinic."])
E1 = dict(age=67, anaemia=1, creatinine_phosphokinase=250, diabetes=1, ejection_fraction=30, high_blood_pressure=1,
          platelets=245000, serum_creatinine=1.8, serum_sodium=134, sex=1, smoking=0, time=95)

# ───────────────────────── 2. scanned PDF (image only) ───────────────────────
R2 = dict(title="Synthetic sample report 02 (scanned)", facility="Sample Heart Institute (fictional)",
  subtitle="Outpatient heart-failure assessment - Report ID SYN-0002",
  demo=[["Patient name", "Test Patient B. Example (fictional)"], ["Patient ID", "SYN-000202"],
        ["Age", "58 years"], ["Sex", "Female"], ["Report date", "2026-02-03"]],
  history=[["Anaemia", "No"], ["Diabetes mellitus", "No"], ["High blood pressure (hypertension)", "No"], ["Smoking status", "Yes (current smoker)"]],
  labs=[["Creatinine Phosphokinase (CPK)", "582", "H", "23 - 300", "mcg/L"],
        ["Platelets", "262000", "", "150000 - 450000", "kiloplatelets/mL"],
        ["Serum Creatinine", "1.1", "", "0.6 - 1.3", "mg/dL"],
        ["Serum Sodium", "138", "", "135 - 145", "mEq/L"],
        ["Hemoglobin", "13.4", "", "12.0 - 15.5", "g/dL"]],
  echo=[["Ejection Fraction", "38 %"], ["Heart rate", "72 bpm"]],
  followup=[["Follow-up period", "140 days"]])
E2 = dict(age=58, anaemia=0, creatinine_phosphokinase=582, diabetes=0, ejection_fraction=38, high_blood_pressure=0,
          platelets=262000, serum_creatinine=1.1, serum_sodium=138, sex=0, smoking=1, time=140)

# ───────────────────────── 3. image (photo-like) with unit conversions ───────
R3 = dict(title="Synthetic sample report 03 (image)", facility="Example Regional Hospital (fictional)",
  subtitle="Cardiology discharge summary - Report ID SYN-0003",
  demo=[["Patient name", "Test Patient C. Sample (fictional)"], ["Patient ID", "SYN-000303"],
        ["Age / Sex", "73 Y / M"], ["Report date", "2026-03-21"]],
  history=[["Anaemia", "Present"], ["Diabetes mellitus", "Absent"], ["Hypertension", "Yes"], ["Smoking status", "Non-smoker"]],
  labs=[["CPK", "1021", "H", "23 - 300", "mcg/L"],
        ["Platelet count", "265", "", "150 - 450", "x10^3/uL"],
        ["Serum Creatinine", "125", "H", "53 - 106", "umol/L"],
        ["Serum Sodium", "129", "L", "135 - 145", "mmol/L"]],
  echo=[["LVEF", "25 %"]],
  followup=[["Follow-up period", "8 weeks"]])
E3_ocr = dict(age=73, anaemia=1, creatinine_phosphokinase=1021, diabetes=0, ejection_fraction=25, high_blood_pressure=1,
          platelets=265000, serum_creatinine=1.41, serum_sodium=129, sex=1, smoking=0, time=56)

# ───────────────────────── 4. incomplete / ambiguous PDF ─────────────────────
R4 = dict(title="Synthetic sample report 04 (incomplete + ambiguous)", facility="Demo Cardiology Clinic (fictional)",
  subtitle="Referral letter with partial results - Report ID SYN-0004",
  demo=[["Patient name", "Test Patient D. Partial (fictional)"], ["Patient ID", "SYN-000404"],
        ["Age", "72 years"], ["Sex", "Male"], ["Report date", "2026-04-09"]],
  history=[["Anaemia", "Yes"], ["Diabetes mellitus", "No"], ["High blood pressure (hypertension)", "borderline"], ["Smoking status", "Ex-smoker (quit 2019)"]],
  labs=[["Platelets", "198000", "", "150000 - 450000", "kiloplatelets/mL"],
        ["Serum Creatinine", "142", "H", "53 - 106", "umol/L"],
        ["Serum Sodium", "133", "L", "135 - 145", "mEq/L"],
        ["Serum Sodium (repeat sample)", "138", "", "135 - 145", "mEq/L"]],
  echo=[["Ejection Fraction", "35 - 40 %"]],
  followup=[["Follow-up period", "12 weeks"]],
  notes=["Creatine kinase was not measured at this visit."])

def raster(pdf_path, dpi=170):
    import pypdfium2 as pdfium
    d = pdfium.PdfDocument(pdf_path); pg = d[0]
    img = pg.render(scale=dpi / 72).to_pil().convert("L"); d.close(); return img

def degrade(img, angle=0.9, noise=7, seed=3):
    import numpy as np
    from PIL import Image, ImageFilter
    rnd = np.random.default_rng(seed)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
    a = np.asarray(img).astype("float32") + rnd.normal(0, noise, (img.height, img.width))
    a = np.clip(a * 0.96 + 4, 0, 255).astype("uint8")
    return Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.6))

def main():
    p1 = os.path.join(OUT, "sample_report_01_text.pdf"); build(p1, R1)
    tmp2 = os.path.join(OUT, "_tmp2.pdf"); build(tmp2, R2)
    img2 = degrade(raster(tmp2, 200)); os.remove(tmp2)
    img2.convert("RGB").save(os.path.join(OUT, "sample_report_02_scanned.pdf"), "PDF", resolution=200)   # image-only PDF
    tmp3 = os.path.join(OUT, "_tmp3.pdf"); build(tmp3, R3)
    img3 = degrade(raster(tmp3, 170), angle=-0.7, noise=9, seed=5); os.remove(tmp3)
    img3.convert("RGB").save(os.path.join(OUT, "sample_report_03_image.jpg"), "JPEG", quality=78)
    p4 = os.path.join(OUT, "sample_report_04_incomplete_ambiguous.pdf"); build(p4, R4)
    exp = {
      "sample_report_01_text.pdf": {"values": E1, "flagged": []},
      "sample_report_02_scanned.pdf": {"values": E2, "flagged": []},
      "sample_report_03_image.jpg": {"values": E3_ocr, "flagged": ["platelets", "serum_creatinine", "time"]},
      "sample_report_04_incomplete_ambiguous.pdf": {"values": {"age": 72, "anaemia": 1, "diabetes": 0, "sex": 1, "platelets": 198000,
                              "serum_creatinine": 1.61, "time": 84},
                              "missing_or_uncertain": ["creatinine_phosphokinase", "ejection_fraction", "high_blood_pressure", "serum_sodium", "smoking"],
                              "flagged": ["serum_creatinine", "time"]},
    }
    json.dump(exp, open(os.path.join(ROOT, "tests", "expected_values.json"), "w"), indent=1)
    print("written:", sorted(os.listdir(OUT)))

if __name__ == "__main__":
    main()
