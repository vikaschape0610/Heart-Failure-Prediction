"""
Patient-report extraction for the XGB-GA heart-failure app.

Turns an uploaded PDF / JPG / JPEG / PNG medical report into the 12 clinical
inputs the XGB-GA model expects (see ``model.FEATURE_META``).

Design rules
------------
* NEVER guess.  A value that is missing, ambiguous, conflicting, low-confidence
  or implausible is returned with ``value=None`` (or flagged ``uncertain``) so
  the UI can force the user to review it.  No defaults are ever inserted.
* Privacy: everything is processed in memory.  Uploaded bytes are never written
  to disk by this module (Tesseract is fed through stdin/stdout) and nothing is
  sent to any network service.  Patient identifiers (name, DOB, MRN...) are not
  extracted.
* No Streamlit import here -> the module is unit-testable on its own.

Pipeline
--------
PDF  : pdfplumber text layer per page -> if a page has too little text, that
       page is rasterised (pypdfium2) and OCR'd with Tesseract.
Image: Pillow decode (+ EXIF rotation, size caps) -> OCR with Tesseract.
Text : line-based label/value parser with synonyms, unit detection and unit
       conversion (every conversion is flagged for review).
"""
from __future__ import annotations

import io
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ────────────────────────── configuration / limits ──────────────────────────
MAX_FILE_BYTES = 15 * 1024 * 1024      # 15 MB per upload
MAX_PDF_PAGES = 10                     # pages inspected per PDF
MAX_IMAGE_PIXELS = 40_000_000          # decompression-bomb guard
MIN_TEXT_CHARS_PER_PAGE = 40           # below this a PDF page is treated as scanned
OCR_DPI = 300
OCR_MIN_LINE_CONF = 70.0               # tesseract word-confidence floor (0-100)
OCR_TIMEOUT_S = 90

SUPPORTED_EXTENSIONS = ("pdf", "jpg", "jpeg", "png")

# Canonical feature order = order requested for the review form.
FEATURE_KEYS = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
    "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium",
    "sex", "smoking", "time",
]
BINARY_KEYS = {"anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"}

# Hard physiological limits: values outside are impossible/unit errors and can
# never be sent to the model.  (The model's *training* range, from
# model.FEATURE_META, is narrower and is only a soft warning.)
HARD_LIMITS = {
    "age": (0, 120),
    "creatinine_phosphokinase": (0, 100000),
    "ejection_fraction": (1, 100),
    "platelets": (1000, 3_000_000),
    "serum_creatinine": (0.05, 40),
    "serum_sodium": (80, 180),
    "time": (0, 3650),
    "anaemia": (0, 1), "diabetes": (0, 1), "high_blood_pressure": (0, 1),
    "sex": (0, 1), "smoking": (0, 1),
}

# Highlight levels.  Only ONE kind of highlight exists: RED = the value is outside the "set range"
# (the range the model was trained on, from model.FEATURE_META).  It is a visual warning only - the prediction is
# still computed for such values (only physiologically impossible values, see HARD_LIMITS, are rejected).
LEVEL_NORMAL = "normal"
LEVEL_OUTSIDE_TRAINING = "outside_training"   # outside the set range -> shown in RED


@dataclass
class RangeFlag:
    key: str
    value: float
    level: str                       # LEVEL_*
    direction: str = ""              # "low" | "high" | ""
    training: Optional[Tuple[float, float]] = None

    @property
    def out_of_range(self) -> bool:
        return self.level != LEVEL_NORMAL


def range_flag(key: str, value: Optional[float],
               training_range: Optional[Tuple[float, float]] = None) -> Optional[RangeFlag]:
    """Classify one value.  Returns None when there is nothing to classify (missing / NaN value)."""
    if value is None or value != value:
        return None
    flag = RangeFlag(key=key, value=float(value), level=LEVEL_NORMAL, training=training_range)
    if key not in BINARY_KEYS and training_range is not None:
        tlo, thi = training_range
        if value < tlo or value > thi:
            flag.level, flag.direction = LEVEL_OUTSIDE_TRAINING, ("low" if value < tlo else "high")
    return flag


def range_flags(values: Dict[str, Optional[float]],
                training_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, RangeFlag]:
    """Classify every available value.  Only keys with a usable value appear in the result."""
    training_ranges = training_ranges or {}
    out: Dict[str, RangeFlag] = {}
    for k in FEATURE_KEYS:
        f = range_flag(k, values.get(k), training_ranges.get(k))
        if f is not None:
            out[k] = f
    return out


# ────────────────────────────── result types ────────────────────────────────
class ExtractionError(Exception):
    """User-facing, safe-to-display error (never contains file content)."""


@dataclass
class FieldResult:
    key: str
    value: Optional[float] = None
    status: str = "missing"          # "ok" | "uncertain" | "missing"
    notes: List[str] = field(default_factory=list)
    source: str = ""                 # the matched line (shown so the user can verify)
    method: str = ""                 # "text" | "ocr"
    confidence: Optional[float] = None   # 0-100 (100 for text layer)

    def flag(self, note: str):
        if self.status == "ok":
            self.status = "uncertain"
        self.notes.append(note)


@dataclass
class ExtractionResult:
    fields: Dict[str, FieldResult]
    methods_used: List[str]          # e.g. ["text"], ["ocr"], ["text", "ocr"]
    pages: int
    warnings: List[str] = field(default_factory=list)

    def values(self) -> Dict[str, Optional[float]]:
        return {k: f.value for k, f in self.fields.items()}

    @property
    def n_ok(self):
        return sum(f.status == "ok" for f in self.fields.values())


@dataclass
class Line:
    text: str
    conf: float = 100.0
    method: str = "text"


# ───────────────────────────── file type / safety ───────────────────────────
def detect_file_type(data: bytes) -> Optional[str]:
    """Identify the file from its magic bytes (the extension is not trusted)."""
    head = data[:2048]
    if b"%PDF-" in head[:1024]:
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    return None


_WIN_DEFAULTS = (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe")


def tesseract_cmd() -> Optional[str]:
    """Locate Tesseract: $TESSERACT_CMD, then PATH, then the default Windows install folders."""
    import os
    env = os.environ.get("TESSERACT_CMD")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("tesseract")
    if found:
        return found
    for cand in _WIN_DEFAULTS:
        if os.path.isfile(cand):
            return cand
    return None


def tesseract_available() -> bool:
    return tesseract_cmd() is not None


# ───────────────────────────────── OCR ──────────────────────────────────────
def _prepare_for_ocr(img):
    """Grayscale + upscale small images + autocontrast.  In-memory only."""
    from PIL import ImageOps, Image
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("L", "RGB"):
        bg = Image.new("RGB", img.size, "white")
        try:
            bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
            img = bg
        except Exception:
            img = img.convert("RGB")
    img = img.convert("L")
    w, h = img.size
    if max(w, h) < 1800:                       # tesseract likes ~300 dpi text
        scale = 1800 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return ImageOps.autocontrast(img, cutoff=1)


def _ocr_image(img) -> List[Line]:
    """Run Tesseract via stdin/stdout (no temp files) and return lines with confidence."""
    if not tesseract_available():
        raise ExtractionError(
            "OCR engine (Tesseract) is not installed, so scanned PDFs and images cannot be read. "
            "Install Tesseract (see README) or use a text-based PDF / manual entry.")
    buf = io.BytesIO()
    _prepare_for_ocr(img).save(buf, format="PNG")
    try:
        proc = subprocess.run(
            [tesseract_cmd() or "tesseract", "stdin", "stdout", "--psm", "6", "-l", "eng", "tsv"],
            input=buf.getvalue(), capture_output=True, timeout=OCR_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        raise ExtractionError("OCR took too long and was cancelled. Try a smaller/clearer file.")
    if proc.returncode != 0:
        raise ExtractionError("OCR failed on this file. It may be corrupted or unreadable.")
    return _lines_from_tsv(proc.stdout.decode("utf-8", errors="replace"))


def _lines_from_tsv(tsv: str) -> List[Line]:
    rows = tsv.splitlines()
    if not rows:
        return []
    header = rows[0].split("\t")
    idx = {n: i for i, n in enumerate(header)}
    groups: Dict[Tuple[str, str, str], List[Tuple[str, float]]] = {}
    order: List[Tuple[str, str, str]] = []
    for r in rows[1:]:
        c = r.split("\t")
        if len(c) < len(header):
            continue
        word = c[idx["text"]].strip()
        if not word:
            continue
        try:
            conf = float(c[idx["conf"]])
        except ValueError:
            continue
        if conf < 0:
            continue
        k = (c[idx["block_num"]], c[idx["par_num"]], c[idx["line_num"]])
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append((word, conf))
    out = []
    for k in order:
        words = groups[k]
        out.append(Line(" ".join(w for w, _ in words),
                        conf=min(sum(c for _, c in words) / len(words),
                                 min(c for _, c in words) + 25),  # one bad word drags the line down
                        method="ocr"))
    return out


# ───────────────────────────── file -> lines ────────────────────────────────
def _open_image(data: bytes):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Image.DecompressionBombError:
        raise ExtractionError("The image is too large to process safely.")
    except Exception:
        raise ExtractionError("The image could not be decoded - the file is corrupted or not a valid image.")
    if img.width * img.height > MAX_IMAGE_PIXELS:
        raise ExtractionError("The image is too large to process safely.")
    return img


def _split_text_lines(text: str) -> List[Line]:
    return [Line(t.strip(), 100.0, "text") for t in text.splitlines() if t.strip()]


def _pdf_to_lines(data: bytes) -> Tuple[List[Line], List[str], int, List[str]]:
    import pdfplumber
    import pypdfium2 as pdfium
    lines: List[Line] = []
    methods: List[str] = []
    warnings: List[str] = []
    try:
        pdf = pdfplumber.open(io.BytesIO(data))
    except Exception as e:
        msg = str(e).lower()
        if "password" in msg or "encrypt" in msg:
            raise ExtractionError("This PDF is password-protected. Remove the password and upload it again.")
        raise ExtractionError("The PDF could not be opened - the file is corrupted or not a valid PDF.")
    with pdf:
        n_pages = len(pdf.pages)
        if n_pages == 0:
            raise ExtractionError("The PDF contains no pages.")
        if n_pages > MAX_PDF_PAGES:
            warnings.append(f"Only the first {MAX_PDF_PAGES} of {n_pages} pages were read.")
        scan_pages = []
        for i, page in enumerate(pdf.pages[:MAX_PDF_PAGES]):
            try:
                txt = page.extract_text(layout=False) or ""
            except Exception:
                txt = ""
            if len(txt.strip()) >= MIN_TEXT_CHARS_PER_PAGE:
                lines.extend(_split_text_lines(txt))
                if "text" not in methods:
                    methods.append("text")
            else:
                scan_pages.append(i)
    if scan_pages:
        try:
            doc = pdfium.PdfDocument(data)
        except Exception:
            raise ExtractionError("The PDF could not be rendered for OCR - it may be corrupted.")
        try:
            for i in scan_pages:
                bitmap = doc[i].render(scale=OCR_DPI / 72)
                pil = bitmap.to_pil()
                lines.extend(_ocr_image(pil))
                if "ocr" not in methods:
                    methods.append("ocr")
        finally:
            doc.close()
    return lines, methods, min(n_pages, MAX_PDF_PAGES), warnings


def file_to_lines(data: bytes, filename: str = "") -> Tuple[List[Line], List[str], int, List[str]]:
    """Validate and read an upload. Returns (lines, methods, pages, warnings)."""
    if not data:
        raise ExtractionError("The uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise ExtractionError(f"The file is larger than the {MAX_FILE_BYTES // (1024 * 1024)} MB limit.")
    ftype = detect_file_type(data)
    if ftype is None:
        raise ExtractionError("Unsupported or corrupted file. Please upload a PDF, JPG, JPEG or PNG report.")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    warnings = []
    if ext and ext in SUPPORTED_EXTENSIONS and ext.replace("jpeg", "jpg") != ftype.replace("jpeg", "jpg"):
        warnings.append(f"The file extension (.{ext}) does not match its actual content ({ftype.upper()}); it was read as {ftype.upper()}.")
    if ftype == "pdf":
        lines, methods, pages, w = _pdf_to_lines(data)
        warnings += w
    else:
        lines = _ocr_image(_open_image(data))
        methods, pages = ["ocr"], 1
    return lines, methods, pages, warnings


# ───────────────────────────── text -> features ─────────────────────────────
# Each entry: regex matching the *label*; the value is searched right after it.
# Negative look-aheads keep e.g. "Creatinine Phosphokinase" away from "creatinine".
_L = {
    "age": r"\bage\b(?!\s*(?:of|at|group|related))",
    "anaemia": r"\ban(?:a)?emi(?:a|c)\b",
    "creatinine_phosphokinase": r"\b(?:creatin(?:e|ine)[\s\-]*(?:phospho\s*kinase|kinase)|cpk|ck(?!\s*-?\s*mb))\b(?!\s*-?\s*mb)",
    "diabetes": r"\b(?:diabetes(?:\s+mellitus)?|diabetic|dm)\b",
    "ejection_fraction": r"\b(?:(?:lv|left\s+ventricular)?\s*ejection\s+fraction|lvef)\b|\bEF\b",
    "high_blood_pressure": r"\b(?:high\s+blood\s+pressure|hypertension|htn)\b",
    "platelets": r"\b(?:platelet(?:s)?(?:\s+count)?|plt)\b",
    "serum_creatinine": r"\b(?:serum\s+creatinine|s\.?\s*creatinine|creatinine)\b(?!\s*[\s\-]*(?:phospho|kinase|clearance|ratio))",
    "serum_sodium": r"\b(?:serum\s+sodium|s\.?\s*sodium|sodium|na\+?)\b",
    "sex": r"\b(?:sex|gender)\b",
    "smoking": r"\b(?:smok(?:ing|er)(?:\s+status)?|smokes|tobacco(?:\s+use)?)\b",
    "time": r"\b(?:follow[\s\-]*up(?:\s+(?:period|time|duration))?|time\s+to\s+follow[\s\-]*up)\b",
}
_LABEL_RE = {k: re.compile(v, re.I if k != "ejection_fraction" else 0) for k, v in _L.items()}
# EF alternation mixes case-insensitive and case-sensitive ("EF"): compile explicitly.
_LABEL_RE["ejection_fraction"] = re.compile(
    r"(?i:\b(?:(?:lv|left\s+ventricular)?\s*ejection\s+fraction|lvef)\b)|\bEF\b")

_AGE_SEX_RE = re.compile(
    r"\bage\s*/\s*(?:sex|gender)\s*[:\-]?\s*(\d{1,3})\s*(?:y(?:ea)?rs?|y\.?o\.?|y)?\s*[/,]\s*(male|female|m|f)\b", re.I)

_NUM = r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?"
_NUM_RE = re.compile(_NUM)
_LEGEND_RE = re.compile(r"\(\s*[^()]*\b0\s*[=:\-]\s*[^()]*\)")   # "(0 = No, 1 = Yes)" legends
_YES = r"(?:yes|y|true|present|positive|pos|known|current(?:ly)?|history\s+of|diagnosed|confirmed)"
_NO = r"(?:no|n|false|absent|negative|neg|none|nil|denied|denies|never|non[\s\-]*smoker|not\s+(?:present|known|diagnosed))"


def _clean_number(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _after_label(line: str, m: re.Match) -> str:
    """Text following the label, with '0=No/1=Yes' legends removed and separators trimmed."""
    rest = line[m.end():]
    rest = _LEGEND_RE.sub(" ", rest)
    return rest.lstrip(" \t:=-–—|.·_")


def _numeric_candidates(line: str, rest: str) -> Optional[Tuple[float, str]]:
    """First number after the label (the 'result' column), plus the text after it."""
    m = _NUM_RE.search(rest)
    if not m:
        return None
    # A range such as "35-40" or "35 to 40" is not a single measurement.
    tail = rest[m.end():m.end() + 12]
    if re.match(r"\s*(?:-|–|—|to)\s*\d", tail, re.I) and not re.match(r"\s*-\s*\d+\s*(?:day|week|month)", tail, re.I):
        return float("nan"), rest[m.start():]
    v = _clean_number(m.group(0))
    return (v, rest[m.end():]) if v is not None else None


def _binary_from_text(rest: str) -> Optional[Tuple[Optional[int], str]]:
    """Return (0|1|None, note).  None => ambiguous."""
    r = rest.strip().lower()
    # "High blood pressure (hypertension): Yes" -> drop the leading parenthetical alias;
    # "Smoking (Yes)" -> if nothing follows the parentheses, read the answer inside them.
    am = re.match(r"\(([^()]*)\)\s*[:=\-]?\s*(.*)$", r)
    if am:
        r = (am.group(2) or am.group(1)).strip()
    if not r:
        return None
    first = re.match(r"[a-z0-9+\-]+(?:\s+[a-z]+)?", r)
    token = first.group(0) if first else r
    if re.match(r"(?:1|1\.0)\b", token):
        return 1, ""
    if re.match(r"(?:0|0\.0)\b", token):
        return 0, ""
    if re.match(_NO + r"\b", token) or re.match(r"non[\s\-]*smoker|never[\s\-]*smoker", token):
        return 0, ""
    if re.match(_YES + r"\b", token):
        return 1, ""
    return None


def _first_phrase(rest: str, n=40) -> str:
    return rest.strip()[:n]


def _parse_line_for(key: str, line: Line) -> Optional[Tuple[Optional[float], List[str], str]]:
    """Try to read one feature from one line.  Returns (value|None, notes, matched-text) or None."""
    text = line.text
    # Combined demographics line, e.g. "Age / Sex: 67 / M" or "Age/Gender: 67 Y / Female"
    if key in ("age", "sex"):
        cm = _AGE_SEX_RE.search(text)
        if cm:
            if key == "age":
                return float(cm.group(1)), [], text
            return (1.0 if cm.group(2).lower() in ("m", "male") else 0.0), [], text
    for m in _LABEL_RE[key].finditer(text):
        rest = _after_label(text, m)
        notes: List[str] = []
        if key == "sex":
            r = rest.lower()
            mm = re.match(r"(male|female|m|f|man|woman)\b", r)
            if mm:
                return (1.0 if mm.group(1) in ("male", "m", "man") else 0.0), notes, text
            if re.match(r"(?:[01])\b", r):
                return float(r[0]), notes, text
            continue
        if key == "age":
            # "Age / Sex: 67 / M"  or  "Age: 67 years"
            nm = _numeric_candidates(text, rest)
            if nm is None:
                continue
            v, tail = nm
            if v != v:
                return None, ["value is a range, not a single number"], text
            if re.match(r"\s*(?:months?|mos?\b|weeks?|days?)", tail, re.I):
                return None, ["age appears to be given in months/weeks/days - please enter years"], text
            return v, notes, text
        if key in BINARY_KEYS:
            b = _binary_from_text(rest)
            if b is None:
                # ambiguous wording such as "ex-smoker", "borderline", "?"
                rest = re.sub(r"^\s*\([^()]*\)\s*[:=\-]?\s*", "", rest) or rest
                if rest.strip() and key == "smoking" and re.match(r"(?:ex|former|quit|stopped|previous)", rest.strip(), re.I):
                    return None, ["former/ex-smoker is ambiguous for the model's 0/1 'Smoking' field - choose manually"], text
                if rest.strip() and re.match(r"[A-Za-z0-9]", rest.strip()):
                    return None, [f"could not interpret '{_first_phrase(rest, 25)}' as Yes/No"], text
                continue
            return float(b[0]), notes, text
        # numeric features
        nm = _numeric_candidates(text, rest)
        if nm is None:
            continue
        v, tail = nm
        if v != v:
            return None, ["value is a range, not a single number"], text
        v, notes = _apply_units(key, v, tail, notes)
        return v, notes, text
    return None


def _apply_units(key: str, v: float, tail: str, notes: List[str]) -> Tuple[Optional[float], List[str]]:
    t = tail.strip().lower()[:40]
    t_norm = t.replace("µ", "u").replace("μ", "u").replace("×", "x").replace("³", "3").replace("^", "")
    if key == "serum_creatinine":
        if re.match(r"(?:u|micro)\s*mol", t_norm) or "umol" in t_norm:
            notes.append(f"converted from {v:g} µmol/L to mg/dL (÷88.4) - please verify")
            return round(v / 88.4, 2), notes
    elif key == "platelets":
        if "lakh" not in t_norm and re.search(r"10.{0,2}3\s*/\s*(?:u|micro)?\s*l|\bk\s*/\s*u?l|thou", t_norm):
            notes.append(f"converted from {v:g} ×10³/µL to platelets/mL-count scale (×1000) - please verify")
            return v * 1000.0, notes
        if "lakh" in t_norm:
            notes.append(f"converted from {v:g} lakh/µL (×100000) - please verify")
            return v * 100000.0, notes
        if v < 1000:
            notes.append(f"{v:g} looks like a ×10³/µL value, but the unit could not be read reliably. "
                         f"The model expects a full count (e.g. {v * 1000:,.0f}) - enter the correct value manually")
            return None, notes
    elif key == "time":
        if re.match(r"weeks?\b|wks?\b", t_norm):
            notes.append(f"converted from {v:g} weeks to days (×7) - please verify")
            return v * 7.0, notes
        if re.match(r"months?\b|mos?\b", t_norm):
            notes.append(f"converted from {v:g} months to days (×30) - approximate, please verify")
            return v * 30.0, notes
    elif key == "ejection_fraction":
        if 0 < v <= 1.0 and "%" not in t_norm[:3]:
            notes.append(f"{v:g} looks like a fraction rather than a percentage - please verify")
            return v, notes
    return v, notes


def parse_features(lines: List[Line], methods: Optional[List[str]] = None) -> Dict[str, FieldResult]:
    """Parse all 12 features from text lines. Conflicting duplicates are NOT resolved by guessing."""
    out: Dict[str, FieldResult] = {}
    for key in FEATURE_KEYS:
        hits = []   # (value, notes, line)
        for ln in lines:
            r = _parse_line_for(key, ln)
            if r is not None:
                hits.append((r, ln))
        fr = FieldResult(key=key)
        if not hits:
            fr.status = "missing"
            fr.notes.append("not found in the report")
            out[key] = fr
            continue
        resolved = [(h, ln) for h, ln in hits if h[0] is not None]
        unresolved = [(h, ln) for h, ln in hits if h[0] is None]
        distinct = sorted({round(h[0], 4) for h, _ in resolved})
        if len(distinct) > 1:
            fr.status = "uncertain"
            fr.value = None
            fr.notes.append("conflicting values found in the report: " + ", ".join(f"{d:g}" for d in distinct) + " - choose the correct one")
            fr.source = " | ".join(ln.text for _, ln in resolved[:3])[:200]
            fr.method = resolved[0][1].method
            out[key] = fr
            continue
        if resolved:
            (val, notes, _), ln = resolved[0]
            fr.value, fr.status = float(val), "ok"
            fr.source, fr.method, fr.confidence = ln.text[:200], ln.method, ln.conf
            for n in notes:
                fr.flag(n)
            if unresolved and key not in BINARY_KEYS:
                pass
            if ln.method == "ocr" and ln.conf < OCR_MIN_LINE_CONF:
                fr.flag(f"low OCR confidence ({ln.conf:.0f}%) - verify against the original")
            elif ln.method == "ocr":
                # OCR can silently drop/swap a digit, so every OCR-derived value must be verified by the user.
                fr.flag(f"read by OCR (confidence {ln.conf:.0f}%) - verify digits against the original")
        else:
            (_, notes, _), ln = unresolved[0]
            fr.status, fr.value = "uncertain", None
            fr.notes += notes
            fr.source, fr.method, fr.confidence = ln.text[:200], ln.method, ln.conf
        # plausibility check against hard limits
        if fr.value is not None:
            lo, hi = HARD_LIMITS[key]
            if not (lo <= fr.value <= hi):
                fr.flag(f"{fr.value:g} is outside the plausible range {lo:,.12g}-{hi:,.12g} (possible unit or OCR error)")
                fr.value = None
        out[key] = fr
    return out


def extract_from_bytes(data: bytes, filename: str = "") -> ExtractionResult:
    """Public entry point: validate -> read -> parse.  Raises ExtractionError (safe message)."""
    lines, methods, pages, warnings = file_to_lines(data, filename)
    if not lines:
        raise ExtractionError("No readable text was found in the file. The scan may be blank, too blurry or too small.")
    fields = parse_features(lines, methods)
    if all(f.status == "missing" for f in fields.values()):
        warnings.append("None of the 12 required clinical values were recognised in this file. "
                        "It may not be a heart-failure clinical report - use manual entry or a clearer report.")
    return ExtractionResult(fields=fields, methods_used=methods, pages=pages, warnings=warnings)


# ─────────────────────────── validation for the UI ──────────────────────────
def validate_confirmed(values: Dict[str, Optional[float]], soft_ranges: Dict[str, Tuple[float, float]],
                       allow_outside_training_range: bool = False) -> Tuple[Dict[str, float], List[str]]:
    """Check user-confirmed values. Returns (clean_patient_dict, errors).

    * every one of the 12 features must be present (no defaults);
    * hard physiological limits always apply;
    * the model's training range is enforced unless the user explicitly allows otherwise.
    """
    errors: List[str] = []
    clean: Dict[str, float] = {}
    for k in FEATURE_KEYS:
        v = values.get(k)
        if v is None or (isinstance(v, float) and v != v):
            errors.append(f"{k}: value is required")
            continue
        lo, hi = HARD_LIMITS[k]
        if not (lo <= v <= hi):
            errors.append(f"{k}: {v:g} is outside the possible range {lo:g}-{hi:g}")
            continue
        if k in BINARY_KEYS and v not in (0, 1):
            errors.append(f"{k}: must be 0 or 1")
            continue
        slo, shi = soft_ranges[k]
        if not allow_outside_training_range and not (slo <= v <= shi):
            errors.append(f"{k}: {v:g} is outside the model's training range {slo:g}-{shi:g}")
            continue
        clean[k] = float(v)
    return clean, errors