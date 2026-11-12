"""
Streamlit UI for "Upload patient report -> extract -> review -> confirm".

Used from app.py (Tab 1).  Returns the confirmed 12-feature patient dict ONLY when
the user has reviewed everything and pressed the confirm button; otherwise None.

Privacy: the uploaded bytes live only in Streamlit's in-memory upload buffer; the
extraction result is kept in st.session_state (server memory) only while the file
is attached and is dropped as soon as the file is removed.  Nothing is written to
disk or sent to any external service.  Text taken from a report is HTML-escaped
before being displayed.
"""
import hashlib
import html

import streamlit as st

import report_extractor as rx

STATE_KEY = "_report_extraction"

_CSS = """
<style>
.rx-pill{display:inline-block;font-size:.72rem;font-weight:700;padding:2px 10px;border-radius:12px;margin:2px 0 2px 0;}
.rx-ok{background:rgba(16,185,129,.15);color:#10b981;border:1px solid rgba(16,185,129,.45);}
.rx-warn{background:rgba(245,158,11,.15);color:#f59e0b;border:1px solid rgba(245,158,11,.6);}
.rx-miss{background:rgba(239,68,68,.15);color:#ef4444;border:1px solid rgba(239,68,68,.6);}
.rx-note{font-size:.74rem;color:#94a3b8;line-height:1.35;margin-bottom:6px;}
.rx-note code{font-size:.72rem;color:#cbd5e1;background:rgba(148,163,184,.12);padding:0 4px;border-radius:4px;}
.rx-rng{display:inline-block;font-size:.72rem;font-weight:700;padding:2px 10px;border-radius:12px;margin:2px 0;}
.rx-out{background:rgba(239,68,68,.22);color:#fca5a5;border:1px solid rgba(239,68,68,.9);}
.rx-lab{font-size:.85rem;color:#94a3b8;font-weight:600;margin-bottom:2px;}
.rx-val{font-size:1.5rem;font-weight:800;color:#e2e8f0;line-height:1.2;margin-bottom:4px;}
.rx-unit{font-size:.8rem;color:#94a3b8;font-weight:600;}
.rx-rows{margin:6px 0 10px 0;}
.rx-row{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;padding:7px 12px;margin:4px 0;border-radius:8px;font-size:.86rem;}
.rx-row b{font-size:.92rem;}
.rx-row.rx-out{border-left:5px solid #ef4444;}
.rx-legend span{margin-right:14px;font-size:.78rem;color:#94a3b8;}
.rx-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:middle;}
.rx-box{border-left:4px solid #00d4aa;background:rgba(0,212,170,.06);padding:10px 14px;border-radius:8px;margin:8px 0;font-size:.85rem;color:#cbd5e1;}
</style>
"""


def _pill(fr: "rx.FieldResult") -> str:
    if fr.status == "ok":
        return '<span class="rx-pill rx-ok">✔ Extracted</span>'
    if fr.status == "uncertain":
        return '<span class="rx-pill rx-warn">⚠ Needs review</span>'
    return '<span class="rx-pill rx-miss">✖ Missing - required</span>'


def _notes_html(fr: "rx.FieldResult") -> str:
    parts = [html.escape(n) for n in fr.notes if not (fr.status == "missing" and n == "not found in the report")]
    if fr.status == "missing":
        parts = ["not found in the report - please enter it"] + parts
    out = "<br>".join(parts)
    if fr.source and fr.status != "missing":
        out += ("<br>" if out else "") + "from report: <code>" + html.escape(fr.source[:120]) + "</code>"
    return f'<div class="rx-note">{out}</div>' if out else ""


_LVL_CLASS = {rx.LEVEL_OUTSIDE_TRAINING: "rx-out"}
_LVL_COLOR = {rx.LEVEL_OUTSIDE_TRAINING: "#ef4444"}


def _fmt_rng(r) -> str:
    return f"{r[0]:,.12g} – {r[1]:,.12g}"


def _flag_text(fl: "rx.RangeFlag", unit: str) -> str:
    arrow = "▲ HIGH" if fl.direction == "high" else "▼ LOW"
    return f"{arrow} - OUT OF SET RANGE ({_fmt_rng(fl.training)} {unit})"


def _range_panel(flags: dict, feature_meta: dict, title: str):
    """Coloured list of every feature whose value is out of range (name + value + why)."""
    bad = {k: f for k, f in flags.items() if f.out_of_range}
    st.markdown(f"**{title}**")
    if not bad:
        st.markdown('<div class="rx-note">✔ All values are inside the set range.</div>',
                    unsafe_allow_html=True)
        return
    rows = []
    for k, fl in bad.items():
        label, unit = feature_meta[k][0], feature_meta[k][1]
        rows.append(f'<div class="rx-row {_LVL_CLASS[fl.level]}"><b>{html.escape(label)}</b>'
                    f'<span><b>{fl.value:,.12g}</b> {html.escape(unit)}</span>'
                    f'<span>{html.escape(_flag_text(fl, unit))}</span></div>')
    st.markdown('<div class="rx-rows">' + "".join(rows) + "</div>", unsafe_allow_html=True)
    st.markdown('<div class="rx-legend"><span><i class="rx-dot" style="background:#ef4444"></i>RED = outside the set range. '
                'The result is still calculated for these values, but treat it with extra care.</span></div>',
                unsafe_allow_html=True)


def _field_css(short: str, flags: dict) -> str:
    """Colour the whole input card of every out-of-range feature (Streamlit container keys -> .st-key-<key>)."""
    css = ""
    for k, fl in flags.items():
        if fl.out_of_range:
            c = _LVL_COLOR[fl.level]
            css += (f".st-key-rxf_{short}_{k}{{border:2px solid {c};border-radius:10px;padding:8px 10px;"
                    f"background:{c}2e;}}")
    return f"<style>{css}</style>" if css else ""


def _value_text(k: str, v, unit: str) -> str:
    """Read-only text for an extracted value (Yes/No for binary fields, no trailing .0)."""
    if v is None:
        return "-"
    if k in rx.BINARY_KEYS:
        no, yes = ("Female", "Male") if k == "sex" else ("No", "Yes")
        return yes if float(v) == 1 else no
    v = float(v)
    return f"{int(v)}" if v == int(v) else f"{v:g}"


def _extract_cached(data: bytes, name: str):
    """Extract once per distinct file; keep result in session memory only."""
    digest = hashlib.sha256(data).hexdigest()
    cache = st.session_state.get(STATE_KEY)
    if cache and cache["digest"] == digest:
        return digest, cache["result"], cache["error"]
    result, error = None, None
    with st.spinner("🔎 Reading report (PDF text / OCR) - processed in memory only..."):
        try:
            result = rx.extract_from_bytes(data, name)
        except rx.ExtractionError as e:
            error = str(e)
        except Exception:                      # never leak internals / file content
            error = "Unexpected problem while reading this file. It may be corrupted - try another file or use manual entry."
    st.session_state[STATE_KEY] = {"digest": digest, "result": result, "error": error}
    return digest, result, error


def render_report_upload(feature_meta: dict):
    """Render uploader + review form. Returns confirmed patient dict or None."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">📄 Upload Patient Report</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rx-box">🔒 <b>Privacy:</b> the report is processed in memory on this server only. '
        'It is not saved and not sent to any external service; extracted values are discarded when you remove the file. '
        'Patient name / ID / date of birth are <b>not</b> extracted.</div>', unsafe_allow_html=True)
    up = st.file_uploader("Upload a patient report (PDF, JPG, JPEG or PNG - max 15 MB)",
                          type=list(rx.SUPPORTED_EXTENSIONS), accept_multiple_files=False, key="report_file")
    if up is None:
        st.session_state.pop(STATE_KEY, None)
        st.info("Upload a report to extract the 12 clinical values automatically. They are shown for checking and cannot "
                "be edited here. Prefer typing the values? Switch to **Enter values manually** above.")
        return None

    digest, result, error = _extract_cached(up.getvalue(), up.name)
    if error:
        st.error(f"❌ {error}")
        st.caption("You can try another file or switch to manual entry.")
        return None

    # ───────── summary
    meth = " + ".join({"text": "PDF text layer", "ocr": "OCR (scanned/image)"}[m] for m in result.methods_used)
    n_ok = result.n_ok
    n_unc = sum(f.status == "uncertain" for f in result.fields.values())
    n_miss = sum(f.status == "missing" for f in result.fields.values())
    st.markdown(f"**Extraction finished** · method: {meth} · pages read: {result.pages}")
    c1, c2, c3 = st.columns(3)
    c1.metric("✔ Extracted", f"{n_ok} / 12")
    c2.metric("⚠ Need review", n_unc)
    c3.metric("✖ Missing", n_miss)
    for w in result.warnings:
        st.warning(w)
    soft = {k: (float(feature_meta[k][2]), float(feature_meta[k][3])) for k in rx.FEATURE_KEYS}
    short = digest[:12]
    panel_ph = st.container()      # filled after the values are drawn
    st.markdown("### 📝 Extracted values")
    st.caption("Nothing is guessed or filled with defaults. RED = outside the set range (e.g. serum sodium below 113 or "
               "above 148) - such values are still used to calculate the result. "
               "Tick **verified** on any flagged field, then press Compute.")

    vals, verified = {}, {}
    flags = {}
    with st.container():
        grid = st.columns(3)
        for i, k in enumerate(rx.FEATURE_KEYS):
            fr = result.fields[k]
            label, unit, lo, hi = feature_meta[k]
            vals[k] = fr.value                                       # read-only: exactly what was extracted
            with grid[i % 3], st.container(key=f"rxf_{short}_{k}"):
                shown = html.escape(_value_text(k, fr.value, unit))
                unit_txt = "" if k in rx.BINARY_KEYS else f' <span class="rx-unit">{html.escape(unit)}</span>'
                st.markdown(f'<div class="rx-lab">{html.escape(label)}</div>'
                            f'<div class="rx-val">{shown}{unit_txt if fr.value is not None else ""}</div>',
                            unsafe_allow_html=True)
                fl = rx.range_flag(k, fr.value, soft[k])
                if fl is not None:
                    flags[k] = fl
                st.markdown(_pill(fr) + _notes_html(fr), unsafe_allow_html=True)
                if fr.status == "uncertain" and fr.value is not None:
                    verified[k] = st.checkbox("I verified this value", key=f"rxv_{short}_{k}")
        st.markdown("---")
        go = st.button("⚡ Compute Patient Prediction", use_container_width=True, key=f"rxgo_{short}")

    with panel_ph:
        _range_panel(flags, feature_meta, "🚩 Out-of-range values found in the report")
    st.markdown(_field_css(short, flags), unsafe_allow_html=True)

    if not go:
        return None

    problems = []
    for k in rx.FEATURE_KEYS:
        if vals[k] is None:
            problems.append(f"**{feature_meta[k][0]}** could not be read from this report. Upload a clearer or more complete "
                            "report, or switch to **Enter values manually**.")
    for k, ok in verified.items():
        if not ok:
            problems.append(f"**{feature_meta[k][0]}** is flagged - tick 'I verified this value' after checking it.")
    clean, errs = rx.validate_confirmed(vals, soft, allow_outside_training_range=True)
    for e in errs:
        key, _, msg = e.partition(": ")
        if "required" in msg:
            continue                                    # already reported above
        problems.append(f"**{feature_meta.get(key, (key,))[0]}**: {msg}")
    if problems:
        st.error("Prediction not run:\n\n" + "\n".join(f"- {p}" for p in problems))
        return None
    return clean