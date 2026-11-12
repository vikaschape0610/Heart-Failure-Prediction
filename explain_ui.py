"""
Simple, human-language explanation of one prediction - written so that the doctor AND the patient can follow it.

Shown under the result of Tab 1:
  * the risk score and what it means,
  * what is raising the risk / what is helping (each with a one-sentence reason in everyday words),
  * a short "in simple words" summary.

No machine-learning vocabulary and no heavy medical jargon is shown to the user.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

# Usual healthy ranges - used only to say "low / normal / high" in the text (NOT for colouring).
NORMAL = {
    "creatinine_phosphokinase": (10.0, 120.0),
    "ejection_fraction": (50.0, 70.0),
    "platelets": (150_000.0, 450_000.0),
    "serum_creatinine": (0.6, 1.3),
    "serum_sodium": (135.0, 145.0),
}

# Everyday names for the tests / conditions.
FRIENDLY = {
    "serum_sodium": "Sodium in the blood",
    "serum_creatinine": "Kidney test (creatinine)",
    "ejection_fraction": "Heart pumping strength (ejection fraction)",
    "creatinine_phosphokinase": "Muscle enzyme (CPK)",
    "platelets": "Platelets (blood-clotting cells)",
    "age": "Age",
    "anaemia": "Anaemia (low red blood cells)",
    "diabetes": "Diabetes",
    "high_blood_pressure": "High blood pressure",
    "smoking": "Smoking",
}

# One-sentence reason for each finding, in simple words, by state.
REASON = {
    "serum_sodium": {
        "low": "Low sodium often means the body is holding on to extra water. This can happen when the heart is not pumping as well as it should.",
        "normal": "Sodium is normal, which is a good sign.",
        "high": "Sodium is higher than usual. This is not common in heart failure and may just be a lack of water in the body."},
    "serum_creatinine": {
        "high": "Creatinine is a waste product that healthy kidneys clear out. A high level means the kidneys are not working as well, and weaker kidneys make heart failure harder to manage.",
        "normal": "The kidney test is normal, so the kidneys are coping well.",
        "low": "Creatinine is lower than usual. This is often harmless and can just mean less muscle in the body."},
    "ejection_fraction": {
        "low": "This number shows how much blood the heart pushes out with each beat. A low number means the heart is pumping weakly.",
        "normal": "The heart is pushing out a healthy amount of blood with each beat.",
        "high": "This is higher than the usual range. It is not a worry by itself, but a heart scan can confirm it."},
    "creatinine_phosphokinase": {
        "high": "CPK is an enzyme that leaks into the blood when muscle, including heart muscle, is strained or damaged.",
        "normal": "The muscle enzyme is normal, so there is no sign of muscle strain.",
        "low": "CPK is lower than usual, which is normally nothing to worry about."},
    "platelets": {
        "high": "Platelets help the blood to clot. Too many of them can make blood clots more likely.",
        "low": "Platelets help the blood to clot. Too few of them can make bleeding more likely.",
        "normal": "The platelet count is normal."},
    "age": {
        "present": "The heart tends to struggle more as we get older, so age adds some risk.",
        "absent": "Age is not adding much risk."},
    "anaemia": {
        "present": "With anaemia there are fewer red blood cells to carry oxygen, so the heart has to work harder.",
        "absent": "There is no anaemia, so the blood carries oxygen well."},
    "diabetes": {
        "present": "Diabetes slowly damages blood vessels and the heart over the years.",
        "absent": "There is no diabetes."},
    "high_blood_pressure": {
        "present": "High blood pressure makes the heart work harder, year after year.",
        "absent": "There is no high blood pressure."},
    "smoking": {
        "present": "Smoking damages the blood vessels and lowers the oxygen in the blood.",
        "absent": "The patient does not smoke."},
}

# Short phrases used in the summary paragraph.
SAY_BAD = {
    "serum_sodium": "the sodium in the blood is low",
    "serum_creatinine": "the kidney test (creatinine) is higher than normal",
    "ejection_fraction": "the heart is pumping less strongly than normal",
    "creatinine_phosphokinase": "a muscle enzyme (CPK) is raised",
    "platelets": "the platelet count is outside the usual range",
    "age": "age adds some risk",
    "anaemia": "there is anaemia, so the heart works harder",
    "diabetes": "there is diabetes",
    "high_blood_pressure": "there is high blood pressure",
    "smoking": "the patient smokes",
}
SAY_GOOD = {
    "serum_sodium": "the sodium level is normal",
    "serum_creatinine": "the kidney test is normal",
    "ejection_fraction": "the heart is pumping a healthy amount of blood",
    "creatinine_phosphokinase": "the muscle enzyme (CPK) is normal",
    "platelets": "the platelet count is normal",
    "age": "age is not adding much risk",
    "anaemia": "there is no anaemia",
    "diabetes": "there is no diabetes",
    "high_blood_pressure": "blood pressure is not high",
    "smoking": "the patient does not smoke",
}
ADVERSE_STATES = {"low", "high", "present"}      # which states count as 'worse' (per finding, see _adverse)

_CSS = """
<style>
.dr-card{border:1px solid #232b45;background:rgba(19,24,43,.6);border-radius:12px;padding:16px 20px;margin:10px 0;color:#e2e8f0;}
.dr-head{font-size:1.25rem;font-weight:800;margin-bottom:4px;}
.dr-sub{font-size:.95rem;color:#cbd5e1;line-height:1.6;}
.dr-hi{color:#ef4444;} .dr-lo{color:#10b981;}
.dr-item{padding:9px 12px;margin:7px 0;border-radius:8px;font-size:.9rem;line-height:1.5;}
.dr-item.up{border-left:5px solid #ef4444;background:rgba(239,68,68,.08);}
.dr-item.down{border-left:5px solid #10b981;background:rgba(16,185,129,.08);}
.dr-item b{font-size:.95rem;}
.dr-chip{display:inline-block;font-size:.7rem;font-weight:700;padding:1px 9px;border-radius:10px;margin-left:8px;
         border:1px solid #64748b;color:#cbd5e1;vertical-align:middle;}
.dr-chip.strong{border-color:#ef4444;color:#fca5a5;} .dr-chip.mod{border-color:#f59e0b;color:#fcd34d;}
.dr-chip.fav{border-color:#10b981;color:#6ee7b7;}
.dr-note{font-size:.82rem;color:#94a3b8;margin-top:3px;}
.dr-say{border-left:4px solid #00d4aa;background:rgba(0,212,170,.07);border-radius:8px;padding:12px 16px;
        font-size:.95rem;line-height:1.7;color:#e2e8f0;}
.dr-step{display:flex;gap:14px;margin:12px 0 0 0;align-items:flex-start;}
.dr-num{flex:0 0 28px;height:28px;border-radius:50%;background:#ef4444;color:#fff;font-weight:800;text-align:center;line-height:28px;font-size:14px;}
.dr-num.g{background:#00d4aa;color:#06231e;}
.dr-t{font-size:.95rem;line-height:1.6;color:#dbe4f0;} .dr-k{font-weight:800;color:#fff;}
.dr-warn{border-left:4px solid #f59e0b;background:rgba(245,158,11,.08);padding:10px 14px;border-radius:8px;
         font-size:.86rem;color:#cbd5e1;margin:8px 0;}
</style>
"""


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _margin(model, sc, cols, patient) -> float:
    """Raw model output for the patient, computed the same way predict_patient does (kept for the unit tests)."""
    row = {c: 0.0 for c in sc.feature_names_in_}
    row.update(patient)
    scaled = pd.DataFrame(sc.transform(pd.DataFrame([row])), columns=sc.feature_names_in_)
    return float(np.ravel(model.predict(scaled[cols].values, output_margin=True))[0])


def _show_value(k: str, v, unit: str) -> str:
    if v is None:
        return "-"
    if "=" in unit:
        no, yes = ("Female", "Male") if k == "sex" else ("No", "Yes")
        return yes if float(v) == 1 else no
    return f"{float(v):,.6g} {unit}"


def _state(k: str, v):
    """-> 'low' | 'high' | 'normal' (tests), 'present' | 'absent' (conditions / age), or None (no rule)."""
    if v is None:
        return None
    v = float(v)
    if k in NORMAL:
        lo, hi = NORMAL[k]
        return "low" if v < lo else ("high" if v > hi else "normal")
    if k == "age":
        return "present" if v >= 65 else "absent"
    if k in ("anaemia", "diabetes", "high_blood_pressure", "smoking"):
        return "present" if v == 1 else "absent"
    return None


def _adverse(k: str, state):
    """True if the state is medically unfavourable, False if favourable / harmless, None when there is no rule."""
    if state is None or k not in FRIENDLY:
        return None
    if state == "present":
        return True
    if state == "absent" or state == "normal":
        return False
    # low / high
    if k in ("serum_sodium", "ejection_fraction"):
        return state == "low"
    if k in ("serum_creatinine", "creatinine_phosphokinase"):
        return state == "high"
    if k == "platelets":
        return True
    return None


def _status_text(k: str, v) -> str:
    s = _state(k, v)
    if s is None or k not in FRIENDLY:
        return ""
    if k in NORMAL:
        lo, hi = NORMAL[k]
        word = {"low": "low", "high": "high", "normal": "normal"}[s]
        return f"{word} (normal is {lo:,.12g} - {hi:,.12g})"
    return {"present": "yes" if k != "age" else "older age (65 or more)",
            "absent": "no" if k != "age" else "under 65"}[s]


def _strength(share: float) -> tuple[str, str]:
    if share >= 0.20:
        return "Big effect", "strong"
    if share >= 0.08:
        return "Medium effect", "mod"
    return "Small effect", ""


def _item(k, v, unit, direction, share) -> str:
    strength_txt, strength_cls = _strength(share)
    st_txt = _status_text(k, v)
    value = "" if k in ("anaemia", "diabetes", "high_blood_pressure", "smoking") else f": {html.escape(_show_value(k, v, unit))}"
    head = f"{html.escape(FRIENDLY[k])}{value}" + (f" - {html.escape(st_txt)}" if st_txt else "")
    chip = f'<span class="dr-chip {strength_cls if direction == "up" else "fav"}">{strength_txt}</span>'
    reason = REASON.get(k, {}).get(_state(k, v), "")
    return f'<div class="dr-item {direction}"><b>{head}</b>{chip}<div>{html.escape(reason)}</div></div>'


def _ok_for_summary(k: str, patient: dict, want_adverse: bool) -> bool:
    """Use a finding in the summary only if it really is unfavourable (or really is normal / absent)."""
    state = _state(k, patient.get(k))
    adv = _adverse(k, state)
    if want_adverse:
        return adv is True
    return adv is False and state in ("normal", "absent")


def _story_steps(patient: dict, up_keys, feature_meta: dict):
    """What is happening in the body, as short steps in plain words.  -> [(marker, title, body_html), ...]
    marker: 'n' = numbered red step, '!' = green note, 'g' = green 'nothing found' step."""
    A = set(up_keys)

    def val(k):
        return html.escape(_show_value(k, patient.get(k), feature_meta[k][1]))

    def stt(k):
        return _state(k, patient.get(k))

    steps = []
    if "ejection_fraction" in A and stt("ejection_fraction") == "low":
        steps.append(("n", "The heart is pumping weakly.",
                      f"The ejection fraction is <b>{val('ejection_fraction')}</b> (normal is 50 - 70%), so less blood is pushed "
                      "out with each beat. Blood then backs up behind the heart, and the other organs, the kidneys included, "
                      "get less blood than they need."))
    if "serum_creatinine" in A:
        steps.append(("n", "The kidneys are under strain.",
                      "When the heart does not supply the kidneys with enough blood, they cannot clear waste properly, so the "
                      f"<b>creatinine is up ({val('serum_creatinine')})</b>. The heart and the kidneys affect each other: "
                      "when one struggles, the other has to work harder."))
    if "serum_sodium" in A:
        steps.append(("n", "The body is holding on to extra water.",
                      "The kidneys keep water and salt when they sense low blood flow. The extra water dilutes the blood, so "
                      f"the <b>sodium reads low ({val('serum_sodium')})</b>. This is a typical sign of fluid build-up in heart "
                      "failure and often means the condition is more advanced."))
    if "creatinine_phosphokinase" in A:
        steps.append(("n", "Muscle, possibly heart muscle, is being stressed.",
                      f"The <b>raised CPK ({val('creatinine_phosphokinase')})</b> means this enzyme is leaking into the blood. "
                      "It can come from the heart, but also from other muscles or from some medicines, so it is worth "
                      "checking where it comes from."))
    if "platelets" in A:
        if stt("platelets") == "high":
            steps.append(("n", "The platelet count is high.",
                          f"Platelets are <b>{val('platelets')}</b>. Too many platelets make blood clots more likely, and a "
                          "high count can go with inflammation or low iron."))
        else:
            steps.append(("n", "The platelet count is low.",
                          f"Platelets are <b>{val('platelets')}</b>. Too few platelets make bleeding more likely, and a low "
                          "count can happen when the liver is congested or the bone marrow is affected."))
    if "anaemia" in A:
        steps.append(("n", "There is anaemia.",
                      "With fewer red blood cells to carry oxygen, the heart has to pump harder to deliver the same amount."))
    chronic = [n for k, n in (("diabetes", "diabetes"), ("high_blood_pressure", "high blood pressure"),
                              ("smoking", "smoking"), ("age", "older age")) if k in A]
    if chronic:
        steps.append(("n", "Long-standing factors are adding to the load.",
                      ", ".join(chronic[:-1]) + (" and " if len(chronic) > 1 else "") + chronic[-1]
                      + " slowly wear down the heart and blood vessels over the years."))
    if (("serum_sodium" in A) or ("serum_creatinine" in A)) and stt("ejection_fraction") in ("normal", "high") \
            and "ejection_fraction" not in A and patient.get("ejection_fraction") is not None:
        lo, hi = feature_meta["ejection_fraction"][2], feature_meta["ejection_fraction"][3]
        e = float(patient["ejection_fraction"])
        note = (" Note that this value is outside what the tool was built on, so treat that one number carefully."
                if not (lo <= e <= hi) else "")
        steps.append(("!", f"Why is the risk high when the pumping number ({val('ejection_fraction')}) looks fine?",
                      "A good pumping number does not rule out heart failure - the heart can pump well but be too stiff to fill "
                      "properly. The signs above point that way, and a heart scan would show which it is." + note))
    if not steps:
        steps.append(("g", "Nothing here points to strain on the heart or kidneys.",
                      "The results do not show clear signs of fluid build-up, kidney strain or weak pumping."))
    return steps


def render_probability_proof(*, model, sc, cols, patient: dict, res: dict, base_value=None, global_shap=None,
                             feature_meta: dict, threshold: float = 0.75,
                             test_accuracy=None, plotly_config=None):
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">🩺 What does this result mean?</div>', unsafe_allow_html=True)

    sv = np.asarray(res["shap_vals"], dtype=float).ravel()
    p = float(res["prob_death"])
    score = int(round(p * 100))
    high = p >= threshold
    thr = int(round(threshold * 100))
    tot = float(np.abs(sv).sum()) or 1.0

    # ── 1. the score ─────────────────────────────────────────────────────────
    if high:
        head = f'<span class="dr-hi">HIGH RISK</span> - score {score} out of 100'
        line = ""
    else:
        head = f'<span class="dr-lo">LOWER RISK</span> - score {score} out of 100'
        line = ""
    st.markdown(
        f'<div class="dr-card"><div class="dr-head">{head}</div></div>',
        unsafe_allow_html=True)

    # ── 2. what is raising the risk / what is helping ───────────────────────
    order = np.argsort(np.abs(sv))[::-1]

    def _conflict(i):
        """The tool leans the opposite way to normal medical thinking (or the field is not a medical finding)."""
        k = cols[i]
        if k in ("sex", "time"):
            return True
        adv = _adverse(k, _state(k, patient.get(k)))
        return (adv is True and sv[i] < 0) or (adv is False and sv[i] > 0)

    sig = [i for i in order if abs(sv[i]) / tot >= 0.02]
    up = [i for i in sig if sv[i] > 0 and not _conflict(i)][:5]
    down = [i for i in sig if sv[i] < 0 and not _conflict(i)][:4]
    odd = [i for i in sig if _conflict(i) and cols[i] not in ("sex", "time")]
    time_i = next((i for i in sig if cols[i] == "time"), None)

    c1, c2 = st.columns(2)
    for col, title, idx, d, empty in (
            (c1, "**🔴 What is raising the risk**", up, "up", "Nothing in these results raises the score noticeably."),
            (c2, "**🟢 What is helping**", down, "down", "Nothing in these results lowers the score noticeably.")):
        with col:
            st.markdown(title)
            if idx:
                st.markdown("".join(_item(cols[i], patient.get(cols[i]), feature_meta[cols[i]][1], d,
                                          abs(sv[i]) / tot) for i in idx), unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="dr-note">{empty}</div>', unsafe_allow_html=True)

    extra = []
    if time_i is not None:
        extra.append(f"<b>Follow-up days</b> ({_show_value('time', patient.get('time'), 'days')}) also changed the score, "
                     "but this is only how long the patient was tracked in the study - it is not a test result - "
                     "so it is not shown above.")
    if odd:
        names = ", ".join(html.escape(FRIENDLY.get(cols[i], feature_meta[cols[i]][0])) for i in odd)
        extra.append(f"<b>{names}</b>: the tool treats this in a way that does not match usual medical thinking, so it is "
                     "not shown above. Please use your own judgement for it.")
    if extra:
        st.markdown('<div class="dr-warn">' + "<br>".join(extra) + "</div>", unsafe_allow_html=True)

    # ── 3. what is happening in the body ─────────────────────────────────────
    rows = []
    n = 0
    for marker, title, body in _story_steps(patient, [cols[i] for i in up], feature_meta):
        if marker == "n":
            n += 1
            badge = f'<div class="dr-num">{n}</div>'
        else:
            badge = f'<div class="dr-num g">{"!" if marker == "!" else "✓"}</div>'
        rows.append(f'<div class="dr-step">{badge}<div class="dr-t"><span class="dr-k">{html.escape(title)}</span> {body}</div></div>')
    st.markdown('<div class="dr-card"><b>🧠 What is happening in this patient\'s body</b>' + "".join(rows) + "</div>",
                unsafe_allow_html=True)

    # ── 4. in simple words ───────────────────────────────────────────────────
    bad = [SAY_BAD[cols[i]] for i in up if cols[i] in SAY_BAD and _ok_for_summary(cols[i], patient, True)]
    good = [SAY_GOOD[cols[i]] for i in down if cols[i] in SAY_GOOD and _ok_for_summary(cols[i], patient, False)]

    def _bullets(xs):
        return "<ul style='margin:6px 0'>" + "".join(f"<li>{html.escape(x[0].upper() + x[1:])}.</li>" for x in xs) + "</ul>"

    if high:
        txt = (f"The tool gives this patient a score of <b>{score} out of 100</b>, which puts them in the "
               "<b>higher-risk group</b>. The main reasons are:"
               + (_bullets(bad) if bad else " several results that together point the same way. "))
        if good:
            txt += "Some good signs:" + _bullets(good)
        txt += ("A high score does <b>not</b> mean something bad will definitely happen. It means this patient needs "
                "to be watched more closely, with regular check-ups and repeat tests.")
    else:
        txt = f"The tool gives this patient a score of <b>{score} out of 100</b>, which puts them in the <b>lower-risk group</b>."
        if good:
            txt += " Good signs:" + _bullets(good)
        if bad:
            txt += "Things to keep an eye on:" + _bullets(bad)
        txt += "This is only one tool, so regular check-ups and the usual treatment are still important."
    st.markdown('<div class="dr-card"><b>💬 In simple words</b>'
                f'<div class="dr-say" style="margin-top:8px">{txt}</div></div>', unsafe_allow_html=True)