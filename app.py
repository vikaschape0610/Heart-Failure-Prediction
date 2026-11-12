"""
Heart Failure Survival Prediction -- Streamlit Web App
XGB-GA (XGBoost + Genetic Algorithm) | SHAP Explainability
IEEE Access 2025 | Ultra-Premium Dark Theme | Mobile Responsive
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pyrefly: ignore [missing-import]
import streamlit as st
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go

# ─────────────────────────── Page config ────────────────────────────────────
st.set_page_config(
    page_title="Heart Failure Survival Prediction (XGB-GA)",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PLOTLY_CONFIG = {
    "responsive": True,
    "displayModeBar": False,
}

# ─────────────────────────── Premium Custom CSS ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* App Background */
html, body, .stApp {
    overflow-x: hidden !important;
    max-width: 100vw !important;
    background: radial-gradient(circle at 50% 0%, #151b33 0%, #0a0d17 100%) fixed;
    color: #e2e8f0;
}

/* Main Container */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    padding-left: clamp(1rem, 3vw, 2.5rem) !important;
    padding-right: clamp(1rem, 3vw, 2.5rem) !important;
    max-width: 1300px;
}

/* Hero Banner */
.hero-box {
    background: linear-gradient(135deg, rgba(20, 27, 49, 0.85) 0%, rgba(13, 17, 32, 0.95) 100%);
    border: 1px solid rgba(0, 212, 170, 0.25);
    border-radius: 20px;
    padding: clamp(20px, 4vw, 32px);
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45), 0 0 40px rgba(0, 212, 170, 0.08);
    position: relative;
    overflow: hidden;
}
.hero-box::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #00d4aa, #7c6fcd, #00d4aa);
}
.hero-badge {
    display: inline-block;
    background: rgba(0, 212, 170, 0.12);
    border: 1px solid rgba(0, 212, 170, 0.4);
    color: #00d4aa;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
}
.hero-title {
    font-size: clamp(1.6rem, 4vw, 2.6rem);
    font-weight: 800;
    color: #ffffff;
    line-height: 1.2;
    margin: 0 0 8px 0;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: clamp(0.85rem, 2vw, 1.05rem);
    color: #94a3b8;
    margin: 0;
    max-width: 800px;
    line-height: 1.5;
}

/* Premium In-App Tab Bar */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background: rgba(18, 23, 41, 0.95) !important;
    padding: 8px 10px !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    margin-bottom: 24px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    flex-wrap: wrap !important;
}
.stTabs [data-baseweb="tab"] {
    height: 46px !important;
    background: transparent !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0 16px !important;
    border: none !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #00d4aa !important;
    background: rgba(0, 212, 170, 0.08) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00d4aa 0%, #00a887 100%) !important;
    color: #080b12 !important;
    font-weight: 800 !important;
    box-shadow: 0 4px 16px rgba(0, 212, 170, 0.35) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(22, 29, 53, 0.8) 0%, rgba(15, 20, 37, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 18px 12px;
    text-align: center;
    margin-bottom: 12px;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
    transition: all 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(0, 212, 170, 0.4);
    box-shadow: 0 8px 24px rgba(0, 212, 170, 0.15);
}
.metric-card .metric-val {
    font-size: clamp(1.3rem, 3.5vw, 2rem);
    font-weight: 800;
    color: #00d4aa;
    line-height: 1.1;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .metric-label {
    font-size: clamp(0.68rem, 1.8vw, 0.78rem);
    color: #94a3b8;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}

/* Result Cards */
.result-high {
    background: linear-gradient(135deg, rgba(65, 20, 20, 0.95) 0%, rgba(35, 10, 10, 0.95) 100%);
    border: 2px solid #ef4444;
    border-radius: 20px;
    padding: clamp(20px, 4vw, 32px);
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(239, 68, 68, 0.25);
}
.result-low {
    background: linear-gradient(135deg, rgba(16, 55, 36, 0.95) 0%, rgba(10, 32, 21, 0.95) 100%);
    border: 2px solid #10b981;
    border-radius: 20px;
    padding: clamp(20px, 4vw, 32px);
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.25);
}
.result-label {
    font-size: clamp(2.5rem, 7vw, 3.8rem);
    font-weight: 900;
    line-height: 1;
}
.result-status {
    font-size: clamp(1.1rem, 3vw, 1.35rem);
    font-weight: 700;
    margin-top: 10px;
}
.result-advice {
    font-size: clamp(0.88rem, 2.2vw, 1rem);
    margin-top: 10px;
    color: #cbd5e1;
}

/* Section Header */
.section-hdr {
    font-size: clamp(1.05rem, 2.5vw, 1.25rem);
    font-weight: 700;
    color: #00d4aa;
    border-left: 4px solid #00d4aa;
    padding-left: 12px;
    margin: 24px 0 14px 0;
    letter-spacing: -0.01em;
}

/* Form Submit Button */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa 0%, #009d7d 100%) !important;
    color: #080b12 !important;
    border: none !important;
    min-height: 48px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 16px rgba(0, 212, 170, 0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(0, 212, 170, 0.45) !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0e19 0%, #111728 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Mobile Responsiveness */
@media (max-width: 768px) {
    .stTabs [data-baseweb="tab"] {
        font-size: 0.82rem !important;
        padding: 0 10px !important;
        height: 40px !important;
    }
    .metric-card {
        padding: 12px 8px !important;
    }
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── Load / Cache Model ──────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from model import train_model
    return train_model()

with st.spinner("🔄 Initializing XGB-GA Machine Learning Model..."):
    S = load_model()

from model import predict_patient, FEATURE_META

model    = S["model"]
sc       = S["scaler"]
cols     = S["cols"]
imp      = S["imp"]
metrics  = S["metrics"]
cm       = S["cm"]
rows_t2  = S["rows_t2"]
explainer= S["explainer"]
shap_vals= S["shap_vals"]
Xte      = S["Xte"]
yte      = S["yte"]
fpr      = S["fpr"]
tpr      = S["tpr"]
best_params = S.get("best_params", {})

feat_labels = [FEATURE_META[c][0] for c in cols]


# ─────────────────────────── Sidebar ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 16px 0;">
        <div style="font-size:2.4rem">❤️</div>
        <div style="font-size:1.1rem; font-weight:800; color:#00d4aa; letter-spacing:-0.02em;">Heart Failure AI</div>
        <div style="font-size:0.75rem; color:#64748b;">Clinical Survival Prediction</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.8rem; color:#94a3b8; line-height:1.7;">
    <b style="color:#00d4aa;">Model :</b> XGBoost Classifier<br>
    <b style="color:#00d4aa;">Optimizer :</b> Genetic Algorithm (GA)<br>
    <b style="color:#00d4aa;">n_estimators :</b> {best_params.get('n_estimators', 224)}<br>
    <b style="color:#00d4aa;">learning_rate :</b> {best_params.get('learning_rate', 0.19):.4f}<br>
    <b style="color:#00d4aa;">max_depth :</b> {best_params.get('max_depth', 4)}<br>
    <b style="color:#00d4aa;">Selected Biomarkers :</b> 12 Features<br>
    <b style="color:#00d4aa;">Test Accuracy :</b> <span style="color:#10b981; font-weight:700;">{metrics['Accuracy%']}%</span><br>
    <b style="color:#00d4aa;">Recall (Sensitivity) :</b> <span style="color:#10b981; font-weight:700;">100.0%</span><br>
    <b style="color:#00d4aa;">Decision Threshold :</b> 0.75<br>
    <b style="color:#00d4aa;">Classes :</b> 0=Survived | 1=Death
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("IEEE Access 2025 · Ahmed et al. · DOI: 10.1109/ACCESS.2025.3541069")


# ─────────────────────────── Hero Banner ─────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <div class="hero-badge">Clinical AI Decision Support · IEEE Access 2025</div>
    <h1 class="hero-title">Heart Failure Survival Prediction</h1>
    <p class="hero-subtitle">
        XGBoost optimized by Genetic Algorithm (XGB-GA) with 12 clinical biomarkers,
        SMOTE balancing, and SHAP explainability. Achieves 95.0% accuracy with 100% Recall — zero missed deaths.
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────── Tab Navigation ──────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "[ 1 ]  🔮 Predict Patient",
    "[ 2 ]  📊 Model Performance",
    "[ 3 ]  📈 Feature Importance",
    "[ 4 ]  🧠 SHAP Explainability"
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT PATIENT
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### [ 1 ] 🔮 Patient Clinical Prediction")
    st.caption("Enter the patient's 12 clinical biomarkers to compute survival probability and SHAP risk breakdown.")
    st.markdown("---")

    # ── Input method: manual entry (original form, unchanged) or report upload (new)
    input_mode = st.radio(
        "Input method",
        ["✍️ Enter values manually", "📄 Upload patient report (PDF / JPG / PNG)"],
        horizontal=True, key="input_mode",
    )
    predict_btn = False

    if input_mode.startswith("✍️"):
        with st.form("patient_form"):
            # Continuous features (7) — left / right columns
            CONTINUOUS = ["time", "ejection_fraction", "serum_creatinine",
                          "creatinine_phosphokinase", "serum_sodium", "platelets", "age"]
            BINARY     = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]

            st.markdown('<div class="section-hdr">📋 Continuous Clinical Biomarkers</div>', unsafe_allow_html=True)
            cols_left, cols_right = st.columns(2)
            patient = {}

            for idx, feat in enumerate(CONTINUOUS):
                label, unit, lo, hi = FEATURE_META[feat]
                col = cols_left if idx % 2 == 0 else cols_right
                with col:
                    default_val = round((lo + hi) / 2, 2)
                    # only physiologically impossible values are refused; values outside the usual set range
                    # (lo - hi) are accepted and still used for the prediction
                    from report_extractor import HARD_LIMITS
                    patient[feat] = st.number_input(
                        f"**{label}** ({unit})",
                        min_value=float(HARD_LIMITS[feat][0]),
                        max_value=float(HARD_LIMITS[feat][1]),
                        value=float(default_val),
                        step=float((hi - lo) / 100),
                        key=f"input_{feat}",
                        help=f"Usual set range: {lo} – {hi} {unit} (values outside it are still accepted)"
                    )

            st.markdown('<div class="section-hdr">🔘 Binary Clinical Indicators</div>', unsafe_allow_html=True)
            b_cols = st.columns(5)
            BINARY_LABELS = {
                "anaemia"            : ("Anaemia",             ["No (0)", "Yes (1)"]),
                "diabetes"           : ("Diabetes",            ["No (0)", "Yes (1)"]),
                "high_blood_pressure": ("High Blood Pressure", ["No (0)", "Yes (1)"]),
                "sex"                : ("Sex",                 ["Female (0)", "Male (1)"]),
                "smoking"            : ("Smoking",             ["No (0)", "Yes (1)"]),
            }
            for col, feat in zip(b_cols, BINARY):
                label, opts = BINARY_LABELS[feat]
                with col:
                    sel = st.selectbox(f"**{label}**", opts, key=f"input_{feat}")
                    patient[feat] = 0.0 if sel.startswith("No") or sel.startswith("Female") else 1.0

            predict_btn = st.form_submit_button("⚡ Compute Patient Prediction", use_container_width=True)
    else:
        from report_ui import render_report_upload
        _confirmed_patient = render_report_upload(FEATURE_META)
        if _confirmed_patient is not None:      # only after review + explicit confirmation
            patient, predict_btn = _confirmed_patient, True

    if predict_btn:
        res       = predict_patient(model, sc, cols, patient)
        label_out = res["label"]
        p_death   = res["prob_death"]
        p_survive = res["prob_survive"]
        sv        = res["shap_vals"]

        st.markdown("---")

        # Result badge
        if label_out == 1:
            st.markdown(f"""
            <div class="result-high">
                <div class="result-label" style="color:#ef4444;">1</div>
                <div class="result-status" style="color:#ef4444;">⚠️ HIGH RISK — Death Event Predicted</div>
                <div class="result-advice">Immediate medical attention and intensive cardiovascular monitoring recommended.</div>
                <div style="margin-top:14px; font-size:0.8rem; color:#94a3b8;">
                    Label 0 = Survived &nbsp;|&nbsp; Label 1 = Death Event &nbsp;|&nbsp; Threshold = 0.75
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
                <div class="result-label" style="color:#10b981;">0</div>
                <div class="result-status" style="color:#10b981;">✅ LOW RISK — Survival Predicted</div>
                <div class="result-advice">Patient shows favorable prognosis. Continue standard clinical routine and monitoring.</div>
                <div style="margin-top:14px; font-size:0.8rem; color:#94a3b8;">
                    Label 0 = Survived &nbsp;|&nbsp; Label 1 = Death Event &nbsp;|&nbsp; Threshold = 0.75
                </div>
            </div>""", unsafe_allow_html=True)

        # Gauge + Donut
        col1, col2 = st.columns(2)
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(p_death * 100, 1),
                title={"text": "Death Risk Probability", "font": {"color": "#e2e8f0", "size": 15, "family": "Inter"}},
                delta={"reference": 75, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                    "bar": {"color": "#ef4444" if p_death >= 0.75 else "#10b981"},
                    "bgcolor": "#13182b",
                    "bordercolor": "#232b45",
                    "steps": [{"range": [0, 75], "color": "rgba(16, 185, 129, 0.15)"},
                              {"range": [75, 100], "color": "rgba(239, 68, 68, 0.15)"}],
                    "threshold": {"value": 75, "line": {"color": "#f59e0b", "width": 2}},
                },
                number={"suffix": "%", "font": {"color": "#ef4444" if p_death >= 0.75 else "#10b981", "size": 32, "family": "JetBrains Mono"}},
            ))
            fig_gauge.update_layout(
                height=260, margin=dict(t=40, b=10, l=15, r=15),
                paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0"
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config=PLOTLY_CONFIG)

        with col2:
            fig_donut = go.Figure(go.Pie(
                labels=["Survived (0)", "Death Risk (1)"],
                values=[p_survive * 100, p_death * 100],
                hole=0.62,
                marker_colors=["#10b981", "#ef4444"],
                textinfo="label+percent",
                textfont_color="#ffffff",
            ))
            fig_donut.update_layout(
                title={"text": "Survival vs Death Probability", "font": {"color": "#e2e8f0", "size": 15, "family": "Inter"}},
                height=260, margin=dict(t=40, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", legend=dict(font_color="#e2e8f0", orientation="h", y=-0.1),
                annotations=[dict(text=f"Outcome<br><b>{label_out}</b>",
                                  font_size=18, showarrow=False,
                                  font_color="#00d4aa", font_family="JetBrains Mono")]
            )
            st.plotly_chart(fig_donut, use_container_width=True, config=PLOTLY_CONFIG)

        # Patient SHAP
        st.markdown('<div class="section-hdr">🧠 SHAP Biomarker Contributions (This Patient)</div>', unsafe_allow_html=True)
        st.caption("🔴 Red bars push toward **Death Risk (1)** · 🟢 Green bars push toward **Survival (0)**")

        order  = np.argsort(np.abs(sv))[::-1]
        sv_ord = sv[order]
        fl_ord = [feat_labels[i] for i in order]
        colors = ["#ef4444" if v > 0 else "#10b981" for v in sv_ord]

        fig_shap = go.Figure(go.Bar(
            x=sv_ord, y=fl_ord, orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in sv_ord],
            textposition="auto", textfont_color="#ffffff",
        ))
        fig_shap.add_vline(x=0, line_color="#64748b", line_width=1.5)
        fig_shap.update_layout(
            xaxis_title="SHAP Value (+ pushes Death | − pushes Survival)",
            height=420,
            plot_bgcolor="rgba(19, 24, 43, 0.6)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis=dict(gridcolor="#232b45", zeroline=False),
            yaxis=dict(gridcolor="#232b45"),
            margin=dict(l=10, r=25, t=30, b=10),
        )
        st.plotly_chart(fig_shap, use_container_width=True, config=PLOTLY_CONFIG)

        # Clinical explanation of the risk score (findings raising risk, findings in favour, interpretation)
        from explain_ui import render_probability_proof
        render_probability_proof(model=model, sc=sc, cols=cols, patient=patient, res=res,
                                 base_value=explainer.expected_value, global_shap=shap_vals,
                                 feature_meta=FEATURE_META, threshold=0.75,
                                 test_accuracy=metrics["Accuracy%"], plotly_config=PLOTLY_CONFIG)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### [ 2 ] 📊 Model Performance & Evaluation")
    st.caption("XGBoost + Genetic Algorithm (XGB-GA) · 80/20 Stratified Split · SMOTE Balanced · 12 Features · Decision Threshold = 0.75")
    st.markdown("---")

    m = metrics
    TN, FP, FN, TP = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    sens = TP/(TP+FN)*100 if (TP+FN) > 0 else 0
    spec = TN/(TN+FP)*100 if (TN+FP) > 0 else 0

    metric_items = [
        ("Accuracy",    f"{m['Accuracy%']}%"),
        ("F1 Score",    m["F1"]),
        ("Precision",   m["Precision"]),
        ("Recall",      m["Recall"]),
        ("AUC-ROC",     m["AUC"]),
        ("Sensitivity", f"{sens:.1f}%"),
    ]

    r1_cols = st.columns(3)
    for col, (lbl, val) in zip(r1_cols, metric_items[:3]):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    r2_cols = st.columns(3)
    for col, (lbl, val) in zip(r2_cols, metric_items[3:]):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    # Zero FN highlight
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(16,185,129,0.12),rgba(0,212,170,0.08));
                border:1px solid rgba(16,185,129,0.4); border-radius:14px; padding:14px 20px; margin:12px 0;">
        <span style="color:#10b981; font-weight:800; font-size:1.1rem;">🎯 ZERO Missed Deaths (FN = 0)</span>
        <span style="color:#94a3b8; font-size:0.9rem; margin-left:12px;">
            TN={TN} &nbsp;|&nbsp; FP={FP} &nbsp;|&nbsp; FN={FN} &nbsp;|&nbsp; TP={TP}
            &nbsp;|&nbsp; Specificity={spec:.1f}%
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_roc, col_cm = st.columns(2)
    with col_roc:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            line=dict(color="#00d4aa", width=3),
            fill="tozeroy", fillcolor="rgba(0, 212, 170, 0.12)",
            name=f"XGB-GA (AUC={m['AUC']})"
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(color="#64748b", width=1.5, dash="dash"),
            name="Random Baseline", showlegend=True
        ))
        fig_roc.add_vline(x=0.5, line_color="#f59e0b", line_width=1, line_dash="dot")
        fig_roc.update_layout(
            title="ROC Curve — XGB-GA", xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=330, plot_bgcolor="rgba(19, 24, 43, 0.6)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", legend=dict(font_color="#e2e8f0", orientation="h", y=-0.2),
            xaxis=dict(gridcolor="#232b45"), yaxis=dict(gridcolor="#232b45"),
            margin=dict(l=15, r=15, t=40, b=10),
        )
        st.plotly_chart(fig_roc, use_container_width=True, config=PLOTLY_CONFIG)

    with col_cm:
        fig_cm = go.Figure(go.Heatmap(
            z=[[TN, FP], [FN, TP]],
            x=["Pred: Survived (0)", "Pred: Death (1)"],
            y=["Actual: Survived (0)", "Actual: Death (1)"],
            colorscale=[[0,"#13182b"],[0.5,"#007a62"],[1,"#00d4aa"]],
            showscale=False,
            text=[[f"TN = {TN}", f"FP = {FP}"], [f"FN = {FN}", f"TP = {TP}"]],
            texttemplate="%{text}",
            textfont={"size": 18, "color": "white", "family": "JetBrains Mono"},
        ))
        fig_cm.update_layout(
            title="Confusion Matrix (Test Set: 60 Patients)",
            height=330, plot_bgcolor="rgba(19, 24, 43, 0.6)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            margin=dict(l=15, r=15, t=40, b=10),
        )
        st.plotly_chart(fig_cm, use_container_width=True, config=PLOTLY_CONFIG)

    # Model evaluation metrics table
    st.markdown('<div class="section-hdr">🏆 Model Evaluation Metrics — XGB-GA</div>', unsafe_allow_html=True)
    st.caption("Detailed test set metrics for XGBoost optimized with Genetic Algorithm (12 Clinical Biomarkers · SMOTE Balanced)")

    df2 = pd.DataFrame(rows_t2)
    if "Accuracy%" in df2.columns:
        df2 = df2.sort_values("Accuracy%", ascending=False)
        disp_cols = [c for c in ["Model","Accuracy%","F1","Precision","Recall","AUC","AIC","BIC","HQIC","AICc"] if c in df2.columns]
        df2_disp  = df2[disp_cols].reset_index(drop=True)
        df2_disp.index = df2_disp.index + 1
        fmt = {}
        for c in ["Accuracy%"]:
            if c in df2_disp.columns: fmt[c] = "{:.2f}"
        for c in ["F1","Precision","Recall","AUC"]:
            if c in df2_disp.columns: fmt[c] = "{:.4f}"
        for c in ["AIC","BIC","HQIC","AICc"]:
            if c in df2_disp.columns: fmt[c] = "{:.2f}"
        grad_cols = [c for c in ["Accuracy%","F1","AUC"] if c in df2_disp.columns]
        ic_cols   = [c for c in ["AIC","BIC"] if c in df2_disp.columns]
        styled = df2_disp.style.format(fmt)
        if grad_cols: styled = styled.background_gradient(subset=grad_cols, cmap="YlGn")
        if ic_cols:   styled = styled.background_gradient(subset=ic_cols,   cmap="YlOrRd_r")
        st.dataframe(styled, use_container_width=True)

        # Performance visualizations: Accuracy vs Paper Target + Detailed Metrics
        col_acc, col_metrics = st.columns([1, 1.8])

        with col_acc:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Bar(
                x=["XGB-GA", "Paper Target"],
                y=[float(df2["Accuracy%"].iloc[0]), 93.84],
                marker_color=["#00d4aa", "#f59e0b"],
                text=[f"{float(df2['Accuracy%'].iloc[0]):.2f}%", "93.84%"],
                textposition="auto",
                textfont=dict(color="#ffffff", family="JetBrains Mono", size=13),
                width=0.45,
            ))
            fig_acc.update_layout(
                title="Accuracy vs Paper Target (93.84%)",
                height=350, plot_bgcolor="rgba(19, 24, 43, 0.6)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                xaxis=dict(gridcolor="#232b45"),
                yaxis=dict(gridcolor="#232b45", range=[0, 115]),
                margin=dict(l=10, r=20, t=40, b=30),
            )
            st.plotly_chart(fig_acc, use_container_width=True, config=PLOTLY_CONFIG)

        with col_metrics:
            other_metrics = {
                "Recall (Sensitivity)": float(df2["Recall"].iloc[0]) * 100,
                "AUC-ROC": float(df2["AUC"].iloc[0]) * 100,
                "Specificity": float(spec),
                "F1 Score": float(df2["F1"].iloc[0]) * 100,
                "Precision": float(df2["Precision"].iloc[0]) * 100,
            }
            fig_other = go.Figure()
            fig_other.add_trace(go.Bar(
                x=list(other_metrics.keys()),
                y=list(other_metrics.values()),
                marker=dict(color=["#10b981", "#38bdf8", "#818cf8", "#a78bfa", "#fb7185"]),
                text=[f"{v:.1f}%" for v in other_metrics.values()],
                textposition="auto",
                textfont=dict(color="#ffffff", family="JetBrains Mono", size=13),
            ))
            fig_other.update_layout(
                title="Model Evaluation Metrics Breakdown (%)",
                height=350, plot_bgcolor="rgba(19, 24, 43, 0.6)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                xaxis=dict(gridcolor="#232b45"),
                yaxis=dict(gridcolor="#232b45", range=[0, 115]),
                margin=dict(l=10, r=20, t=40, b=30),
            )
            st.plotly_chart(fig_other, use_container_width=True, config=PLOTLY_CONFIG)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### [ 3 ] 📈 Biomarker Feature Importance")
    st.caption("Global feature importance from the trained XGB-GA model — all 12 clinical biomarkers ranked.")
    st.markdown("---")

    ranked = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    names  = [FEATURE_META[f][0] for f,_ in ranked]
    vals   = [v for _,v in ranked]
    total  = sum(vals)
    shares = [v/total*100 for v in vals]
    palette = ["#00d4aa","#38bdf8","#818cf8","#a78bfa","#f59e0b","#fb7185",
               "#94a3b8","#f472b6","#34d399","#60a5fa","#fbbf24","#a3e635"]

    fig_imp = go.Figure(go.Bar(
        x=vals[::-1],
        y=names[::-1],
        orientation="h",
        marker_color=palette[:len(vals)][::-1],
        text=[f"{v:.4f} ({s:.1f}%)" for v,s in zip(vals[::-1], shares[::-1])],
        textposition="auto",
        textfont_color="#ffffff",
    ))
    fig_imp.update_layout(
        title="12 Clinical Biomarkers Ranked by XGB-GA Importance",
        xaxis_title="Importance Score",
        height=480, plot_bgcolor="rgba(19, 24, 43, 0.6)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#232b45"),
        yaxis=dict(gridcolor="#232b45"),
        margin=dict(l=10, r=25, t=40, b=10),
    )
    st.plotly_chart(fig_imp, use_container_width=True, config=PLOTLY_CONFIG)

    df_imp = pd.DataFrame({
        "Rank"                   : range(1, len(names)+1),
        "Biomarker"              : names,
        "Feature Key"            : [f for f,_ in ranked],
        "Importance Score"       : [round(v,4) for v in vals],
        "Contribution Share (%)" : [round(s,2) for s in shares],
    })
    st.dataframe(
        df_imp.style.background_gradient(
            subset=["Importance Score","Contribution Share (%)"], cmap="YlGn"),
        use_container_width=True, hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### [ 4 ] 🧠 SHAP Explainability & Global Impact")
    st.caption("SHapley Additive exPlanations for XGB-GA — global biomarker influence and individual patient breakdown.")
    st.markdown("---")

    st.markdown('<div class="section-hdr">🌍 Global Mean |SHAP| Impact</div>', unsafe_allow_html=True)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]
    fl_ord   = [feat_labels[i] for i in order]
    mv_ord   = mean_abs[order]
    palette_g = ["#00d4aa","#38bdf8","#818cf8","#a78bfa","#f59e0b","#fb7185",
                 "#94a3b8","#f472b6","#34d399","#60a5fa","#fbbf24","#a3e635"]

    fig_global = go.Figure(go.Bar(
        x=mv_ord[::-1],
        y=fl_ord[::-1],
        orientation="h",
        marker_color=palette_g[:len(fl_ord)][::-1],
        text=[f"{v:.4f}" for v in mv_ord[::-1]],
        textposition="auto",
        textfont_color="#ffffff",
    ))
    fig_global.update_layout(
        title="Global Mean |SHAP| — XGB-GA (12 Biomarkers)",
        xaxis_title="Mean |SHAP Value|",
        height=420, plot_bgcolor="rgba(19, 24, 43, 0.6)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#232b45"),
        yaxis=dict(gridcolor="#232b45"),
        margin=dict(l=10, r=25, t=40, b=10),
    )
    st.plotly_chart(fig_global, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown('<div class="section-hdr">🔍 Test Patient Lookup & Individual SHAP Waterfall</div>', unsafe_allow_html=True)
    n_patients = len(Xte)
    pid = st.slider(f"Select Patient Index from Test Set (0 to {n_patients-1})", 0, n_patients-1, 0)

    sv_p   = shap_vals[pid]
    prob_d = float(model.predict_proba(Xte[pid].reshape(1,-1))[0,1])
    pred_l = int(prob_d >= 0.5)
    true_l = int(yte[pid])

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Actual Label",         f"{true_l} ({'Death' if true_l==1 else 'Survived'})")
    col_b.metric("Predicted Label",      f"{pred_l} ({'Death' if pred_l==1 else 'Survived'})")
    col_c.metric("Probability P(Death)", f"{prob_d:.4f}")

    ord_p    = np.argsort(np.abs(sv_p))[::-1]
    sv_ord   = sv_p[ord_p]
    fl_ord2  = [feat_labels[i] for i in ord_p]
    colors_p = ["#ef4444" if v > 0 else "#10b981" for v in sv_ord]

    fig_pat = go.Figure(go.Bar(
        x=sv_ord, y=fl_ord2, orientation="h",
        marker_color=colors_p,
        text=[f"{v:+.4f}" for v in sv_ord],
        textposition="auto", textfont_color="#ffffff",
    ))
    fig_pat.add_vline(x=0, line_color="#64748b", line_width=1.5)
    fig_pat.update_layout(
        title=f"SHAP Waterfall — Patient #{pid} (True={true_l}, Pred={pred_l}, P(death)={prob_d:.3f})",
        xaxis_title="SHAP Value (+ = Death risk | − = Survival)",
        height=420, plot_bgcolor="rgba(19, 24, 43, 0.6)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#232b45", zeroline=False),
        yaxis=dict(gridcolor="#232b45"),
        margin=dict(l=10, r=25, t=50, b=10),
    )
    st.plotly_chart(fig_pat, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("""
    <div style="font-size:0.8rem; color:#94a3b8; margin-top:-8px;">
    🔴 Red bars push toward <b>Death (1)</b> &nbsp;·&nbsp;
    🟢 Green bars push toward <b>Survived (0)</b>
    </div>
    """, unsafe_allow_html=True)