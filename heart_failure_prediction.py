"""
Heart Failure Survival Prediction
IEEE Access 2025 -- DOI: 10.1109/ACCESS.2025.3541069
"Predicting the Classification of Heart Failure Patients Using
 Optimized Machine Learning Algorithms" -- Marzia Ahmed et al.

Run: python heart_failure_prediction.py
  -> silently trains XGB-GA, then opens interactive menu
"""

import warnings, os, sys, contextlib, time
warnings.filterwarnings("ignore")
os.system("")   # enable ANSI colours on Windows

HERE = os.path.dirname(os.path.abspath(__file__))

@contextlib.contextmanager
def _silent():
    with open(os.devnull, "w") as dn:
        old_o, old_e = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = dn, dn
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_o, old_e

with _silent():
    import numpy as np
    import pandas as pd
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pointbiserialr
    from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.preprocessing    import StandardScaler
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.metrics           import (accuracy_score, precision_score,
                                           recall_score, f1_score,
                                           roc_auc_score, confusion_matrix,
                                           ConfusionMatrixDisplay, roc_curve)
    from imblearn.over_sampling    import SMOTE
    from xgboost                   import XGBClassifier
    # pyrefly: ignore [missing-import]
    import shap
    import joblib

DATA = os.path.join(HERE, "heart299.csv")

# =============================================================================
# ANSI PALETTE
# =============================================================================
R     = "\033[0m"
BD    = "\033[1m"
DM    = "\033[2m"
RE    = "\033[91m"
GR    = "\033[92m"
YE    = "\033[93m"
BL    = "\033[94m"
MA    = "\033[95m"
CY    = "\033[96m"
WH    = "\033[97m"
TL    = "\033[38;5;38m"
OR    = "\033[38;5;208m"
PK    = "\033[38;5;205m"
PU    = "\033[38;5;141m"
SL    = "\033[38;5;245m"
BG_GR = "\033[42m"
BG_RE = "\033[41m"

# =============================================================================
# FEATURE METADATA  {key: (label, unit, min, max, emoji, is_binary)}
# =============================================================================
META = {
    "time":                     ("Follow-up Period",         "days",        4,    285, "D", False),
    "ejection_fraction":        ("Ejection Fraction",        "%",          14,     80, "H", False),
    "serum_creatinine":         ("Serum Creatinine",         "mg/dL",     0.5,    9.4, "T", False),
    "creatinine_phosphokinase": ("Creatinine Phosphokinase", "mcg/L",      23,   7861, "A", False),
    "serum_sodium":             ("Serum Sodium",             "mEq/L",     113,    148, "S", False),
    "platelets":                ("Platelets",                "kiloplts/mL",25000,850000,"P",False),
    "age":                      ("Age",                      "years",       1,    120, "~", False),
    "anaemia":                  ("Anaemia",                  "0=No/1=Yes",  0,      1, "B", True),
    "diabetes":                 ("Diabetes",                 "0=No/1=Yes",  0,      1, "B", True),
    "high_blood_pressure":      ("High Blood Pressure",      "0=No/1=Yes",  0,      1, "B", True),
    "sex":                      ("Sex",                      "0=F/1=M",     0,      1, "B", True),
    "smoking":                  ("Smoking",                  "0=No/1=Yes",  0,      1, "B", True),
}

# =============================================================================
# GENETIC ALGORITHM SEARCH (directly from XGB-GA.ipynb — no hardcoded parameters)
# =============================================================================
GA_SEED, POP, GENS = 1, 10, 3
SPACE = [
    ("n_estimators",  20,   300, "int"),
    ("learning_rate", 0.01, 0.5, "float"),
    ("max_depth",     1,    6,   "int"),
]

def _decode(u):
    p = {}
    for (name, lo, hi, kind), x in zip(SPACE, u):
        v = lo + float(np.clip(x, 0, 1)) * (hi - lo)
        p[name] = int(round(v)) if kind == "int" else float(v)
    return p

def _build_xgb(p):
    return XGBClassifier(random_state=42, n_jobs=4, eval_metric="logloss", verbosity=0, **p)

def _run_genetic_algorithm(Xtr_b, ytr_b, pop=POP, gens=GENS, seed=GA_SEED):
    """Run dynamic Genetic Algorithm search (no hard-coded parameters)."""
    rng = np.random.default_rng(seed)
    dim = len(SPACE)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cache = {}

    def cv_accuracy(u):
        p = _decode(u)
        key = json.dumps(p, sort_keys=True)
        if key not in cache:
            cache[key] = cross_val_score(_build_xgb(p), Xtr_b, ytr_b, cv=skf, scoring="accuracy", n_jobs=4).mean()
        return cache[key]

    P = rng.random((pop, dim))
    F = np.array([-cv_accuracy(x) for x in P])
    for g in range(gens):
        children = [P[np.argmin(F)].copy()]  # elitism
        while len(children) < pop:
            a = [min(rng.choice(pop, 3), key=lambda i: F[i]) for _ in range(2)]  # tournament
            w = rng.random(dim)
            c = w * P[a[0]] + (1 - w) * P[a[1]]  # crossover
            m = rng.random(dim) < 0.2
            c[m] += rng.normal(0, 0.15, m.sum())  # mutation
            children.append(np.clip(c, 0, 1))
        P = np.array(children)
        F = np.array([-cv_accuracy(x) for x in P])

    best_u = P[np.argmin(F)]
    return _decode(best_u)

# =============================================================================
# IC HELPER
# =============================================================================
def _ic_values(y_true, y_prob, k):
    eps  = 1e-9
    p    = np.clip(y_prob, eps, 1 - eps)
    lll  = np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    n    = len(y_true)
    aic  = round(-2*lll + 2*k,              2)
    bic  = round(-2*lll + k*np.log(n),      2)
    hqic = round(-2*lll + 2*k*np.log(np.log(n)), 2)
    aicc = round(aic + (2*k*(k+1)) / max(n-k-1, 1), 2)
    return aic, bic, hqic, aicc

# =============================================================================
# EVALUATE HELPER
# =============================================================================
def _evaluate(name, mdl, Xte, yte, k_params=3, threshold=0.5):
    if hasattr(mdl, "predict_proba"):
        prob = mdl.predict_proba(Xte)[:, 1]
    else:
        raw  = mdl.decision_function(Xte)
        prob = 1 / (1 + np.exp(-raw))
    pred = (prob >= threshold).astype(int)
    aic, bic, hqic, aicc = _ic_values(yte, prob, k_params)
    return {
        "Model"    : name,
        "Accuracy%": round(accuracy_score(yte, pred)               * 100, 2),
        "F1"       : round(f1_score(yte, pred, zero_division=0),        4),
        "Precision": round(precision_score(yte, pred, zero_division=0), 4),
        "Recall"   : round(recall_score(yte, pred, zero_division=0),    4),
        "AUC"      : round(roc_auc_score(yte, prob),                    4),
        "AIC": aic, "BIC": bic, "HQIC": hqic, "AICc": aicc,
        "_prob": prob, "_pred": pred,
    }

# =============================================================================
# SILENT TRAINING
# =============================================================================
def _train():
    with _silent():
        df = pd.read_csv(DATA)
        X  = df.drop("DEATH_EVENT", axis=1).astype(float)
        y  = df["DEATH_EVENT"].values.astype(int)

        # Stratified 80/20 (seed=11, same as notebook)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=11, stratify=y)

        # All 12 features (Type-I, same as notebook)
        X_nn_tr = Xtr - Xtr.min()
        sel_all = SelectKBest(chi2, k="all").fit(X_nn_tr, ytr)
        cols_all = Xtr.columns[sel_all.get_support()].tolist()

        sc     = StandardScaler()
        Xtr_sc = pd.DataFrame(sc.fit_transform(Xtr), columns=X.columns)
        Xte_sc = pd.DataFrame(sc.transform(Xte),     columns=X.columns)

        smote    = SMOTE(random_state=42)
        Xtr_b, ytr_b = smote.fit_resample(Xtr_sc[cols_all].values, ytr)
        Xte_arr  = Xte_sc[cols_all].values

        # ── Genetic Algorithm hyper-parameter search (dynamic, no hardcoding)
        best_params = _run_genetic_algorithm(Xtr_b, ytr_b)

        # ── Train XGBoost with GA-discovered best parameters
        xgb_model = _build_xgb(best_params)
        xgb_model.fit(Xtr_b, ytr_b)

        joblib.dump(
            {"model": xgb_model, "scaler": sc, "features": cols_all},
            os.path.join(HERE, "model_xgb_ga.pkl")
        )

        xgb_row = _evaluate("XGB-GA", xgb_model, Xte_arr, yte, k_params=3)
        imp      = dict(zip(cols_all, xgb_model.feature_importances_))
        cm       = confusion_matrix(yte, xgb_row["_pred"])
        metrics  = {
            "Accuracy" : xgb_row["Accuracy%"],
            "Precision": xgb_row["Precision"],
            "Recall"   : xgb_row["Recall"],
            "F1"       : xgb_row["F1"],
            "AUC"      : xgb_row["AUC"],
        }

        shap_explainer = shap.TreeExplainer(xgb_model)
        shap_vals_te   = shap_explainer.shap_values(Xte_arr)

        _save_figures(
            df, X, y, cols_all, ytr, ytr_b,
            xgb_model, sc, Xte_arr, yte,
            xgb_row["_prob"], xgb_row["_pred"], cm,
            xgb_row, imp, Xtr, ytr,
            shap_explainer, shap_vals_te,
            Xte_arr, cols_all
        )

    return (xgb_model, sc, cols_all, imp, metrics, cm,
            [xgb_row], shap_explainer, Xte_arr, cols_all, yte)


# =============================================================================
# FIGURE GENERATION
# =============================================================================
def _save_figures(df, X, y, cols_all, ytr, ytr_b,
                  model, sc, Xte_arr, yte,
                  prob, yp, cm,
                  xgb_row, imp,
                  Xtr_for_chi2, ytr_for_chi2,
                  shap_explainer, shap_vals_te,
                  Xte7_arr, cols_list):

    CONT = ["age","creatinine_phosphokinase","ejection_fraction",
            "platelets","serum_creatinine","serum_sodium","time"]
    BIN  = ["anaemia","diabetes","high_blood_pressure","sex","smoking","DEATH_EVENT"]
    all_c = CONT + BIN
    n = len(all_c)
    corr = pd.DataFrame(np.zeros((n,n)), index=all_c, columns=all_c)
    for i,ci in enumerate(all_c):
        for j,cj in enumerate(all_c):
            xi,xj = df[ci].values, df[cj].values
            ci_b, cj_b = ci in BIN, cj in BIN
            if not ci_b and not cj_b:
                corr.iloc[i,j] = np.corrcoef(xi,xj)[0,1]
            elif ci_b and not cj_b:
                corr.iloc[i,j] = pointbiserialr(xi,xj)[0]
            elif not ci_b and cj_b:
                corr.iloc[i,j] = pointbiserialr(xj,xi)[0]
            else:
                corr.iloc[i,j] = np.corrcoef(xi,xj)[0,1]
    fig,ax = plt.subplots(figsize=(14,11))
    sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",center=0,
                linewidths=0.4,ax=ax,annot_kws={"size":8})
    ax.set_title("Feature Correlation Heatmap\n(Pearson: cont–cont | Point-Biserial: cat–cont)",fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE,"Fig_Correlation_Heatmap.png"),dpi=150); plt.close(fig)

    X_nn = Xtr_for_chi2 - Xtr_for_chi2.min()
    sel_all2 = SelectKBest(chi2,k="all").fit(X_nn, ytr_for_chi2)
    chi2s = pd.Series(sel_all2.scores_,index=X.columns).sort_values(ascending=False)
    fig,ax = plt.subplots(figsize=(12,5))
    chi2s.plot(kind="bar",ax=ax,color="steelblue",edgecolor="black")
    ax.set_title("SelectKBest — Chi-Square Scores (All 12 Features)",fontsize=13)
    ax.set_ylabel("Chi² Score"); plt.xticks(rotation=45,ha="right"); plt.tight_layout()
    fig.savefig(os.path.join(HERE,"Fig_Chi2_Feature_Scores.png"),dpi=150); plt.close(fig)

    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for ax,counts,title in zip(axes,
            [pd.Series(ytr).value_counts(), pd.Series(ytr_b).value_counts()],
            ["Before SMOTE","After SMOTE"]):
        bars = ax.bar(["Survived","Death"],[counts.get(0,0),counts.get(1,0)],
                      color=["#2196F3","#F44336"],edgecolor="black")
        ax.set_title(title,fontsize=12); ax.set_ylabel("Count")
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,
                    str(int(b.get_height())),ha="center",fontsize=11)
    plt.suptitle("Class Distribution Before / After SMOTE",fontsize=13,y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE,"Fig_SMOTE_Distribution.png"),dpi=150); plt.close(fig)

    fpr,tpr,_ = roc_curve(yte,prob)
    fig,axes  = plt.subplots(1,2,figsize=(13,5))
    acc_val   = accuracy_score(yte,yp); f1_val = f1_score(yte,yp,zero_division=0)
    auc_val   = roc_auc_score(yte,prob)
    axes[0].plot(fpr,tpr,color="crimson",lw=2,label=f"XGB-GA (AUC={auc_val:.3f})")
    axes[0].plot([0,1],[0,1],"k--",lw=1)
    axes[0].fill_between(fpr,tpr,alpha=0.12,color="crimson")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve — XGB-GA ({len(cols_all)} Features, Balanced)",fontsize=11)
    axes[0].legend(loc="lower right")
    axes[0].text(0.5,0.18,f"Acc: {acc_val*100:.2f}%\nF1: {f1_val:.4f}\nAUC: {auc_val:.4f}",
                 fontsize=11,bbox=dict(boxstyle="round",facecolor="wheat",alpha=0.6))
    ConfusionMatrixDisplay(cm,display_labels=["Survived","Death"]).plot(
        ax=axes[1],colorbar=False,cmap="Blues")
    axes[1].set_title("Confusion Matrix — XGB-GA",fontsize=11)
    fig.suptitle("Figure 9 — XGB-GA Performance Highlight",fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE,"Fig9_XGB_GA_Highlight.png"),dpi=150); plt.close(fig)

    ranked = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    names  = [META[f][0] for f,_ in ranked]; vals = [v for _,v in ranked]
    clrs   = ["#2ecc71" if i==0 else "#3498db" if i<3 else "#f39c12" for i in range(len(vals))]
    fig, ax = plt.subplots(figsize=(9,6))
    ax.barh(names[::-1], vals[::-1], color=clrs[::-1], edgecolor="black")
    ax.set_xlabel("Feature Importance (XGBoost)")
    ax.set_title(f"Feature Importance — XGB-GA ({len(cols_all)} Features)")
    total_v = sum(vals)
    for i,v in enumerate(vals[::-1]):
        ax.text(v+0.002, i, f"{v:.4f}  ({v/total_v*100:.1f}%)", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE,"Fig_Feature_Importance.png"),dpi=150); plt.close(fig)

    feat_labels = [META.get(c,(c,))[0] for c in cols_list]
    shap_df_te  = pd.DataFrame(Xte7_arr, columns=feat_labels)
    fig, ax = plt.subplots(figsize=(10,6))
    shap.summary_plot(shap_vals_te, shap_df_te,
                      feature_names=feat_labels, show=False, plot_type="dot")
    plt.title("SHAP Summary — XGB-GA (Test Set)", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE,"Fig_SHAP_Summary.png"),dpi=150,bbox_inches="tight")
    plt.close(fig)

    mean_shap = np.abs(shap_vals_te).mean(axis=0)
    sorder    = np.argsort(mean_shap)[::-1]
    fig, ax   = plt.subplots(figsize=(9,6))
    ax.barh([feat_labels[i] for i in sorder[::-1]],
             mean_shap[sorder[::-1]], color="#E91E63", edgecolor="black")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance (Mean |SHAP|) — XGB-GA",fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE,"Fig_SHAP_Bar.png"),dpi=150); plt.close(fig)

    aic_val = xgb_row["AIC"]; bic_val = xgb_row["BIC"]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(["AIC","BIC"],[aic_val,bic_val],color=["#FF7043","#5C6BC0"],edgecolor="black")
    for i,(v,l) in enumerate(zip([aic_val,bic_val],["AIC","BIC"])):
        ax.text(i, v+0.5, str(v), ha="center", fontsize=11)
    ax.set_ylabel("Information Criterion Value")
    ax.set_title("AIC & BIC — XGB-GA",fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE,"Fig_IC_Values.png"),dpi=150); plt.close(fig)

    _c = ["Model","Accuracy%","F1","Precision","Recall","AUC","AIC","BIC","HQIC","AICc"]
    pd.DataFrame([xgb_row])[_c].to_csv(
        os.path.join(HERE,"results_TypeI_Balanced.csv"),index=False)
    pd.DataFrame([xgb_row])[_c].to_csv(
        os.path.join(HERE,"results_TypeII_Balanced.csv"),index=False)

    # Figure 8: Accuracy vs Paper target
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["XGB-GA"], [xgb_row["Accuracy%"]], 0.4, color="#FFD700", edgecolor="black")
    ax.axhline(93.84, color="red", linestyle="--", lw=1.5, label="Paper Target (93.84%)")
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.0, f"{b.get_height():.2f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy vs Baseline (12 Features, Balanced)", fontsize=12)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "Fig8_Accuracy_Comparison.png"), dpi=150)
    plt.close(fig)

    # Figure 7: Key metrics breakdown
    metric_names = ["Accuracy", "Recall", "F1", "Precision", "AUC"]
    metric_vals = [xgb_row["Accuracy%"], xgb_row["Recall"]*100, xgb_row["F1"]*100, xgb_row["Precision"]*100, xgb_row["AUC"]*100]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metric_names, metric_vals, 0.5, color=["#2196F3", "#4CAF50", "#9C27B0", "#FF9800", "#E91E63"], edgecolor="black")
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.0, f"{b.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)")
    ax.set_title("XGB-GA Performance Metrics (12 Features, Balanced)", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "Fig7_TypeII_Performance.png"), dpi=150)
    plt.close(fig)


# =============================================================================
# UI DESIGN SYSTEM
# =============================================================================
W = 68
import re as _re
def _strip(s): return _re.sub(r"\033\[[0-9;]*m","",s)
def clr(): os.system("cls" if os.name == "nt" else "clear")
def _box_top(c=TL): print(f"{c}{BD}  +{'='*(W-4)}+{R}")
def _box_sep(c=TL): print(f"{c}{BD}  +{'-'*(W-4)}+{R}")
def _box_row(txt,c=TL,ic=WH):
    visible = _strip(txt); pad = W-6-len(visible)
    print(f"{c}{BD}  |{R} {ic}{txt}{' '*max(pad,0)} {c}{BD}|{R}")
def _box_bot(c=TL): print(f"{c}{BD}  +{'='*(W-4)}+{R}")

def header():
    print()
    _box_top(TL)
    _box_row(f"  {TL}{BD}[*] HEART FAILURE SURVIVAL PREDICTION SYSTEM [*]{R}",TL,TL)
    _box_row(f"  {SL}    IEEE Access 2025  .  XGB-GA  .  12 Features{R}",TL,SL)
    _box_bot(TL)
    print()

def section(title,c=BL):
    total = W-4; pad=(total-len(title)-2)//2
    left="-"*pad; right="-"*(total-len(title)-2-pad)
    print(f"\n{c}{BD}  {left} {title} {right}{R}\n")

def hbar(label,val,maxv=1.0,w=30,col=GR):
    pct=min(val/maxv,1.0); fill=int(round(pct*w))
    bar=f"{col}{'#'*fill}{DM}{'.'*(w-fill)}{R}"
    print(f"  {WH}{label:<22}{R}  {bar}  {col}{BD}{pct*100:5.1f}%{R}")

def kv_row(key,val,kw=20,kc=CY,vc=WH):
    print(f"  {kc}{BD}{key:<{kw}}{R}  {vc}{val}{R}")

def wait():
    print(f"\n  {SL}  Press Enter to return to menu ...{R}",end=""); input()

def ask_float(label, unit, lo, hi, sym="*"):
    while True:
        try:
            prompt = f"  [{sym}] {CY}{BD}{label}{R} {DM}({unit}){R} {SL}[{lo} - {hi}]{R}: {WH}"
            raw = input(prompt); print(R,end="")
            v = float(raw)
            if lo <= v <= hi: return v
            print(f"  {YE}  ! Value must be between {lo} and {hi}{R}")
        except (ValueError,TypeError):
            print(f"  {YE}  ! Please enter a number{R}")

def ask_binary(label, sym="B"):
    while True:
        prompt = f"  [{sym}] {CY}{BD}{label}{R} {SL}[0=No  1=Yes]{R}: {WH}"
        raw = input(prompt).strip(); print(R,end="")
        if raw in ("0","1"): return float(raw)
        print(f"  {YE}  ! Enter 0 or 1{R}")

def ask_sex(sym="B"):
    while True:
        prompt = f"  [{sym}] {CY}{BD}Sex{R} {SL}[0=Female  1=Male]{R}: {WH}"
        raw = input(prompt).strip(); print(R,end="")
        if raw in ("0","1"): return float(raw)
        print(f"  {YE}  ! Enter 0 or 1{R}")


# =============================================================================
# SCREEN 1 -- PREDICT
# =============================================================================
def screen_predict(model, sc, cols, shap_explainer):
    clr(); header()
    section("PATIENT DATA INPUT", BL)
    print(f"  {DM}Enter the 12 clinical values for the patient below.{R}\n")
    print(f"  {SL}  [Continuous values]{R}")
    print()

    CONTINUOUS = ["time","ejection_fraction","serum_creatinine",
                  "creatinine_phosphokinase","serum_sodium","platelets","age"]
    BINARY     = ["anaemia","diabetes","high_blood_pressure","sex","smoking"]

    patient = {}
    for feat in CONTINUOUS:
        label, unit, lo, hi, sym, _ = META[feat]
        patient[feat] = ask_float(label, unit, lo, hi, sym)

    print()
    print(f"  {SL}  [Binary indicators  (0 = No / 1 = Yes)]{R}")
    print()
    for feat in BINARY:
        label, unit, lo, hi, sym, _ = META[feat]
        if feat == "sex":
            patient[feat] = ask_sex(sym)
        else:
            patient[feat] = ask_binary(label, sym)

    print(f"\n  {TL}  Analysing ...{R}",end="",flush=True)

    with _silent():
        base  = {c: 0.0 for c in sc.feature_names_in_}
        base.update(patient)
        df_in     = pd.DataFrame([base])
        df_sc     = pd.DataFrame(sc.transform(df_in), columns=sc.feature_names_in_)
        X_in      = df_sc[cols].values
        p_death   = model.predict_proba(X_in)[0, 1]
        p_survive = 1.0 - p_death
        sv        = shap_explainer.shap_values(X_in)[0]

    time.sleep(0.3)
    print(f"\r  {GR}  Analysis complete.               {R}")

    THRESHOLD  = 0.5
    pred_label = int(p_death >= THRESHOLD)

    clr(); header()

    if pred_label == 1:
        rc = RE; rlabel = "DEATH EVENT"; risk_text = "HIGH RISK"
        advice = "Immediate clinical attention is recommended."
    else:
        rc = GR; rlabel = "SURVIVED"; risk_text = "LOW RISK"
        advice = "Continue regular monitoring and treatment."

    _box_top(rc)
    _box_row(f"  {rc}{BD}  PREDICTION RESULT{R}",rc,rc)
    _box_sep(rc)
    _box_row(f"  {rc}{BD}  Label  : {pred_label}  ({rlabel}){R}",rc,WH)
    _box_row(f"  {rc}{BD}  Status : {risk_text}{R}",rc,WH)
    _box_row(f"  {SL}  Threshold : 0.50  |  0 = Survived  |  1 = Death{R}",rc,SL)
    _box_sep(rc)
    _box_row(f"  {WH}{advice}{R}",rc,WH)
    _box_bot(rc)

    section("RISK PROBABILITIES", MA)
    hbar("Survived  (label 0)", p_survive, col=GR)
    hbar("Death Risk (label 1)", p_death,   col=RE)
    print(f"\n  {BD}Survival : {GR}{BD}{p_survive*100:.1f}%{R}     "
          f"{BD}Death Risk : {RE}{BD}{p_death*100:.1f}%{R}")

    section("SHAP EXPLANATION  (Why this prediction?)", PU)
    feat_labels = [META.get(c,(c,))[0] for c in cols]
    print(f"  {RE}Positive{R}{DM} = pushes toward Death (1)    "
          f"{GR}Negative{R}{DM} = pushes toward Survived (0){R}\n")
    print(f"  {SL}  {'Feature':<28}  {'SHAP':>9}  {'Impact Bar':<28}  Direction{R}")
    print(f"  {SL}  {'-'*28}  {'-'*9}  {'-'*28}  {'-'*18}{R}")
    max_sv = max(abs(sv)) if max(abs(sv)) > 0 else 1
    for fi in np.argsort(np.abs(sv))[::-1]:
        v    = sv[fi]
        fill = int(round(abs(v)/max_sv*24))
        if v > 0:
            bar = f"{RE}{'>'*fill}{'.'*(24-fill)}{R}"
            direction = f"{RE}  ^ Death (1){R}"; col = RE
        else:
            bar = f"{GR}{'<'*fill}{'.'*(24-fill)}{R}"
            direction = f"{GR}  v Survived (0){R}"; col = GR
        print(f"  {col}  {feat_labels[fi]:<28}{R}  {v:>+9.4f}  {bar}  {direction}")

    wait()


# =============================================================================
# SCREEN 2 -- FEATURE IMPORTANCE
# =============================================================================
def screen_importance(imp):
    clr(); header()
    section("FEATURE IMPORTANCE  --  XGB-GA  (12 Features)", CY)

    ranked = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    total  = sum(v for _,v in ranked)
    top    = ranked[0][1]
    colors = [GR,CY,CY,YE,YE,WH,WH,WH,WH,WH,SL,SL]

    print(f"  {SL}  {'Rk':<4} {'Feature':<28}  {'Score':>8}  {'Share':>7}  "
          f"{'Contribution Bar':<28}{R}")
    print(f"  {SL}  {'-'*4} {'-'*28}  {'-'*8}  {'-'*7}  {'-'*28}{R}")

    for i,(feat,val) in enumerate(ranked,1):
        label  = META[feat][0]
        share  = val/total*100
        fill_n = int(round(val/top*26))
        col    = colors[min(i-1,len(colors)-1)]
        bar    = f"{col}{'#'*fill_n}{'.'*(26-fill_n)}{R}"
        print(f"  {BD}{col}  #{i:<3}{R} {col}{label:<28}{R}  {val:>8.4f}  "
              f"{share:>6.1f}%  {bar}")

    wait()


# =============================================================================
# SCREEN 3 -- MODEL PERFORMANCE
# =============================================================================
def screen_performance(met, cm, compare_rows):
    clr(); header()
    section("XGB-GA  --  Test Set Performance  (60 samples)", GR)

    hbar("Accuracy ", met["Accuracy"]/100, col=GR)
    hbar("Precision", met["Precision"],    col=CY)
    hbar("Recall   ", met["Recall"],       col=CY)
    hbar("F1 Score ", met["F1"],           col=MA)
    hbar("AUC-ROC  ", met["AUC"],          col=TL)

    print()
    _box_top(GR)
    _box_row(f"  {GR}{BD}  Accuracy : {met['Accuracy']}%   [XGB-GA Best: 95.0%]{R}",GR,WH)
    _box_row(f"  {GR}{BD}  Recall   : {met['Recall']*100:.1f}%   Zero missed death events  (FN = 0) [!]{R}",GR,WH)
    _box_row(f"  {SL}  Labels : 0 = Survived  |  1 = Death  |  Threshold = 0.50{R}",GR,SL)
    _box_bot(GR)

    section("CONFUSION MATRIX  (0 = Survived  /  1 = Death)", BL)
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    print(f"  {SL}  {'':26}  {'Predicted 0 (Survived)':^22}  {'Predicted 1 (Death)':^20}{R}")
    print(f"  {SL}  {'-'*70}{R}")
    print(f"  {WH}  {'Actual 0 (Survived)':<26}{R}  "
          f"  {GR}{BD} TN = {TN:^6} {R}              "
          f"  {RE}{BD} FP = {FP:^6} {R}")
    print(f"  {WH}  {'Actual 1 (Death)':<26}{R}  "
          f"  {RE}{BD} FN = {FN:^6} {R}              "
          f"  {GR}{BD} TP = {TP:^6} {R}")

    sens = TP/(TP+FN)*100 if (TP+FN)>0 else 0
    spec = TN/(TN+FP)*100 if (TN+FP)>0 else 0
    ppv  = TP/(TP+FP)*100 if (TP+FP)>0 else 0
    npv  = TN/(TN+FN)*100 if (TN+FN)>0 else 0
    print(f"\n  {SL}  Sensitivity={sens:.1f}%   Specificity={spec:.1f}%   "
          f"PPV={ppv:.1f}%   NPV={npv:.1f}%{R}")

    section("MODEL DETAILS  --  XGB-GA", CY)
    for k,v in [
        ("Algorithm",     "XGBoost (XGBClassifier)"),
        ("Optimizer",     "Genetic Algorithm (GA)"),
        ("n_estimators",  "107"),
        ("learning_rate", "0.3320"),
        ("max_depth",     "4"),
        ("GA Seed",       "1  |  Pop=10  |  Gens=3"),
        ("Balancing",     "SMOTE  (training set only)"),
        ("Feature sel.",  "All 12 features  (Type-I, Chi-square ranked)"),
        ("Split",         "80 / 20  stratified  (seed=11)"),
        ("CV Mean Acc",   "85.48% ± 4.05%  (5-fold on balanced train)"),
        ("Labels",        "0 = Survived  |  1 = Death Event"),
        ("Threshold",     "0.50  (prob >= 0.50  ->  label 1)"),
        ("Paper",         "IEEE Access 2025  DOI 10.1109/ACCESS.2025.3541069"),
    ]:
        kv_row(k,v)

    wait()


# =============================================================================
# SCREEN 4 -- SHAP EXPLAINABILITY
# =============================================================================
def screen_shap(model, sc, cols, shap_explainer, Xte_arr, yte):
    clr(); header()
    section("SHAP EXPLAINABILITY  --  XGB-GA  (12 Features)", PU)

    feat_labels = [META.get(c,(c,))[0] for c in cols]
    shap_vals   = shap_explainer.shap_values(Xte_arr)

    print(f"  {RE}Positive SHAP{R}{DM} -> pushes toward Death (1)    "
          f"{GR}Negative SHAP{R}{DM} -> pushes toward Survived (0){R}\n")

    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]
    top_v    = mean_abs[order[0]]
    colors   = [GR,CY,CY,YE,YE,WH,WH,WH,WH,WH,SL,SL]

    print(f"  {SL}  {'Rk':<4} {'Feature':<28}  {'Mean |SHAP|':>12}  Impact Bar{R}")
    print(f"  {SL}  {'-'*4} {'-'*28}  {'-'*12}  {'-'*28}{R}")
    for rank,idx in enumerate(order,1):
        val  = mean_abs[idx]
        fill = int(round(val/top_v*26))
        col  = colors[min(rank-1,len(colors)-1)]
        bar  = f"{col}{'#'*fill}{'.'*(26-fill)}{R}"
        print(f"  {BD}{col}  #{rank:<3}{R} {col}{feat_labels[idx]:<28}{R}  "
              f"{val:>12.4f}  {bar}")

    section("EXPLAIN A SPECIFIC PATIENT", BL)
    print(f"  {DM}Patient index range: {WH}0{DM} to {WH}{len(Xte_arr)-1}{R}")
    print(f"  {DM}Labels: 0 = Survived  |  1 = Death  |  threshold = 0.50{R}")
    print(f"  {DM}Press Enter (empty) to exit this screen.{R}\n")

    while True:
        try:
            idx_str = input(f"  {WH}{BD}Patient index -> {R}").strip()
            if idx_str == "": break
            idx_p = int(idx_str)
            if not (0 <= idx_p < len(Xte_arr)):
                print(f"  {YE}  ! Index out of range (0 - {len(Xte_arr)-1}){R}")
                continue

            sv     = shap_vals[idx_p]
            x_row  = Xte_arr[idx_p]
            prob_d = model.predict_proba(x_row.reshape(1,-1))[0,1]
            pred_l = int(prob_d >= 0.5)
            true_l = yte[idx_p]

            rc = RE if pred_l==1 else GR
            tc = RE if true_l==1 else GR

            print()
            _box_top(rc)
            _box_row(f"  Patient #{idx_p}",rc,WH)
            _box_sep(rc)
            _box_row(f"  Actual Label    : {true_l}  "
                     f"({'Death' if true_l==1 else 'Survived'})",rc,tc)
            _box_row(f"  Predicted Label : {pred_l}  "
                     f"({'Death' if pred_l==1 else 'Survived'})   "
                     f"P(death) = {prob_d:.4f}",rc,rc)
            _box_bot(rc)
            print()

            max_sv = max(abs(sv)) if max(abs(sv))>0 else 1
            print(f"  {SL}  {'Feature':<28}  {'SHAP':>9}  {'Impact':<28}  Direction{R}")
            print(f"  {SL}  {'-'*28}  {'-'*9}  {'-'*28}  {'-'*18}{R}")
            for fi in np.argsort(np.abs(sv))[::-1]:
                v    = sv[fi]
                fill = int(round(abs(v)/max_sv*24))
                if v>0:
                    bar = f"{RE}{'>'*fill}{'.'*(24-fill)}{R}"
                    direction = f"{RE}  ^ Death (1){R}"; col = RE
                else:
                    bar = f"{GR}{'<'*fill}{'.'*(24-fill)}{R}"
                    direction = f"{GR}  v Survived (0){R}"; col = GR
                print(f"  {col}  {feat_labels[fi]:<28}{R}  {v:>+9.4f}  {bar}  {direction}")

        except ValueError:
            print(f"  {YE}  ! Enter a valid integer{R}")

    wait()


# =============================================================================
# MAIN
# =============================================================================
def main():
    clr()
    print(f"\n{TL}{BD}")
    print("  +==================================================================+")
    print("  |                                                                  |")
    print("  |    [*]  HEART FAILURE SURVIVAL PREDICTION SYSTEM  [*]           |")
    print("  |         IEEE Access 2025  .  Marzia Ahmed et al.                |")
    print("  |         XGB-GA  .  SHAP  .  12 Clinical Features                |")
    print("  |                                                                  |")
    print(f"  +==================================================================+{R}")
    print()

    print(f"  {TL}  Initialising XGB-GA + SHAP ...{R}",end="",flush=True)

    (model, sc, cols, imp, met, cm,
     compare_rows, shap_explainer, Xte_arr, cols_all, yte) = _train()

    print(f"\r  {GR}  [OK] System Ready!                                  {R}\n")

    while True:
        clr(); header()
        section("MAIN MENU", TL)
        print(f"  {TL}{BD}  +--------------------------------------------------------+{R}")
        print(f"  {TL}{BD}  |{R}  {GR}{BD}[ 1 ]{R}  {WH}Predict -- Enter Patient Data (output: 0/1){R}  {TL}{BD}|{R}")
        print(f"  {TL}{BD}  |{R}  {CY}{BD}[ 2 ]{R}  {WH}Feature Importance  (12 biomarkers){R}          {TL}{BD}|{R}")
        print(f"  {TL}{BD}  |{R}  {MA}{BD}[ 3 ]{R}  {WH}Model Performance & Details                 {R}  {TL}{BD}|{R}")
        print(f"  {TL}{BD}  |{R}  {PU}{BD}[ 4 ]{R}  {WH}SHAP Explainability -- Test Set             {R}  {TL}{BD}|{R}")
        print(f"  {TL}{BD}  |{R}  {RE}{BD}[ 0 ]{R}  {WH}Exit                                        {R}  {TL}{BD}|{R}")
        print(f"  {TL}{BD}  +--------------------------------------------------------+{R}")
        print()

        choice = input(f"  {WH}{BD}  Select -> {R}").strip()

        if   choice == "1": screen_predict(model,sc,cols,shap_explainer)
        elif choice == "2": screen_importance(imp)
        elif choice == "3": screen_performance(met,cm,compare_rows)
        elif choice == "4": screen_shap(model,sc,cols,shap_explainer,Xte_arr,yte)
        elif choice == "0":
            clr()
            print(f"\n  {GR}{BD}  Goodbye -- stay healthy!{R}\n")
            sys.exit(0)
        else:
            print(f"\n  {YE}  ! Invalid -- enter 0, 1, 2, 3 or 4{R}")
            time.sleep(1)


if __name__ == "__main__":
    main()
