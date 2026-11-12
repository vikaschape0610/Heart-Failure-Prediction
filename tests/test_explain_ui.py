"""The 'how is this % calculated' panel relies on SHAP additivity: base + sum(SHAP) -> sigmoid == model probability."""
import os, sys
import numpy as np, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import model as M
import explain_ui as ex


@pytest.fixture(scope="module")
def S():
    return M.train_model()


PATIENTS = [
    dict(age=67, anaemia=1, creatinine_phosphokinase=250, diabetes=0, ejection_fraction=30, high_blood_pressure=1,
         platelets=245000, serum_creatinine=1.8, serum_sodium=134, sex=1, smoking=0, time=95),
    dict(age=45, anaemia=0, creatinine_phosphokinase=120, diabetes=0, ejection_fraction=60, high_blood_pressure=0,
         platelets=300000, serum_creatinine=0.9, serum_sodium=140, sex=0, smoking=0, time=200),
]


@pytest.mark.parametrize("patient", PATIENTS)
def test_base_plus_shap_equals_model_output_and_probability(S, patient):
    res = M.predict_patient(S["model"], S["scaler"], S["cols"], patient)
    base = float(np.ravel(np.asarray(S["explainer"].expected_value, dtype=float))[0])
    sv = np.asarray(res["shap_vals"], dtype=float).ravel()
    margin = ex._margin(S["model"], S["scaler"], S["cols"], patient)
    assert base + sv.sum() == pytest.approx(margin, abs=1e-3)
    assert ex._sigmoid(base + sv.sum()) == pytest.approx(res["prob_death"], abs=1e-4)
    assert sv[sv > 0].sum() + sv[sv < 0].sum() == pytest.approx(sv.sum())


def test_value_text_for_binary_and_numeric():
    assert ex._show_value("sex", 1, "0=F/1=M") == "Male" and ex._show_value("sex", 0, "0=F/1=M") == "Female"
    assert ex._show_value("smoking", 1, "0=No/1=Yes") == "Yes"
    assert ex._show_value("serum_sodium", 134, "mEq/L") == "134 mEq/L" and ex._show_value("age", None, "years") == "-"


@pytest.mark.parametrize("ef", [5, 95])
def test_values_outside_set_range_still_give_a_valid_probability(S, ef):
    p = dict(PATIENTS[0], ejection_fraction=ef)
    res = M.predict_patient(S["model"], S["scaler"], S["cols"], p)
    assert 0.0 <= res["prob_death"] <= 1.0 and np.isfinite(res["shap_vals"]).all()


# ── "what is happening in the body" story ───────────────────────────────────
META = {k: M.FEATURE_META[k] for k in M.FEATURE_META}
PT = dict(PATIENTS[0], serum_creatinine=1.8, serum_sodium=105, creatinine_phosphokinase=250, ejection_fraction=85,
          platelets=900000)

def _titles(steps):
    return [t for _, t, _ in steps]

def test_story_for_kidney_sodium_cpk_with_good_pumping_number():
    steps = ex._story_steps(PT, ["serum_creatinine", "serum_sodium", "creatinine_phosphokinase", "platelets"], META)
    t = " | ".join(_titles(steps))
    assert "kidneys are under strain" in t and "extra water" in t and "Muscle" in t and "platelet count is high" in t
    assert "looks fine" in t                                     # EF 85 is fine but congestion signs -> explains why risk is high
    assert any("outside what the tool was built on" in b for _, _, b in steps)    # EF 85 > 80

def test_story_weak_heart_gets_pumping_step_and_no_pumping_paradox():
    steps = ex._story_steps(dict(PT, ejection_fraction=25), ["ejection_fraction", "serum_creatinine"], META)
    t = " | ".join(_titles(steps))
    assert "pumping weakly" in t and "looks fine" not in t

def test_story_with_nothing_raising_risk_says_so():
    steps = ex._story_steps(PT, [], META)
    assert len(steps) == 1 and steps[0][0] == "g"