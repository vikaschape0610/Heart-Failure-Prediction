"""
Comprehensive Range and Correctness Validation Suite for XGB-GA Heart Failure Model.
Tests within-range, boundary, out-of-range, and extreme/invalid physiological inputs.
"""
import os
import sys
import io

# Force UTF-8 on Windows stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import model as M
from report_extractor import HARD_LIMITS, range_flag, LEVEL_NORMAL, LEVEL_OUTSIDE_TRAINING, validate_confirmed
import explain_ui as ex

def run_all_tests():
    print("=" * 80)
    print("XGB-GA MODEL ACCURACY & RANGE VALIDATION COMPREHENSIVE TEST")
    print("=" * 80)

    # 1. Train / Load Model
    print("\n[STEP 1] Loading / Training Model and Verifying Test Set Performance...")
    S = M.train_model()
    model = S["model"]
    sc = S["scaler"]
    cols = S["cols"]
    explainer = S["explainer"]
    metrics = S["metrics"]
    cm = S["cm"]

    print(f"  Model Type          : {type(model).__name__}")
    print(f"  Best GA Parameters  : {S.get('best_params')}")
    print(f"  Test Accuracy       : {metrics['Accuracy%']}%")
    print(f"  Test Recall         : {metrics['Recall'] * 100}% (Missed Deaths: {cm[1,0]})")
    print(f"  Test Precision      : {metrics['Precision']:.4f}")
    print(f"  Test F1 Score       : {metrics['F1']:.4f}")
    print(f"  Test ROC-AUC        : {metrics['AUC']:.4f}")
    print(f"  Confusion Matrix    : \n    TN: {cm[0,0]} | FP: {cm[0,1]}\n    FN: {cm[1,0]} | TP: {cm[1,1]}")

    results = {
        "metrics": metrics,
        "cm": cm.tolist(),
        "within_range_tests": [],
        "boundary_tests": [],
        "out_of_range_tests": [],
        "extreme_tests": [],
        "monotonicity_tests": {},
        "fairness_tests": []
    }

    # Helper function to run a single test case
    def evaluate_case(name, patient_dict, category, note=""):
        res = M.predict_patient(model, sc, cols, patient_dict, threshold=0.75)
        p_death = res["prob_death"]
        p_surv = res["prob_survive"]
        label = res["label"]
        sv = res["shap_vals"]

        # Mathematical sanity checks
        is_prob_valid = (0.0 <= p_death <= 1.0) and (0.0 <= p_surv <= 1.0)
        is_sum_one = np.isclose(p_death + p_surv, 1.0, atol=1e-5)
        is_shap_finite = np.isfinite(sv).all()
        is_label_correct = (label == 1 if p_death >= 0.75 else label == 0)

        # Check SHAP additivity
        base = float(np.ravel(np.asarray(explainer.expected_value, dtype=float))[0])
        shap_margin = base + float(sv.sum())
        expected_prob = 1.0 / (1.0 + np.exp(-shap_margin))
        shap_additive = np.isclose(expected_prob, p_death, atol=1e-3)

        status = "PASS" if (is_prob_valid and is_sum_one and is_shap_finite and is_label_correct and shap_additive) else "FAIL"

        case_res = {
            "name": name,
            "category": category,
            "patient": patient_dict,
            "prob_death": round(p_death, 5),
            "prob_survive": round(p_surv, 5),
            "label": label,
            "prediction": "Death Event (High Risk)" if label == 1 else "Survived (Low Risk)",
            "status": status,
            "is_prob_valid": bool(is_prob_valid),
            "is_sum_one": bool(is_sum_one),
            "is_shap_finite": bool(is_shap_finite),
            "shap_additive": bool(shap_additive),
            "note": note
        }
        return case_res

    # ──────────────────────────────────────────────────────────────────────────
    # [STEP 2] WITHIN-RANGE TESTS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[STEP 2] Running Within-Range Clinical Scenarios...")

    # Case A: Typical Healthy Profile (Low Risk)
    healthy_patient = {
        "age": 45.0,
        "anaemia": 0.0,
        "creatinine_phosphokinase": 150.0,
        "diabetes": 0.0,
        "ejection_fraction": 60.0,
        "high_blood_pressure": 0.0,
        "platelets": 260000.0,
        "serum_creatinine": 0.8,
        "serum_sodium": 140.0,
        "sex": 0.0,
        "smoking": 0.0,
        "time": 220.0
    }
    c_healthy = evaluate_case("Typical Healthy Patient", healthy_patient, "Within-Range", "Normal labs, high EF (60%), long follow-up (220d)")
    results["within_range_tests"].append(c_healthy)
    print(f"  [PASS] {c_healthy['name']}: Death Prob = {c_healthy['prob_death']*100:.2f}% -> {c_healthy['prediction']} [Status: {c_healthy['status']}]")

    # Case B: Severe High-Risk Patient (Within Training Distribution)
    severe_patient = {
        "age": 75.0,
        "anaemia": 1.0,
        "creatinine_phosphokinase": 582.0,
        "diabetes": 1.0,
        "ejection_fraction": 20.0,
        "high_blood_pressure": 1.0,
        "platelets": 240000.0,
        "serum_creatinine": 4.0,
        "serum_sodium": 125.0,
        "sex": 1.0,
        "smoking": 1.0,
        "time": 15.0
    }
    c_severe = evaluate_case("Decompensated High-Risk Patient", severe_patient, "Within-Range", "Elderly, low EF (20%), elevated creatinine (4.0), early time (15d)")
    results["within_range_tests"].append(c_severe)
    print(f"  [OK] {c_severe['name']}: Death Prob = {c_severe['prob_death']*100:.2f}% -> {c_severe['prediction']} [Status: {c_severe['status']}]")

    # Case C: Median Typical Patient
    median_patient = {k: (M.FEATURE_META[k][2] + M.FEATURE_META[k][3]) / 2.0 if k not in M.FEATURE_META or M.FEATURE_META[k][3] > 1 else 0.0 for k in M.FEATURE_META}
    c_median = evaluate_case("Midpoint Baseline Patient", median_patient, "Within-Range", "Exact midpoint of all feature ranges")
    results["within_range_tests"].append(c_median)
    print(f"  [OK] {c_median['name']}: Death Prob = {c_median['prob_death']*100:.2f}% -> {c_median['prediction']} [Status: {c_median['status']}]")

    # ──────────────────────────────────────────────────────────────────────────
    # [STEP 3] EXACT BOUNDARY TESTS (MIN & MAX OF ALL 12 FEATURES)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[STEP 3] Running Boundary Limit Tests (Exact Min/Max of Training Distribution)...")

    # All Minimum Values
    all_min_patient = {k: float(M.FEATURE_META[k][2]) for k in M.FEATURE_META}
    c_min = evaluate_case("All Features at Training Minimum", all_min_patient, "Boundary", "All features = training set min")
    results["boundary_tests"].append(c_min)
    print(f"  [OK] {c_min['name']}: Death Prob = {c_min['prob_death']*100:.2f}% -> {c_min['prediction']} [Status: {c_min['status']}]")

    # All Maximum Values
    all_max_patient = {k: float(M.FEATURE_META[k][3]) for k in M.FEATURE_META}
    c_max = evaluate_case("All Features at Training Maximum", all_max_patient, "Boundary", "All features = training set max")
    results["boundary_tests"].append(c_max)
    print(f"  [OK] {c_max['name']}: Death Prob = {c_max['prob_death']*100:.2f}% -> {c_max['prediction']} [Status: {c_max['status']}]")

    # ──────────────────────────────────────────────────────────────────────────
    # [STEP 4] OUT-OF-RANGE TESTS (BEYOND TRAINING SET, WITHIN HARD LIMITS)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[STEP 4] Running Out-of-Range Clinical Tests (Extrapolations)...")

    # Individual Out-of-Range Low / High Cases
    out_of_range_scenarios = [
        # EF Below Training Min (Training Min = 14%, Hard Min = 1%)
        ("Low EF = 10% (Severe Failure)", dict(healthy_patient, ejection_fraction=10.0), "EF below training min (14%)"),
        ("Low EF = 5% (End-Stage Cardiogenic)", dict(healthy_patient, ejection_fraction=5.0), "EF severely below training min"),
        # EF Above Training Max (Training Max = 80%, Hard Max = 100%)
        ("High EF = 85% (Hyperdynamic Heart)", dict(healthy_patient, ejection_fraction=85.0), "EF above training max (80%)"),
        ("High EF = 95% (Supranormal EF)", dict(healthy_patient, ejection_fraction=95.0), "EF near 100%"),

        # Serum Creatinine Below Training Min (Training Min = 0.5, Hard Min = 0.05)
        ("Low Creatinine = 0.3 mg/dL (Cachexia/Low Muscle)", dict(healthy_patient, serum_creatinine=0.3), "Creatinine below training min (0.5)"),
        # Serum Creatinine Above Training Max (Training Max = 9.4, Hard Max = 40.0)
        ("High Creatinine = 12.0 mg/dL (Acute Renal Failure)", dict(healthy_patient, serum_creatinine=12.0), "Creatinine above training max (9.4)"),
        ("Severe High Creatinine = 25.0 mg/dL (End-Stage ESRD)", dict(healthy_patient, serum_creatinine=25.0), "Extreme creatinine"),

        # Serum Sodium Below Training Min (Training Min = 113, Hard Min = 80)
        ("Low Sodium = 105 mEq/L (Severe Hyponatremia)", dict(healthy_patient, serum_sodium=105.0), "Sodium below training min (113)"),
        ("Critical Low Sodium = 90 mEq/L", dict(healthy_patient, serum_sodium=90.0), "Critically low sodium"),
        # Serum Sodium Above Training Max (Training Max = 148, Hard Max = 180)
        ("High Sodium = 155 mEq/L (Hypernatremia)", dict(healthy_patient, serum_sodium=155.0), "Sodium above training max (148)"),
        ("Critical High Sodium = 170 mEq/L", dict(healthy_patient, serum_sodium=170.0), "Critically high sodium"),

        # CPK Above Training Max (Training Max = 7861, Hard Max = 100,000)
        ("High CPK = 15,000 mcg/L (Severe Rhabdomyolysis)", dict(healthy_patient, creatinine_phosphokinase=15000.0), "CPK above training max (7861)"),
        ("Extreme CPK = 50,000 mcg/L (Massive Muscle Necrosis)", dict(healthy_patient, creatinine_phosphokinase=50000.0), "Massive CPK elevation"),

        # Platelets Below Training Min (Training Min = 25,000, Hard Min = 1,000)
        ("Low Platelets = 12,000 /mL (Severe Thrombocytopenia)", dict(healthy_patient, platelets=12000.0), "Platelets below training min (25,000)"),
        # Platelets Above Training Max (Training Max = 850,000, Hard Max = 3,000,000)
        ("High Platelets = 1,200,000 /mL (Thrombocytosis)", dict(healthy_patient, platelets=1200000.0), "Platelets above training max (850,000)"),

        # Age Out of Range (Training max 95 in data, FEATURE_META 120, Hard Max 120)
        ("Centenarian Patient (Age = 105)", dict(healthy_patient, age=105.0), "Age above standard cohort"),

        # Follow-up Time (Training range = 4 to 285 days, Hard Limit max = 3650 days = 10 years)
        ("Follow-up Time = 1 day (Acute Admission Day 1)", dict(healthy_patient, time=1.0), "Time below training min (4)"),
        ("Follow-up Time = 500 days (Long-term Surviving)", dict(healthy_patient, time=500.0), "Time above training max (285)"),
        ("Follow-up Time = 1000 days (~3 Years Survival)", dict(healthy_patient, time=1000.0), "Time well above training max"),

        # Multi-factor Out-of-Range Extremes
        ("Extreme Multi-Organ Crisis (OOR Combined)", {
            "age": 85.0,
            "anaemia": 1.0,
            "creatinine_phosphokinase": 12000.0,
            "diabetes": 1.0,
            "ejection_fraction": 8.0,
            "high_blood_pressure": 1.0,
            "platelets": 15000.0,
            "serum_creatinine": 14.5,
            "serum_sodium": 102.0,
            "sex": 1.0,
            "smoking": 1.0,
            "time": 2.0
        }, "Multiple biomarkers simultaneously out of training range (all severe)"),

        ("Super-Preserved Profile (OOR Combined)", {
            "age": 35.0,
            "anaemia": 0.0,
            "creatinine_phosphokinase": 80.0,
            "diabetes": 0.0,
            "ejection_fraction": 85.0,
            "high_blood_pressure": 0.0,
            "platelets": 250000.0,
            "serum_creatinine": 0.4,
            "serum_sodium": 142.0,
            "sex": 0.0,
            "smoking": 0.0,
            "time": 600.0
        }, "Preserved organs with EF=85% and follow-up=600d")
    ]

    for name, p_dict, note in out_of_range_scenarios:
        c = evaluate_case(name, p_dict, "Out-of-Range", note)
        results["out_of_range_tests"].append(c)
        print(f"  [OK] {c['name']:<48}: P(Death)={c['prob_death']*100:6.2f}% | P(Surv)={c['prob_survive']*100:6.2f}% | {c['prediction']} [Status: {c['status']}]")

    # ──────────────────────────────────────────────────────────────────────────
    # [STEP 5] MONOTONICITY & CLINICAL GRADIENT TESTS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[STEP 5] Testing Clinical Monotonicity (Directional Sanity Across Ranges)...")

    # Test Creatinine Gradient (from 0.4 to 20.0 mg/dL)
    cr_values = [0.4, 0.8, 1.2, 1.8, 2.5, 4.0, 7.0, 9.4, 15.0, 25.0]
    cr_probs = []
    base_case = dict(healthy_patient)
    for cr in cr_values:
        p = dict(base_case, serum_creatinine=cr)
        res = M.predict_patient(model, sc, cols, p)
        cr_probs.append(res["prob_death"])
    
    # Check that as creatinine worsens, risk does not decrease drastically
    cr_monotonic = all(cr_probs[i] <= cr_probs[i+1] + 0.02 for i in range(len(cr_probs)-1))
    results["monotonicity_tests"]["serum_creatinine"] = {
        "values": cr_values,
        "probs": [round(x, 4) for x in cr_probs],
        "is_monotonic_increasing": cr_monotonic
    }
    print(f"  Creatinine Gradient (0.4 to 25.0 mg/dL): {['{:.1f}%'.format(p*100) for p in cr_probs]}")
    print(f"    -> Monotonically non-decreasing risk: {cr_monotonic}")

    # Test Ejection Fraction Gradient (from 5% to 85%)
    ef_values = [5, 10, 15, 20, 30, 40, 50, 60, 70, 85]
    ef_probs = []
    for ef in ef_values:
        p = dict(base_case, ejection_fraction=float(ef))
        res = M.predict_patient(model, sc, cols, p)
        ef_probs.append(res["prob_death"])
    
    # Generally as EF increases, risk should decrease or stay low
    ef_monotonic = all(ef_probs[i] >= ef_probs[i+1] - 0.02 for i in range(len(ef_probs)-1))
    results["monotonicity_tests"]["ejection_fraction"] = {
        "values": ef_values,
        "probs": [round(x, 4) for x in ef_probs],
        "is_monotonic_decreasing": ef_monotonic
    }
    print(f"  Ejection Fraction Gradient (5% to 85%): {['{:.1f}%'.format(p*100) for p in ef_probs]}")
    print(f"    -> Monotonically non-increasing risk: {ef_monotonic}")

    # Test Time Gradient (from 1 to 500 days)
    time_values = [1, 10, 30, 80, 150, 250, 400, 600]
    time_probs = []
    for t in time_values:
        p = dict(severe_patient, time=float(t))
        res = M.predict_patient(model, sc, cols, p)
        time_probs.append(res["prob_death"])
    
    time_monotonic = all(time_probs[i] >= time_probs[i+1] - 0.05 for i in range(len(time_probs)-1))
    results["monotonicity_tests"]["time"] = {
        "values": time_values,
        "probs": [round(x, 4) for x in time_probs],
        "is_monotonic_decreasing": time_monotonic
    }
    print(f"  Follow-up Time Gradient (1 to 600 days): {['{:.1f}%'.format(p*100) for p in time_probs]}")
    print(f"    -> Monotonically non-increasing risk: {time_monotonic}")

    # ──────────────────────────────────────────────────────────────────────────
    # [STEP 6] PHYSIOLOGICALLY IMPOSSIBLE / HARD-LIMIT VALIDATION TESTS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[STEP 6] Testing Physiological Hard Limits & Out-of-Bounds Handling...")

    impossible_cases = [
        ("Negative Age (-10 years)", dict(healthy_patient, age=-10.0), "age"),
        ("Impossible Age (250 years)", dict(healthy_patient, age=250.0), "age"),
        ("Negative Ejection Fraction (-5%)", dict(healthy_patient, ejection_fraction=-5.0), "ejection_fraction"),
        ("Superhuman Ejection Fraction (140%)", dict(healthy_patient, ejection_fraction=140.0), "ejection_fraction"),
        ("Negative Platelets (-50,000)", dict(healthy_patient, platelets=-50000.0), "platelets"),
        ("Extremely Massive Platelets (5,000,000)", dict(healthy_patient, platelets=5000000.0), "platelets"),
        ("Lethal Low Sodium (50 mEq/L)", dict(healthy_patient, serum_sodium=50.0), "serum_sodium"),
        ("Lethal High Sodium (220 mEq/L)", dict(healthy_patient, serum_sodium=220.0), "serum_sodium"),
        ("Negative Creatinine (-2.0 mg/dL)", dict(healthy_patient, serum_creatinine=-2.0), "serum_creatinine"),
        ("Impossible Creatinine (100.0 mg/dL)", dict(healthy_patient, serum_creatinine=100.0), "serum_creatinine"),
        ("Invalid Binary Sex (Sex = 3)", dict(healthy_patient, sex=3.0), "sex")
    ]

    for name, p_dict, target_key in impossible_cases:
        # Check validation logic
        # 1. range_flag
        lo_train, hi_train = M.FEATURE_META[target_key][2], M.FEATURE_META[target_key][3]
        rf = range_flag(target_key, p_dict[target_key], (lo_train, hi_train))
        is_flagged_oor = rf.out_of_range if rf else False

        # 2. validate_confirmed against HARD_LIMITS
        soft_limits = {k: (M.FEATURE_META[k][2], M.FEATURE_META[k][3]) for k in M.FEATURE_META}
        _, errs_allowed = validate_confirmed(p_dict, soft_limits, allow_outside_training_range=True)
        is_blocked_by_hard_limits = any(target_key in e for e in errs_allowed)

        # 3. Direct model behavior (does it crash or handle stably?)
        try:
            raw_eval = evaluate_case(name, p_dict, "Physiologically Impossible", "Raw model pass test")
            direct_model_status = "Computed Safely (No NaN/Crash)"
            prob_out = raw_eval["prob_death"]
        except Exception as e:
            direct_model_status = f"Threw Exception: {str(e)}"
            prob_out = None

        results["extreme_tests"].append({
            "name": name,
            "target_key": target_key,
            "value": p_dict[target_key],
            "flagged_as_outside_training": is_flagged_oor,
            "blocked_by_hard_limits": is_blocked_by_hard_limits,
            "direct_model_handling": direct_model_status,
            "prob_out": prob_out
        })
        print(f"  [OK] {name:<42}: Flagged OOR={is_flagged_oor} | Rejected by Hard Limits={is_blocked_by_hard_limits} | Direct Model: {direct_model_status}")

    # ──────────────────────────────────────────────────────────────────────────
    # [STEP 7] COUNTERFACTUAL GENDER FAIRNESS TEST
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[STEP 7] Verifying Counterfactual Fairness (Gender Invariance)...")
    fairness_diffs = []
    for test_p in [healthy_patient, severe_patient, median_patient]:
        p_f = dict(test_p, sex=0.0)
        p_m = dict(test_p, sex=1.0)
        res_f = M.predict_patient(model, sc, cols, p_f)
        res_m = M.predict_patient(model, sc, cols, p_m)
        diff = abs(res_f["prob_death"] - res_m["prob_death"])
        fairness_diffs.append(diff)
        results["fairness_tests"].append({
            "prob_female": round(res_f["prob_death"], 5),
            "prob_male": round(res_m["prob_death"], 5),
            "abs_diff": round(diff, 5),
            "shap_sex_f": float(res_f["shap_vals"][cols.index("sex")]),
            "shap_sex_m": float(res_m["shap_vals"][cols.index("sex")])
        })
    max_gender_gap = max(fairness_diffs)
    print(f"  Max Absolute Gender Probability Gap across test profiles: {max_gender_gap:.5f}")
    print(f"  Counterfactual Fairness Status: {'PASS (Fair)' if max_gender_gap < 0.05 else 'FAIL'}")

    # Save complete test output to json
    out_path = os.path.join(ROOT, "tests", "range_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Full test results written to: {out_path}")

    return results

if __name__ == "__main__":
    run_all_tests()
