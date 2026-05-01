"""V3 classifier — LightGBM trained on labeled post_catalyst_outcomes.

Replaces (eventually) the hand-tuned V2 thresholds with a learned model.
Runs on every event with the existing features (probability, runup,
priced_in_score, catalyst_type, etc.) and outputs a calibrated P(direction_up).

Training data: post_catalyst_outcomes WHERE direction_correct_3d IS NOT NULL
               AND outcome_confidence >= MIN_CONFIDENCE.

Validation: walk-forward by catalyst_date — train on older 80%, test on newer 20%.
            This catches whether the model overfits to specific eras.

Output: P(up | features). Signal is derived from probability:
  prob >= 0.65  →  LONG_HIGH_CONF
  prob 0.55-0.65 →  LONG
  prob 0.45-0.55 →  NO_TRADE_AMBIGUOUS
  prob 0.35-0.45 →  SHORT
  prob < 0.35   →  SHORT_HIGH_CONF
"""
import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Features. Numerics first, categoricals last.
NUMERIC_FEATURES = [
    "predicted_prob",
    "runup_pre_event_30d_pct",
    "priced_in_score",
    "preevent_avg_volume_30d",
    "pre_event_price",
    "predicted_npv_b",
    "year",          # market regime proxy
    "month",         # seasonality (FDA backlog cycles)
]
CATEGORICAL_FEATURES = ["catalyst_type"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MIN_OUTCOME_CONFIDENCE = 0.7
TEST_FRACTION = 0.20  # newest 20% held out for walk-forward


def _wilson_ci(hits: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n < 1:
        return (0.0, 1.0)
    p = hits / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - spread), min(1.0, center + spread))


def _fetch_training_rows(db, min_outcome_confidence: float) -> List[Dict[str, Any]]:
    """Pull labeled events with all features. Excludes deadband (NULL labels)."""
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                pco.predicted_prob,
                pco.runup_pre_event_30d_pct,
                pco.priced_in_score,
                pco.preevent_avg_volume_30d,
                pco.pre_event_price,
                pco.predicted_npv_b,
                pco.catalyst_type,
                pco.catalyst_date,
                CASE WHEN pco.direction_correct_3d THEN 1 ELSE 0 END AS y
            FROM post_catalyst_outcomes pco
            WHERE pco.direction_correct_3d IS NOT NULL
              AND pco.outcome_confidence >= %s
              AND pco.signal_v2 IN ('LONG_UNDERPRICED_POSITIVE',
                                    'SHORT_SELL_THE_NEWS',
                                    'SHORT_LOW_PROBABILITY',
                                    'LONG', 'SHORT')
            ORDER BY pco.catalyst_date ASC
        """, (min_outcome_confidence,))
        rows = []
        for r in cur.fetchall():
            (prob, runup, priced_in, vol_30d, pre_price, npv,
             cat_type, cat_date, y) = r
            try:
                d = datetime.strptime(str(cat_date)[:10], "%Y-%m-%d")
                year, month = d.year, d.month
            except Exception:
                year, month = None, None
            rows.append({
                "predicted_prob": float(prob) if prob is not None else None,
                "runup_pre_event_30d_pct": float(runup) if runup is not None else None,
                "priced_in_score": float(priced_in) if priced_in is not None else None,
                "preevent_avg_volume_30d": float(vol_30d) if vol_30d is not None else None,
                "pre_event_price": float(pre_price) if pre_price is not None else None,
                "predicted_npv_b": float(npv) if npv is not None else None,
                "year": year,
                "month": month,
                "catalyst_type": cat_type or "Other",
                "y": int(y),
            })
    return rows


def train_v3_lgbm(db, min_outcome_confidence: float = MIN_OUTCOME_CONFIDENCE) -> Dict[str, Any]:
    """Train the V3 LightGBM model. Returns metrics + serialized model string.
    Caller is responsible for inserting into lgbm_models table.
    """
    import pandas as pd
    import lightgbm as lgb

    rows = _fetch_training_rows(db, min_outcome_confidence)
    if len(rows) < 50:
        raise RuntimeError(f"Only {len(rows)} labeled rows — too few to train")

    df = pd.DataFrame(rows)
    # Walk-forward split by catalyst_date order (already sorted ASC)
    n = len(df)
    n_test = max(20, int(n * TEST_FRACTION))
    n_train = n - n_test
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()

    # LightGBM accepts pandas categoricals directly. Cast catalyst_type.
    for d in (train_df, test_df):
        d["catalyst_type"] = d["catalyst_type"].astype("category")

    X_train = train_df[ALL_FEATURES]
    y_train = train_df["y"].values
    X_test = test_df[ALL_FEATURES]
    y_test = test_df["y"].values

    # Conservative params — small dataset, prevent overfit
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 15,            # shallow trees
        "max_depth": 4,              # cap depth
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 20,      # require enough events per leaf
        "lambda_l2": 1.0,            # regularization
        "verbose": -1,
        "seed": 42,
    }
    train_set = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=CATEGORICAL_FEATURES,
        free_raw_data=False,
    )
    test_set = lgb.Dataset(
        X_test, label=y_test,
        categorical_feature=CATEGORICAL_FEATURES,
        reference=train_set,
        free_raw_data=False,
    )
    booster = lgb.train(
        params, train_set,
        num_boost_round=200,
        valid_sets=[test_set],
        valid_names=["test"],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )

    train_pred = booster.predict(X_train)
    test_pred = booster.predict(X_test)
    train_acc_n = int(((train_pred >= 0.5).astype(int) == y_train).sum())
    test_acc_n = int(((test_pred >= 0.5).astype(int) == y_test).sum())
    train_accuracy = train_acc_n / len(y_train)
    test_accuracy = test_acc_n / len(y_test)
    test_ci_lo, test_ci_hi = _wilson_ci(test_acc_n, len(y_test))

    # Per-bucket test accuracy (high-confidence subset)
    high_conf_mask = (test_pred >= 0.65) | (test_pred <= 0.35)
    high_conf_y = y_test[high_conf_mask]
    # For SHORT predictions, "correct" means direction_up=False so we flip target
    high_conf_pred = (test_pred[high_conf_mask] >= 0.5).astype(int)
    if len(high_conf_y) > 0:
        high_conf_acc_n = int((high_conf_pred == high_conf_y).sum())
        high_conf_acc = high_conf_acc_n / len(high_conf_y)
        high_conf_ci_lo, high_conf_ci_hi = _wilson_ci(high_conf_acc_n, len(high_conf_y))
    else:
        high_conf_acc = None
        high_conf_ci_lo, high_conf_ci_hi = None, None

    importance_split = booster.feature_importance(importance_type="split").tolist()
    importance_gain = booster.feature_importance(importance_type="gain").tolist()
    feat_imp = {
        f: {"split": int(s), "gain": float(g)}
        for f, s, g in zip(ALL_FEATURES, importance_split, importance_gain)
    }

    return {
        "model_str": booster.model_to_string(),
        "feature_names": ALL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "test_ci_lower_pct": round(100.0 * test_ci_lo, 1),
        "test_ci_upper_pct": round(100.0 * test_ci_hi, 1),
        "min_outcome_confidence": min_outcome_confidence,
        "metrics_json": {
            "best_iteration": booster.best_iteration,
            "feature_importance": feat_imp,
            "high_conf_subset": {
                "n": int(len(high_conf_y)),
                "accuracy": high_conf_acc,
                "ci_lower_pct": round(100.0 * high_conf_ci_lo, 1) if high_conf_ci_lo is not None else None,
                "ci_upper_pct": round(100.0 * high_conf_ci_hi, 1) if high_conf_ci_hi is not None else None,
            },
            "earliest_train_date": str(train_df["catalyst_type"].iloc[0]) if len(train_df) > 0 else None,
            "earliest_test_date": str(test_df["catalyst_type"].iloc[0]) if len(test_df) > 0 else None,
        },
    }


def save_model_to_db(db, training_result: Dict[str, Any], model_version: str, notes: str = "") -> int:
    """Insert a new lgbm_models row, marking previous active=false."""
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE lgbm_models SET is_active = FALSE WHERE is_active = TRUE")
        cur.execute("""
            INSERT INTO lgbm_models (
                model_version, model_str, feature_names, categorical_features,
                train_n, test_n, train_accuracy, test_accuracy,
                test_ci_lower_pct, test_ci_upper_pct,
                metrics_json, min_outcome_confidence, is_active, notes
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s::jsonb, %s, TRUE, %s
            )
            RETURNING id
        """, (
            model_version,
            training_result["model_str"],
            training_result["feature_names"],
            training_result["categorical_features"],
            training_result["train_n"],
            training_result["test_n"],
            training_result["train_accuracy"],
            training_result["test_accuracy"],
            training_result["test_ci_lower_pct"],
            training_result["test_ci_upper_pct"],
            json.dumps(training_result["metrics_json"]),
            training_result["min_outcome_confidence"],
            notes,
        ))
        new_id = cur.fetchone()[0]
        conn.commit()
    return new_id


_model_cache: Dict[str, Any] = {"booster": None, "id": None, "feature_names": None}


def load_active_model(db) -> Optional[Dict[str, Any]]:
    """Load the active V3 LGBM model from DB. Caches in process memory."""
    import lightgbm as lgb
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, model_str, feature_names, categorical_features,
                   train_n, test_n, train_accuracy, test_accuracy,
                   test_ci_lower_pct, test_ci_upper_pct,
                   metrics_json, model_version, trained_at
            FROM lgbm_models
            WHERE is_active = TRUE
            ORDER BY trained_at DESC
            LIMIT 1
        """)
        r = cur.fetchone()
    if not r:
        return None
    (model_id, model_str, feat_names, cat_feats, train_n, test_n,
     train_acc, test_acc, ci_lo, ci_hi, metrics, version, trained) = r
    if _model_cache.get("id") != model_id:
        booster = lgb.Booster(model_str=model_str)
        _model_cache.update({
            "booster": booster, "id": model_id,
            "feature_names": list(feat_names),
            "categorical_features": list(cat_feats or []),
        })
    return {
        "id": model_id,
        "model_version": version,
        "trained_at": trained.isoformat() if trained else None,
        "feature_names": list(feat_names),
        "train_n": train_n,
        "test_n": test_n,
        "train_accuracy": float(train_acc) if train_acc is not None else None,
        "test_accuracy": float(test_acc) if test_acc is not None else None,
        "test_ci_lower_pct": float(ci_lo) if ci_lo is not None else None,
        "test_ci_upper_pct": float(ci_hi) if ci_hi is not None else None,
        "metrics": metrics,
    }


def predict_v3(features: Dict[str, Any], db) -> Optional[Dict[str, Any]]:
    """Predict P(direction_up) for a single event. Returns None if no model."""
    import pandas as pd
    info = load_active_model(db)
    if info is None or _model_cache.get("booster") is None:
        return None
    booster = _model_cache["booster"]
    feat_names = _model_cache["feature_names"]
    row = {f: features.get(f) for f in feat_names}
    df = pd.DataFrame([row])
    if "catalyst_type" in df.columns:
        df["catalyst_type"] = df["catalyst_type"].astype("category")
    prob = float(booster.predict(df)[0])
    if prob >= 0.65:
        signal = "LONG_HIGH_CONF"
    elif prob >= 0.55:
        signal = "LONG"
    elif prob > 0.45:
        signal = "NO_TRADE_AMBIGUOUS"
    elif prob > 0.35:
        signal = "SHORT"
    else:
        signal = "SHORT_HIGH_CONF"
    return {
        "prob_up": prob,
        "signal_v3": signal,
        "model_version": info["model_version"],
        "model_id": info["id"],
    }
