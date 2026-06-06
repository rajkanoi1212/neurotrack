"""
NeuroTrack — Step 3: Train RandomForest Models
===============================================
Trains three separate RandomForest classifiers:
  Model A — activity_type   (11 classes)
  Model B — project         (10 classes)
  Model C — cognitive_state (6 classes)

Why RandomForest?
  - Handles non-linear compound rules (AI + topic = research, not just AI)
  - No feature scaling needed
  - Gives reliable probability estimates at each leaf
  - Trees export cleanly to JSON for browser inference
  - Interpretable — can print feature importances

Exports:
  data/model.json       — all 3 models in one JS-compatible JSON
  data/model_A.pkl      — sklearn pickle (Python reuse)
  data/model_B.pkl
  data/model_C.pkl

Run:
    python ml/step3_train.py
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
import warnings

from collections              import Counter
from sklearn.ensemble         import RandomForestClassifier
from sklearn.preprocessing    import LabelEncoder
from sklearn.model_selection  import StratifiedKFold, cross_val_score
from sklearn.metrics          import classification_report
from sklearn.tree             import _tree

warnings.filterwarnings("ignore")

# ── PATHS ─────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

print("NeuroTrack | Step 3: Train RandomForest Models")
print("=" * 55)

# ── LOAD ──────────────────────────────────────────────────────────
X_df = pd.read_csv(os.path.join(DATA, "features.csv"))
y_df = pd.read_csv(os.path.join(DATA, "labels.csv"))

X          = X_df.values.astype(np.float32)
feat_names = list(X_df.columns)

print(f"Feature matrix : {X.shape[0]} rows × {X.shape[1]} features")


# ══════════════════════════════════════════════════════════════════
# HELPER: EXPORT TREE TO JS-COMPATIBLE DICT
# Each tree node is either:
#   {"f": feature_index, "t": threshold, "l": left_child, "r": right_child}
# or a leaf:
#   {"v": [prob_class_0, prob_class_1, ...]}
# ══════════════════════════════════════════════════════════════════

def export_tree_to_dict(estimator) -> dict:
    """
    Recursively converts a fitted sklearn DecisionTree into a
    nested Python dict that can be serialised to JSON and walked
    in JavaScript with the same logic as Python.
    """
    tree = estimator.tree_

    def recurse(node_id: int) -> dict:
        # Leaf node
        if tree.feature[node_id] == _tree.TREE_UNDEFINED:
            values = tree.value[node_id][0]
            total  = values.sum()
            probs  = (values / total).tolist() if total > 0 else values.tolist()
            return {"v": probs}

        # Decision node
        return {
            "f": int(tree.feature[node_id]),             # feature index
            "t": float(tree.threshold[node_id]),          # split threshold
            "l": recurse(tree.children_left[node_id]),    # <= threshold
            "r": recurse(tree.children_right[node_id]),   # >  threshold
        }

    return recurse(0)


# ══════════════════════════════════════════════════════════════════
# HELPER: TRAIN ONE RF MODEL
# ══════════════════════════════════════════════════════════════════

def train_model(X: np.ndarray,
                y_raw: np.ndarray,
                name: str,
                n_estimators: int = 200,
                max_depth: int = 14) -> tuple:
    """
    Trains a RandomForestClassifier on y_raw (string labels).
    Returns (fitted_rf, label_encoder, cross_val_scores).
    """
    # Drop classes with fewer than 3 samples — can't stratify
    counts      = Counter(y_raw)
    valid_tags  = [tag for tag, cnt in counts.items() if cnt >= 3]
    mask        = np.isin(y_raw, valid_tags)
    X_tr        = X[mask]
    y_tr        = y_raw[mask]

    le   = LabelEncoder()
    y_enc = le.fit_transform(y_tr)

    rf = RandomForestClassifier(
        n_estimators  = n_estimators,
        max_depth     = max_depth,
        min_samples_leaf = 1,
        class_weight  = "balanced",   # handle imbalanced classes
        random_state  = 42,
        n_jobs        = 1,            # use single core to avoid multiprocessing hang on Windows
    )

    # 5-fold stratified cross-validation
    n_folds = min(5, min(counts[t] for t in valid_tags))
    cv      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores  = cross_val_score(rf, X_tr, y_enc, cv=cv, scoring="accuracy")

    # Final fit on full training set
    rf.fit(X_tr, y_enc)

    print(f"\n  [{name}]")
    print(f"    Classes   : {list(le.classes_)}")
    print(f"    CV acc    : {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"    Train acc : {rf.score(X_tr, y_enc):.4f}")
    print(f"\n  Classification Report:")
    report = classification_report(
        y_enc, rf.predict(X_tr),
        target_names=le.classes_,
        zero_division=0
    )
    for line in report.split("\n"):
        print(f"    {line}")

    return rf, le, scores


# ══════════════════════════════════════════════════════════════════
# TRAIN 3 MODELS
# ══════════════════════════════════════════════════════════════════

print("\nTraining Model A — activity_type")
rf_A, le_A, scores_A = train_model(X, y_df["activity_type"].values,   "activity_type")

print("\nTraining Model B — project")
rf_B, le_B, scores_B = train_model(X, y_df["project"].values,          "project")

print("\nTraining Model C — cognitive_state")
rf_C, le_C, scores_C = train_model(X, y_df["cognitive_state"].values,  "cognitive_state")


# ══════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE  (from Model A — most interpretable)
# ══════════════════════════════════════════════════════════════════

print("\nTop 15 Feature Importances (Model A — activity_type):")
importances = rf_A.feature_importances_
top_idx     = np.argsort(importances)[::-1][:15]
for i in top_idx:
    bar = "█" * int(importances[i] * 200)
    print(f"  {feat_names[i]:<28} {importances[i]:.4f}  {bar}")


# ══════════════════════════════════════════════════════════════════
# EXPORT TREES → JSON  (browser-runnable)
# ══════════════════════════════════════════════════════════════════

print("\nExporting trees to JSON...")

model_pkg = {
    "activity":  {
        "trees":    [export_tree_to_dict(e) for e in rf_A.estimators_],
        "classes":  le_A.classes_.tolist(),
        "n_classes": len(le_A.classes_),
    },
    "project": {
        "trees":    [export_tree_to_dict(e) for e in rf_B.estimators_],
        "classes":  le_B.classes_.tolist(),
        "n_classes": len(le_B.classes_),
    },
    "cognitive": {
        "trees":    [export_tree_to_dict(e) for e in rf_C.estimators_],
        "classes":  le_C.classes_.tolist(),
        "n_classes": len(le_C.classes_),
    },
    "feature_names": feat_names,
    "n_features":    len(feat_names),
    "accuracy": {
        "activity_type":   round(float(scores_A.mean()), 4),
        "project":         round(float(scores_B.mean()), 4),
        "cognitive_state": round(float(scores_C.mean()), 4),
    },
}

model_json_path = os.path.join(DATA, "model.json")
with open(model_json_path, "w") as f:
    json.dump(model_pkg, f, separators=(",", ":"))

json_size = os.path.getsize(model_json_path) / 1024
print(f"  model.json   : {json_size:.0f} KB")


# ══════════════════════════════════════════════════════════════════
# VERIFY: JS-style tree walk matches sklearn exactly
# ══════════════════════════════════════════════════════════════════

def js_predict(vec: list, model_dict: dict) -> str:
    """
    Mirrors the JavaScript rfPredict() function exactly.
    Walks each tree and averages leaf probability vectors.
    """
    trees   = model_dict["trees"]
    classes = model_dict["classes"]
    n_cls   = len(classes)
    avg     = np.zeros(n_cls)

    for tree in trees:
        node = tree
        while "f" in node:
            node = node["l"] if vec[node["f"]] <= node["t"] else node["r"]
        avg += np.array(node["v"])

    avg /= len(trees)
    return classes[avg.argmax()]


print("\nVerifying JS walk == sklearn predict (must be 100%)...")

# Use the full feature matrix to verify
mismatches = 0
for i in range(len(X)):
    vec = X[i].tolist()
    sk  = le_A.inverse_transform(rf_A.predict(X[i:i+1]))[0]
    js  = js_predict(vec, model_pkg["activity"])
    if sk != js:
        mismatches += 1
        print(f"  MISMATCH at row {i}: sklearn={sk}  js={js}")

print(f"  Checked {len(X)} rows — mismatches: {mismatches}")
if mismatches == 0:
    print("  ✓ Perfect match — JS inference is identical to sklearn")


# ── SAVE SKLEARN PICKLES ──────────────────────────────────────────
for label, rf, le in [("A", rf_A, le_A), ("B", rf_B, le_B), ("C", rf_C, le_C)]:
    path = os.path.join(DATA, f"model_{label}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": rf, "label_encoder": le, "feature_names": feat_names}, f)
    print(f"  model_{label}.pkl saved")

print("\n✓ Training complete — all models saved to data/")
