# backend.py
"""
FastAPI Backend for Exoplanet Classification (UPDATED)
- CatBoost defaults match original script exactly
- Master dataset accumulation (uploaded_data/master_dataset.csv)
- Generalized detection of disposition/status columns
- /api/train appends uploaded CSV to master_dataset.csv (default) and re-trains ensemble + CatBoost
- Added /api/explain endpoint (SHAP + simple ELI5) - FIXED preprocessing for explain
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import math
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import io
import traceback
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print(
    "🚀 STARTING EXOPLANET CLASSIFIER API (MASTER + GENERALIZED DISPOSITION + CATBOOST DEFAULTS)"
)
print("=" * 80)

app = FastAPI(title="Exoplanet Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "saved_models"
DATA_DIR = "uploaded_data"
MASTER_DATA_PATH = os.path.join(DATA_DIR, "master_dataset.csv")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------
# Utility: find disposition-like column and map to binary label
# ---------------------------
def find_disposition_column(df: pd.DataFrame) -> Optional[str]:
    # Find column name containing 'disposition' or 'status'
    for col in df.columns:
        low = col.lower()
        if "disposition" in low or "status" in low:
            return col
    return None


def classify_disposition_value(val: Any) -> Optional[int]:
    # Return 1 for planet (candidate/confirmed/etc), 0 for false positive
    if pd.isna(val):
        return None
    # If already numeric 0/1
    try:
        if float(val) in (0.0, 1.0):
            return int(float(val))
    except Exception:
        pass
    s = str(val).upper()
    # Treat anything with 'FALSE' or 'FP' as NOT_PLANET (0)
    if "FALSE" in s or " FP" in s or s.strip() == "FP":
        return 0
    # Otherwise treat as planet
    return 1


# ---------------------------
# CatBoost detector (defaults exactly as requested)
# ---------------------------
class BinaryExoplanetDetector:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model: Optional[CatBoostClassifier] = None
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.class_names = ["FALSE POSITIVE", "EXOPLANET"]
        self.X_test = None
        self.y_test = None

    def prepare_from_dataframe(self, df: pd.DataFrame):
        # detect disposition-like column
        disp_col = find_disposition_column(df)
        if disp_col is None:
            raise ValueError("No disposition/status column found in CSV.")
        df = df.copy()
        df["target"] = df[disp_col].apply(classify_disposition_value)
        # drop rows where target couldn't be determined
        df = df[~df["target"].isnull()]

        # priority features (same as your list)
        priority_features = [
            "koi_period",
            "koi_duration",
            "koi_depth",
            "koi_prad",
            "koi_teq",
            "koi_insol",
            "koi_model_snr",
            "koi_steff",
            "koi_slogg",
            "koi_srad",
            "koi_impact",
            "koi_time0bk",
            "koi_time0",
            "koi_eccen",
            "koi_longp",
            "koi_trans_depth",
            "koi_ror",
            "koi_dor",
            "koi_incl",
            "koi_limbdark_mod",
        ]

        available_features = []
        for feat in priority_features:
            if feat in df.columns:
                missing_pct = df[feat].isnull().sum() / len(df) * 100
                if missing_pct < 80:
                    available_features.append(feat)
        # fallback to numeric columns if not enough prioritized features
        if len(available_features) < 3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_candidates = [c for c in numeric_cols if c != "target"]
            if len(numeric_candidates) < 3:
                raise ValueError("Not enough numeric features for CatBoost training.")
            available_features = numeric_candidates[: min(20, len(numeric_candidates))]

        X = df[available_features].copy()
        y = df["target"].astype(int).copy()
        # fill missing values with median and drop remaining NaNs
        X = X.fillna(X.median())
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]

        self.feature_names = available_features
        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size=0.3,
        use_smote=True,
        catboost_params: Optional[Dict[str, Any]] = None,
    ):
        catboost_params = catboost_params or {}
        iterations = int(catboost_params.pop("iterations", 500))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if use_smote:
            smote = SMOTE(random_state=self.random_state)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)

        class_counts = (
            np.bincount(y_train)
            if len(np.unique(y_train)) > 1
            else np.array([len(y_train), 0])
        )
        if len(class_counts) < 2 or class_counts[1] == 0:
            class_weights = None
        else:
            class_weights = {
                0: len(y_train) / (2 * class_counts[0]),
                1: len(y_train) / (2 * class_counts[1]),
            }

        # default params exactly as in your script
        default_params = dict(
            iterations=iterations,
            learning_rate=catboost_params.pop("learning_rate", 0.05),
            depth=catboost_params.pop("depth", 8),
            l2_leaf_reg=catboost_params.pop("l2_leaf_reg", 3),
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=self.random_state,
            verbose=catboost_params.pop("verbose", 100),
            early_stopping_rounds=catboost_params.pop("early_stopping_rounds", 50),
            use_best_model=catboost_params.pop("use_best_model", True),
            task_type=catboost_params.pop("task_type", "CPU"),
            border_count=catboost_params.pop("border_count", 128),
        )
        # merge any other user-supplied catboost_params
        default_params.update(catboost_params or {})
        if class_weights is not None:
            default_params["class_weights"] = list(class_weights.values())

        self.model = CatBoostClassifier(**default_params)
        self.model.fit(
            X_train_scaled, y_train, eval_set=(X_test_scaled, y_test), plot=False
        )
        self.X_test = X_test_scaled
        self.y_test = y_test
        return self.model

    def predict(self, X: pd.DataFrame):
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict(X_scaled).flatten()

    def predict_proba(self, X: pd.DataFrame):
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict_proba(X_scaled)

    def evaluate(self):
        if self.X_test is None or self.y_test is None:
            raise ValueError("No test set available. Train the model first.")
        y_pred = self.model.predict(self.X_test).flatten()
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        auc = roc_auc_score(self.y_test, y_proba)
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
        }

    def save_model(self, filepath):
        self.model.save_model(filepath)
        return filepath

    def load_model(self, filepath):
        model = CatBoostClassifier()
        model.load_model(filepath)
        self.model = model
        return self.model


# ---------------------------
# Model store (saves ensemble metadata + catboost meta), includes master dataset path implicitly
# ---------------------------
class ModelStore:
    def __init__(self):
        self.models: Dict[str, Optional[Any]] = {}
        self.scaler: Optional[StandardScaler] = None
        self.imputer: Optional[SimpleImputer] = None
        self.feature_names: List[str] = []
        self.training_history: List[Dict] = []
        self.last_train: Optional[Dict[str, Any]] = None
        self.catboost_detector: Optional[BinaryExoplanetDetector] = None
        self.load_initial_models()

    def load_initial_models(self):
        try:
            with open(os.path.join(MODEL_DIR, "models.pkl"), "rb") as f:
                data = pickle.load(f)
                self.models = data.get("models", {})
                self.scaler = data.get("scaler", None)
                self.imputer = data.get("imputer", None)
                self.feature_names = data.get("feature_names", [])
                self.training_history = data.get("history", [])
                cat_meta = data.get("catboost_meta", None)
                if cat_meta:
                    cat_path = cat_meta.get("model_path")
                    feat_names = cat_meta.get("feature_names")
                    scaler_obj = cat_meta.get("scaler")
                    if cat_path and os.path.exists(cat_path):
                        detector = BinaryExoplanetDetector()
                        detector.scaler = scaler_obj
                        detector.feature_names = feat_names
                        detector.load_model(cat_path)
                        self.catboost_detector = detector
                        self.models["CatBoost"] = "loaded"
            print("✓ Loaded pre-trained models from disk (if present)")
        except FileNotFoundError:
            print("⚠ No pre-trained models found. Will train on first data upload.")
            self.models = {
                "HistGradientBoosting": None,
                "RandomForest": None,
                "XGBoost": None,
                "CatBoost": None,
            }

    def save_models(self):
        data = {
            "models": self.models,
            "scaler": self.scaler,
            "imputer": self.imputer,
            "feature_names": self.feature_names,
            "history": self.training_history,
        }
        if (
            self.catboost_detector is not None
            and self.catboost_detector.model is not None
        ):
            cat_path = os.path.join(MODEL_DIR, "catboost_model.cbm")
            self.catboost_detector.save_model(cat_path)
            cat_meta = {
                "model_path": cat_path,
                "feature_names": self.catboost_detector.feature_names,
                "scaler": self.catboost_detector.scaler,
            }
            data["catboost_meta"] = cat_meta
        with open(os.path.join(MODEL_DIR, "models.pkl"), "wb") as f:
            pickle.dump(data, f)
        print("✓ Models metadata saved to disk")


model_store = ModelStore()


# ---------------------------
# Ensemble helpers (preprocessing/train functions adapted to use any disposition col)
# ---------------------------
class PredictionInput(BaseModel):
    features: Dict[str, float]


class HyperparametersInput(BaseModel):
    model_name: str
    hyperparameters: Dict[str, Any]


def preprocess_data(
    df: pd.DataFrame, is_training: bool = False, disposition_col: Optional[str] = None
):
    """
    Preprocess for the ensemble models.
    - If disposition_col provided, uses it to create binary_label; otherwise tries to find one.
    - Returns X_scaled_df (DataFrame) and y (np.array or None)
    """
    print(f"Preprocessing data: {df.shape}")
    df = df.copy()
    disp_col = disposition_col or find_disposition_column(df)
    if disp_col is not None:
        df["binary_label"] = df[disp_col].apply(classify_disposition_value)
        df = df[~df["binary_label"].isnull()]
        y = df["binary_label"].astype(int).values
        print(f"  Created labels from {disp_col}: samples={len(y)}")
    else:
        y = None

    exclude_cols = [
        disp_col,
        "binary_label",
        "kepoi_name",
        "kepler_name",
        "kepid",
        "label",
    ]
    exclude_cols = [c for c in exclude_cols if c]
    X = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]
    print(f"  Numeric features detected: {numeric_cols}")

    if X_numeric.shape[1] == 0:
        raise ValueError(
            "No numeric features found in input. Provide numeric KOI-like features (period, duration, depth, prad, etc.)"
        )

    if is_training:
        model_store.imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(
            model_store.imputer.fit_transform(X_numeric), columns=numeric_cols
        )
        model_store.scaler = StandardScaler()
        X_scaled = model_store.scaler.fit_transform(X_imputed)
        model_store.feature_names = numeric_cols[:]
        X_scaled_df = pd.DataFrame(X_scaled, columns=model_store.feature_names)
        print(
            f"  Fitted imputer+scaler; feature count={len(model_store.feature_names)}"
        )
        return X_scaled_df, y
    else:
        if (
            model_store.imputer is None
            or model_store.scaler is None
            or not model_store.feature_names
        ):
            raise ValueError("Model preprocessing artifacts missing. Train first.")
        X_imputed = pd.DataFrame(
            model_store.imputer.transform(X_numeric), columns=numeric_cols
        )
        for feat in model_store.feature_names:
            if feat not in X_imputed.columns:
                X_imputed[feat] = 0.0
        X_imputed = X_imputed[model_store.feature_names]
        X_scaled = model_store.scaler.transform(X_imputed)
        X_scaled_df = pd.DataFrame(X_scaled, columns=model_store.feature_names)
        print(f"  Transformed using existing scaler; shape={X_scaled_df.shape}")
        return X_scaled_df, y


def train_models(X_train, y_train, hyperparameters=None):
    print("Training ensemble models with SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {X_resampled.shape}")

    default_params = {
        "HistGradientBoosting": {
            "max_iter": 100,
            "learning_rate": 0.1,
            "max_depth": 10,
        },
        "RandomForest": {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5},
        "XGBoost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
    }

    if hyperparameters:
        for k, v in (
            hyperparameters.items() if isinstance(hyperparameters, dict) else []
        ):
            default_params[k] = {**default_params.get(k, {}), **v}

    models = {}
    # HGB
    h = default_params["HistGradientBoosting"]
    models["HistGradientBoosting"] = HistGradientBoostingClassifier(
        max_iter=int(h.get("max_iter", 100)),
        learning_rate=float(h.get("learning_rate", 0.1)),
        max_depth=int(h.get("max_depth", 10)),
        random_state=42,
    )
    models["HistGradientBoosting"].fit(X_resampled, y_resampled)
    # RF
    r = default_params["RandomForest"]
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=int(r.get("n_estimators", 100)),
        max_depth=int(r.get("max_depth", 20)),
        min_samples_split=int(r.get("min_samples_split", 2)),
        random_state=42,
        n_jobs=-1,
    )
    models["RandomForest"].fit(X_resampled, y_resampled)
    # XGBoost
    x = default_params["XGBoost"]
    models["XGBoost"] = XGBClassifier(
        n_estimators=int(x.get("n_estimators", 100)),
        learning_rate=float(x.get("learning_rate", 0.1)),
        max_depth=int(x.get("max_depth", 6)),
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    models["XGBoost"].fit(X_resampled, y_resampled)
    print("✓ Ensemble models trained")
    return models


def train_single_model(model_name: str, X_train, y_train, hyperparams: Dict[str, Any]):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    if model_name == "HistGradientBoosting":
        p = hyperparams or {}
        model = HistGradientBoostingClassifier(
            max_iter=int(p.get("max_iter", 100)),
            learning_rate=float(p.get("learning_rate", 0.1)),
            max_depth=int(p.get("max_depth", 10)),
            random_state=42,
        )
    elif model_name == "RandomForest":
        p = hyperparams or {}
        model = RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 100)),
            max_depth=int(p.get("max_depth", 20)),
            min_samples_split=int(p.get("min_samples_split", 2)),
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "XGBoost":
        p = hyperparams or {}
        model = XGBClassifier(
            n_estimators=int(p.get("n_estimators", 100)),
            learning_rate=float(p.get("learning_rate", 0.1)),
            max_depth=int(p.get("max_depth", 6)),
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )
    else:
        raise ValueError("Unknown model name")
    model.fit(X_resampled, y_resampled)
    return model


# ---------------------------
# API endpoints
# ---------------------------


@app.get("/")
def read_root():
    return {
        "message": "Exoplanet Classifier API",
        "version": "1.4",
        "status": "running",
        "endpoints": [
            "/api/model-status",
            "/api/statistics",
            "/api/predict",
            "/api/predict-batch",
            "/api/train",
            "/api/feature-importance",
            "/api/update-hyperparameters",
            "/api/train-catboost",
            "/api/predict-catboost",
            "/api/catboost-status",
            "/api/explain",
        ],
    }


@app.get("/api/model-status")
def get_model_status():
    is_trained = (
        any(m is not None for m in model_store.models.values())
        or model_store.catboost_detector is not None
    )
    return {
        "trained": is_trained,
        "models": list(model_store.models.keys()),
        "feature_count": (
            len(model_store.feature_names)
            if model_store.feature_names
            else (
                len(model_store.catboost_detector.feature_names)
                if model_store.catboost_detector
                else 0
            )
        ),
        "training_history": model_store.training_history[-10:],
    }


@app.get("/api/statistics")
def get_statistics():
    if not model_store.training_history:
        return {"error": "No training history available"}
    return model_store.training_history[-1]


@app.post("/api/predict")
async def predict_single(data: PredictionInput):
    if (
        not any(m is not None for m in model_store.models.values())
        or model_store.imputer is None
        or model_store.scaler is None
    ):
        raise HTTPException(
            status_code=400,
            detail="Model and preprocessing not trained yet. Please train models first.",
        )
    try:
        df = pd.DataFrame([data.features])
        X_scaled_df, _ = preprocess_data(df, is_training=False)
        preds_by_model = {}
        probas_by_model = {}
        for name, model in model_store.models.items():
            if model is not None and hasattr(model, "predict"):
                pred = int(model.predict(X_scaled_df)[0])
                proba = model.predict_proba(X_scaled_df)[0]
                preds_by_model[name] = pred
                probas_by_model[name] = proba
        if not preds_by_model:
            raise ValueError("No trained ensemble models available.")
        final_pred = int(np.round(np.mean(list(preds_by_model.values()))))
        avg_proba = np.mean(list(probas_by_model.values()), axis=0)
        return {
            "prediction": "PLANET" if final_pred == 1 else "NOT_PLANET",
            "confidence": float(max(avg_proba)),
            "probability_planet": float(avg_proba[1]),
            "probability_not_planet": float(avg_proba[0]),
            "individual_predictions": preds_by_model,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if (
        not any(m is not None for m in model_store.models.values())
        or model_store.imputer is None
        or model_store.scaler is None
    ):
        raise HTTPException(
            status_code=400, detail="Model and preprocessing not trained yet"
        )
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        X_scaled_df, _ = preprocess_data(df, is_training=False)
        all_predictions = []
        for name, model in model_store.models.items():
            if model is not None and hasattr(model, "predict"):
                preds = model.predict(X_scaled_df)
                all_predictions.append(preds)
        final_predictions = np.round(np.mean(all_predictions, axis=0)).astype(int)
        proba_model = None
        for candidate in ["HistGradientBoosting", "RandomForest", "XGBoost"]:
            if model_store.models.get(candidate) is not None:
                proba_model = model_store.models[candidate]
                break
        if proba_model is None:
            raise ValueError("No model available for probabilities")
        probabilities = proba_model.predict_proba(X_scaled_df)
        results = []
        for i, (pred, proba) in enumerate(zip(final_predictions, probabilities)):
            results.append(
                {
                    "index": i,
                    "prediction": "PLANET" if int(pred) == 1 else "NOT_PLANET",
                    "confidence": float(max(proba)),
                    "probability_planet": float(proba[1]),
                }
            )
        return {"predictions": results, "total": len(results)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/train")
async def train_model(
    file: UploadFile = File(...),
    hyperparameters: Optional[str] = Form(None),
    append: Optional[str] = Form("true"),
):
    """
    Full-train: ensemble + CatBoost (optional)
    - If append (form) is "true" (default), uploaded CSV is appended to master_dataset.csv
    - hyperparameters: JSON string including optional "CatBoost" key for CatBoost hyperparams
    """
    try:
        contents = await file.read()
        uploaded_df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = os.path.join(DATA_DIR, f"data_{timestamp}.csv")
        uploaded_df.to_csv(raw_path, index=False)

        do_append = True
        if isinstance(append, str) and append.lower() in ("false", "0", "no"):
            do_append = False

        # Build combined dataset: either uploaded alone or appended to master
        if do_append and os.path.exists(MASTER_DATA_PATH):
            try:
                master_df = pd.read_csv(MASTER_DATA_PATH)
                combined_df = pd.concat(
                    [master_df, uploaded_df], ignore_index=True, sort=False
                )
                # drop exact duplicate rows
                combined_df = combined_df.drop_duplicates()
            except Exception:
                combined_df = pd.concat([uploaded_df], ignore_index=True, sort=False)
        else:
            combined_df = uploaded_df.copy()

        # Save/overwrite master dataset if append True
        if do_append:
            combined_df.to_csv(MASTER_DATA_PATH, index=False)
            print(f"Appended uploaded CSV to master dataset: {MASTER_DATA_PATH}")
        else:
            print(
                "Append disabled: training on uploaded CSV only (not saved to master)."
            )

        # Preprocess ensemble (fits imputer+scaler using combined_df)
        X_scaled_df, y = preprocess_data(combined_df, is_training=True)
        if y is None:
            raise HTTPException(
                status_code=400, detail="No disposition/status column found in CSV."
            )
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
        )

        hyper_dict = None
        if hyperparameters:
            try:
                hyper_dict = json.loads(hyperparameters)
            except Exception:
                hyper_dict = None

        # Train ensemble
        models = train_models(X_train, y_train, hyper_dict)
        model_store.models.update(models)

        # Evaluate ensemble models
        metrics = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics[name] = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
            }

        # Try to train CatBoost on combined_df (if disposition found & features available)
        cat_metrics = None
        try:
            cat_params = (
                hyper_dict.get("CatBoost") if isinstance(hyper_dict, dict) else {}
            )
            detector = BinaryExoplanetDetector()
            X_cat, y_cat = detector.prepare_from_dataframe(combined_df)
            if len(X_cat) >= 10:
                detector.train(
                    X_cat,
                    y_cat,
                    test_size=0.2,
                    use_smote=True,
                    catboost_params=cat_params,
                )
                cat_metrics = detector.evaluate()
                model_store.catboost_detector = detector
                model_store.models["CatBoost"] = "trained"
                metrics["CatBoost"] = cat_metrics
            else:
                print("CatBoost skipped: not enough usable rows for CatBoost training.")
        except Exception as e:
            print(f"Warning: CatBoost training skipped due to: {e}")
            traceback.print_exc()

        # Save training history and last_train
        training_record = {
            "timestamp": timestamp,
            "samples": len(combined_df),
            "features": len(model_store.feature_names),
            "metrics": metrics,
            "hyperparameters": hyper_dict,
            "master_dataset_saved": do_append,
        }
        model_store.training_history.append(training_record)
        model_store.last_train = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "raw_df_path": raw_path,
        }

        model_store.save_models()
        return {
            "status": "success",
            "message": "Models trained (ensemble + optional CatBoost)",
            "metrics": metrics,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


# The /api/train-catboost and /api/predict-catboost endpoints still exist for one-off CatBoost flows:
@app.post("/api/train-catboost")
async def train_catboost(
    file: UploadFile = File(...),
    hyperparameters: Optional[str] = Form(None),
    append: Optional[str] = Form("true"),
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        # optionally append to master if requested
        do_append = True
        if isinstance(append, str) and append.lower() in ("false", "0", "no"):
            do_append = False
        if do_append and os.path.exists(MASTER_DATA_PATH):
            master_df = pd.read_csv(MASTER_DATA_PATH)
            combined_df = pd.concat(
                [master_df, df], ignore_index=True, sort=False
            ).drop_duplicates()
            combined_df.to_csv(MASTER_DATA_PATH, index=False)
        else:
            combined_df = df.copy()
            if do_append:
                combined_df.to_csv(MASTER_DATA_PATH, index=False)
        detector = BinaryExoplanetDetector()
        X, y = detector.prepare_from_dataframe(combined_df)
        if len(X) < 10:
            raise HTTPException(
                status_code=400, detail="Not enough samples to train CatBoost."
            )
        cat_params = {}
        if hyperparameters:
            try:
                cat_params = json.loads(hyperparameters)
            except Exception:
                cat_params = {}
        detector.train(X, y, test_size=0.2, use_smote=True, catboost_params=cat_params)
        metrics = detector.evaluate()
        model_store.catboost_detector = detector
        model_store.models["CatBoost"] = "trained"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_record = {
            "timestamp": timestamp,
            "samples": int(len(X)),
            "features": len(detector.feature_names),
            "metrics": {"CatBoost": metrics},
            "hyperparameters": {"CatBoost": cat_params},
        }
        model_store.training_history.append(training_record)
        model_store.save_models()
        return {
            "status": "success",
            "model": "CatBoost",
            "metrics": metrics,
            "features": detector.feature_names,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict-catboost")
async def predict_catboost(data: PredictionInput):
    if (
        model_store.catboost_detector is None
        or model_store.catboost_detector.model is None
    ):
        raise HTTPException(status_code=400, detail="CatBoost model not trained.")
    try:
        detector = model_store.catboost_detector
        df = pd.DataFrame([data.features])
        for feat in detector.feature_names:
            if feat not in df.columns:
                df[feat] = 0.0
        df = df[detector.feature_names]
        proba = detector.predict_proba(df)[0]
        pred = int(detector.predict(df)[0])
        return {
            "prediction": "EXOPLANET" if pred == 1 else "FALSE_POSITIVE",
            "probability_exoplanet": float(proba[1]),
            "probability_false_positive": float(proba[0]),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/catboost-status")
def catboost_status():
    if (
        model_store.catboost_detector is None
        or model_store.catboost_detector.model is None
    ):
        return {"trained": False}
    return {"trained": True, "features": model_store.catboost_detector.feature_names}


@app.post("/api/update-hyperparameters")
async def update_hyperparameters(data: HyperparametersInput):
    if data.model_name not in model_store.models:
        raise HTTPException(status_code=404, detail="Model not found")
    if model_store.last_train is None:
        raise HTTPException(
            status_code=400,
            detail="No previous training dataset available. Retrain full models first.",
        )
    try:
        if data.model_name == "CatBoost":
            # Retrain CatBoost using master/raw path if present
            raw_path = model_store.last_train.get("raw_df_path")
            # If master exists, prefer master (because that's the cumulative dataset)
            train_df_path = (
                MASTER_DATA_PATH if os.path.exists(MASTER_DATA_PATH) else raw_path
            )
            if train_df_path is None or not os.path.exists(train_df_path):
                raise HTTPException(
                    status_code=400, detail="No available CSV for CatBoost retrain."
                )
            df = pd.read_csv(train_df_path)
            detector = BinaryExoplanetDetector()
            X_cat, y_cat = detector.prepare_from_dataframe(df)
            cat_params = data.hyperparameters or {}
            detector.train(
                X_cat, y_cat, test_size=0.2, use_smote=True, catboost_params=cat_params
            )
            metrics = detector.evaluate()
            model_store.catboost_detector = detector
            model_store.models["CatBoost"] = "trained"
            record = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "samples": int(len(X_cat)),
                "updated_model": "CatBoost",
                "metrics": {"CatBoost": metrics},
                "hyperparameters": {"CatBoost": data.hyperparameters},
            }
            model_store.training_history.append(record)
            model_store.save_models()
            return {"status": "success", "model": "CatBoost", "metrics": metrics}
        else:
            X_train = model_store.last_train["X_train"]
            X_test = model_store.last_train["X_test"]
            y_train = model_store.last_train["y_train"]
            y_test = model_store.last_train["y_test"]
            new_model = train_single_model(
                data.model_name, X_train, y_train, data.hyperparameters
            )
            model_store.models[data.model_name] = new_model
            y_pred = new_model.predict(X_test)
            y_proba = new_model.predict_proba(X_test)[:, 1]
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
            }
            record = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "samples": int(len(y_train) + len(y_test)),
                "updated_model": data.model_name,
                "metrics": {data.model_name: metrics},
                "hyperparameters": {data.model_name: data.hyperparameters},
            }
            model_store.training_history.append(record)
            model_store.save_models()
            return {"status": "success", "model": data.model_name, "metrics": metrics}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/feature-importance")
def get_feature_importance():
    if model_store.models.get("RandomForest") is None:
        raise HTTPException(
            status_code=400,
            detail="Model not trained yet or RandomForest not available",
        )
    importances = model_store.models["RandomForest"].feature_importances_
    features = model_store.feature_names
    feature_importance = [
        {"feature": feat, "importance": float(imp)}
        for feat, imp in sorted(
            zip(features, importances), key=lambda x: x[1], reverse=True
        )
    ]
    return {"feature_importance": feature_importance[:20]}


# ---------- NEW/REPLACEMENT: /api/explain endpoint (SHAP + ELI5) ----------
@app.post("/api/explain")
async def explain(data: PredictionInput):
    """
    Returns SHAP explanations (top 5) for a single-row prediction.
    Uses the same preprocessing pipeline as /api/predict so probabilities and SHAP
    are consistent with prediction outputs. Returns both a full 'eli5' (with prob)
    and 'eli5_short' (no probability) for UI display.
    """
    try:
        try:
            import shap
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="shap package not installed on server. Run `pip install shap`.",
            )

        df = pd.DataFrame([data.features])

        # Helper: safely convert a possibly-nested value into a Python float
        def to_scalar(x):
            try:
                arr = np.asarray(x)
                if arr.size == 0:
                    raise ValueError("empty array")
                if arr.size == 1:
                    return float(arr.item())
                return float(arr.flatten()[0])
            except Exception:
                return float(x)

        def make_eli5_full(
            pred_label: str, prob: float, top_feats: List[Dict[str, float]]
        ):
            if not top_feats:
                return f"Model predicted {pred_label} with probability {prob:.2f}."
            top_names = [f["feature"] for f in top_feats[:2]]
            directions = []
            for f in top_feats[:2]:
                sign = "increases" if f["value"] > 0 else "decreases"
                directions.append(f"{f['feature']} {sign} the chance")
            return (
                f"The model predicted {pred_label} (prob {prob:.2f}). "
                f"The strongest influences were {', '.join(top_names)}. "
                f"In short: {directions[0]} and {directions[1]}."
            )

        def make_eli5_short(pred_label: str, top_feats: List[Dict[str, float]]):
            if not top_feats:
                return f"The model predicted {pred_label}."
            top_names = [f["feature"] for f in top_feats[:2]]
            directions = []
            for f in top_feats[:2]:
                sign = "increases" if f["value"] > 0 else "decreases"
                directions.append(f"{f['feature']} {sign} the chance")
            return (
                f"The model predicted {pred_label}. "
                f"The strongest influences were {', '.join(top_names)}. "
                f"In short: {directions[0]} and {directions[1]}."
            )

        # Prefer RandomForest for SHAP explanations if available
        rf = model_store.models.get("RandomForest")
        if rf is not None:
            # Ensure preprocessing artifacts exist
            if (
                model_store.imputer is None
                or model_store.scaler is None
                or not model_store.feature_names
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Model preprocessing artifacts missing. Train models first.",
                )

            # Use same preprocessing as /api/predict to get scaled features
            X_scaled_df, _ = preprocess_data(df, is_training=False)

            # Use the scaled features for prediction + SHAP so outputs match /api/predict
            try:
                proba_raw = rf.predict_proba(X_scaled_df)[0]
            except Exception as e:
                # if RF predict_proba fails, raise a meaningful error
                raise HTTPException(
                    status_code=400, detail=f"RF predict_proba failed: {e}"
                )

            proba_arr = np.asarray(proba_raw).flatten()
            planet_prob = (
                to_scalar(proba_arr[1])
                if proba_arr.size > 1
                else to_scalar(proba_arr[0])
            )

            pred_raw = rf.predict(X_scaled_df)[0]
            pred = int(np.asarray(pred_raw).item())

            explainer = shap.TreeExplainer(rf)
            raw_shap = explainer.shap_values(X_scaled_df)

            if isinstance(raw_shap, list):
                shap_for_pos = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
            else:
                shap_for_pos = raw_shap

            shap_row = np.asarray(shap_for_pos[0]).flatten()
            features = model_store.feature_names or []
            shap_list = []
            for f, v in zip(features, shap_row):
                try:
                    val = to_scalar(v)
                except Exception:
                    val = 0.0
                shap_list.append({"feature": f, "value": float(val)})

            top5 = sorted(shap_list, key=lambda x: abs(x["value"]), reverse=True)[:5]
            eli5_full = make_eli5_full(
                "PLANET" if pred == 1 else "NOT_PLANET", planet_prob, top5
            )
            eli5_short = make_eli5_short("PLANET" if pred == 1 else "NOT_PLANET", top5)

            return {
                "prediction": "PLANET" if pred == 1 else "NOT_PLANET",
                "probabilities": {
                    "planet": float(planet_prob),
                    "not_planet": float(to_scalar(proba_arr[0])),
                },
                "shap": top5,
                "eli5": eli5_full,
                "eli5_short": eli5_short,
            }

        # Fallback to CatBoost detector (it already handles its own scaling)
        detector = model_store.catboost_detector
        if detector is not None and detector.model is not None:
            feats = detector.feature_names
            for feat in feats:
                if feat not in df.columns:
                    df[feat] = 0.0
            X = df[feats]

            proba_raw = detector.predict_proba(X)[0]
            proba_arr = np.asarray(proba_raw).flatten()
            planet_prob = (
                to_scalar(proba_arr[1])
                if proba_arr.size > 1
                else to_scalar(proba_arr[0])
            )

            pred_raw = detector.predict(X)[0]
            pred = int(np.asarray(pred_raw).item())

            explainer = shap.TreeExplainer(detector.model)
            raw_shap = explainer.shap_values(X)
            if isinstance(raw_shap, list):
                shap_for_pos = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
            else:
                shap_for_pos = raw_shap

            shap_row = np.asarray(shap_for_pos[0]).flatten()
            shap_list = []
            for f, v in zip(feats, shap_row):
                try:
                    val = to_scalar(v)
                except Exception:
                    val = 0.0
                shap_list.append({"feature": f, "value": float(val)})

            top5 = sorted(shap_list, key=lambda x: abs(x["value"]), reverse=True)[:5]
            eli5_full = make_eli5_full(
                "EXOPLANET" if pred == 1 else "FALSE_POSITIVE", planet_prob, top5
            )
            eli5_short = make_eli5_short(
                "EXOPLANET" if pred == 1 else "FALSE_POSITIVE", top5
            )
            return {
                "prediction": "EXOPLANET" if pred == 1 else "FALSE_POSITIVE",
                "probabilities": {
                    "planet": float(planet_prob),
                    "not_planet": float(to_scalar(proba_arr[0])),
                },
                "shap": top5,
                "eli5": eli5_full,
                "eli5_short": eli5_short,
            }

        raise HTTPException(
            status_code=400,
            detail="No suitable model available for explanation. Train models first.",
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


# ---------- existing manual data endpoint ---------- #
class ManualDataInput(BaseModel):
    koi_period: float
    koi_duration: float
    koi_depth: float
    koi_prad: float
    koi_teq: Optional[float] = None
    koi_insol: Optional[float] = None
    koi_model_snr: Optional[float] = None
    koi_steff: Optional[float] = None
    koi_slogg: Optional[float] = None
    koi_srad: Optional[float] = None
    label: str  # "PLANET" or "NOT_PLANET"


@app.post("/api/add-manual-data")
async def add_manual_data(data: ManualDataInput):
    try:
        data_dict = data.dict()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame([data_dict])
        # append to master dataset too
        if os.path.exists(MASTER_DATA_PATH):
            master_df = pd.read_csv(MASTER_DATA_PATH)
            new_master = pd.concat(
                [master_df, df], ignore_index=True, sort=False
            ).drop_duplicates()
            new_master.to_csv(MASTER_DATA_PATH, index=False)
        else:
            df.to_csv(MASTER_DATA_PATH, index=False)
        # also store manual_data.csv
        if os.path.exists(os.path.join(DATA_DIR, "manual_data.csv")):
            df.to_csv(
                os.path.join(DATA_DIR, "manual_data.csv"),
                mode="a",
                header=False,
                index=False,
            )
        else:
            df.to_csv(os.path.join(DATA_DIR, "manual_data.csv"), index=False)
        return {
            "status": "success",
            "message": "Data point added successfully",
            "data": data_dict,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


def to_json_number(x):
    """
    Convert a numeric value to a JSON-safe Python float or None.
    Replaces NaN/Inf with None.
    """
    try:
        # convert numpy scalar -> Python float
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def sanitize_array(arr):
    """Convert array-like (numpy/list) to list of JSON-safe floats (None for non-finite)."""
    if arr is None:
        return None
    a = np.asarray(arr)
    out = []
    for v in a.flat:
        out.append(to_json_number(v))
    # if 1-D originally, return as list; if multi-dimensional, still flattened (we use only 1D arrays here)
    return out


# --- Robust eval metrics endpoint ---
@app.get("/api/eval-metrics")
def api_eval_metrics():
    """
    Return evaluation metrics for the frontend (JSON-safe).
    - metrics: accuracy, precision, recall, f1_score, roc_auc (numbers or null)
    - confusion: tn, fp, fn, tp (ints)
    - roc: { fpr: [...], tpr: [...], thresholds: [...], auc: float|null }
    - feature_importance: [{feature, importance}, ...]
    - samples: number of test samples
    """
    try:
        if model_store.last_train is None:
            return {"error": "No training split available. Run /api/train first."}

        # Prefer RandomForest for feature importance and probabilities; otherwise use any model that supports predict_proba
        rf = model_store.models.get("RandomForest")
        if rf is None:
            for m in model_store.models.values():
                if m is not None:
                    rf = m
                    break

        if rf is None:
            return {"error": "No trained models available."}

        # Pull last train/test
        X_test = model_store.last_train.get("X_test")
        y_test = model_store.last_train.get("y_test")

        if X_test is None or y_test is None:
            return {"error": "Training split incomplete: missing X_test or y_test."}

        # Convert to numpy arrays (handles pandas/numpy)
        X_test_vals = np.asarray(X_test)
        y_test_vals = np.asarray(y_test)

        # Ensure y_test is 1-D
        if y_test_vals.ndim > 1:
            y_test_vals = y_test_vals.ravel()

        # If y_test are not numeric, try to convert (e.g., bool/str)
        try:
            y_test_vals = y_test_vals.astype(int)
        except Exception:
            # fallback: try numeric conversion
            y_test_vals = np.array(
                [
                    int(float(v)) if (v is not None and str(v) != "") else 0
                    for v in y_test_vals
                ]
            )

        samples = int(len(y_test_vals))

        # Predictions (wrap in try/except to guard odd models)
        try:
            y_pred = rf.predict(X_test_vals)
            y_pred = np.asarray(y_pred).ravel()
        except Exception as e:
            # If predict fails, return partial info
            return {"error": f"Model prediction failed for eval metrics: {str(e)}"}

        # Confusion matrix (safe)
        try:
            cm = confusion_matrix(y_test_vals, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = (
                    int(cm[0, 0]),
                    int(cm[0, 1]),
                    int(cm[1, 0]),
                    int(cm[1, 1]),
                )
            else:
                # handle unexpected shapes
                tn = int(cm[0, 0]) if cm.size > 0 else 0
                fp = int(cm[0, 1]) if cm.size > 1 else 0
                fn = int(cm[1, 0]) if cm.size > 2 else 0
                tp = int(cm[1, 1]) if cm.size > 3 else 0
        except Exception:
            tn = fp = fn = tp = 0

        # Basic metrics (guard divisions and non-finite values)
        def safe_metric(fn, y_true, y_pred):
            try:
                v = fn(y_true, y_pred)
                return to_json_number(v)
            except Exception:
                return None

        accuracy = safe_metric(accuracy_score, y_test_vals, y_pred)
        precision = safe_metric(
            lambda a, b: precision_score(a, b, zero_division=0), y_test_vals, y_pred
        )
        recall = safe_metric(
            lambda a, b: recall_score(a, b, zero_division=0), y_test_vals, y_pred
        )
        f1s = safe_metric(
            lambda a, b: f1_score(a, b, zero_division=0), y_test_vals, y_pred
        )

        # ROC (only if we have probabilities and both classes present)
        roc_data = None
        try:
            if len(np.unique(y_test_vals)) > 1 and hasattr(rf, "predict_proba"):
                y_proba = np.asarray(rf.predict_proba(X_test_vals)[:, 1]).ravel()

                # mask out non-finite probabilities
                finite_mask = np.isfinite(y_proba)
                if (
                    np.sum(finite_mask) >= 2
                    and np.unique(y_test_vals[finite_mask]).size > 1
                ):
                    fpr, tpr, thresholds = roc_curve(
                        y_test_vals[finite_mask], y_proba[finite_mask]
                    )
                    auc = roc_auc_score(y_test_vals[finite_mask], y_proba[finite_mask])
                    roc_data = {
                        "fpr": sanitize_array(fpr),
                        "tpr": sanitize_array(tpr),
                        "thresholds": sanitize_array(thresholds),
                        "auc": to_json_number(auc),
                    }
                else:
                    # probabilities were not usable (constant or non-finite)
                    roc_data = None
            else:
                roc_data = None
        except Exception as e:
            # If predict_proba / ROC computation fails, skip ROC
            print("Warning: ROC computation failed:", str(e))
            roc_data = None

        # Feature importance (if available)
        feature_importance = []
        try:
            if hasattr(rf, "feature_importances_") and model_store.feature_names:
                importances = np.asarray(rf.feature_importances_)
                features = list(model_store.feature_names)
                # align lengths defensively
                n = min(len(features), importances.size)
                feature_importance = [
                    {
                        "feature": features[i],
                        "importance": to_json_number(importances[i]),
                    }
                    for i in range(n)
                ]
        except Exception:
            feature_importance = []

        response = {
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1s,
                "roc_auc": roc_data["auc"] if roc_data else None,
            },
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "roc": roc_data,
            "feature_importance": feature_importance,
            "samples": samples,
        }

        return response

    except Exception as e:
        import traceback as _tb

        _tb.print_exc()
        # Return a 500 with message
        raise HTTPException(status_code=500, detail=f"Eval metrics error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 80)
    print("🚀 STARTING SERVER")
    print("=" * 80 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
