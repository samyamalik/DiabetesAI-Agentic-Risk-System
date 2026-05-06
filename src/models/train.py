"""
Model Training Pipeline for Type 2 Diabetes Risk Stratification (Agent B).

This module implements a complete ML training pipeline:
  1. Load cleaned dataset
  2. Encode categorical features
  3. Train-test split (80/20)
  4. Handle class imbalance (class_weight='balanced')
  5. Train 3 models: Logistic Regression, Random Forest, XGBoost
  6. Evaluate with Accuracy, Precision, Recall, F1, ROC-AUC
  7. Select best model (by ROC-AUC)
  8. Save best model to disk
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import warnings

# Try importing XGBoost (optional dependency)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ==================== PATH CONFIGURATION ====================
# Resolve paths relative to this file: src/models/train.py
_THIS_DIR = Path(__file__).resolve().parent            # src/models/
_SRC_DIR = _THIS_DIR.parent                            # src/
_PROJECT_ROOT = _SRC_DIR.parent                        # AgentMinor/

DATA_PATH = _PROJECT_ROOT / "data" / "processed" / "cleaned_diabetes.csv"
BEST_MODEL_PATH = _SRC_DIR / "models" / "best_model.pkl"
SCALER_PATH = _SRC_DIR / "models" / "scaler.pkl"
ENCODER_PATH = _SRC_DIR / "models" / "encoders.pkl"

# Constants
TARGET_COLUMN = "Outcome"
CATEGORICAL_COLS = ["BMI_Category", "Glucose_Category"]
RANDOM_STATE = 42
TEST_SIZE = 0.20


class ModelTrainer:
    """
    End-to-end training pipeline for diabetes risk prediction.

    Usage:
        trainer = ModelTrainer()
        results = trainer.run_training_pipeline()
    """

    def __init__(self, data_path: str = None):
        """
        Initialize the trainer.

        Args:
            data_path: Optional override for the dataset CSV path.
        """
        self.data_path = Path(data_path) if data_path else DATA_PATH
        self.scaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns = []       # stored after encoding
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0
        self.results_table = None       # DataFrame with per-model metrics

    # ------------------------------------------------------------------
    # STEP 1: Load Data
    # ------------------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """Load the cleaned diabetes dataset."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Please run the EDA pipeline first."
            )

        df = pd.read_csv(self.data_path)
        df = df.dropna()
        logger.info(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        return df

    # ------------------------------------------------------------------
    # STEP 2: Feature Preparation (encode categoricals)
    # ------------------------------------------------------------------
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features (X) and target (y).
        Encode categorical columns using LabelEncoder.

        Returns:
            X: Encoded feature DataFrame
            y: Target Series
        """
        # Separate target
        y = df[TARGET_COLUMN].copy()
        X = df.drop(columns=[TARGET_COLUMN]).copy()

        # Encode categorical columns
        for col in CATEGORICAL_COLS:
            if col in X.columns:
                le = LabelEncoder()
                if col == "BMI_Category":
                    le.fit(["Normal", "Obese", "Overweight"])
                elif col == "Glucose_Category":
                    le.fit(["Diabetic", "Normal", "Prediabetic"])
                else:
                    le.fit(X[col].astype(str))
                X[col] = le.transform(X[col].astype(str))
                self.encoders[col] = le
                logger.info(f"Encoded '{col}': {list(le.classes_)}")

        self.feature_columns = list(X.columns)
        logger.info(f"Features ({len(self.feature_columns)}): {self.feature_columns}")
        return X, y

    # ------------------------------------------------------------------
    # STEP 3: Train-Test Split
    # ------------------------------------------------------------------
    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """80/20 stratified split."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        logger.info(
            f"Split → Train: {X_train.shape[0]}, Test: {X_test.shape[0]}"
        )
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # STEP 4: Check Class Imbalance
    # ------------------------------------------------------------------
    @staticmethod
    def check_class_distribution(y: pd.Series) -> None:
        """Print class distribution and imbalance ratio."""
        counts = y.value_counts()
        total = len(y)
        print("\n📊 Class Distribution:")
        for label, count in counts.items():
            pct = count / total * 100
            print(f"   Class {label}: {count} ({pct:.1f}%)")
        ratio = counts.min() / counts.max()
        print(f"   Imbalance ratio (minority/majority): {ratio:.2f}")
        if ratio < 0.8:
            print("   ⚠️  Imbalanced → using class_weight='balanced'\n")
        else:
            print("   ✅ Balanced distribution\n")

    # ------------------------------------------------------------------
    # STEP 5: Train Multiple Models
    # ------------------------------------------------------------------
    def train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, object]:
        """
        Train Logistic Regression, Random Forest, and XGBoost.
        Uses class_weight='balanced' to handle class imbalance.
        """
        models: Dict[str, object] = {}

        # --- 1. Logistic Regression (baseline) ---
        print("🔹 Training Logistic Regression …")
        lr = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        lr.fit(X_train, y_train)
        models["Logistic Regression"] = lr
        logger.info("Logistic Regression trained.")

        # --- 2. Random Forest Classifier ---
        print("🔹 Training Random Forest …")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        models["Random Forest"] = rf
        logger.info("Random Forest trained.")

        # --- 3. XGBoost Classifier ---
        if XGB_AVAILABLE:
            print("🔹 Training XGBoost …")
            # Compute scale_pos_weight for imbalance handling
            neg_count = int(np.sum(y_train == 0))
            pos_count = int(np.sum(y_train == 1))
            scale_pos = neg_count / max(pos_count, 1)

            xgb_clf = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                verbosity=0,
            )
            xgb_clf.fit(X_train, y_train)
            models["XGBoost"] = xgb_clf
            logger.info("XGBoost trained.")
        else:
            print("⚠️  XGBoost not installed – skipping.")

        return models

    # ------------------------------------------------------------------
    # STEP 6: Evaluate Models
    # ------------------------------------------------------------------
    def evaluate_models(
        self,
        models: Dict[str, object],
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """
        Evaluate each model and return a comparison DataFrame.

        Metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC.
        """
        rows = []

        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_proba = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba")
                else y_pred
            )

            rows.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, y_proba),
            })

        results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
        self.results_table = results
        return results

    # ------------------------------------------------------------------
    # STEP 7: Select Best Model
    # ------------------------------------------------------------------
    def select_best_model(
        self, models: Dict[str, object], results: pd.DataFrame
    ) -> None:
        """Pick the model with the highest ROC-AUC."""
        best_row = results.iloc[0]
        self.best_model_name = best_row["Model"]
        self.best_score = best_row["ROC-AUC"]
        self.best_model = models[self.best_model_name]
        print(f"\n🏆 Best Model: {self.best_model_name}  (ROC-AUC = {self.best_score:.4f})")

    # ------------------------------------------------------------------
    # STEP 8: Save Model
    # ------------------------------------------------------------------
    def save_model(self) -> Path:
        """Save best model, scaler, and label encoders to disk."""
        if self.best_model is None:
            raise ValueError("No model to save – run training first.")

        BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.best_model, BEST_MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.encoders, ENCODER_PATH)

        print(f"💾 Model saved → {BEST_MODEL_PATH}")
        print(f"💾 Scaler saved → {SCALER_PATH}")
        print(f"💾 Encoders saved → {ENCODER_PATH}")
        logger.info(f"Best model ({self.best_model_name}) saved to {BEST_MODEL_PATH}")
        return BEST_MODEL_PATH

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------
    def run_training_pipeline(self) -> Dict:
        """
        Execute the complete training pipeline end-to-end.

        Returns:
            dict with status, best_model name, scores, and model_path.
        """
        print("=" * 60)
        print("  🧠 MODEL TRAINING PIPELINE – Agent B")
        print("=" * 60)

        # Step 1: Load data
        print("\n📥 Step 1: Loading data …")
        df = self.load_data()

        # Step 2: Feature preparation
        print("🔧 Step 2: Preparing features …")
        X, y = self.prepare_features(df)

        # Step 3: Train-test split
        print("🔀 Step 3: Splitting data (80/20) …")
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Step 4: Check class imbalance
        print("⚖️  Step 4: Checking class imbalance …")
        self.check_class_distribution(y_train)

        # Scale features (needed for Logistic Regression; tree models are
        # scale-invariant, but we scale uniformly for consistency).
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Step 5: Train models
        print("🤖 Step 5: Training models …")
        models = self.train_models(X_train_scaled, y_train)

        # Step 6: Evaluate models
        print("\n📊 Step 6: Evaluating models …")
        results = self.evaluate_models(models, X_test_scaled, y_test)
        print("\n" + results.to_string(index=False))

        # Step 7: Select best model
        self.select_best_model(models, results)

        # Step 8: Save model
        model_path = self.save_model()

        print("\n" + "=" * 60)
        print("  ✅ TRAINING PIPELINE COMPLETE")
        print("=" * 60)

        return {
            "status": "success",
            "best_model": self.best_model_name,
            "best_roc_auc": float(self.best_score),
            "all_scores": results.to_dict(orient="records"),
            "model_path": str(model_path),
            "feature_columns": self.feature_columns,
        }


# ======================================================================
# Standalone execution
# ======================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    trainer = ModelTrainer()
    result = trainer.run_training_pipeline()
