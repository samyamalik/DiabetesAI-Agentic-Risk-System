"""
Risk Stratification Agent (Agent B) for Type 2 Diabetes.

Loads the best trained model and predicts diabetes risk for individual
patients or batches.  Converts model probability into human-readable
risk categories:

    0.00 – 0.25  →  Low
    0.25 – 0.50  →  Moderate
    0.50 – 0.75  →  High
    0.75 – 1.00  →  Very High

This agent is consumed by downstream agents:
    • Explainability Agent
    • Recommendation Agent
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ==================== PATH CONFIGURATION ====================
_THIS_DIR = Path(__file__).resolve().parent          # src/agents/
_SRC_DIR = _THIS_DIR.parent                          # src/
_PROJECT_ROOT = _SRC_DIR.parent                      # AgentMinor/

DEFAULT_MODEL_PATH = _SRC_DIR / "models" / "best_model.pkl"
DEFAULT_SCALER_PATH = _SRC_DIR / "models" / "scaler.pkl"
DEFAULT_ENCODER_PATH = _SRC_DIR / "models" / "encoders.pkl"

# ==================== RISK THRESHOLDS ====================
RISK_THRESHOLDS = {
    "Low":       (0.00, 0.25),
    "Moderate":  (0.25, 0.50),
    "High":      (0.50, 0.75),
    "Very High": (0.75, 1.00),
}


class RiskAgent:
    """
    Agent responsible for diabetes risk prediction and stratification.

    Usage:
        agent = RiskAgent()                        # loads saved model
        prob, category = agent.predict(patient)    # dict or DataFrame row
    """

    # ------------------------------------------------------------------
    # __init__: load trained model, scaler, and encoders
    # ------------------------------------------------------------------
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        encoder_path: Optional[str] = None,
    ):
        """
        Initialize the RiskAgent and load the trained model from disk.

        Args:
            model_path:  Path to the saved model .pkl file.
            scaler_path: Path to the saved StandardScaler .pkl file.
            encoder_path: Path to the saved label-encoders dict .pkl file.
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.scaler_path = Path(scaler_path) if scaler_path else DEFAULT_SCALER_PATH
        self.encoder_path = Path(encoder_path) if encoder_path else DEFAULT_ENCODER_PATH

        self.model = None
        self.scaler = None
        self.encoders: Dict = {}
        self.model_loaded = False

        # Feature names expected by the model (set after loading encoders)
        self.feature_names: Optional[List[str]] = None

        # Attempt to load model at init
        self.load_model()

    # ------------------------------------------------------------------
    # load_model
    # ------------------------------------------------------------------
    def load_model(self) -> bool:
        """
        Load the trained model, scaler, and encoders from disk.

        Returns:
            True if everything loaded successfully.
        """
        try:
            if not self.model_path.exists():
                logger.warning(f"Model file not found: {self.model_path}")
                return False

            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded: {type(self.model).__name__} from {self.model_path}")

            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Scaler loaded.")

            if self.encoder_path.exists():
                self.encoders = joblib.load(self.encoder_path)
                logger.info(f"Encoders loaded for: {list(self.encoders.keys())}")

            # Determine expected feature names from the scaler if available
            if self.scaler is not None and hasattr(self.scaler, "feature_names_in_"):
                self.feature_names = list(self.scaler.feature_names_in_)

            self.model_loaded = True
            return True

        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")
            return False

    # ------------------------------------------------------------------
    # _preprocess_input: align a patient dict/row to model features
    # ------------------------------------------------------------------
    def _preprocess_input(
        self, data: Union[Dict, pd.Series, pd.DataFrame]
    ) -> np.ndarray:
        """
        Convert a patient record (dict, Series, or single-row DataFrame)
        into a scaled numpy array ready for model.predict_proba().
        """
        # Convert to single-row DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            df = data.to_frame().T
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(f"Unsupported input type: {type(data)}")

        # Drop target column if accidentally present
        if "Outcome" in df.columns:
            df = df.drop(columns=["Outcome"])

        # Encode categorical columns using the saved encoders
        for col, le in self.encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))

        # If we know the expected feature order, reorder columns
        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0  # fill missing feature with 0
            df = df[self.feature_names]

        # Scale
        if self.scaler is not None:
            arr = self.scaler.transform(df)
        else:
            arr = df.values.astype(float)

        return arr

    # ------------------------------------------------------------------
    # predict: core prediction method
    # ------------------------------------------------------------------
    def predict(
        self, data: Union[Dict, pd.Series, pd.DataFrame]
    ) -> Tuple[float, str]:
        """
        Predict diabetes risk for a single patient.

        Args:
            data: Patient features as a dict, pandas Series, or
                  single-row DataFrame.

        Returns:
            (probability, risk_category)
            probability  – float in [0, 1]
            risk_category – one of 'Low', 'Moderate', 'High', 'Very High'
        """
        if not self.model_loaded:
            raise RuntimeError(
                "Model not loaded. Ensure best_model.pkl exists "
                "(run ModelTrainer first)."
            )

        # Preprocess
        X = self._preprocess_input(data)

        # Predict probability of positive class (diabetes)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0, 1]
        else:
            proba = float(self.model.predict(X)[0])

        # Map probability to risk category
        category = self._classify_risk(proba)

        return float(proba), category

    # ------------------------------------------------------------------
    # predict_batch: convenience for multiple patients
    # ------------------------------------------------------------------
    def predict_batch(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict risk for every row in a DataFrame.

        Returns a DataFrame with columns:
            Probability, Risk_Category
        """
        probas = []
        cats = []
        for _, row in df.iterrows():
            p, c = self.predict(row)
            probas.append(p)
            cats.append(c)

        return pd.DataFrame({
            "Probability": probas,
            "Risk_Category": cats,
        })

    # ------------------------------------------------------------------
    # _classify_risk: probability → category string
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_risk(probability: float) -> str:
        """
        Convert a probability score to a risk category.

        Risk levels:
            0.00 – 0.25  → Low
            0.25 – 0.50  → Moderate
            0.50 – 0.75  → High
            0.75 – 1.00  → Very High
        """
        for label, (lo, hi) in RISK_THRESHOLDS.items():
            if lo <= probability < hi:
                return label
        # Edge case: probability == 1.0
        return "Very High"

    # ------------------------------------------------------------------
    # Convenience accessors for downstream agents
    # ------------------------------------------------------------------
    def get_model(self):
        """Return the underlying sklearn/xgb model object."""
        return self.model

    def get_feature_names(self) -> Optional[List[str]]:
        """Return the feature names the model expects."""
        return self.feature_names

    def get_risk_thresholds(self) -> Dict:
        """Return the risk threshold mapping."""
        return RISK_THRESHOLDS


# ======================================================================
# STEP 10: Quick self-test when run directly
# ======================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    print("=" * 60)
    print("  🧪 RISK AGENT – Self-Test")
    print("=" * 60)

    # Instantiate agent (loads model automatically)
    agent = RiskAgent()

    if not agent.model_loaded:
        print("❌ Model not found. Please run train.py first.")
        exit(1)

    # Sample patient record
    sample_patient = {
        "Pregnancies": 5,
        "Glucose": 166.0,
        "BloodPressure": 72.0,
        "SkinThickness": 19.0,
        "Insulin": 175.0,
        "BMI": 25.8,
        "DiabetesPedigreeFunction": 0.587,
        "Age": 51,
        "BMI_Category": "Overweight",
        "Glucose_Category": "Prediabetic",
    }

    print(f"\n📋 Sample Patient: {sample_patient}")
    probability, risk_category = agent.predict(sample_patient)
    print(f"\n🎯 Prediction Results:")
    print(f"   Probability : {probability:.4f}")
    print(f"   Risk Level  : {risk_category}")
    print("=" * 60)
