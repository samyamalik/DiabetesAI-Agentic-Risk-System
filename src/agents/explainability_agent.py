"""
Explainability & Clinical Transparency Agent (Agent C).

Explains diabetes risk predictions using SHAP and generates
clinician-friendly natural language explanations.

Consumed by:
    - Recommendation Agent
    - Dashboard / Streamlit UI
"""

import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.config.settings import FEATURE_IMPORTANCE_TOP_N, RISK_CATEGORIES

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────
_AGENTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _AGENTS_DIR.parent
_PROJECT_ROOT = _SRC_DIR.parent
_PLOTS_DIR = _PROJECT_ROOT / "logs" / "shap_plots"
_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_PATH = _SRC_DIR / "models" / "best_model.pkl"
DEFAULT_SCALER_PATH = _SRC_DIR / "models" / "scaler.pkl"
DEFAULT_ENCODER_PATH = _SRC_DIR / "models" / "encoders.pkl"
DATA_PATH = _PROJECT_ROOT / "data" / "processed" / "cleaned_diabetes.csv"

# ── Clinical context for features ───────────────────────────────────
FEATURE_CLINICAL = {
    "Glucose": {"label": "Glucose level", "unit": "mg/dL",
                "high": 126, "desc_high": "prediabetic/diabetic range"},
    "BMI": {"label": "BMI", "unit": "",
            "high": 30, "desc_high": "obesity – a major diabetes risk factor"},
    "Age": {"label": "Age", "unit": "years",
            "high": 45, "desc_high": "older age is associated with higher risk"},
    "BloodPressure": {"label": "Blood Pressure", "unit": "mmHg",
                      "high": 90, "desc_high": "elevated blood pressure"},
    "Insulin": {"label": "Insulin", "unit": "μU/mL",
                "high": 166, "desc_high": "elevated insulin levels"},
    "SkinThickness": {"label": "Skin Thickness", "unit": "mm",
                      "high": 40, "desc_high": "increased subcutaneous fat"},
    "DiabetesPedigreeFunction": {"label": "Family History Score", "unit": "",
                                 "high": 0.5, "desc_high": "strong genetic predisposition"},
    "Pregnancies": {"label": "Number of Pregnancies", "unit": "",
                    "high": 6, "desc_high": "gestational history may elevate risk"},
    "BMI_Category": {"label": "BMI Category", "unit": "", "high": None, "desc_high": ""},
    "Glucose_Category": {"label": "Glucose Category", "unit": "", "high": None, "desc_high": ""},
}

DISCLAIMER = (
    "Note: This is a decision-support explanation, NOT a medical diagnosis. "
    "Always consult a qualified healthcare provider."
)


class ExplainabilityAgent:
    """
    Agent C — generates SHAP-based explanations for diabetes risk predictions.

    Usage (standalone):
        agent = ExplainabilityAgent()
        result = agent.explain(patient_dict)

    Usage (pipeline):
        result = agent.run_pipeline(model, X, risk_scores, risk_categories)
    """

    # ── __init__ ─────────────────────────────────────────────────────
    def __init__(self):
        """Load model, scaler, encoders and initialise SHAP explainer."""
        self.model = None
        self.scaler = None
        self.encoders: Dict = {}
        self.feature_names: List[str] = []
        self.shap_explainer = None
        self.shap_values = None
        self.explanations: List[Dict] = []

        # Try to load saved artefacts
        self._load_artefacts()

    # ── Private helpers ──────────────────────────────────────────────
    def _load_artefacts(self) -> None:
        """Load model, scaler, and encoders from disk."""
        try:
            if DEFAULT_MODEL_PATH.exists():
                self.model = joblib.load(DEFAULT_MODEL_PATH)
                logger.info(f"Model loaded: {type(self.model).__name__}")
            if DEFAULT_SCALER_PATH.exists():
                self.scaler = joblib.load(DEFAULT_SCALER_PATH)
            if DEFAULT_ENCODER_PATH.exists():
                self.encoders = joblib.load(DEFAULT_ENCODER_PATH)
            if self.scaler and hasattr(self.scaler, "feature_names_in_"):
                self.feature_names = list(self.scaler.feature_names_in_)
        except Exception as exc:
            logger.warning(f"Could not load artefacts: {exc}")

    def _init_shap(self, model, X_background: pd.DataFrame) -> bool:
        """Initialise the appropriate SHAP explainer for *model*."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not installed – using fallback.")
            return False
        try:
            name = type(model).__name__
            if any(t in name for t in ("RandomForest", "XGB", "GradientBoosting", "DecisionTree")):
                self.shap_explainer = shap.TreeExplainer(model)
                logger.info(f"TreeExplainer initialised for {name}")
            elif "Logistic" in name or "Linear" in name:
                self.shap_explainer = shap.LinearExplainer(model, X_background)
                logger.info(f"LinearExplainer initialised for {name}")
            else:
                self.shap_explainer = shap.KernelExplainer(model.predict_proba, X_background)
                logger.info(f"KernelExplainer initialised for {name}")
            return True
        except Exception as exc:
            logger.warning(f"SHAP init failed: {exc}")
            return False

    def _preprocess_patient(self, data) -> pd.DataFrame:
        """Convert a patient dict/Series into a scaled DataFrame row."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            df = data.to_frame().T
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(f"Unsupported type: {type(data)}")

        if "Outcome" in df.columns:
            df = df.drop(columns=["Outcome"])

        for col, le in self.encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))

        if self.feature_names:
            for c in self.feature_names:
                if c not in df.columns:
                    df[c] = 0
            df = df[self.feature_names]

        if self.scaler is not None:
            arr = self.scaler.transform(df)
            df = pd.DataFrame(arr, columns=self.feature_names)
        return df

    # ── Core: explain() ──────────────────────────────────────────────
    def explain(self, patient_data, model=None) -> Dict:
        """
        Explain a single patient prediction.

        Args:
            patient_data: dict, Series, or single-row DataFrame.
            model: optional model override (uses self.model if None).

        Returns:
            Dict with keys: shap_values, top_features,
            explanation_text, probability, risk_category, disclaimer.
        """
        mdl = model if model is not None else self.model
        if mdl is None:
            raise RuntimeError("No model available. Load one first.")

        # Preprocess
        X_row = self._preprocess_patient(patient_data)

        # Ensure explainer is ready
        if self.shap_explainer is None:
            bg = self._get_background_data()
            self._init_shap(mdl, bg)

        # Probability & category
        prob = float(mdl.predict_proba(X_row)[0, 1]) if hasattr(mdl, "predict_proba") else float(mdl.predict(X_row)[0])
        category = self._classify_risk(prob)

        # SHAP values
        shap_vals, top_pos, top_neg = self._compute_shap(X_row)

        # Natural-language explanation
        raw_features = patient_data if isinstance(patient_data, dict) else patient_data.to_dict()
        explanation_text = self._generate_explanation(prob, category, top_pos, top_neg, raw_features)

        return {
            "probability": prob,
            "risk_category": category,
            "shap_values": shap_vals,
            "top_positive_features": top_pos,
            "top_negative_features": top_neg,
            "top_features": top_pos,  # backward compat with main_pipeline
            "explanation_text": explanation_text,
            "disclaimer": DISCLAIMER,
        }

    # ── SHAP computation ─────────────────────────────────────────────
    def _compute_shap(self, X_row: pd.DataFrame):
        """Return (shap_dict, top_positive, top_negative)."""
        features = list(X_row.columns)

        if self.shap_explainer is None or not SHAP_AVAILABLE:
            return self._fallback_importance(X_row)

        try:
            sv = self.shap_explainer.shap_values(X_row)
            # TreeExplainer for RF returns [arr_class0, arr_class1]; pick positive class
            if isinstance(sv, list):
                sv = np.array(sv[1]) if len(sv) > 1 else np.array(sv[0])
            else:
                sv = np.array(sv)
            # Flatten to 1-D (single-row input → shape (1, n_features))
            vals = sv.flatten()

            shap_dict = {f: float(vals[i]) for i, f in enumerate(features)}
            sorted_feats = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
            top_pos = [(f, v) for f, v in sorted_feats if v > 0][:5]
            top_neg = [(f, v) for f, v in sorted_feats if v < 0][-5:]
            return shap_dict, top_pos, top_neg
        except Exception as exc:
            logger.warning(f"SHAP computation failed: {exc}; using fallback.")
            return self._fallback_importance(X_row)

    def _fallback_importance(self, X_row):
        """Variance-based fallback when SHAP is unavailable."""
        features = list(X_row.columns)
        scores = {f: float(abs(X_row[f].values[0])) for f in features}
        sorted_f = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return scores, sorted_f[:5], []

    # ── Background data helper ────────────────────────────────────────
    def _get_background_data(self) -> pd.DataFrame:
        """Load & preprocess a background sample for SHAP."""
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            if "Outcome" in df.columns:
                df = df.drop(columns=["Outcome"])
            for col, le in self.encoders.items():
                if col in df.columns:
                    df[col] = le.transform(df[col].astype(str))
            if self.feature_names:
                df = df[[c for c in self.feature_names if c in df.columns]]
            if self.scaler:
                df = pd.DataFrame(self.scaler.transform(df), columns=df.columns)
            return df.sample(min(100, len(df)), random_state=42)
        return pd.DataFrame()

    # ── Natural-language explanation ──────────────────────────────────
    @staticmethod
    def _classify_risk(prob: float) -> str:
        if prob < 0.25:   return "Low"
        if prob < 0.50:   return "Moderate"
        if prob < 0.75:   return "High"
        return "Very High"

    def _generate_explanation(self, prob, category, top_pos, top_neg, raw_feats) -> str:
        """Build a clinician-friendly paragraph."""
        lines = []
        display = RISK_CATEGORIES.get(category.lower(), category)
        lines.append(f"RISK ASSESSMENT: {display} (probability {prob:.1%})\n")

        # Top risk-increasing features
        if top_pos:
            contributors = []
            for fname, _ in top_pos[:5]:
                clin = FEATURE_CLINICAL.get(fname, {})
                label = clin.get("label", fname)
                val = raw_feats.get(fname)
                unit = clin.get("unit", "")
                if val is not None and isinstance(val, (int, float)):
                    high_thresh = clin.get("high")
                    if high_thresh and val > high_thresh:
                        contributors.append(f"elevated {label} ({val}{' ' + unit if unit else ''} — {clin.get('desc_high', '')})")
                    else:
                        contributors.append(f"{label} ({val}{' ' + unit if unit else ''})")
                else:
                    contributors.append(label)

            summary = ", ".join(contributors[:-1])
            if len(contributors) > 1:
                summary += f", and {contributors[-1]}"
            else:
                summary = contributors[0]

            lines.append(f"KEY RISK DRIVERS: The patient's risk is primarily driven by {summary}.")
            lines.append("   These factors significantly contributed to the prediction.\n")

        # Protective factors
        if top_neg:
            protectors = []
            for fname, _ in top_neg[:3]:
                label = FEATURE_CLINICAL.get(fname, {}).get("label", fname)
                protectors.append(label.lower())
            lines.append(f"PROTECTIVE FACTORS: {', '.join(protectors)} helped lower the risk score.\n")

        # Category-specific guidance
        lines.append("CLINICAL SUMMARY:")
        guidance = {
            "Low":       "Current metrics suggest low diabetes risk. Maintain a healthy lifestyle with regular checkups.",
            "Moderate":  "Moderate risk detected. Lifestyle modifications (diet, exercise) and periodic monitoring are recommended.",
            "High":      "High risk identified. Proactive clinical intervention and close monitoring are strongly recommended.",
            "Very High": "Very high risk. Immediate medical consultation and comprehensive metabolic evaluation are advised.",
        }
        lines.append(f"   {guidance.get(category, '')}")
        lines.append(f"\n{DISCLAIMER}")
        return "\n".join(lines)

    # ── Visualisation ────────────────────────────────────────────────
    def generate_summary_plot(self, model=None, save_path=None) -> str:
        """Generate and save a SHAP summary (beeswarm) plot."""
        mdl = model or self.model
        if not SHAP_AVAILABLE or mdl is None:
            return ""
        bg = self._get_background_data()
        if bg.empty:
            return ""
        if self.shap_explainer is None:
            self._init_shap(mdl, bg)
        try:
            sv = self.shap_explainer.shap_values(bg)
            if isinstance(sv, list):
                sv = sv[1]
            plt.figure(figsize=(10, 6))
            shap.summary_plot(sv, bg, show=False, plot_size=None)
            plt.title("SHAP Feature Importance (Global)", fontsize=14, pad=15)
            plt.tight_layout()
            path = save_path or str(_PLOTS_DIR / "shap_summary.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Summary plot saved → {path}")
            return path
        except Exception as exc:
            logger.warning(f"Summary plot failed: {exc}")
            return ""

    def generate_waterfall_plot(self, patient_data, model=None, save_path=None) -> str:
        """Generate and save a SHAP waterfall plot for one patient."""
        mdl = model or self.model
        if not SHAP_AVAILABLE or mdl is None:
            return ""
        X_row = self._preprocess_patient(patient_data)
        if self.shap_explainer is None:
            bg = self._get_background_data()
            self._init_shap(mdl, bg)
        try:
            explanation = self.shap_explainer(X_row)
            if hasattr(explanation, "__getitem__") and len(explanation.shape) > 1:
                # Binary classification: pick positive class
                try:
                    expl_obj = explanation[0, :, 1]
                except Exception:
                    expl_obj = explanation[0]
            else:
                expl_obj = explanation[0]
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(expl_obj, show=False)
            plt.title("SHAP Waterfall — Individual Prediction", fontsize=13, pad=15)
            plt.tight_layout()
            path = save_path or str(_PLOTS_DIR / "shap_waterfall.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Waterfall plot saved → {path}")
            return path
        except Exception as exc:
            logger.warning(f"Waterfall plot failed: {exc}")
            return ""

    # ── Pipeline interface (backward-compatible with main_pipeline) ──
    def run_pipeline(self, model, X: pd.DataFrame, risk_scores: np.ndarray,
                     risk_categories: List[str],
                     X_background: Optional[pd.DataFrame] = None) -> Dict:
        """
        Full explainability pipeline called by main_pipeline.py.

        Returns dict with: status, shap_analysis, explanations, top_features.
        """
        logger.info("=" * 50)
        logger.info("EXPLAINABILITY AGENT: Starting pipeline")
        logger.info("=" * 50)

        # Init SHAP
        if X_background is None:
            X_background = X.sample(min(100, len(X)), random_state=42)
        self._init_shap(model, X_background)

        # Global SHAP
        shap_report = self._global_shap(model, X, X_background)

        # Per-sample explanations
        explanations = []
        for idx in range(min(3, len(X))):
            sample = X.iloc[idx].to_dict()
            category = risk_categories[idx] if idx < len(risk_categories) else "Unknown"
            score = float(risk_scores[idx]) if idx < len(risk_scores) else 0.0
            top_pos = self.get_top_features(shap_report["feature_importance"])
            text = self._generate_explanation(score, category, top_pos, [], sample)
            explanations.append({
                "sample_index": idx,
                "explanation": text,
                "risk_score": score,
                "risk_category": category,
            })

        report = {
            "status": "success",
            "shap_analysis": shap_report,
            "explanations": explanations,
            "top_features": self.get_top_features(shap_report["feature_importance"], 5),
        }
        logger.info("EXPLAINABILITY AGENT: Pipeline completed")
        return report

    def _global_shap(self, model, X, X_bg) -> Dict:
        """Compute global feature importance via SHAP."""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return self._generate_fallback_importance(X)
        try:
            sv = self.shap_explainer.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            importance = np.abs(sv).mean(axis=0)
            return {
                "method": "SHAP (TreeExplainer)",
                "feature_importance": dict(zip(X.columns, importance.tolist())),
                "shap_values_available": True,
            }
        except Exception as exc:
            logger.warning(f"Global SHAP failed: {exc}")
            return self._generate_fallback_importance(X)

    @staticmethod
    def _generate_fallback_importance(X: pd.DataFrame) -> Dict:
        importance = {}
        total_var = X.select_dtypes(include=[np.number]).var().sum()
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                importance[col] = float(X[col].var() / total_var) if total_var > 0 else 0.1
            else:
                importance[col] = float(X[col].nunique() / len(X))
        return {"method": "Fallback (variance)", "feature_importance": importance, "shap_values_available": False}

    @staticmethod
    def get_top_features(feature_importance: Dict, top_n: int = FEATURE_IMPORTANCE_TOP_N) -> List[Tuple[str, float]]:
        return sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]


# ══════════════════════════════════════════════════════════════════════
# Standalone test
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(levelname)s  %(message)s")

    agent = ExplainabilityAgent()

    sample = {
        "Pregnancies": 5, "Glucose": 166.0, "BloodPressure": 72.0,
        "SkinThickness": 19.0, "Insulin": 175.0, "BMI": 25.8,
        "DiabetesPedigreeFunction": 0.587, "Age": 51,
        "BMI_Category": "Overweight", "Glucose_Category": "Prediabetic",
    }

    print("\n" + "=" * 65)
    print("  🔬 EXPLAINABILITY AGENT — Self-Test")
    print("=" * 65)

    result = agent.explain(sample)

    print(f"\n📊 Probability: {result['probability']:.4f}")
    print(f"📊 Risk Level : {result['risk_category']}")
    print(f"\n🔝 Top Risk-Increasing Features:")
    for fname, val in result["top_positive_features"]:
        print(f"   {fname:30s}  SHAP = {val:+.4f}")
    if result["top_negative_features"]:
        print(f"\n🛡️  Top Protective Features:")
        for fname, val in result["top_negative_features"]:
            print(f"   {fname:30s}  SHAP = {val:+.4f}")
    print(f"\n{result['explanation_text']}")

    # Generate plots
    print("\n📈 Generating SHAP plots …")
    s_path = agent.generate_summary_plot()
    if s_path:
        print(f"   Summary plot   → {s_path}")
    w_path = agent.generate_waterfall_plot(sample)
    if w_path:
        print(f"   Waterfall plot → {w_path}")

    print("\n" + "=" * 65)
    print("  ✅ EXPLAINABILITY AGENT TEST COMPLETE")
    print("=" * 65)
