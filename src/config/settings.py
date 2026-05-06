"""
Configuration settings for the Agentic AI Diabetes Risk Stratification System.
Centralized configuration for paths, model parameters, and thresholds.
"""

import os
from pathlib import Path

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "src" / "models" / "saved_models"
LOGS_DIR = PROJECT_ROOT / "logs"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, LOGS_DIR, VECTOR_DB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== DATA CONFIGURATION ====================
DATASET_PATH = DATA_RAW_DIR / "diabetes_data.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "diabetes_processed.csv"
TRAIN_DATA_PATH = DATA_PROCESSED_DIR / "train_data.csv"
TEST_DATA_PATH = DATA_PROCESSED_DIR / "test_data.csv"

# Data features
TARGET_COLUMN = "Outcome"
FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    "BMI_Category", "Glucose_Category"
]

# ==================== MODEL CONFIGURATION ====================
MODEL_TYPE = "xgboost"  # Options: "logistic", "random_forest", "xgboost"
MODEL_PATH = MODELS_DIR / f"{MODEL_TYPE}_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"

# Model hyperparameters
MODEL_PARAMS = {
    "random_state": 42,
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "verbosity": 0
}

# ==================== RISK STRATIFICATION THRESHOLDS ====================
RISK_THRESHOLDS = {
    "low": (0.0, 0.25),
    "moderate": (0.25, 0.50),
    "high": (0.50, 0.75),
    "very_high": (0.75, 1.0)
}

RISK_CATEGORIES = {
    "low": "Low Risk",
    "moderate": "Moderate Risk",
    "high": "High Risk",
    "very_high": "Very High Risk"
}

# ==================== EXPLAINABILITY CONFIGURATION ====================
SHAP_SAMPLE_SIZE = 100  # Number of samples for SHAP background
FEATURE_IMPORTANCE_TOP_N = 5  # Top N features to explain

# ==================== MONITORING CONFIGURATION ====================
MONITORING_WINDOW_MONTHS = 3  # Track data over last N months
ALERT_THRESHOLD = 0.05  # Alert if risk increases by 5% or more

# ==================== RECOMMENDATION SYSTEM ====================
VECTOR_DB_TYPE = "faiss"  # Options: "faiss", "chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default recommendations by risk level
RECOMMENDATIONS_BASE = {
    "low": {
        "diet": ["Maintain balanced diet", "Include whole grains and vegetables"],
        "exercise": ["150 mins moderate activity per week", "Include strength training"],
        "monitoring": ["Annual health checkup", "Monitor weight quarterly"]
    },
    "moderate": {
        "diet": ["Reduce sugar and refined carbs", "Increase fiber intake"],
        "exercise": ["200 mins moderate activity per week", "Add resistance training"],
        "monitoring": ["Semi-annual health checkup", "Monitor weight bi-monthly"]
    },
    "high": {
        "diet": ["Follow Mediterranean or DASH diet", "Consult dietitian"],
        "exercise": ["Consult doctor before exercise", "Gradual activity increase"],
        "monitoring": ["Quarterly health checkup", "Monthly blood tests"]
    },
    "very_high": {
        "diet": ["Medical nutrition therapy required", "Strict carb limitation"],
        "exercise": ["Medical clearance required", "Supervised exercise program"],
        "monitoring": ["Monthly physician visits", "Regular lab monitoring"]
    }
}

# ==================== LOGGING CONFIGURATION ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "diabetes_ai_system.log"

# ==================== MEDICAL DISCLAIMERS ====================
MEDICAL_DISCLAIMER = """
⚠️ MEDICAL DISCLAIMER ⚠️
This AI system is intended for DECISION SUPPORT ONLY and is NOT a medical diagnosis tool.
- Predictions are based on statistical models and should not replace clinical judgment.
- Always consult with qualified healthcare providers before taking medical actions.
- This system is not a substitute for professional medical advice, diagnosis, or treatment.
- For medical emergencies, contact your local emergency services immediately.
"""

# ==================== SYSTEM CONFIGURATION ====================
RANDOM_SEED = 42
VERBOSE = True
TEST_MODE = False  # Set to True for development/testing
