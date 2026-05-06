"""
Data Ingestion & Preprocessing Agent (Agent A).

Entry point of the agentic pipeline. Converts raw PIMA Diabetes data
into a clean, ML-ready format.

Raw input:   data/raw/diabetes.csv
Clean output: data/processed/cleaned_diabetes.csv

Pipeline position:
  **Ingestion** → Risk → Explain → Recommend → Monitoring
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from src.config.settings import (
    DATASET_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN
)

logger = logging.getLogger(__name__)

# ── Columns where zero is clinically invalid ─────────────────────────
ZERO_INVALID_COLS = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]


class IngestionAgent:
    """
    Agent A — loads, cleans, engineers, and saves the diabetes dataset.

    Usage:
        agent = IngestionAgent()
        df = agent.run()                     # full pipeline

        # Or with custom paths:
        agent = IngestionAgent(
            input_path="data/raw/diabetes.csv",
            output_path="data/processed/cleaned_diabetes.csv"
        )
        df = agent.run()
    """

    # ── __init__ ─────────────────────────────────────────────────────
    def __init__(self, input_path: Optional[str] = None,
                 output_path: Optional[str] = None):
        """
        Initialize the Ingestion Agent.

        Args:
            input_path:  path to raw CSV  (default: settings.DATASET_PATH)
            output_path: path to save cleaned CSV (default: settings.PROCESSED_DATA_PATH)
        """
        self.input_path = Path(input_path) if input_path else DATASET_PATH
        self.output_path = Path(output_path) if output_path else PROCESSED_DATA_PATH
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.data_quality_report: Dict = {}

    # ── 1. Load Data ─────────────────────────────────────────────────
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load the raw CSV dataset.

        Args:
            file_path: optional override for input path.

        Returns:
            Raw DataFrame.
        """
        path = Path(file_path) if file_path else self.input_path

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")

        self.raw_data = pd.read_csv(path)
        logger.info(f"Loaded dataset from {path}")
        logger.info(f"  Shape: {self.raw_data.shape}")
        logger.info(f"  Columns: {list(self.raw_data.columns)}")
        return self.raw_data

    # ── 2. Clean Data ────────────────────────────────────────────────
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the PIMA dataset:
          1. Replace invalid zeros with NaN in clinical columns
          2. Fill NaN using median (robust to outliers)

        Args:
            df: raw DataFrame.

        Returns:
            Cleaned DataFrame with no missing values.
        """
        df = df.copy()

        # Step 2a: Replace invalid zeros → NaN
        zeros_replaced = {}
        for col in ZERO_INVALID_COLS:
            if col in df.columns:
                count = (df[col] == 0).sum()
                if count > 0:
                    df[col] = df[col].replace(0, np.nan)
                    zeros_replaced[col] = int(count)

        if zeros_replaced:
            logger.info(f"  Replaced invalid zeros → NaN: {zeros_replaced}")

        # Step 2b: Fill NaN with median
        missing_before = df.isnull().sum()
        missing_cols = missing_before[missing_before > 0]

        if not missing_cols.empty:
            logger.info(f"  Missing values before fill:\n{missing_cols.to_string()}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        remaining = df.isnull().sum().sum()
        logger.info(f"  Missing values after median fill: {remaining}")

        self.data_quality_report["zeros_replaced"] = zeros_replaced
        self.data_quality_report["missing_filled"] = True

        return df

    # ── 3. Feature Engineering ───────────────────────────────────────
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features:
          • BMI_Category:     Normal / Overweight / Obese
          • Glucose_Category: Normal / Prediabetic / Diabetic

        Args:
            df: cleaned DataFrame.

        Returns:
            DataFrame with new feature columns.
        """
        df = df.copy()

        # BMI_Category
        if "BMI" in df.columns:
            df["BMI_Category"] = pd.cut(
                df["BMI"],
                bins=[0, 25, 30, float("inf")],
                labels=["Normal", "Overweight", "Obese"],
                right=False,
            )
            df["BMI_Category"] = df["BMI_Category"].astype(str)
            logger.info(f"  BMI_Category distribution:\n"
                        f"{df['BMI_Category'].value_counts().to_string()}")

        # Glucose_Category
        if "Glucose" in df.columns:
            df["Glucose_Category"] = pd.cut(
                df["Glucose"],
                bins=[0, 140, 200, float("inf")],
                labels=["Normal", "Prediabetic", "Diabetic"],
                right=False,
            )
            df["Glucose_Category"] = df["Glucose_Category"].astype(str)
            logger.info(f"  Glucose_Category distribution:\n"
                        f"{df['Glucose_Category'].value_counts().to_string()}")

        self.data_quality_report["features_engineered"] = True
        return df

    # ── 4. Data Validation ───────────────────────────────────────────
    def validate(self, df: pd.DataFrame) -> Dict:
        """
        Validate the processed DataFrame.

        Checks:
          • No remaining missing values
          • Correct dtypes
          • Expected columns present

        Returns:
            Validation report dict.
        """
        report = {
            "total_samples": len(df),
            "total_features": len(df.columns),
            "columns": list(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
        }

        # Assertions
        assert report["missing_values"] == 0, \
            f"Still {report['missing_values']} missing values!"

        expected = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
                    "Outcome", "BMI_Category", "Glucose_Category"]
        for col in expected:
            assert col in df.columns, f"Expected column '{col}' not found!"

        logger.info(f"  Validation passed — {report['total_samples']} samples, "
                    f"{report['total_features']} features, 0 missing values")

        self.data_quality_report["validation"] = report
        return report

    # ── 5. Save Data ─────────────────────────────────────────────────
    def save_data(self, df: pd.DataFrame,
                  output_path: Optional[str] = None) -> str:
        """
        Save cleaned DataFrame to CSV.

        Args:
            df: processed DataFrame.
            output_path: override path.

        Returns:
            Path string where the file was saved.
        """
        path = Path(output_path) if output_path else self.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"  Saved processed data → {path}")
        return str(path)

    # ── 6. run() — Full Pipeline ─────────────────────────────────────
    def run(self, input_file: Optional[str] = None) -> pd.DataFrame:
        """
        Execute the full ingestion pipeline:
            Load → Clean → Feature Engineer → Validate → Save

        Args:
            input_file: optional override for raw CSV path.

        Returns:
            Processed DataFrame (also saved to disk).
        """
        logger.info("=" * 55)
        logger.info("INGESTION AGENT: Starting data ingestion pipeline")
        logger.info("=" * 55)

        # Step 1: Load
        logger.info("Step 1 — Loading raw data …")
        df = self.load_data(input_file)

        # Step 2: Clean
        logger.info("Step 2 — Cleaning data (zeros → NaN → median) …")
        df = self.clean_data(df)

        # Step 3: Feature engineering
        logger.info("Step 3 — Feature engineering …")
        df = self.feature_engineering(df)

        # Step 4: Validate
        logger.info("Step 4 — Validating processed data …")
        self.validate(df)

        # Step 5: Save
        logger.info("Step 5 — Saving processed data …")
        self.save_data(df)

        self.processed_data = df
        logger.info("=" * 55)
        logger.info("INGESTION AGENT: Pipeline completed ✅")
        logger.info(f"  Output: {self.output_path}")
        logger.info(f"  Shape:  {df.shape}")
        logger.info("=" * 55)

        return df

    # ── Backward-compatible run_pipeline() for main_pipeline.py ──────
    def run_pipeline(self, input_file: Optional[str] = None) -> Dict:
        """
        Wrapper that returns a dict (expected by main_pipeline.py).
        """
        df = self.run(input_file)

        return {
            "status": "success",
            "processed_data": df,
            "data_shape": df.shape,
            "quality_report": self.data_quality_report,
        }

    # ── Accessors ────────────────────────────────────────────────────
    def get_processed_data(self) -> pd.DataFrame:
        """Return processed DataFrame."""
        return self.processed_data

    def get_quality_report(self) -> Dict:
        """Return data quality report."""
        return self.data_quality_report


# ══════════════════════════════════════════════════════════════════════
# Standalone test
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    agent = IngestionAgent(
        input_path="data/raw/diabetes.csv",
        output_path="data/processed/cleaned_diabetes.csv",
    )

    df = agent.run()

    print("\n" + "=" * 55)
    print("  📥 INGESTION AGENT — Results")
    print("=" * 55)
    print(f"\nShape: {df.shape}")
    print(f"\nHead:\n{df.head().to_string()}")
    print(f"\nMissing values:\n{df.isnull().sum().to_string()}")
    print(f"\nDtypes:\n{df.dtypes.to_string()}")
    print(f"\n✅ Agent A complete — data ready for downstream agents.")
