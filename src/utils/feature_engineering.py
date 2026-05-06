"""
Feature engineering utilities for diabetes risk stratification.
Creates derived features and domain-specific transformations.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering operations."""

    @staticmethod
    def calculate_bmi(height_cm: float, weight_kg: float) -> float:
        """
        Calculate Body Mass Index.
        BMI = weight (kg) / (height (m))^2
        """
        if height_cm <= 0 or weight_kg <= 0:
            return np.nan
        height_m = height_cm / 100
        return weight_kg / (height_m ** 2)

    @staticmethod
    def categorize_bmi(bmi: float) -> str:
        """Categorize BMI into health categories."""
        if np.isnan(bmi):
            return "Unknown"
        elif bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 25:
            return "Normal"
        elif 25 <= bmi < 30:
            return "Overweight"
        else:
            return "Obese"

    @staticmethod
    def categorize_glucose(glucose_mg_dl: float) -> str:
        """Categorize fasting glucose levels."""
        if np.isnan(glucose_mg_dl):
            return "Unknown"
        elif glucose_mg_dl < 100:
            return "Normal"
        elif 100 <= glucose_mg_dl < 126:
            return "Prediabetic"
        else:
            return "Diabetic"

    @staticmethod
    def categorize_blood_pressure(systolic: float, diastolic: float) -> str:
        """Categorize blood pressure."""
        if np.isnan(systolic) or np.isnan(diastolic):
            return "Unknown"
        
        if systolic < 120 and diastolic < 80:
            return "Normal"
        elif systolic < 130 and diastolic < 80:
            return "Elevated"
        elif systolic < 140 or diastolic < 90:
            return "Stage1_Hypertension"
        else:
            return "Stage2_Hypertension"

    @staticmethod
    def calculate_pulse_pressure(systolic: float, diastolic: float) -> float:
        """Calculate pulse pressure (systolic - diastolic)."""
        if np.isnan(systolic) or np.isnan(diastolic):
            return np.nan
        return systolic - diastolic

    @staticmethod
    def calculate_mean_arterial_pressure(systolic: float, diastolic: float) -> float:
        """Calculate Mean Arterial Pressure."""
        if np.isnan(systolic) or np.isnan(diastolic):
            return np.nan
        return (systolic + 2 * diastolic) / 3

    @staticmethod
    def create_age_groups(age: float) -> str:
        """Create age group categories."""
        if np.isnan(age):
            return "Unknown"
        elif age < 30:
            return "Youth"
        elif 30 <= age < 45:
            return "Early_Middle_Age"
        elif 45 <= age < 60:
            return "Middle_Age"
        else:
            return "Senior"

    @staticmethod
    def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features from existing columns.
        
        Args:
            df: Input dataframe
        
        Returns:
            DataFrame with additional interaction features
        """
        df_features = df.copy()
        
        # Age-BMI interaction
        if "age" in df_features.columns and "bmi" in df_features.columns:
            df_features["age_bmi_interaction"] = df_features["age"] * df_features["bmi"]
        
        # Glucose-BMI interaction
        if "fasting_glucose" in df_features.columns and "bmi" in df_features.columns:
            df_features["glucose_bmi_interaction"] = df_features["fasting_glucose"] * df_features["bmi"]
        
        # Blood pressure interaction
        if "blood_pressure_systolic" in df_features.columns and "blood_pressure_diastolic" in df_features.columns:
            df_features["bp_ratio"] = (
                df_features["blood_pressure_systolic"] / 
                (df_features["blood_pressure_diastolic"] + 1)  # Add 1 to avoid division by zero
            )
        
        logger.info("Interaction features created")
        return df_features

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            df: Input dataframe
        
        Returns:
            Dataframe with engineered features
        """
        df_engineered = df.copy()
        
        # BMI categorization
        if "bmi" in df_engineered.columns:
            df_engineered["bmi_category"] = df_engineered["bmi"].apply(
                FeatureEngineer.categorize_bmi
            )
        
        # Glucose categorization
        if "fasting_glucose" in df_engineered.columns:
            df_engineered["glucose_category"] = df_engineered["fasting_glucose"].apply(
                FeatureEngineer.categorize_glucose
            )
        
        # Blood pressure categorization
        if "blood_pressure_systolic" in df_engineered.columns and "blood_pressure_diastolic" in df_engineered.columns:
            df_engineered["bp_category"] = df_engineered.apply(
                lambda row: FeatureEngineer.categorize_blood_pressure(
                    row["blood_pressure_systolic"], 
                    row["blood_pressure_diastolic"]
                ),
                axis=1
            )
            
            # Pulse pressure
            df_engineered["pulse_pressure"] = df_engineered.apply(
                lambda row: FeatureEngineer.calculate_pulse_pressure(
                    row["blood_pressure_systolic"],
                    row["blood_pressure_diastolic"]
                ),
                axis=1
            )
            
            # Mean arterial pressure
            df_engineered["mean_arterial_pressure"] = df_engineered.apply(
                lambda row: FeatureEngineer.calculate_mean_arterial_pressure(
                    row["blood_pressure_systolic"],
                    row["blood_pressure_diastolic"]
                ),
                axis=1
            )
        
        # Age groups
        if "age" in df_engineered.columns:
            df_engineered["age_group"] = df_engineered["age"].apply(
                FeatureEngineer.create_age_groups
            )
        
        # Interaction features
        df_engineered = FeatureEngineer.create_interaction_features(df_engineered)
        
        logger.info("All feature engineering transformations applied")
        return df_engineered
