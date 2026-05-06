"""
Data preprocessing utilities for diabetes risk stratification.
Handles missing values, outliers, and data normalization.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing operations."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        self.numeric_columns = []
        self.categorical_columns = []

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input dataframe
            strategy: "mean", "median", "forward_fill", or "drop"
        
        Returns:
            DataFrame with handled missing values
        """
        df_processed = df.copy()
        
        missing_count = df_processed.isnull().sum()
        if missing_count.sum() > 0:
            logger.warning(f"Missing values detected:\n{missing_count[missing_count > 0]}")
        
        if strategy == "mean":
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
        elif strategy == "median":
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
        elif strategy == "forward_fill":
            df_processed = df_processed.fillna(method="ffill").fillna(method="bfill")
        elif strategy == "drop":
            df_processed = df_processed.dropna()
        
        logger.info(f"Missing values handled using '{strategy}' strategy")
        return df_processed

    def detect_outliers(self, df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> dict:
        """
        Detect outliers in numeric columns.
        
        Args:
            df: Input dataframe
            method: "iqr" or "zscore"
            threshold: IQR multiplier or z-score threshold
        
        Returns:
            Dictionary with outlier information
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols:
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                outlier_mask = (df[col] < lower) | (df[col] > upper)
            elif method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > threshold
            
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                outliers[col] = {
                    "count": outlier_count,
                    "percentage": (outlier_count / len(df)) * 100
                }
        
        if outliers:
            logger.warning(f"Outliers detected:\n{outliers}")
        
        return outliers

    def normalize_data(self, X_train: np.ndarray, X_test: np.ndarray = None) -> tuple:
        """
        Normalize numeric features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
        
        Returns:
            Tuple of normalized arrays (X_train_scaled, X_test_scaled or None)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Data normalization completed")
        return X_train_scaled, X_test_scaled

    def encode_categorical(self, df: pd.DataFrame, columns: list, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input dataframe
            columns: List of categorical columns to encode
            fit: If True, fit new encoders; if False, use existing
        
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        
        for col in columns:
            if fit:
                self.encoders[col] = LabelEncoder()
                df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                if col in self.encoders:
                    df_encoded[col] = self.encoders[col].transform(df_encoded[col].astype(str))
        
        logger.info(f"Encoded {len(columns)} categorical columns")
        return df_encoded

    def get_scaler(self):
        """Return the fitted scaler."""
        return self.scaler

    def get_encoders(self):
        """Return all fitted encoders."""
        return self.encoders
