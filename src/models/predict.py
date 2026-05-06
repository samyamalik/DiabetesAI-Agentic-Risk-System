"""
Model prediction module for diabetes risk stratification.
Loads trained model and generates predictions.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Union, Tuple

from src.config.settings import MODEL_PATH, SCALER_PATH, ENCODER_PATH

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Predictor for diabetes risk using trained model.
    
    Loads pre-trained model and generates predictions on new data.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = None
        self.model_loaded = False

    def load_model_artifacts(self, model_path: str = None, 
                            scaler_path: str = None,
                            encoder_path: str = None) -> bool:
        """
        Load model, scaler, and encoders.
        
        Args:
            model_path: Path to model file
            scaler_path: Path to scaler file
            encoder_path: Path to encoder file
        
        Returns:
            True if successful
        """
        try:
            model_p = Path(model_path) if model_path else MODEL_PATH
            scaler_p = Path(scaler_path) if scaler_path else SCALER_PATH
            encoder_p = Path(encoder_path) if encoder_path else ENCODER_PATH
            
            # Load model
            if model_p.exists():
                self.model = joblib.load(model_p)
                logger.info(f"Model loaded from {model_p}")
            else:
                logger.warning(f"Model not found at {model_p}")
                return False
            
            # Load scaler
            if scaler_p.exists():
                self.scaler = joblib.load(scaler_p)
                logger.info(f"Scaler loaded from {scaler_p}")
            else:
                logger.info("Scaler file not found. Using default StandardScaler.")
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
            
            # Load encoders
            if encoder_p.exists():
                self.encoders = joblib.load(encoder_p)
                logger.info(f"Encoders loaded from {encoder_p}")
            else:
                self.encoders = {}
            
            self.model_loaded = True
            return True
        
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {str(e)}")
            return False

    def preprocess_input(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Preprocess input for prediction.
        
        Args:
            X: Input features
        
        Returns:
            Preprocessed features
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        X_processed = X.copy()
        
        # Encode categorical variables
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in self.encoders:
                X_processed[col] = self.encoders[col].transform(X_processed[col].astype(str))
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_processed)
        else:
            X_scaled = X_processed.values
        
        return X_scaled

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Input features
        
        Returns:
            Prediction probabilities
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model_artifacts() first.")
        
        # Preprocess
        X_processed = self.preprocess_input(X)
        
        # Predict
        if hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(X_processed)
            # Return probability of positive class
            return predictions[:, 1] if predictions.shape[1] > 1 else predictions[:, 0]
        else:
            return self.model.predict(X_processed)

    def predict_sample(self, sample: pd.Series) -> Tuple[float, str]:
        """
        Predict for a single sample.
        
        Args:
            sample: Single sample as Series
        
        Returns:
            Tuple of (prediction_probability, risk_category)
        """
        # Convert to DataFrame for preprocessing
        X = sample.to_frame().T
        prediction = self.predict(X)[0]
        
        # Classify
        if prediction < 0.25:
            category = "low"
        elif prediction < 0.50:
            category = "moderate"
        elif prediction < 0.75:
            category = "high"
        else:
            category = "very_high"
        
        return prediction, category


class Tuple:
    """Type hint for tuple return."""
    pass
