# src/feature_engineering.py
import pandas as pd
import logging
from src.config import logger

class FeatureEngineer:
    def __init__(self):
        logger.debug("FeatureEngineer initialized.")

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Creating new features.")
            df['amount_bin'] = pd.qcut(df['amount'], q=10, duplicates='drop')
            logger.debug("Features created. Data shape: %s", df.shape)
            return df
        except Exception as e:
            logger.exception("Error creating features: %s", e)
            raise