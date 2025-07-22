import pandas as pd
import numpy as np
import logging
import joblib
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config import logger
from src.config import CLV_PRED_CONFIG_GB

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class CLVPredictionGB:
    def __init__(self):
        # Create a pipeline that scales features and applies Gradient Boosting Regressor.
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('gb', GradientBoostingRegressor(random_state=CLV_PRED_CONFIG_GB.get('random_state', 42)))
        ])
        logger.debug("CLVPredictionGB pipeline initialized with StandardScaler and GradientBoostingRegressor.")

    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Train a regression model (Gradient Boosting) to predict advanced CLV.
        Applies log transformation to the target before training, and converts predictions back.
        """
        try:
            logger.info("Training advanced CLV prediction model using Gradient Boosting with log transformation.")
            test_size = CLV_PRED_CONFIG_GB.get('test_size', 0.2)
            random_state = CLV_PRED_CONFIG_GB.get('random_state', 42)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            # Apply log transformation to the training target
            y_train_log = np.log1p(y_train)
            
            param_grid = {
                'gb__n_estimators': CLV_PRED_CONFIG_GB.get('param_grid', {}).get('gb__n_estimators', [100, 200, 300]),
                'gb__learning_rate': CLV_PRED_CONFIG_GB.get('param_grid', {}).get('gb__learning_rate', [0.01, 0.05, 0.1]),
                'gb__max_depth': CLV_PRED_CONFIG_GB.get('param_grid', {}).get('gb__max_depth', [3, 4, 5])
            }
            n_iter = CLV_PRED_CONFIG_GB.get('n_iter', 20)
            scoring = CLV_PRED_CONFIG_GB.get('scoring', 'neg_mean_absolute_error')
            
            randomized_search = RandomizedSearchCV(
                estimator=self.pipeline,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=scoring,
                random_state=random_state,
                n_jobs=-1
            )
            
            start_time = time.time()
            randomized_search.fit(X_train, y_train_log)
            duration = time.time() - start_time
            logger.info("Gradient Boosting tuning completed in %.2f seconds.", duration)
            
            # Update the pipeline with the best estimator found
            self.pipeline = randomized_search.best_estimator_
            logger.debug("Best hyperparameters: %s", randomized_search.best_params_)
            
            # Predict on the test set in log-space and convert back
            predictions_log = self.pipeline.predict(X_test)
            predictions = np.expm1(predictions_log)
            
            mae = mean_absolute_error(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions)
            logger.info("Advanced CLV Prediction MAE on validation set: %.2f", mae)
            logger.info("MAPE on validation set: %.2f%%", mape)
            return mae
        except Exception as e:
            logger.exception("Error training advanced CLV prediction model with GB: %s", e)
            raise

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict CLV using the trained Gradient Boosting pipeline.
        """
        try:
            logger.info("Predicting advanced CLV using Gradient Boosting pipeline.")
            predictions_log = self.pipeline.predict(X)
            predictions = np.expm1(predictions_log)
            logger.debug("Advanced CLV predictions (GB) generated.")
            return predictions
        except Exception as e:
            logger.exception("Error predicting advanced CLV with GB: %s", e)
            raise

    def save_model(self, model_dir: str) -> None:
        """
        Save the entire Gradient Boosting pipeline (scaler and model) to the specified directory.
        """
        try:
            logger.info("Saving advanced CLV prediction pipeline (GB).")
            joblib.dump(self.pipeline, f"{model_dir}/clv_model_gb.pkl")
            logger.debug("Advanced CLV prediction pipeline (GB) saved in %s", model_dir)
        except Exception as e:
            logger.exception("Error saving advanced CLV prediction model with GB: %s", e)
            raise
