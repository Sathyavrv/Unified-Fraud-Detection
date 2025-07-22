import pandas as pd
import numpy as np
import logging
import joblib
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config import CLV_PRED_CONFIG, logger

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class CLVPrediction:
    def __init__(self):
        # Use a pipeline that first scales features then applies Ridge Regression.
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])
        logger.debug("CLVPrediction pipeline initialized with StandardScaler and Ridge Regression.")

    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Train a regression model to predict advanced CLV using a pipeline that includes scaling.
        The training uses RandomizedSearchCV to optimize hyperparameters.
        A log transformation is applied to the target to reduce the effect of extreme values.
        """
        try:
            logger.info("Training advanced CLV prediction model using Ridge Regression with log transformation.")
            test_size = CLV_PRED_CONFIG.get('test_size', 0.2)
            random_state = CLV_PRED_CONFIG.get('random_state', 42)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            # Apply log transformation to the training target (using log1p to handle zero values safely)
            y_train_log = np.log1p(y_train)
            
            param_grid = {
                'ridge__alpha': CLV_PRED_CONFIG.get('param_grid', {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]})['alpha'],
                'ridge__solver': CLV_PRED_CONFIG.get('param_grid', {'solver': ['auto', 'svd', 'cholesky']}).get('solver', ['auto'])
            }
            n_iter = CLV_PRED_CONFIG.get('n_iter', 20)
            scoring = CLV_PRED_CONFIG.get('scoring', 'neg_mean_absolute_error')
            
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
            logger.info("Ridge Regression tuning completed in %.2f seconds.", duration)
            
            # Update the pipeline with the best estimator found.
            self.pipeline = randomized_search.best_estimator_
            logger.debug("Best hyperparameters: %s", randomized_search.best_params_)
            
            # Predict on the test set (in log-space) and convert back to original scale.
            predictions_log = self.pipeline.predict(X_test)
            predictions = np.expm1(predictions_log)
            
            mae = mean_absolute_error(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions)
            logger.info("Advanced CLV Prediction MAE on validation set: %.2f", mae)
            logger.info("MAPE on validation set: %.2f%%", mape)
            
            return mae
        except Exception as e:
            logger.exception("Error training advanced CLV prediction model: %s", e)
            raise

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict CLV using the trained pipeline.
        The predictions are transformed back to the original scale using np.expm1.
        """
        try:
            logger.info("Predicting advanced CLV using the trained pipeline.")
            predictions_log = self.pipeline.predict(X)
            predictions = np.expm1(predictions_log)
            logger.debug("Advanced CLV predictions generated.")
            return predictions
        except Exception as e:
            logger.exception("Error predicting advanced CLV: %s", e)
            raise

    def save_model(self, model_dir: str) -> None:
        """
        Save the entire pipeline (scaler and model) to the specified directory.
        """
        try:
            logger.info("Saving advanced CLV prediction pipeline.")
            joblib.dump(self.pipeline, f"{model_dir}/clv_model.pkl")
            logger.debug("Advanced CLV prediction pipeline saved in %s", model_dir)
        except Exception as e:
            logger.exception("Error saving advanced CLV prediction model: %s", e)
            raise
