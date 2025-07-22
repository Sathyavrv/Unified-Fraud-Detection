# src/main.py
import time
import os
from typing import Any, Dict, Optional, Tuple, List, Union

import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, classification_report
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.config import MODELS_DIR, XGB_TRAIN_PARAMS, XGB_TUNE_CONFIG, IFOREST_CONFIG, logger
from sklearn.model_selection import StratifiedGroupKFold

class FraudDetectionModelError(Exception):
    """Custom exception for errors in the FraudDetectionModel."""
    pass

class FraudDetectionModel:
    """
    A class for training and tuning fraud detection models using both supervised
    (XGBoost) and unsupervised (Isolation Forest) methods.
    """
    
    # Default XGB tuning config is taken from config if not provided.
    
    def __init__(self) -> None:
        """
        Initializes the FraudDetectionModel with placeholders for the XGBoost
        and Isolation Forest models.
        """
        self.xgb_model: Optional[XGBClassifier] = None
        self.iforest_model: Optional[IsolationForest] = None
        logger.debug("FraudDetectionModel initialized.")

    def _split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2, 
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        logger.info("Data split: %d training samples, %d test samples", len(X_train), len(X_test))
        return X_train, X_test, y_train, y_test

    def train_supervised(self, X: pd.DataFrame, y: pd.Series) -> str:
        try:
            logger.info("Training supervised fraud detection model (XGBoost).")
            X_train, X_test, y_train, y_test = self._split_data(X, y, test_size=0.2, seed=42)
            eval_set = [(X_train, y_train), (X_test, y_test)]
            self.xgb_model = XGBClassifier(**XGB_TRAIN_PARAMS)
            start_time = time.time()
            self.xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
            duration = time.time() - start_time
            logger.info("XGBoost model training completed in %.2f seconds.", duration)
            y_pred = self.xgb_model.predict(X_test)
            precision = precision_score(y_test, y_pred)
            logger.info("XGBoost Fraud Detection Precision: %.2f", precision)
            report = classification_report(y_test, y_pred)
            logger.info("Classification Report:\n%s", report)
            return report
        except Exception as e:
            logger.exception("Error training supervised fraud detection model: %s", e)
            raise FraudDetectionModelError("Error in train_supervised") from e

    def tune_supervised_xgb(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        xgb_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            if xgb_config is None:
                xgb_config = XGB_TUNE_CONFIG.copy()
            seed: int = xgb_config.get('seed', 42)
            X_train_all, X_holdout, y_train_all, y_holdout = self._split_data(X, y, test_size=0.2, seed=seed)
            logger.info("Starting XGBoost hyperparameter tuning with RandomizedSearchCV on the training set.")
            xgb_params: Dict[str, Any] = xgb_config.get('xgb_params', {})
            n_estimators: int = xgb_config.get('rounds', 500)
            early_stopping_rounds: int = xgb_config.get('early_stopping_rounds', 10)
            verbose_eval: Union[bool, int] = xgb_config.get('verbose_eval', True)
            folds: int = xgb_config.get('folds', 5)
            n_iter: int = xgb_config.get('n_iter', 50)
            scoring: str = xgb_config.get('scoring', 'roc_auc')
            # Remove explicit eval_metric to avoid conflicts with **xgb_params.
            base_xgb = XGBClassifier(
                n_estimators=n_estimators,
                random_state=seed,
                use_label_encoder=False,
                **xgb_params
            )
            kfold = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
            param_dist: Dict[str, Any] = xgb_params
            randomized_search = RandomizedSearchCV(
                estimator=base_xgb,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring=scoring,
                cv=kfold,
                random_state=seed,
                verbose=1,
                n_jobs=2
            )
            start_time = time.time()
            randomized_search.fit(X_train_all, y_train_all,
                                  eval_set=[(X_train_all, y_train_all)],
                                  early_stopping_rounds=early_stopping_rounds,
                                  verbose=verbose_eval)
            tuning_duration = time.time() - start_time
            logger.info("Hyperparameter tuning completed in %.2f seconds.", tuning_duration)
            best_params: Dict[str, Any] = randomized_search.best_params_
            best_score: float = randomized_search.best_score_
            logger.info("Best XGBoost parameters: %s", best_params)
            logger.info("Best XGBoost CV score: %.4f", best_score)
            self.xgb_model = randomized_search.best_estimator_
            y_pred = self.xgb_model.predict(X_holdout)
            report: str = classification_report(y_holdout, y_pred)
            logger.info("Classification Report on hold-out set:\n%s", report)
            return {"best_params": best_params, "best_score": best_score, "classification_report": report}
        except Exception as e:
            logger.exception("Error during XGBoost hyperparameter tuning: %s", e)
            raise FraudDetectionModelError("Error in tune_supervised_xgb") from e

    def train_unsupervised(self, X: pd.DataFrame) -> List[int]:
        try:
            logger.info("Training unsupervised fraud detection model (Isolation Forest).")
            self.iforest_model = IsolationForest(**IFOREST_CONFIG)
            self.iforest_model.fit(X)
            preds = self.iforest_model.predict(X)
            preds_binary = [1 if x == -1 else 0 for x in preds]
            logger.debug("Isolation Forest predictions generated.")
            return preds_binary
        except Exception as e:
            logger.exception("Error training unsupervised fraud detection model: %s", e)
            raise FraudDetectionModelError("Error in train_unsupervised") from e

    def visualize_iforest(self, X: pd.DataFrame, output_dir: Optional[str] = None) -> None:
        try:
            logger.info("Visualizing Isolation Forest predictions.")
            preds = self.iforest_model.predict(X)
            binary_preds = [1 if x == -1 else 0 for x in preds]
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X)
            plt.figure(figsize=(8,6))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=binary_preds, cmap='coolwarm', alpha=0.6)
            plt.title("Isolation Forest Predictions (Anomaly Detection)")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.colorbar(label="Anomaly (1) vs Normal (0)")
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, "isolation_forest_visualization.png")
                plt.savefig(output_file)
                logger.info("Isolation Forest visualization saved to %s", output_file)
            else:
                plt.show()
            plt.close()
        except Exception as e:
            logger.exception("Error visualizing Isolation Forest predictions: %s", e)
            raise FraudDetectionModelError("Error in visualize_iforest") from e

    def save_models(self, model_dir: str) -> None:
        try:
            logger.info("Saving fraud detection models.")
            joblib.dump(self.xgb_model, f"{model_dir}/xgb_fraud_model.pkl")
            joblib.dump(self.iforest_model, f"{model_dir}/iforest_model.pkl")
            logger.debug("Fraud detection models saved in %s", model_dir)
        except Exception as e:
            logger.exception("Error saving fraud detection models: %s", e)
            raise FraudDetectionModelError("Error in save_models") from e

    def load_models(self, model_dir: str) -> None:
        try:
            logger.info("Loading fraud detection models.")
            self.xgb_model = joblib.load(f"{model_dir}/xgb_fraud_model.pkl")
            self.iforest_model = joblib.load(f"{model_dir}/iforest_model.pkl")
            logger.debug("Fraud detection models loaded successfully.")
        except Exception as e:
            logger.exception("Error loading fraud detection models: %s", e)
            raise FraudDetectionModelError("Error in load_models") from e

    def xgb_predict(
        self, 
        X: pd.DataFrame, 
        return_proba: bool = False, 
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            logger.info("Making predictions with XGBoost model.")
            y_pred = self.xgb_model.predict(X)
            result: Dict[str, Any] = {"predictions": y_pred}
            if return_proba:
                result["probabilities"] = self.xgb_model.predict_proba(X)
            logger.debug("XGBoost predictions generated.")
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                df_pred = pd.DataFrame(result["predictions"], columns=["prediction"])
                if return_proba and "probabilities" in result:
                    df_proba = pd.DataFrame(result["probabilities"], columns=["prob_class0", "prob_class1"])
                    df_pred = pd.concat([df_pred, df_proba], axis=1)
                output_file = os.path.join(output_dir, "xgb_predictions.csv")
                df_pred.to_csv(output_file, index=False)
                logger.info("XGBoost predictions saved to %s", output_file)
            return result
        except Exception as e:
            logger.exception("Error making XGBoost predictions: %s", e)
            raise FraudDetectionModelError("Error in xgb_predict") from e
