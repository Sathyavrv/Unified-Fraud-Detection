# src/main.py
import os
import sys
import logging
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import DATA_DIR, TUNE_MODEL, MODELS_DIR, logger, SAVE_PROCESSED_DATA
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from src.fraud_detection import FraudDetectionModel
from src.customer_segmentation import CustomerSegmentation
from src.clv_prediction import CLVPrediction
from src.clv_prediction_gb import CLVPredictionGB
from src.utils import compute_advanced_clv, compute_rfm_metrics


def main() -> None:
    try:
        logger.info("Starting main pipeline execution.")
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # 1. Load data
        loader = DataLoader()
        transactions = loader.load_transactions()
        cards = loader.load_cards()
        users = loader.load_users()
        fraud_labels = loader.load_fraud_labels()
        gc.collect()

        # 2. Preprocess and merge data
        preprocessor = Preprocessor()
        transactions = preprocessor.preprocess_transactions(transactions)
        cards = preprocessor.preprocess_cards(cards)
        df_merged = preprocessor.merge_data(transactions, cards, users, fraud_labels)
        gc.collect()

        df_train = df_merged[df_merged['fraud_label'].notna()]
        df_test = df_merged[df_merged['fraud_label'].isna()]
        gc.collect()
        
        # Split processing into supervised and unsupervised pipelines:
        df_supervised = preprocessor.process_for_supervised(df_train)
        gc.collect()
        
        del df_train
        gc.collect()

        df_unsupervised = preprocessor.process_for_unsupervised(df_merged)
        gc.collect()

        
        '''# 3. Feature Engineering (optional)
        feature_engineer = FeatureEngineer()
        df_supervised = feature_engineer.create_features(df_supervised)
        df_unsupervised = feature_engineer.create_features(df_unsupervised)'''
        
        # Define feature columns for supervised model (exclude target & non-features)
        X_supervised = df_supervised.drop(columns='fraud_label')
        supervised_features = X_supervised.columns.tolist()
        y_supervised = preprocessor.get_target_vector(df_supervised)
        gc.collect()
        
        # 4. Fraud Detection – Supervised: Tune and Predict using XGBoost
        fraud_model = FraudDetectionModel()
        if TUNE_MODEL:
            tuning_results = fraud_model.tune_supervised_xgb(X_supervised, y_supervised)
        else:
            tuning_results = fraud_model.train_supervised(X_supervised, y_supervised)
        logger.info("XGBoost Tuning Results: %s", tuning_results)
                
        X_test_supervised = df_test[supervised_features]
        predictions = fraud_model.xgb_predict(
            X_test_supervised, 
            return_proba=True, 
            output_dir=os.path.join(DATA_DIR, "outputs")
        )
        gc.collect()

        del X_supervised, y_supervised, X_test_supervised, df_test
        gc.collect()
        
        # 5. Fraud Detection – Unsupervised: Train Isolation Forest and Visualize
        unsupervised_preds = fraud_model.train_unsupervised(df_unsupervised)
        df_unsupervised['unsupervised_fraud_score'] = unsupervised_preds
        unsupervised_output_file = os.path.join(DATA_DIR, 'unsupervised_predictions.csv')
        df_unsupervised.to_csv(unsupervised_output_file, index=False)
        logger.info("Unsupervised predictions saved to %s", unsupervised_output_file)
        #fraud_model.visualize_iforest(df_unsupervised, output_dir=os.path.join(DATA_DIR, "outputs"))
        gc.collect()

        fraud_model.save_models(MODELS_DIR)
        gc.collect()

        # 7. Customer Lifetime Value (CLV) Prediction – Train and Save Model
        
        logger.info("Starting advanced CLV computation using RFM metrics.")
        gc.collect()   

        # Compute RFM metrics
        rfm = compute_rfm_metrics(df_unsupervised)
        logger.debug("Computed RFM metrics (first 5 rows):\n%s", rfm.head().to_string())

        rfm = compute_advanced_clv(rfm)
        logger.debug("RFM metrics with advanced CLV computed (first 5 rows):\n%s", rfm.head().to_string())

        # Log the average CLV
        avg_clv = rfm['clv'].mean()
        logger.info("Average CLV: %.2f", avg_clv)

        # Merge additional customer features
        customer_info = df_unsupervised.groupby('client_id').agg({
            'current_age': 'first',
            'per_capita_income': 'first'
        }).reset_index()
        logger.debug("Customer info aggregated (first 5 rows):\n%s", customer_info.head().to_string())

        rfm = rfm.merge(customer_info, on='client_id', how='left')
        logger.debug("Merged RFM with customer info (first 5 rows):\n%s", rfm.head().to_string())
        logger.info("RFM DataFrame shape after merging: %s", rfm.shape)

        # Build feature set for CLV prediction.
        clv_features = rfm[['recency', 'frequency', 'monetary', 'current_age', 'per_capita_income']]
        clv_target = rfm['clv']
        logger.info("CLV feature set shape: %s", clv_features.shape)
        logger.debug("CLV features sample:\n%s", clv_features.head().to_string())
        logger.info("CLV target sample (first 5 values): %s", clv_target.head().to_list())

        # Option 1: Train Ridge-based CLV model.
        clv_model_ridge = CLVPrediction()
        mae_ridge = clv_model_ridge.train(clv_features, clv_target)
        logger.info("Ridge-based Advanced CLV Prediction MAE: %.2f", mae_ridge)
        clv_model_ridge.save_model(MODELS_DIR)

        # Option 2: Train Gradient Boosting-based CLV model.
        clv_model_gb = CLVPredictionGB()
        mae_gb = clv_model_gb.train(clv_features, clv_target)
        logger.info("Gradient Boosting Advanced CLV Prediction MAE: %.2f", mae_gb)
        clv_model_gb.save_model(MODELS_DIR)

        gc.collect()

        
        # 6. Customer Segmentation – Tune and Predict using KMeans
        segmentation_features = df_unsupervised[[
            'amount', 'use_chip', 'merchant_city', 'merchant_state', 'mcc', 'errors',
            'card_brand', 'card_type', 'credit_limit', 'year_pin_last_changed',
            'card_on_dark_web', 'current_age', 'retirement_age', 'per_capita_income',
            'yearly_income', 'total_debt', 'credit_score', 'num_credit_cards',
            'transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year',
            'transaction_weekday', 'transaction_quarter', 'day_of_year', 'week_of_year',
            'expires_month', 'expires_year'
        ]]
        segmentation = CustomerSegmentation(n_clusters=5)
        best_k = segmentation.tune(segmentation_features)
        logger.info("Best number of clusters found: %d", best_k)
        segmentation_labels = segmentation.predict(segmentation_features)        
        df_unsupervised['customer_segment'] = segmentation_labels

        cluster_info = segmentation.interpret_clusters(segmentation_features)
        logger.info("Cluster interpretation:\n%s", cluster_info.to_string())

        segmentation.save_model(MODELS_DIR)
        gc.collect()                
        
        # 8. Save the processed dataset (optional)
        if SAVE_PROCESSED_DATA:
            processed_file = os.path.join(DATA_DIR, 'processed_data.csv')
            df_unsupervised.to_csv(processed_file, index=False)
            logger.info("Processed dataset saved to %s", processed_file)
        else:
            logger.info("SAVE_PROCESSED_DATA is False. Skipping saving processed data.")
        
        logger.info("Pipeline execution completed successfully.")        
    except Exception as e:
        logger.exception("An error occurred in the main pipeline: %s", e)
    finally:
        gc.collect()

if __name__ == '__main__':
    main()
