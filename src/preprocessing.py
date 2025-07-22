# src/preprocessing.py
import pandas as pd
import logging
import gc
from sklearn.preprocessing import LabelEncoder
from src.config import logger

class Preprocessor:
    def __init__(self):
        logger.debug("Preprocessor initialized.")
    
    def preprocess_transactions(self, transactions: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Preprocessing transactions.")
            transactions['date'] = pd.to_datetime(transactions['date'], errors='coerce')
            # Replace missing values more selectively if possible.
            transactions.fillna("NA", inplace=True)
            logger.debug("Transactions preprocessed. Shape: %s", transactions.shape)
            gc.collect()
            return transactions
        except Exception as e:
            logger.exception("Error preprocessing transactions: %s", e)
            raise

    def preprocess_cards(self, cards: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Preprocessing cards.")
            cards['acct_open_date'] = pd.to_datetime(cards['acct_open_date'], errors='coerce')
            logger.debug("Cards preprocessed. Shape: %s", cards.shape)
            gc.collect()
            return cards
        except Exception as e:
            logger.exception("Error preprocessing cards: %s", e)
            raise

    def preprocess_users(self, users: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Preprocessing users.")
            logger.debug("Users preprocessed. Shape: %s", users.shape)
            return users
        except Exception as e:
            logger.exception("Error preprocessing users: %s", e)
            raise

    def merge_data(self, transactions: pd.DataFrame, cards: pd.DataFrame,
                   users: pd.DataFrame, fraud_labels: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Merging datasets.")
            transactions['id'] = transactions['id'].astype(str)
            transactions['client_id'] = transactions['client_id'].astype(str)
            transactions['card_id'] = transactions['card_id'].astype(str)
            cards['id'] = cards['id'].astype(str)
            fraud_labels['transaction_id'] = fraud_labels['transaction_id'].astype(str)
            users.rename(columns={'id': 'client_id'}, inplace=True)
            users['client_id'] = users['client_id'].astype(str)
            df = transactions.merge(cards, left_on='card_id', right_on='id', suffixes=('', '_card'))
            df = df.merge(fraud_labels, left_on='id', right_on='transaction_id', how='left')
            df = df.merge(users, on='client_id', how='left')
            del transactions, cards, fraud_labels, users
            gc.collect()
            df_processed = self.process_common(df.copy())
            logger.debug("Data merged successfully. Final shape: %s", df_processed.shape)
            gc.collect()
            return df_processed
        except Exception as e:
            logger.exception("Error merging data: %s", e)
            raise

    def process_common(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs common processing on the dataframe: handling dates, splitting columns,
        formatting monetary values, and label encoding.
        """
        logger.info("Performing common processing on dataframe.")
        
        if 'date' in df.columns:
            logger.debug("Processing 'date' column.")
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['transaction_hour'] = df['date'].dt.hour.astype(int)
            df['transaction_day'] = df['date'].dt.day.astype(int)
            df['transaction_month'] = df['date'].dt.month.astype(int)
            df['transaction_year'] = df['date'].dt.year.astype(int)
            df['transaction_weekday'] = df['date'].dt.weekday.astype(int)
            df['transaction_quarter'] = df['date'].dt.quarter.astype(int)
            df['day_of_year'] = df['date'].dt.dayofyear.astype(int)
            df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
            logger.debug("Processed 'date' column.")

        if 'acct_open_date' in df.columns and df['acct_open_date'].dtype == object:
            logger.debug("Processing 'acct_open_date' column.")
            split_acct = df['acct_open_date'].str.split('/', expand=True)
            df['acct_open_month'] = split_acct[0].astype(int)
            df['acct_open_year'] = split_acct[1].astype(int)
            logger.debug("Processed 'acct_open_date' column.")

        if 'expires' in df.columns:
            logger.debug("Processing 'expires' column.")
            split_expires = df['expires'].str.split('/', expand=True)
            df['expires_month'] = split_expires[0].astype(int)
            df['expires_year'] = split_expires[1].astype(int)
            logger.debug("Processed 'expires' column.")

        cols_to_drop = ["date", "acct_open_date", "expires", "card_number", "cvv", "transaction_id"]
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        logger.debug("Dropped columns: %s", cols_to_drop)
        gc.collect()
        
        monetary_cols = ['amount', 'per_capita_income', 'yearly_income', 'total_debt', 'credit_limit']
        for col in monetary_cols:
            if col in df.columns:
                logger.debug("Processing monetary column: %s", col)
                df[col] = df[col].replace('[\$,]', '', regex=True).astype('float64')
                logger.debug("Processed monetary column: %s", col)
        gc.collect()
        
        label_cols = ['use_chip', 'merchant_city', 'merchant_state', 'errors', 
                      'card_brand', 'card_type', 'has_chip', 'card_on_dark_web', 'gender']
        for col in label_cols:
            if col in df.columns:
                logger.debug("Label encoding column: %s", col)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                logger.debug("Label encoded column: %s", col)            
        gc.collect()
        logger.debug("Common processing completed. Shape: %s", df.shape)
        return df

    def process_for_supervised(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the dataframe for supervised learning (e.g., XGBoost).
        Keeps the target variable.
        """    
        cols_to_drop = ["id", "client_id", "card_id", "merchant_id",
                        "id_card", "client_id_card", "zip", "address"]

        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        logger.info("Processed dataframe for supervised learning. Shape: %s", df.shape)
        gc.collect()
        return df

    def process_for_unsupervised(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the dataframe for unsupervised learning (e.g., clustering, IsolationForest).
        Drops the target variable if present.
        """
        if 'zip' and 'address' in df.columns:
            df = df.drop(columns=['zip', 'address'])
        if 'fraud_label' in df.columns:
            df = df.drop(columns=['fraud_label'])
        logger.info("Processed dataframe for unsupervised learning. Shape: %s", df.shape)
        gc.collect()
        return df

    def get_target_vector(self, df: pd.DataFrame) -> pd.Series:
        try:
            logger.info("Extracting target vector for fraud detection.")
            target = df['fraud_label'].apply(lambda x: 1 if str(x).lower() in ['yes', 'true', '1'] else 0)
            logger.debug("Target vector length: %s", len(target))
            gc.collect()
            return target
        except Exception as e:
            logger.exception("Error extracting target vector: %s", e)
            raise
