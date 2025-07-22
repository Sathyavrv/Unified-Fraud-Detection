# src/data_loader.py
import pandas as pd
import json
import logging
from src.config import DATA_FILES, logger

class DataLoader:
    def __init__(self):
        try:
            self.transactions_file = DATA_FILES['transactions']
            self.cards_file = DATA_FILES['cards']
            self.mcc_file = DATA_FILES['mcc']
            self.fraud_labels_file = DATA_FILES['fraud_labels']
            self.users_file = DATA_FILES['users']
            logger.debug("DataLoader initialized with file paths.")
        except Exception as e:
            logger.exception("Error initializing DataLoader: %s", e)
    
    def load_transactions(self) -> pd.DataFrame:
        try:
            logger.info("Loading transactions from %s", self.transactions_file)
            transactions = pd.read_csv(self.transactions_file)
            logger.debug("Transactions loaded with shape %s", transactions.shape)
            return transactions
        except Exception as e:
            logger.exception("Error loading transactions: %s", e)
            raise
    
    def load_cards(self) -> pd.DataFrame:
        try:
            logger.info("Loading cards from %s", self.cards_file)
            cards = pd.read_csv(self.cards_file)
            logger.debug("Cards loaded with shape %s", cards.shape)
            return cards
        except Exception as e:
            logger.exception("Error loading cards: %s", e)
            raise
    
    def load_mcc(self) -> dict:
        try:
            logger.info("Loading MCC codes from %s", self.mcc_file)
            with open(self.mcc_file, 'r') as f:
                mcc_codes = json.load(f)
            logger.debug("MCC codes loaded with keys: %s", list(mcc_codes.keys()))
            return mcc_codes
        except Exception as e:
            logger.exception("Error loading MCC codes: %s", e)
            raise

    def load_fraud_labels(self) -> pd.DataFrame:
        try:
            logger.info("Loading fraud labels from %s", self.fraud_labels_file)
            with open(self.fraud_labels_file, 'r') as f:
                fraud_labels = json.load(f)
            fraud_labels_df = pd.DataFrame(list(fraud_labels['target'].items()),
                                           columns=['transaction_id', 'fraud_label'])
            logger.debug("Fraud labels loaded with shape %s", fraud_labels_df.shape)
            return fraud_labels_df
        except Exception as e:
            logger.exception("Error loading fraud labels: %s", e)
            raise

    def load_users(self) -> pd.DataFrame:
        try:
            logger.info("Loading users from %s", self.users_file)
            users = pd.read_csv(self.users_file)
            logger.debug("Users loaded with shape %s", users.shape)
            return users
        except Exception as e:
            logger.exception("Error loading users: %s", e)
            raise
