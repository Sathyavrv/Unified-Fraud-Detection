import os
import logging

# Define base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
SAVE_PROCESSED_DATA = True
TUNE_MODEL = False

# Ensure the logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Data file paths
DATA_FILES = {
    'transactions': os.path.join(DATA_DIR, 'transactions_data.csv'),
    'cards': os.path.join(DATA_DIR, 'cards_data.csv'),
    'mcc': os.path.join(DATA_DIR, 'mcc_codes.json'),
    'fraud_labels': os.path.join(DATA_DIR, 'train_fraud_labels.json'),
    'users': os.path.join(DATA_DIR, 'users_data.csv')
}

# Logging configuration
LOG_FILE = os.path.join(LOG_DIR, 'app.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# XGBoost training parameters
XGB_TRAIN_PARAMS = {
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'random_state': 42,
    'eval_metric': 'logloss',  
    'early_stopping_rounds': 50,
    'scoring': 'f1',    
}

# XGBoost tuning configuration (default used if none provided)
XGB_TUNE_CONFIG = {
    'xgb_params': {
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2, 0.3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1.0, 1.5, 2.0],
        'eval_metric': ['aucpr'],  # Evaluates performance based on the precision-recall curve
        'learning_rate': [0.05, 0.1],
        'scale_pos_weight': [90, 600, 675, 750]  # Tune around the computed imbalance ratio
    },
    'rounds': 500,
    'early_stopping_rounds': 50,
    'verbose_eval': True,
    'folds': 5,
    'seed': 42,
    'n_iter': 5,
    'scoring': 'f1'  
}


# Isolation Forest configuration
IFOREST_CONFIG = {
    'contamination': 0.01,
    'max_samples': 'auto',
    'random_state': 42
}

# Customer segmentation (KMeans) configuration
CUSTOMER_SEG_CONFIG = {
    'n_clusters': [3, 5, 7, 9],
    'init': ['k-means++'],    
    'max_iter': [300],
    'random_state': 42
}

# CLV prediction configuration using Ridge regression
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

# Configuration for Ridge Regression (linear baseline)
CLV_PRED_CONFIG = {
    'model': Ridge,
    'param_grid': {
        'ridge__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        'ridge__solver': ['auto', 'svd', 'cholesky']
    },
    'test_size': 0.2,
    'random_state': 42,
    'n_iter': 20,
    'scoring': 'neg_mean_absolute_error'
}

# Configuration for Gradient Boosting Regressor (captures non-linear relationships)
CLV_PRED_CONFIG_GB = {
    'model': GradientBoostingRegressor,
    'param_grid': {
        'gb__n_estimators': [100, 200, 300],
        'gb__learning_rate': [0.01, 0.05, 0.1],
        'gb__max_depth': [3, 4, 5]
    },
    'test_size': 0.2,
    'random_state': 42,
    'n_iter': 20,
    'scoring': 'neg_mean_absolute_error'
}

 