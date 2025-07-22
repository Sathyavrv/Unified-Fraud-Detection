# Unified Fraud Detection

Unified Fraud Detection is a modular Python project for detecting fraudulent transactions, predicting customer lifetime value (CLV), and segmenting customers using machine learning techniques. The project is designed for extensibility and ease of experimentation with different models and data pipelines.

## Features
- **Fraud Detection**: Supervised (XGBoost) and unsupervised (Isolation Forest) models for transaction fraud detection.
- **Customer Segmentation**: KMeans-based segmentation with feature engineering and cluster interpretation.
- **CLV Prediction**: Predicts customer lifetime value using Ridge Regression and Gradient Boosting.
- **Feature Engineering**: Automated feature creation for improved model performance.
- **Data Preprocessing**: Modular preprocessing for transactions, cards, and users data.
- **Model Persistence**: Save and load trained models for reuse.
- **Exploratory Analysis**: Jupyter notebook for data exploration.

## Project Structure
```
Unified-Fraud-Detection/
├── data/                # Sample data files (CSV, JSON)
├── logs/                # Application logs
├── models/              # Saved model files (pkl)
├── notebooks/           # Jupyter notebooks for analysis
├── src/                 # Source code
│   ├── clv_prediction.py        # CLV prediction (Ridge)
│   ├── clv_prediction_gb.py     # CLV prediction (Gradient Boosting)
│   ├── config.py                # Configuration (placeholder)
│   ├── customer_segmentation.py # Customer segmentation logic
│   ├── data_loader.py           # Data loading utilities
│   ├── feature_engineering.py   # Feature engineering
│   ├── fraud_detection.py       # Fraud detection models
│   ├── main.py                  # Main entry point
│   ├── preprocessing.py         # Data preprocessing
│   └── utils.py                 # Utility functions
├── requirements.txt     # Python dependencies
├── setup.py             # (Optional) Setup script
└── README.md            # Project documentation
```

## Setup
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Unified-Fraud-Detection
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   - Required packages: pandas, numpy, xgboost, scikit-learn, joblib

3. **Prepare data:**
   - Place your data files in the `data/` directory. Example files:
     - `cards_data.csv`
     - `users_data.csv`
     - `mcc_codes.json`

## Usage

### Running the Main Pipeline
The main entry point is `src/main.py`. You can run it as:
```bash
python src/main.py
```

### Notebooks
- Explore the data and models interactively using the Jupyter notebook in `notebooks/exploratory_analysis.ipynb`.

### Modules
- **Fraud Detection**: `src/fraud_detection.py`
- **CLV Prediction**: `src/clv_prediction.py`, `src/clv_prediction_gb.py`
- **Customer Segmentation**: `src/customer_segmentation.py`
- **Feature Engineering**: `src/feature_engineering.py`

## Model Files
Trained models are saved in the `models/` directory as `.pkl` files:
- `clv_model.pkl`, `clv_model_gb.pkl`, `iforest_model.pkl`, `kmeans_segmentation.pkl`, `scaler.pkl`, `xgb_fraud_model.pkl`

## Logging
- Logs are written to `logs/app.log`.

## Extending the Project
- Add new models or preprocessing steps by creating new modules in `src/`.
- Update `main.py` to integrate new workflows.

## License
[MIT License](LICENSE) *(Add a LICENSE file if needed)*

## Contact
For questions or contributions, please open an issue or submit a pull request. 