# src/utils.py
import pandas as pd

def compute_rfm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM metrics for each client based on transaction data.
    Assumes df contains columns: client_id, transaction_year, transaction_month, transaction_day, amount.
    """
    # Create a transaction_date column from year, month, and day.
    df_renamed = df.rename(columns={
        'transaction_year': 'year',
        'transaction_month': 'month',
        'transaction_day': 'day'
    })
    # Now create the transaction_date column using the renamed columns
    df['transaction_date'] = pd.to_datetime(df_renamed[['year', 'month', 'day']])
    # Use the latest transaction date in the dataset as the reference.
    max_date = df['transaction_date'].max()
    rfm = df.groupby('client_id').agg({
        'transaction_date': lambda x: (max_date - x.max()).days,  # Recency: days since last transaction
        'client_id': 'count',  # Frequency: count of transactions
        'amount': 'sum'        # Monetary: total spent
    }).rename(columns={
        'transaction_date': 'recency',
        'client_id': 'frequency',
        'amount': 'monetary'
    }).reset_index()
    return rfm

def compute_advanced_clv(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Compute an advanced proxy for CLV using RFM metrics.
    Here, we use a proxy: CLV = monetary * (frequency / (recency + 1))
    In practice, you might include discount factors or churn probabilities.
    """
    rfm['clv'] = rfm['monetary'] * (rfm['frequency'] / (rfm['recency'] + 1))
    return rfm
