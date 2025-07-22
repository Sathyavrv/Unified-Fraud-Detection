import pandas as pd
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from src.config import logger, CUSTOMER_SEG_CONFIG

class CustomerSegmentation:
    def __init__(self, n_clusters: int = 5, segmentation_features: list = None):
        """
        Parameters:
            n_clusters (int): Initial number of clusters.
            segmentation_features (list): List of column names to use for segmentation.
                If None, a default list based on credit/risk attributes is used.
        """
        # Default feature set focusing on credit risk and behavioral attributes.
        if segmentation_features is None:
            segmentation_features = [
                'amount', 'use_chip', 'merchant_city', 'merchant_state', 'mcc', 'errors',
                'card_brand', 'card_type', 'credit_limit', 'year_pin_last_changed',
                'card_on_dark_web', 'current_age', 'retirement_age', 'per_capita_income',
                'yearly_income', 'total_debt', 'credit_score', 'num_credit_cards',
                'transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year',
                'transaction_weekday', 'transaction_quarter', 'day_of_year', 'week_of_year',
                'expires_month', 'expires_year'
            ]
        self.n_clusters = n_clusters
        self.segmentation_features = segmentation_features
        self.scaler = StandardScaler()
        # Extract the first element from the 'init' list
        init_method = CUSTOMER_SEG_CONFIG.get('init', ['k-means++'])[0]
        max_iter = CUSTOMER_SEG_CONFIG.get('max_iter', [300])[0]
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            init=init_method,
            max_iter=max_iter,
            random_state=CUSTOMER_SEG_CONFIG.get('random_state', 42)
        )
        logger.debug("CustomerSegmentation initialized with %d clusters and features: %s",
                     self.n_clusters, self.segmentation_features)

    def _prepare_features(self, X: pd.DataFrame, fit: bool = True):
        """
        Selects segmentation features and scales them.
        
        Parameters:
            X (pd.DataFrame): Input data.
            fit (bool): If True, fit the scaler; otherwise, transform using the existing scaler.
            
        Returns:
            X_scaled (np.array): Scaled feature matrix.
        """
        X_features = X[self.segmentation_features]
        if fit:
            X_scaled = self.scaler.fit_transform(X_features)
        else:
            X_scaled = self.scaler.transform(X_features)
        return X_scaled

    def fit(self, X: pd.DataFrame) -> list:
        """
        Fits the KMeans model on the selected and scaled features.
        
        Parameters:
            X (pd.DataFrame): Input data.
            
        Returns:
            labels (list): Cluster labels for each sample.
        """
        try:
            logger.info("Fitting KMeans for customer segmentation.")
            X_scaled = self._prepare_features(X, fit=True)
            self.kmeans_model.fit(X_scaled)
            labels = self.kmeans_model.labels_
            logger.debug("Customer segmentation labels generated.")
            return labels
        except Exception as e:
            logger.exception("Error fitting customer segmentation: %s", e)
            raise

    def tune(self, X: pd.DataFrame) -> int:
        """
        Tunes the number of clusters by evaluating inertia on scaled features.
        
        Parameters:
            X (pd.DataFrame): Input data.
        
        Returns:
            best_k (int): Best number of clusters found.
        """
        try:
            logger.info("Tuning customer segmentation model using inertia metric.")
            X_scaled = self._prepare_features(X, fit=True)
            best_k = None
            best_inertia = float('inf')
            candidate_clusters = CUSTOMER_SEG_CONFIG.get('n_clusters', list(range(3, 11)))
            for k in candidate_clusters:
                kmeans = KMeans(
                    n_clusters=k,
                    init=CUSTOMER_SEG_CONFIG.get('init', ['k-means++'])[0],
                    max_iter=CUSTOMER_SEG_CONFIG.get('max_iter', [300])[0],
                    random_state=CUSTOMER_SEG_CONFIG.get('random_state', 42)
                )
                kmeans.fit(X_scaled)
                inertia = kmeans.inertia_
                logger.info("KMeans with k=%d has inertia: %.2f", k, inertia)
                if inertia < best_inertia:
                    best_inertia = inertia
                    best_k = k
            logger.info("Best number of clusters found: %d with inertia: %.2f", best_k, best_inertia)
            self.n_clusters = best_k
            self.kmeans_model = KMeans(
                n_clusters=best_k,
                init=CUSTOMER_SEG_CONFIG.get('init', ['k-means++'])[0],
                max_iter=CUSTOMER_SEG_CONFIG.get('max_iter', [300])[0],
                random_state=CUSTOMER_SEG_CONFIG.get('random_state', 42)
            )
            self.kmeans_model.fit(X_scaled)
            return best_k
        except Exception as e:
            logger.exception("Error tuning customer segmentation: %s", e)
            raise

    def predict(self, X: pd.DataFrame) -> list:
        """
        Predicts the cluster labels for new data.
        
        Parameters:
            X (pd.DataFrame): New input data.
            
        Returns:
            labels (list): Cluster labels for each sample.
        """
        try:
            logger.info("Predicting customer segments.")
            X_scaled = self._prepare_features(X, fit=False)
            labels = self.kmeans_model.predict(X_scaled)
            logger.debug("Customer segments predicted.")
            return labels
        except Exception as e:
            logger.exception("Error predicting customer segments: %s", e)
            raise

    def interpret_clusters(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Interprets the clusters by computing the cluster centroids (in the original feature space)
        and counting the number of samples in each cluster.
        
        Parameters:
            X (pd.DataFrame): The data used to compute cluster assignments.
            
        Returns:
            df_centroids (pd.DataFrame): DataFrame with centroids and counts for each cluster.
        """
        try:
            logger.info("Interpreting clusters.")
            centroids = self.kmeans_model.cluster_centers_
            centroids_original = self.scaler.inverse_transform(centroids)
            df_centroids = pd.DataFrame(centroids_original, columns=self.segmentation_features)
            df_centroids['cluster'] = range(self.n_clusters)
            X_scaled = self._prepare_features(X, fit=False)
            labels = self.kmeans_model.predict(X_scaled)
            counts = pd.Series(labels).value_counts().sort_index()
            df_centroids['count'] = counts.values
            logger.info("Cluster interpretation completed.")
            logger.debug("Cluster centroids and counts:\n%s", df_centroids)
            return df_centroids
        except Exception as e:
            logger.exception("Error interpreting clusters: %s", e)
            raise

    def save_model(self, model_dir: str) -> None:
        """
        Saves the KMeans model and the scaler to the specified directory.
        
        Parameters:
            model_dir (str): Directory path where the models will be saved.
        """
        try:
            logger.info("Saving customer segmentation model and scaler.")
            joblib.dump(self.kmeans_model, f"{model_dir}/kmeans_segmentation.pkl")
            joblib.dump(self.scaler, f"{model_dir}/scaler.pkl")
            logger.debug("Customer segmentation model and scaler saved in %s", model_dir)
        except Exception as e:
            logger.exception("Error saving customer segmentation model: %s", e)
            raise
