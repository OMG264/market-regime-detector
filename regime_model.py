import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import warnings

# Suppress minor data handling warnings
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    def __init__(self, ticker, start_date, end_date):
        """
        Initializes the regime detector pipeline.
        Uses 3 components to organically capture Bull, Bear, and Sideways markets.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.features = None
        
        # Using a Gaussian Mixture Model to handle fluid probability distributions
        self.model = GaussianMixture(n_components=3, 
                                     covariance_type='full', 
                                     random_state=42)
        
        # Color & metadata configurations for mapping and plotting
        self.regime_meta = {
            0: {'label': 'Bull', 'color': 'green'},
            1: {'label': 'Bear', 'color': 'red'},
            2: {'label': 'Sideways', 'color': 'blue'}
        }

    def fetch_data(self):
        """Step 1: Reliable historical data ingestion."""
        print(f" Fetching market data for {self.ticker}...")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
        # Isolate closing prices and ensure no gaps exist
        self.data = self.data[['Close']]
        self.data.dropna(inplace=True)
        return self.data

    def engineer_features(self):
        """Step 2: Calculate independent statistical features."""
        print(" Engineering independent features (Returns & Volatility)...")
        
        # Calculate daily returns
        self.data['Returns'] = self.data['Close'].pct_change()
        
        # Calculate 21-day rolling volatility (Standard trading month)
        self.data['Volatility'] = self.data['Returns'].rolling(window=21).std()
        
        # Strip rows that don't have enough history for the rolling window
        self.data.dropna(inplace=True)
        
        self.features = self.data[['Returns', 'Volatility']].values
        return self.features

    def fit_predict(self):
        """Step 3: Fit the GMM and sort cluster labels intelligently."""
        print(" Fitting Gaussian Mixture Model across 3 structural components...")
        
        # Get raw cluster assignments from GMM (0, 1, or 2 randomly ordered)
        raw_clusters = self.model.fit_predict(self.features)
        
        # Extract mean returns per cluster to apply your sorting logic
        cluster_means = self.data['Returns'].groupby(raw_clusters).mean()
        
        # Map raw clusters based on historical characteristics
        bull_cluster = cluster_means.idxmax()
        bear_cluster = cluster_means.idxmin()
        sideways_cluster = [c for c in cluster_means.index if c not in [bull_cluster, bear_cluster]][0]
        
        # Create strict mapping protocol (0=Bull, 1=Bear, 2=Sideways)
        sorting_map = {bull_cluster: 0, bear_cluster: 1, sideways_cluster: 2}
        
        # Overwrite arbitrary flags with sorted financial regimes
        self.data['Regime'] = [sorting_map[c] for c in raw_clusters]
        self.data['Regime_Label'] = self.data['Regime'].map(lambda x: self.regime_meta[x]['label'])
        
        print(" Pipeline execution successful!")
        return self.data