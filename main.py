import matplotlib.pyplot as plt
from regime_model import MarketRegimeDetector

def plot_regimes(df, regime_meta, ticker):
    """
    Takes the processed dataframe and plots the asset price 
    with vertical background shading for the market regimes.
    """
    print(" Generating structural regime chart...")
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot the solid price line
    ax.plot(df.index, df['Close'], label=f'{ticker} Price', color='black', lw=1.2)

    # Shade the background based on the regime
    for regime_id, meta in regime_meta.items():
        # Create a boolean mask for where this specific regime is active
        mask = (df['Regime'] == regime_id)
        
        ax.fill_between(df.index, 
                        df['Close'].min() * 0.95, # Slight padding on the bottom
                        df['Close'].max() * 1.05, # Slight padding on the top
                        where=mask, 
                        alpha=0.2, # 20% opacity for the background shade
                        color=meta['color'], 
                        label=meta['label'])

    # Build the custom legend
    handles = [plt.Line2D([0], [0], color=meta['color'], lw=4, label=meta['label']) 
               for regime_id, meta in regime_meta.items()]
    
    ax.legend(handles=handles, loc='upper left')
    ax.set_title(f'{ticker} Market Regimes (Hybrid GMM Architecture)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print(" Displaying chart window...")
    plt.show()

def main():
    ticker_symbol = "SPY"
    
    # Instantiate the detector pipeline
    detector = MarketRegimeDetector(ticker=ticker_symbol, 
                                    start_date="2000-01-01", 
                                    end_date=None) # End=None fetches up to today
    
    # Step 1: Data Ingestion with Error Handling
    df = detector.fetch_data()
    if df is None or df.empty:
        print(f"❌ Error: Failed to download data for {ticker_symbol}. Check connection.")
        return

    # Step 2: Feature Engineering
    features = detector.engineer_features()
    if len(features) == 0:
        print(" Error: Not enough data points to calculate rolling features.")
        return

    # Step 3: Fit, Predict, and Label
    processed_df = detector.fit_predict()
    if processed_df is None or 'Regime' not in processed_df.columns:
        print(" Error: Model failed to classify market regimes.")
        return

    # Step 4: Visualization
    # Notice how we pass detector.regime_meta so the chart knows what colors to use!
    plot_regimes(processed_df, detector.regime_meta, ticker_symbol)

if __name__ == '__main__':
    main()