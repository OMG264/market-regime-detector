# 📈 Market Regime Detector (GMM Hybrid Architecture)

A production-grade algorithmic tool that identifies historical stock market environments (Bull, Bear, and Sideways) using statistical probability. 

Unlike traditional trading indicators that rely on rigid, hard-coded thresholds, this project utilizes a **Gaussian Mixture Model (GMM)** to understand the fluid and overlapping nature of financial markets, outputting mathematically robust regime classifications.

---

## ✨ Core Features

* **🧠 Unsupervised Machine Learning:** Leverages `scikit-learn`'s Gaussian Mixture Model to organically cluster daily market data based on volatility and momentum.
* **📊 Smart Labeling Logic:** Automatically maps abstract statistical clusters to human-readable financial regimes (Bull = High Return/Low Vol, Bear = Low Return/High Vol).
* **🏗️ Object-Oriented Pipeline:** Built with a modular `class` architecture, cleanly separating the mathematical engine (`regime_model.py`) from the execution and charting script (`main.py`).
* **🎨 Professional Data Visualization:** Generates macro-economic style charts with vertical background shading to distinctly highlight structural market shifts over time.
* **🛡️ Robust Error Handling:** Includes graceful API failure safety nets for live data ingestion.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.x
* **Data Ingestion:** `yfinance`
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Visualization:** `matplotlib`

---

## 🚀 How to Run

### 1. Setup the Environment
Clone the repository and install the required dependencies using the provided `requirements.txt` file.

```bash
git clone [https://github.com/YourUsername/market-regime-detector.git](https://github.com/YourUsername/market-regime-detector.git)
cd market-regime-detector
pip install -r requirements.txt
