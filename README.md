# IBM Stock Analysis: ARIMA+GARCH Forecasting and EDA (1967–Present)

## Project Description
This project performs a comprehensive time series analysis of IBM stock prices from 1967 to present using ARIMA and GARCH models. The analysis includes exploratory data analysis, stationarity testing, volatility modeling, forecasting, and evaluation to provide insights into IBM's historical price patterns and future price predictions.

## Detailed Steps Performed

### 1. Data Acquisition
- Downloaded historical daily IBM stock price data from Yahoo Finance (1967-present)
- Handled missing data and column name variations
- Prepared time series data for analysis

### 2. Exploratory Visualization & Statistical Summaries
- Generated visualizations of historical price movements
- Calculated descriptive statistics (mean, median, standard deviation, etc.)
- Analyzed skewness and kurtosis of the distribution

### 3. Train-Test Split
- Split data into training set (1967-2022) and testing set (2023-present)
- Visualized the split to confirm proper separation

### 4. ARIMA Parameter Selection
- Performed stationarity tests (ADF, KPSS) on the original series
- Applied differencing to achieve stationarity
- Analyzed ACF and PACF plots to determine optimal ARIMA parameters
- Selected ARIMA(1,1,1) based on the analysis

### 5. ARIMA Model Fitting & Residual Analysis
- Fitted ARIMA model on training data
- Extracted and analyzed residuals
- Performed diagnostic checks on residuals

### 6. Volatility Modeling with GARCH
- Applied GARCH(1,1) to model conditional volatility
- Analyzed GARCH parameters and persistence
- Visualized conditional volatility over time
- Identified and annotated major volatility regimes (e.g., 1987 crash, 2008 crisis, COVID-19)

### 7. Residual Distribution Analysis
- Analyzed the distribution of ARIMA residuals
- Performed normality tests (Jarque-Bera)
- Visualized residual distribution with histograms and Q-Q plots

### 8. Forecasting with ARIMA + GARCH
- Combined ARIMA for mean forecasting and GARCH for volatility forecasting
- Generated point forecasts and confidence intervals
- Visualized forecasts against actual test data

### 9. Forecast Evaluation
- Calculated performance metrics (MAE, RMSE, MAPE)
- Assessed forecast accuracy and confidence interval coverage
- Performed rolling-window cross-validation for robustness

### 10. Recommendations and Insights
- Summarized findings from the analysis
- Discussed model limitations and potential improvements
- Suggested alternative models for future analysis

## How to Run

### Prerequisites
- Python 3.8 or higher
- Required packages listed in `requirements.txt`

### Installation and Execution
1. Clone this repository or download the source code
2. Run the automated script:
   - Double-click `run_analysis.bat` (Windows only)

Alternatively, you can manually install dependencies and run the analysis:
```bash
pip install -r requirements.txt
python ibm_arima_garch_analysis.py
python generate_html_report.py
python serve_report.py 59169
```

### Output
The script generates several visualization files and an HTML report:
- Historical price plots
- Train-test split visualization
- ACF/PACF plots
- Residual analysis plots
- Conditional volatility visualization
- Forecast vs. actual comparison

The HTML report (`report.html`) provides a comprehensive view of all analysis results in a well-formatted document. When you run the analysis script, a web server will automatically start to serve the report at http://localhost:59169/report.html.

## Results and Conclusions
The ARIMA(1,1,1)+GARCH(1,1) model provides a reasonable baseline for forecasting IBM stock prices, capturing both trend and volatility components. The model performs well in cross-validation on historical data but shows limitations for long-term forecasting. Advanced models like Silverkite, Orbit, or TimesFM might provide better forecasting accuracy, especially for capturing complex market dynamics.

## License
This project is provided for educational purposes only. Use at your own risk.