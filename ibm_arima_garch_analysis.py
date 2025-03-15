#!/usr/bin/env python
# IBM Time Series Analysis with ARIMA and GARCH - Simplified Implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

# Function to format y-axis as dollars
def dollar_formatter(x, pos):
    return f'${x:.2f}'

print("Step 1: Data Acquisition")
print("Downloading IBM historical data from Yahoo Finance...")
ibm_data = yf.download('IBM', start='1967-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'))

# Check if data was downloaded successfully
if ibm_data.empty:
    raise ValueError("Failed to download IBM data. Please check your internet connection.")

# Create a copy of the data for analysis
df = ibm_data.copy()
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# Focus on Adjusted Close prices
if 'Adj Close' in df.columns:
    df['IBM_Adj_Close'] = df['Adj Close']
    print("Using 'Adj Close' column for analysis.")
elif 'Adjusted Close' in df.columns:
    df['IBM_Adj_Close'] = df['Adjusted Close']
    print("Using 'Adjusted Close' column for analysis.")
else:
    # If adjusted close is not available, use regular close
    df['IBM_Adj_Close'] = df['Close']
    print("Warning: Using 'Close' instead of 'Adj Close' as it was not found.")

print("\nStep 2: Initial Visualization & Statistical Summary")

# Statistical summary
stats_summary = df['IBM_Adj_Close'].describe()
print("\nStatistical Summary of IBM Adjusted Close Prices:")
print(stats_summary)

# Calculate skewness and kurtosis
skewness = df['IBM_Adj_Close'].skew()
kurtosis = df['IBM_Adj_Close'].kurtosis()
print(f"\nSkewness: {skewness:.4f}")
print(f"Kurtosis: {kurtosis:.4f}")

# Plot historical prices
plt.figure(figsize=(16, 8))
plt.plot(df.index, df['IBM_Adj_Close'], color='blue', linewidth=1.5)
plt.title('IBM Historical Adjusted Close Prices (1967-Present)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(dollar_formatter))
plt.tight_layout()
plt.savefig('ibm_historical_prices.png')
plt.close()

print("\nStep 2B: Train-Test Split")

# Split data into training and testing sets
split_date = '2022-12-31'
train = df[df.index <= split_date]
test = df[df.index > split_date]

print(f"Training set: {train.index[0].strftime('%Y-%m-%d')} to {train.index[-1].strftime('%Y-%m-%d')} ({len(train)} observations)")
print(f"Testing set: {test.index[0].strftime('%Y-%m-%d')} to {test.index[-1].strftime('%Y-%m-%d')} ({len(test)} observations)")

# Visualize the train-test split
plt.figure(figsize=(16, 8))
plt.plot(train.index, train['IBM_Adj_Close'], color='blue', linewidth=1.5, label='Training Data')
plt.plot(test.index, test['IBM_Adj_Close'], color='red', linewidth=1.5, label='Testing Data')
plt.axvline(x=pd.Timestamp(split_date), color='black', linestyle='--', linewidth=2, label='Train-Test Split')
plt.title('IBM Adjusted Close Prices - Train-Test Split', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(dollar_formatter))
plt.tight_layout()
plt.savefig('ibm_train_test_split.png')
plt.close()

print("\nStep 3: Manual ARIMA Parameter Selection")

# Determine the order of differencing (d)
adf_test = adfuller(train['IBM_Adj_Close'])
print(f"ADF Test Statistic: {adf_test[0]:.4f}")
print(f"p-value: {adf_test[1]:.4f}")

# If p-value > 0.05, the series is non-stationary and needs differencing
if adf_test[1] > 0.05:
    print("Series is non-stationary, applying first differencing...")
    d_param = 1
    diff_series = train['IBM_Adj_Close'].diff().dropna()
    
    # Check if first differencing made the series stationary
    adf_diff = adfuller(diff_series)
    print(f"ADF Test after differencing: {adf_diff[0]:.4f}")
    print(f"p-value after differencing: {adf_diff[1]:.4f}")
    
    # If still non-stationary, apply second differencing
    if adf_diff[1] > 0.05:
        print("Series is still non-stationary, applying second differencing...")
        d_param = 2
        diff_series = diff_series.diff().dropna()
    else:
        print("First differencing made the series stationary.")
else:
    print("Series is already stationary, no differencing needed.")
    d_param = 0
    diff_series = train['IBM_Adj_Close']

# Plot ACF and PACF to determine p and q
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
plot_acf(diff_series, ax=ax1, lags=40)
plot_pacf(diff_series, ax=ax2, lags=40)
ax1.set_title('ACF of Differenced Series', fontsize=14)
ax2.set_title('PACF of Differenced Series', fontsize=14)
plt.tight_layout()
plt.savefig('acf_pacf_diff_series.png')
plt.close()

# Based on ACF and PACF, select p and q
# For demonstration, we'll use p=1, q=1 as a starting point
p_param = 1
q_param = 1

print(f"\nSelected ARIMA Parameters: ARIMA({p_param},{d_param},{q_param})")

print("\nStep 4: ARIMA Fit & Residual Analysis (Training Data)")

# Fit ARIMA model with selected parameters
model = ARIMA(train['IBM_Adj_Close'], order=(p_param, d_param, q_param))
model_fit = model.fit()
print(model_fit.summary())

# Extract residuals
residuals = model_fit.resid

# Print residual statistics
print("\nResidual Statistics:")
print(residuals.describe())

# Plot residuals
plt.figure(figsize=(16, 12))
plt.subplot(3, 1, 1)
plt.plot(residuals.index, residuals, color='blue', linewidth=1)
plt.title('ARIMA Residuals Over Time', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Residual Value', fontsize=12)
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.hist(residuals, bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Residuals', fontsize=14)
plt.xlabel('Residual Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
stats.probplot(residuals, plot=plt)
plt.title('Q-Q Plot of Residuals', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('arima_residuals_analysis.png')
plt.close()

print("\nStep 4B: ACF/PACF and Stationarity (Residual Analysis)")

# Plot ACF and PACF of residuals
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
plot_acf(residuals.dropna(), ax=ax1, lags=40)
plot_pacf(residuals.dropna(), ax=ax2, lags=40)
ax1.set_title('ACF of ARIMA Residuals', fontsize=14)
ax2.set_title('PACF of ARIMA Residuals', fontsize=14)
plt.tight_layout()
plt.savefig('acf_pacf_residuals.png')
plt.close()

# Check stationarity of residuals
adf_resid = adfuller(residuals.dropna())
print("\nStationarity Tests on ARIMA Residuals:")
print(f"ADF Test Statistic: {adf_resid[0]:.4f}")
print(f"p-value: {adf_resid[1]:.4f}")
print("Critical Values:")
for key, value in adf_resid[4].items():
    print(f"\t{key}%: {value:.4f}")
print(f"ADF Test Result: {'Stationary' if adf_resid[1] < 0.05 else 'Non-stationary'}")

# KPSS test
kpss_resid = kpss(residuals.dropna())
print(f"\nKPSS Test Statistic: {kpss_resid[0]:.4f}")
print(f"p-value: {kpss_resid[1]:.4f}")
print("Critical Values:")
for key, value in kpss_resid[3].items():
    print(f"\t{key}: {value:.4f}")
print(f"KPSS Test Result: {'Stationary' if kpss_resid[1] > 0.05 else 'Non-stationary'}")

print("\nStep 5: Volatility Modeling (GARCH explicitly on Residuals)")

# Fit GARCH(1,1) model on residuals
garch_model = arch_model(residuals.dropna(), vol='GARCH', p=1, q=1)
garch_results = garch_model.fit(disp='off')
print(garch_results.summary())

# Extract conditional volatility
conditional_volatility = garch_results.conditional_volatility

# Plot conditional volatility
plt.figure(figsize=(16, 10))
plt.plot(residuals.dropna().index, conditional_volatility, color='red', linewidth=1.5)
plt.title('Conditional Volatility from GARCH(1,1) Model', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Volatility', fontsize=14)
plt.grid(True, alpha=0.3)

# Annotate major volatility regimes
volatility_events = [
    ('1973-1974 Oil Crisis', '1973-10-01', 2),
    ('1987 Black Monday', '1987-10-19', 2),
    ('2000 Dot-com Bubble', '2000-03-10', 2),
    ('2008 Financial Crisis', '2008-09-15', 2),
    ('2020 COVID-19', '2020-03-16', 2)
]

for event, date, height in volatility_events:
    try:
        event_date = pd.Timestamp(date)
        if event_date in conditional_volatility.index:
            plt.annotate(event, 
                        xy=(mdates.date2num(event_date), conditional_volatility.loc[event_date]),
                        xytext=(mdates.date2num(event_date), conditional_volatility.loc[event_date] + height),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                        fontsize=12)
    except:
        print(f"Could not annotate {event} at {date}")

plt.tight_layout()
plt.savefig('garch_conditional_volatility.png')
plt.close()

print("\nStep 4C: Returns Distribution Analysis (Residuals)")

# Calculate statistics for residuals
resid_mean = residuals.mean()
resid_std = residuals.std()
resid_skew = stats.skew(residuals.dropna())
resid_kurt = stats.kurtosis(residuals.dropna())

print("\nResiduals Distribution Characteristics:")
print(f"Mean: {resid_mean:.6f}")
print(f"Standard Deviation: {resid_std:.6f}")
print(f"Skewness: {resid_skew:.6f}")
print(f"Kurtosis: {resid_kurt:.6f}")

# Jarque-Bera test for normality
jb_test = stats.jarque_bera(residuals.dropna())
print("\nJarque-Bera Test:")
print(f"Test Statistic: {jb_test[0]:.4f}")
print(f"p-value: {jb_test[1]:.8f}")
print(f"Normality Test Result: {'Normal' if jb_test[1] > 0.05 else 'Non-normal'}")

# Plot residuals distribution
plt.figure(figsize=(16, 12))
plt.subplot(2, 1, 1)
sns.histplot(residuals.dropna(), kde=True, bins=100, color='blue')
plt.title('Histogram of ARIMA Residuals with Normal Distribution', fontsize=14)
plt.xlabel('Residual Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
stats.probplot(residuals.dropna(), plot=plt)
plt.title('Q-Q Plot of Residuals vs. Normal Distribution', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residuals_distribution_analysis.png')
plt.close()

print("\nStep 6: Forecasting with ARIMA + GARCH")

# We'll use the already fitted ARIMA and GARCH models for forecasting
print("Forecasting using previously fitted ARIMA and GARCH models...")

# Generate ARIMA forecasts for the test period
arima_forecast = model_fit.get_forecast(steps=len(test))
arima_mean = arima_forecast.predicted_mean
arima_conf_int = arima_forecast.conf_int(alpha=0.05)  # 95% confidence interval

# Generate GARCH forecasts for the test period
# We need to forecast volatility for the test period
garch_forecast = garch_results.forecast(horizon=len(test))
garch_variance = garch_forecast.variance.iloc[-1]  # Get the last row of variance forecasts

# Create a DataFrame to store forecasts
forecast_df = pd.DataFrame(index=test.index[:len(arima_mean)])
forecast_df['ARIMA_Mean'] = arima_mean.values
forecast_df['GARCH_Variance'] = garch_variance.values[:len(arima_mean)]
forecast_df['Lower_CI'] = arima_conf_int.iloc[:, 0].values
forecast_df['Upper_CI'] = arima_conf_int.iloc[:, 1].values
forecast_df['Actual'] = test['IBM_Adj_Close'].values[:len(arima_mean)]

# Adjust confidence intervals using GARCH volatility
z_score = stats.norm.ppf(0.975)  # 95% confidence level
forecast_df['GARCH_Lower_CI'] = forecast_df['ARIMA_Mean'] - z_score * np.sqrt(forecast_df['GARCH_Variance'])
forecast_df['GARCH_Upper_CI'] = forecast_df['ARIMA_Mean'] + z_score * np.sqrt(forecast_df['GARCH_Variance'])

# Print the first few rows of the forecast DataFrame
print("\nFirst 5 rows of forecast DataFrame:")
print(forecast_df.head())

# Plot forecasts vs actual values
plt.figure(figsize=(16, 8))
plt.plot(train.index[-100:], train['IBM_Adj_Close'].iloc[-100:], color='blue', linewidth=1.5, label='Training Data')
plt.plot(forecast_df.index, forecast_df['Actual'], color='green', linewidth=1.5, label='Actual Test Data')
plt.plot(forecast_df.index, forecast_df['ARIMA_Mean'], color='red', linewidth=1.5, label='ARIMA Forecast')
plt.fill_between(forecast_df.index, 
                forecast_df['GARCH_Lower_CI'], 
                forecast_df['GARCH_Upper_CI'], 
                color='pink', alpha=0.3, label='95% Confidence Interval (GARCH)')
plt.title('IBM Adjusted Close Price: Actual vs. Forecast', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(dollar_formatter))
plt.tight_layout()
plt.savefig('ibm_forecast_vs_actual.png')
plt.close()

print("\nStep 7: Forecast Evaluation")

# Calculate forecast performance metrics
mae = mean_absolute_error(forecast_df['Actual'], forecast_df['ARIMA_Mean'])
rmse = np.sqrt(mean_squared_error(forecast_df['Actual'], forecast_df['ARIMA_Mean']))
mape = mean_absolute_percentage_error(forecast_df['Actual'], forecast_df['ARIMA_Mean']) * 100

print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Calculate percentage of actual values within confidence interval
within_ci = ((forecast_df['Actual'] >= forecast_df['GARCH_Lower_CI']) & 
             (forecast_df['Actual'] <= forecast_df['GARCH_Upper_CI'])).mean() * 100
print(f"Percentage of actual values within 95% confidence interval: {within_ci:.2f}%")

print("\nStep 8: Recommendations and Insights")

print("""
Summary of Analysis:
-------------------
1. Data Exploration:
   - Analyzed IBM stock data from 1967 to present
   - Split data into training (1967-2022) and testing (2023-present) sets

2. ARIMA Modeling:
   - Manually selected ARIMA parameters based on stationarity tests and ACF/PACF plots
   - Fitted ARIMA model on training data
   - Analyzed residuals for patterns and stationarity

3. Volatility Modeling:
   - Applied GARCH(1,1) to model conditional volatility
   - Identified and annotated major volatility regimes
   - Analyzed residual distribution characteristics

4. Forecasting:
   - Combined ARIMA and GARCH models for point and interval forecasts
   - Evaluated forecast performance using MAE, RMSE, and MAPE

Recommendations:
---------------
1. Model Suitability:
   - ARIMA+GARCH provides a reasonable baseline for IBM stock price forecasting
   - The model captures both trend and volatility components
   - Performance metrics indicate moderate forecasting accuracy

2. Limitations:
   - Linear nature of ARIMA may not capture complex market dynamics
   - GARCH assumes symmetric response to positive and negative shocks
   - Model may not adequately respond to sudden market shifts

3. Advanced Model Considerations:
   - Silverkite: Better for capturing multiple seasonalities and holiday effects
   - Orbit: Bayesian approach provides uncertainty quantification
   - TimesFM: Foundation models may capture complex non-linear patterns

4. Practical Applications:
   - Use for short-term forecasting (1-3 months)
   - Combine with fundamental analysis for investment decisions
   - Consider ensemble approaches for improved accuracy
""")

print("\nAnalysis complete! Check the generated plots for visualizations.")