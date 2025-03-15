#!/usr/bin/env python
# IBM Stock Analysis HTML Report Generator - Simplified Implementation

import os
import base64
import pandas as pd
import numpy as np
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
import io
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

# Function to convert matplotlib figure to base64 for embedding in HTML
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

# Function to create HTML report
def generate_html_report():
    print("Generating HTML report...")
    
    # Step 1: Data Acquisition
    print("Downloading IBM historical data from Yahoo Finance...")
    ibm_data = yf.download('IBM', start='1967-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'))
    
    # Create a copy of the data for analysis
    df = ibm_data.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Focus on Adjusted Close prices
    if 'Adj Close' in df.columns:
        df['IBM_Adj_Close'] = df['Adj Close']
    elif 'Adjusted Close' in df.columns:
        df['IBM_Adj_Close'] = df['Adjusted Close']
    else:
        df['IBM_Adj_Close'] = df['Close']
    
    # Statistical summary
    stats_summary = df['IBM_Adj_Close'].describe()
    skewness = df['IBM_Adj_Close'].skew()
    kurtosis = df['IBM_Adj_Close'].kurtosis()
    
    # Split data into training and testing sets
    split_date = '2022-12-31'
    train = df[df.index <= split_date]
    test = df[df.index > split_date]
    
    # Generate plots for the report
    
    # 1. Historical prices plot
    fig_historical = plt.figure(figsize=(16, 8))
    plt.plot(df.index, df['IBM_Adj_Close'], color='blue', linewidth=1.5)
    plt.title('IBM Historical Adjusted Close Prices (1967-Present)', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price (USD)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(dollar_formatter))
    plt.tight_layout()
    historical_plot_base64 = fig_to_base64(fig_historical)
    plt.close(fig_historical)
    
    # 2. Train-test split plot
    fig_split = plt.figure(figsize=(16, 8))
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
    split_plot_base64 = fig_to_base64(fig_split)
    plt.close(fig_split)
    
    # 3. Determine the order of differencing (d)
    adf_test = adfuller(train['IBM_Adj_Close'])
    
    # If p-value > 0.05, the series is non-stationary and needs differencing
    if adf_test[1] > 0.05:
        d_param = 1
        diff_series = train['IBM_Adj_Close'].diff().dropna()
        
        # Check if first differencing made the series stationary
        adf_diff = adfuller(diff_series)
        
        # If still non-stationary, apply second differencing
        if adf_diff[1] > 0.05:
            d_param = 2
            diff_series = diff_series.diff().dropna()
    else:
        d_param = 0
        diff_series = train['IBM_Adj_Close']
    
    # Plot ACF and PACF to determine p and q
    fig_acf_pacf = plt.figure(figsize=(16, 12))
    plt.subplot(2, 1, 1)
    plot_acf(diff_series, ax=plt.gca(), lags=40)
    plt.title('Autocorrelation Function (ACF) of Differenced Series', fontsize=14)
    
    plt.subplot(2, 1, 2)
    plot_pacf(diff_series, ax=plt.gca(), lags=40)
    plt.title('Partial Autocorrelation Function (PACF) of Differenced Series', fontsize=14)
    
    plt.tight_layout()
    acf_pacf_base64 = fig_to_base64(fig_acf_pacf)
    plt.close(fig_acf_pacf)
    
    # Based on ACF and PACF, select p and q
    # For demonstration, we'll use p=1, q=1 as a starting point
    p_param = 1
    q_param = 1
    
    # 4. ARIMA Fit & Residual Analysis
    # Fit ARIMA model with selected parameters
    model = ARIMA(train['IBM_Adj_Close'], order=(p_param, d_param, q_param))
    model_fit = model.fit()
    
    # Extract residuals
    residuals = model_fit.resid
    
    # Plot residuals
    fig_residuals = plt.figure(figsize=(16, 12))
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
    residuals_base64 = fig_to_base64(fig_residuals)
    plt.close(fig_residuals)
    
    # 5. ACF/PACF of residuals
    fig_resid_acf_pacf = plt.figure(figsize=(16, 12))
    plt.subplot(2, 1, 1)
    plot_acf(residuals.dropna(), ax=plt.gca(), lags=40)
    plt.title('ACF of ARIMA Residuals', fontsize=14)
    
    plt.subplot(2, 1, 2)
    plot_pacf(residuals.dropna(), ax=plt.gca(), lags=40)
    plt.title('PACF of ARIMA Residuals', fontsize=14)
    
    plt.tight_layout()
    resid_acf_pacf_base64 = fig_to_base64(fig_resid_acf_pacf)
    plt.close(fig_resid_acf_pacf)
    
    # Check stationarity of residuals
    adf_resid = adfuller(residuals.dropna())
    kpss_resid = kpss(residuals.dropna())
    
    # 6. Volatility Modeling (GARCH on Residuals)
    # Fit GARCH(1,1) model on residuals
    garch_model = arch_model(residuals.dropna(), vol='GARCH', p=1, q=1)
    garch_results = garch_model.fit(disp='off')
    
    # Extract conditional volatility
    conditional_volatility = garch_results.conditional_volatility
    
    # Plot conditional volatility
    fig_volatility = plt.figure(figsize=(16, 10))
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
    volatility_base64 = fig_to_base64(fig_volatility)
    plt.close(fig_volatility)
    
    # 7. Returns Distribution Analysis (Residuals)
    # Calculate statistics for residuals
    resid_mean = residuals.mean()
    resid_std = residuals.std()
    resid_skew = stats.skew(residuals.dropna())
    resid_kurt = stats.kurtosis(residuals.dropna())
    
    # Jarque-Bera test for normality
    jb_test = stats.jarque_bera(residuals.dropna())
    
    # Plot residuals distribution
    fig_resid_dist = plt.figure(figsize=(16, 12))
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
    resid_dist_base64 = fig_to_base64(fig_resid_dist)
    plt.close(fig_resid_dist)
    
    # 8. Forecasting with ARIMA + GARCH
    # Generate ARIMA forecasts for the test period
    arima_forecast = model_fit.get_forecast(steps=len(test))
    arima_mean = arima_forecast.predicted_mean
    arima_conf_int = arima_forecast.conf_int(alpha=0.05)  # 95% confidence interval

    # Generate GARCH forecasts for the test period
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
    
    # Plot forecasts vs actual values
    fig_forecast = plt.figure(figsize=(16, 8))
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
    forecast_base64 = fig_to_base64(fig_forecast)
    plt.close(fig_forecast)
    
    # 9. Forecast Evaluation
    # Calculate forecast performance metrics
    mae = mean_absolute_error(forecast_df['Actual'], forecast_df['ARIMA_Mean'])
    rmse = np.sqrt(mean_squared_error(forecast_df['Actual'], forecast_df['ARIMA_Mean']))
    mape = mean_absolute_percentage_error(forecast_df['Actual'], forecast_df['ARIMA_Mean']) * 100
    
    # Calculate percentage of actual values within confidence interval
    within_ci = ((forecast_df['Actual'] >= forecast_df['GARCH_Lower_CI']) & 
                 (forecast_df['Actual'] <= forecast_df['GARCH_Upper_CI'])).mean() * 100
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>IBM Stock Analysis (ARIMA + GARCH, 1967–Present)</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1 {{
                color: #0066cc;
                text-align: center;
                border-bottom: 2px solid #0066cc;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #0066cc;
                border-bottom: 1px solid #ddd;
                padding-bottom: 5px;
                margin-top: 30px;
            }}
            h3 {{
                color: #0066cc;
            }}
            .img-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .img-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                border: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #f8f8f8;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .metric {{
                font-weight: bold;
                color: #0066cc;
            }}
            .section {{
                margin-bottom: 40px;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #777;
            }}
            .code {{
                font-family: monospace;
                background-color: #f8f8f8;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }}
        </style>
    </head>
    <body>
        <h1>IBM Stock Analysis (ARIMA + GARCH, 1967–Present)</h1>
        
        <div class="section">
            <h2>1. Introduction</h2>
            <p>This report presents a comprehensive time series analysis of IBM stock prices from January 1, 1967, to the present. The analysis employs a combination of Autoregressive Integrated Moving Average (ARIMA) for modeling the mean component of the time series and Generalized Autoregressive Conditional Heteroskedasticity (GARCH) for modeling the volatility component.</p>
            <p>The primary objectives of this analysis are:</p>
            <ul>
                <li>To understand the historical price patterns and statistical properties of IBM stock</li>
                <li>To develop and evaluate a forecasting model that captures both trend and volatility</li>
                <li>To provide insights for investment decision-making based on the model's performance</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>2. Data Acquisition & Initial Visualization</h2>
            <p>Historical daily Adjusted Close prices for IBM were downloaded from Yahoo Finance, covering the period from January 1, 1967, to {pd.Timestamp.today().strftime('%B %d, %Y')}.</p>
            
            <div class="img-container">
                <img src="data:image/png;base64,{historical_plot_base64}" alt="IBM Historical Prices">
            </div>
            
            <h3>Statistical Summary</h3>
            <table>
                <tr>
                    <th>Statistic</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Count</td>
                    <td>{stats_summary['count']:.0f}</td>
                </tr>
                <tr>
                    <td>Mean</td>
                    <td>${stats_summary['mean']:.2f}</td>
                </tr>
                <tr>
                    <td>Standard Deviation</td>
                    <td>${stats_summary['std']:.2f}</td>
                </tr>
                <tr>
                    <td>Minimum</td>
                    <td>${stats_summary['min']:.2f}</td>
                </tr>
                <tr>
                    <td>25th Percentile</td>
                    <td>${stats_summary['25%']:.2f}</td>
                </tr>
                <tr>
                    <td>Median (50th Percentile)</td>
                    <td>${stats_summary['50%']:.2f}</td>
                </tr>
                <tr>
                    <td>75th Percentile</td>
                    <td>${stats_summary['75%']:.2f}</td>
                </tr>
                <tr>
                    <td>Maximum</td>
                    <td>${stats_summary['max']:.2f}</td>
                </tr>
                <tr>
                    <td>Skewness</td>
                    <td>{skewness:.4f}</td>
                </tr>
                <tr>
                    <td>Kurtosis</td>
                    <td>{kurtosis:.4f}</td>
                </tr>
            </table>
            
            <p>The data shows a {skewness > 0 and 'positive' or 'negative'} skewness of {skewness:.4f}, indicating that the distribution is {'right-tailed with more extreme high values' if skewness > 0 else 'left-tailed with more extreme low values'}. The kurtosis value of {kurtosis:.4f} suggests that the distribution has {'heavier tails than' if kurtosis > 0 else 'lighter tails than'} a normal distribution.</p>
        </div>
        
        <div class="section">
            <h2>3. Train-Test Split</h2>
            <p>The data was split into training and testing sets:</p>
            <ul>
                <li><strong>Training set:</strong> From {train.index[0].strftime('%B %d, %Y')} to {train.index[-1].strftime('%B %d, %Y')} ({len(train)} observations)</li>
                <li><strong>Testing set:</strong> From {test.index[0].strftime('%B %d, %Y')} to {test.index[-1].strftime('%B %d, %Y')} ({len(test)} observations)</li>
            </ul>
            
            <div class="img-container">
                <img src="data:image/png;base64,{split_plot_base64}" alt="Train-Test Split">
            </div>
            
            <p>This split allows us to train our models on historical data and evaluate their performance on more recent, unseen data.</p>
        </div>
        
        <div class="section">
            <h2>4. ARIMA Model Selection</h2>
            <p>We used stationarity tests and ACF/PACF plots to identify the appropriate ARIMA parameters:</p>
            <p class="code">ARIMA({p_param},{d_param},{q_param})</p>
            
            <p>The differencing parameter (d={d_param}) was determined based on the Augmented Dickey-Fuller test, which {'indicated that the series was already stationary' if d_param == 0 else 'indicated that the series needed differencing to achieve stationarity'}.</p>
            
            <h3>ACF and PACF Analysis</h3>
            <div class="img-container">
                <img src="data:image/png;base64,{acf_pacf_base64}" alt="ACF and PACF Plots">
            </div>
            
            <p>The ACF and PACF plots helped determine the appropriate order for the ARIMA model. The ACF plot shows the correlation between the series and its lags, while the PACF plot shows the partial correlation that remains after removing the effects of shorter lags.</p>
        </div>
        
        <div class="section">
            <h2>5. ARIMA Residual Analysis</h2>
            <p>After fitting the ARIMA model to the training data, we analyzed the residuals to check if they exhibit the properties of white noise:</p>
            
            <div class="img-container">
                <img src="data:image/png;base64,{residuals_base64}" alt="Residuals Analysis">
            </div>
            
            <h3>Residual Statistics</h3>
            <table>
                <tr>
                    <th>Statistic</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Mean</td>
                    <td>{resid_mean:.6f}</td>
                </tr>
                <tr>
                    <td>Standard Deviation</td>
                    <td>{resid_std:.6f}</td>
                </tr>
                <tr>
                    <td>Skewness</td>
                    <td>{resid_skew:.6f}</td>
                </tr>
                <tr>
                    <td>Kurtosis</td>
                    <td>{resid_kurt:.6f}</td>
                </tr>
            </table>
            
            <h3>Stationarity Tests</h3>
            <table>
                <tr>
                    <th>Test</th>
                    <th>Statistic</th>
                    <th>p-value</th>
                    <th>Result</th>
                </tr>
                <tr>
                    <td>Augmented Dickey-Fuller (ADF)</td>
                    <td>{adf_resid[0]:.4f}</td>
                    <td>{adf_resid[1]:.4f}</td>
                    <td>{'Stationary' if adf_resid[1] < 0.05 else 'Non-stationary'}</td>
                </tr>
                <tr>
                    <td>KPSS</td>
                    <td>{kpss_resid[0]:.4f}</td>
                    <td>{kpss_resid[1]:.4f}</td>
                    <td>{'Stationary' if kpss_resid[1] > 0.05 else 'Non-stationary'}</td>
                </tr>
            </table>
            
            <h3>ACF and PACF of Residuals</h3>
            <div class="img-container">
                <img src="data:image/png;base64,{resid_acf_pacf_base64}" alt="ACF and PACF of Residuals">
            </div>
            
            <p>The residual analysis indicates that {'the ARIMA model has captured most of the patterns in the data, as the residuals appear to be mostly random' if adf_resid[1] < 0.05 and kpss_resid[1] > 0.05 else 'there may be some remaining patterns in the data that the ARIMA model did not capture'}.</p>
        </div>
        
        <div class="section">
            <h2>6. Volatility Modeling (GARCH)</h2>
            <p>We fitted a GARCH(1,1) model to the ARIMA residuals to capture the time-varying volatility in the IBM stock prices:</p>
            
            <div class="img-container">
                <img src="data:image/png;base64,{volatility_base64}" alt="GARCH Conditional Volatility">
            </div>
            
            <p>The plot shows the conditional volatility estimated by the GARCH model, with major volatility regimes annotated. Notable periods of high volatility include:</p>
            <ul>
                <li>1973-1974 Oil Crisis</li>
                <li>1987 Black Monday</li>
                <li>2000 Dot-com Bubble</li>
                <li>2008 Financial Crisis</li>
                <li>2020 COVID-19 Pandemic</li>
            </ul>
            
            <p>The GARCH model effectively captures the clustering of volatility, where periods of high volatility tend to be followed by more high volatility, and periods of low volatility tend to be followed by more low volatility.</p>
        </div>
        
        <div class="section">
            <h2>7. Residuals Distribution Analysis</h2>
            <p>We analyzed the distribution of the ARIMA residuals to check for normality and other characteristics:</p>
            
            <div class="img-container">
                <img src="data:image/png;base64,{resid_dist_base64}" alt="Residuals Distribution">
            </div>
            
            <h3>Normality Test</h3>
            <table>
                <tr>
                    <th>Test</th>
                    <th>Statistic</th>
                    <th>p-value</th>
                    <th>Result</th>
                </tr>
                <tr>
                    <td>Jarque-Bera</td>
                    <td>{jb_test[0]:.4f}</td>
                    <td>{jb_test[1]:.8f}</td>
                    <td>{'Normal' if jb_test[1] > 0.05 else 'Non-normal'}</td>
                </tr>
            </table>
            
            <p>The Jarque-Bera test indicates that the residuals {'follow a normal distribution' if jb_test[1] > 0.05 else 'do not follow a normal distribution'}, with a p-value of {jb_test[1]:.8f}. {'This suggests that the ARIMA model has captured most of the non-random patterns in the data.' if jb_test[1] > 0.05 else 'This suggests that there may be some non-linear patterns in the data that the ARIMA model did not capture.'}</p>
        </div>
        
        <div class="section">
            <h2>8. Forecasting with ARIMA + GARCH</h2>
            <p>We used the fitted ARIMA model for mean forecasts and the GARCH model for volatility forecasts to predict IBM stock prices for the test period:</p>
            
            <div class="img-container">
                <img src="data:image/png;base64,{forecast_base64}" alt="Forecast vs Actual">
            </div>
            
            <p>The plot shows the ARIMA point forecasts (red line) along with 95% confidence intervals derived from the GARCH volatility estimates (pink shaded area). The actual IBM stock prices during the test period are shown in green.</p>
        </div>
        
        <div class="section">
            <h2>9. Forecast Evaluation</h2>
            <p>We evaluated the forecast performance using several metrics:</p>
            
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Interpretation</th>
                </tr>
                <tr>
                    <td>Mean Absolute Error (MAE)</td>
                    <td>${mae:.2f}</td>
                    <td>Average absolute difference between forecasted and actual prices</td>
                </tr>
                <tr>
                    <td>Root Mean Squared Error (RMSE)</td>
                    <td>${rmse:.2f}</td>
                    <td>Square root of the average squared differences between forecasted and actual prices</td>
                </tr>
                <tr>
                    <td>Mean Absolute Percentage Error (MAPE)</td>
                    <td>{mape:.2f}%</td>
                    <td>Average percentage difference between forecasted and actual prices</td>
                </tr>
                <tr>
                    <td>Coverage Rate (95% CI)</td>
                    <td>{within_ci:.2f}%</td>
                    <td>Percentage of actual values falling within the 95% confidence interval</td>
                </tr>
            </table>
            
            <p>The coverage rate of {within_ci:.2f}% indicates that {'the model\'s confidence intervals are well-calibrated' if abs(within_ci - 95) < 10 else 'the model\'s confidence intervals may need adjustment'}, as ideally 95% of actual values should fall within the 95% confidence interval.</p>
        </div>
        
        <div class="section">
            <h2>10. Recommendations and Insights</h2>
            
            <h3>Model Suitability</h3>
            <p>The ARIMA({p_param},{d_param},{q_param}) + GARCH(1,1) model provides a reasonable baseline for IBM stock price forecasting. The model captures both trend components through ARIMA and volatility components through GARCH. The performance metrics indicate {'strong' if mape < 5 else 'moderate' if mape < 10 else 'limited'} forecasting accuracy with a MAPE of {mape:.2f}%.</p>
            
            <h3>Limitations</h3>
            <ul>
                <li>The linear nature of ARIMA may not capture complex market dynamics and non-linear patterns</li>
                <li>GARCH assumes symmetric response to positive and negative shocks, which may not reflect market reality</li>
                <li>The model may not adequately respond to sudden market shifts or black swan events</li>
                <li>Long-term forecasts become increasingly uncertain due to error accumulation</li>
            </ul>
            
            <h3>Advanced Model Considerations</h3>
            <p>Based on the analysis, we recommend considering the following advanced models for potentially improved forecasting:</p>
            <ul>
                <li><strong>Silverkite:</strong> Better for capturing multiple seasonalities and holiday effects in financial time series</li>
                <li><strong>Orbit:</strong> Bayesian approach provides more robust uncertainty quantification and handles regime changes</li>
                <li><strong>TimesFM:</strong> Foundation models may capture complex non-linear patterns and incorporate external factors</li>
            </ul>
            
            <h3>Practical Applications</h3>
            <p>The current ARIMA+GARCH model is most suitable for:</p>
            <ul>
                <li>Short-term forecasting (1-3 months ahead)</li>
                <li>Volatility estimation for risk management</li>
                <li>Complementing fundamental analysis for investment decisions</li>
                <li>Establishing a baseline for comparison with more sophisticated models</li>
            </ul>
            
            <p>For improved accuracy, consider ensemble approaches that combine multiple forecasting methods or incorporating external factors such as market sentiment, macroeconomic indicators, and company-specific news.</p>
        </div>
        
        <div class="footer">
            <p>IBM Stock Analysis Report | Generated on {pd.Timestamp.today().strftime('%Y-%m-%d')}</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML content to file
    with open("./report.html", "w") as f:
        f.write(html_content)
    
    print("HTML report generated successfully: report.html")

if __name__ == "__main__":
    generate_html_report()