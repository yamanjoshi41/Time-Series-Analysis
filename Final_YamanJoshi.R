import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load data
data = pd.read_csv("/Users/yamanjoshi/Downloads/TEMPHELPN.csv")

# Prepare date and tsibble equivalent
data['DATE'] = pd.to_datetime(data['DATE'])
data = data.set_index('DATE')
data = data['2000-01-01':]
data['TEMPHELPN'] = data['TEMPHELPN'].astype(float)

# Plot data
data['TEMPHELPN'].plot(title='Number of people employed in temporary help services', ylabel='Thousand of Persons')
plt.show()

# Using differencing to achieve stationarity in time series data
data['log_TEMPHELPN'] = np.log(data['TEMPHELPN'])
data['temp_rate_sd'] = data['log_TEMPHELPN'].diff(12)
data['temp_rate_dd'] = data['temp_rate_sd'].diff(1)

# Conduct unit root tests
def kpss_test(series, **kw):
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    return p_value

print('KPSS Test for log:', kpss_test(data['log_TEMPHELPN'].dropna(), regression='c'))
print('KPSS Test for seasonal difference:', kpss_test(data['temp_rate_sd'].dropna(), regression='c'))
print('KPSS Test for double difference:', kpss_test(data['temp_rate_dd'].dropna(), regression='c'))

# Create plots of the ACF and PACF
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(data['temp_rate_sd'].dropna(), lags=36, ax=ax[0])
plot_pacf(data['temp_rate_sd'].dropna(), lags=36, ax=ax[1])
ax[0].set_title('Seasonal Difference')
ax[1].set_title('Partial ACF for Seasonal Difference')
plt.show()

fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(data['temp_rate_dd'].dropna(), lags=36, ax=ax[0])
plot_pacf(data['temp_rate_dd'].dropna(), lags=36, ax=ax[1])
ax[0].set_title('Double Difference')
ax[1].set_title('Partial ACF for Double Difference')
plt.show()

# Identify two ARIMA models and fit for the time series
arima1 = ARIMA(data['log_TEMPHELPN'], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit()
arima2 = ARIMA(data['log_TEMPHELPN'], order=(1, 1, 0), seasonal_order=(0, 1, 1, 12)).fit()
auto_arima = ARIMA(data['log_TEMPHELPN'], order=(2, 0, 0), seasonal_order=(0, 1, 1, 12)).fit()

# Compare models
models = [arima1, arima2, auto_arima]
model_names = ['ARIMA(0,1,1)(0,1,1)[12]', 'ARIMA(1,1,0)(0,1,1)[12]', 'Auto ARIMA']
aic_values = [model.aic for model in models]
bic_values = [model.bic for model in models]

model_comparison = pd.DataFrame({'Model': model_names, 'AIC': aic_values, 'BIC': bic_values})
print(model_comparison.sort_values(by='AIC'))

# Create training data set
train_data = data[:'2020-12-31']
test_data = data['2021-01-01':]

# Fit the "auto" ARIMA model to the training data
best_fit_auto = ARIMA(train_data['log_TEMPHELPN'], order=(2, 0, 0), seasonal_order=(0, 1, 1, 12)).fit()

# Report the model output
print(best_fit_auto.summary())

# Generate residual diagnostics
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
sns.residplot(best_fit_auto.resid, ax=ax)
ax.set_title('Residual Diagnostics for Best ARIMA Model')
plt.show()

# Autocorrelation test
from statsmodels.stats.diagnostic import acorr_ljungbox
ljung_box_result = acorr_ljungbox(best_fit_auto.resid, lags=[24], return_df=True)
print(ljung_box_result)

# Fit an ETS model
ets_model = ExponentialSmoothing(train_data['log_TEMPHELPN'], seasonal='add', seasonal_periods=12).fit()

# Report the ETS model output
print(ets_model.summary())

# Residual diagnostics for ETS model
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
sns.residplot(ets_model.resid, ax=ax)
ax.set_title('Residual Diagnostics for ETS Model')
plt.show()

# Autocorrelation test for ETS
ljung_box_result_ets = acorr_ljungbox(ets_model.resid, lags=[24], return_df=True)
print(ljung_box_result_ets)

# Compare the ARIMA and the ETS models based on test data set
arima_forecast = best_fit_auto.get_forecast(steps=len(test_data))
ets_forecast = ets_model.forecast(steps=len(test_data))

def accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)  # ME
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mpe = np.mean((forecast - actual) / actual)  # MPE
    return pd.DataFrame({'MAPE': [mape], 'ME': [me], 'MAE': [mae], 'MPE': [mpe]})

arima_accuracy = accuracy(arima_forecast.predicted_mean, test_data['log_TEMPHELPN'])
ets_accuracy = accuracy(ets_forecast, test_data['log_TEMPHELPN'])

print('ARIMA Model Accuracy:')
print(arima_accuracy)
print('ETS Model Accuracy:')
print(ets_accuracy)

# Use bootstrapping to generate 1000 simulations of the log of the time series
stl = STL(data['log_TEMPHELPN'], seasonal=13)
res = stl.fit()

sim_temp = np.zeros((1000, len(data)))
for i in range(1000):
    sim_temp[i, :] = res.trend + np.random.normal(0, res.resid.std(), len(data))

# ETS model for the bootstrapped series
ets_bootstrapped_forecasts = []
for i in range(1000):
    model = ExponentialSmoothing(sim_temp[i, :], seasonal='add', seasonal_periods=12).fit()
    ets_forecast = model.forecast(steps=36)
    ets_bootstrapped_forecasts.append(ets_forecast)

# Plot the ETS bootstrapped forecasts
plt.figure(figsize=(12, 6))
for forecast in ets_bootstrapped_forecasts:
    plt.plot(forecast, color='r', alpha=0.1)
plt.plot(data['log_TEMPHELPN'], color='b')
plt.title('Bootstrapped forecasts')
plt.show()

# Generate ARIMA bagged forecasts for the three years
arima_bagged_forecasts = []
for i in range(1000):
    model = ARIMA(sim_temp[i, :], order=(2, 0, 0), seasonal_order=(0, 1, 1, 12)).fit()
    forecast = model.get_forecast(steps=36)
    arima_bagged_forecasts.append(forecast.predicted_mean)

# Plot ARIMA bagged forecasts
plt.figure(figsize=(12, 6))
for forecast in arima_bagged_forecasts:
    plt.plot(forecast, color='r', alpha=0.1)
plt.plot(data['log_TEMPHELPN'], color='b')
plt.title('ARIMA Bootstrapped forecasts')
plt.show()
