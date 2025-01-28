library(fpp3)

data <- readr::read_csv("/Users/yamanjoshi/Downloads/TEMPHELPN.csv")
data


#date preparing
################################################################################

#tsibble result
data2 <- data %>%
  mutate(month = yearmonth(DATE)) %>% 
  filter(month >= yearmonth("2000 Jan")) %>%  
  select(-DATE) %>%
  as_tsibble(index = month)
data2

view(data2)


#plot data
autoplot(data2, TEMPHELPN) + 
  labs(y="Thousand of Persons", 
       title = "Number of people employed in temporary help services")


# Using differencing to achieve stationarity in time series data
 data2 %>%
  transmute(
    temp_rate = TEMPHELPN,
    log_temp_rate = log(temp_rate),
    temp_rate_sd = difference(log(temp_rate), 12),
    temp_rate_dd = difference(difference(log(temp_rate), 12),1))


#1.	Conduct unit root tests for the log, the seasonal difference, and the double
#   difference of the time series. 
data2 %>%
  features(log(TEMPHELPN), unitroot_kpss)

data2 %>%
  features(difference(log(TEMPHELPN), 12), unitroot_kpss)

data2 %>%
  features(difference(difference(log(TEMPHELPN), 12),1), unitroot_kpss)




################################################################################


#2.	Create plots of the ACF and PACF for the seasonal difference and the double 
#   difference of the log of the time series.

# seasonal difference
data2 %>%
  gg_tsdisplay(difference(log(TEMPHELPN), 12),
               plot_type = 'partial', lag = 36) +
  labs(title = "Seasonal Difference", y = "")

# double difference
data2 %>%
  gg_tsdisplay(difference(difference(log(TEMPHELPN), 12)),
               plot_type = 'partial', lag = 36) +
  labs(title = "Double Difference", y = "")


################################################################################


#3 and 4. Identify two ARIMA models and fit for the time series. 


fit <- data2 %>%
  model(
    arima1 = ARIMA((log(TEMPHELPN)) ~ pdq(0,1,1) + PDQ(0,1,1)), 
    arima2 = ARIMA((log(TEMPHELPN)) ~ pdq(1,1,0) + PDQ(0,1,1)),  
    auto = ARIMA((log(TEMPHELPN)), stepwise = FALSE, approx = FALSE)
  )


fit %>% pivot_longer(everything(), names_to = "Model Name", values_to = "Order")



#Information criteria
glance(fit) %>% arrange(AICc) %>% select(.model:BIC)



################################################################################

#5.	Create a training data set from the start of the series to December 2020.

# Split the data into training and test sets
train_data <- data2 |>
  filter_index("2000 Jan" ~ "2020 Dec")
  train_data <- train_data %>% drop_na()


data3 <- train_data %>%
  mutate(log_temp = log(TEMPHELPN)) %>%
  select(-TEMPHELPN)
data3 # THIS IS MY TRAINING DATA NOW


test_data <- data2 |>
  filter_index("2021 Jan" ~ .)
test_data <- test_data %>% drop_na()



data4 <- test_data %>%
  mutate(log_temp = log(TEMPHELPN)) %>%
  select(-TEMPHELPN)
data4 # THIS IS MY Testing DATA NOW


# Fit the "auto" ARIMA model to the training data

best_fit_auto <- data3 %>%
  model(
    auto = ARIMA(log_temp ~ pdq(2, 0, 0) + PDQ(0, 1, 1))
  )


# Report the model output
report(best_fit_auto)



# Generate residual diagnostics
best_fit_auto %>%
  gg_tsresiduals(lag = 36) +
  labs(title = "Residual Diagnostics for Best ARIMA Model")


# Autocorrelation test
best_fit_auto %>%
  augment() %>%
  features(.innov, ljung_box, lag = 24, dof = 3)



################################################################################

#6.	Fit an ETS model 

# Fit the ETS model to the training data set
ets_fit <- data3 %>%
  model(ets = ETS(log_temp))
report(ets_fit)


#residual diagnostic
ets_fit %>%
  gg_tsresiduals(lag = 36) +
  labs(title = "Residual Diagnostics for ETS Model")

# Autocorrelation test
ets_fit %>%
  augment() %>%
  features(.innov, ljung_box, lag = 24, dof = 4)




################################################################################

#7.	Based on the test data set in item 5, compare the ARIMA and the ETS models


bind_rows(
  best_fit_auto |> accuracy(),
  ets_fit |> accuracy(),
  best_fit_auto |> forecast(h = 36) |> accuracy(data4),
  ets_fit |> forecast(h = 36) |> accuracy(data4)
) |>
  select(-ME, -MPE, -ACF1)





################################################################################

#8.	Use bootstrapping to generate 1000 simulations of the log of the time series


datanew<- data2%>%
  mutate('LogTEMPHELPN'=log(TEMPHELPN)) %>%
  select(-TEMPHELPN)
datanew

temp_stl <- datanew %>%
  model(stl = STL(LogTEMPHELPN))

temp_stl %>%
  components() %>%
  autoplot()

sim_temp <- temp_stl %>%
  generate(new_data = datanew, times = 1000, bootstrap_block_size = 24) %>%
  select(-.model, -LogTEMPHELPN)
sim_temp

ets_forecasts <- sim_temp %>%
  model(ets = ETS(.sim)) %>% 
  forecast(h = 36)

ets_forecasts |>
  update_tsibble(key = .rep) |>
  autoplot(.mean) +
  autolayer(datanew,LogTEMPHELPN) +
  guides(colour = "none") +
  labs(title = "Bootstrapped forecasts",
       y="-LogTEMPHELPN")




################################################################################

#9.	Create a plot that shows three-year forecasts of the better model 

#bagged mean for the ETS bootstrapped forecasts
bagged_ets <- ets_forecasts %>%
  summarise(bagged_mean = mean(.mean))  

# Print the bagged mean for the first 36 periods 
print(bagged_ets, n = 36)


#print the tsibble that shows the ETS bagged forecasts for the three years 

ets_forecast <- datanew%>%
  model(ets = ETS(LogTEMPHELPN)) %>%
  forecast(h = 36)


# Plotting the results
ets_forecast %>%
  autoplot(datanew) +  
  autolayer(bagged_ets, bagged_mean, col = "#D55E00") + 
  labs(title = "Comparing bagged ETS forecasts and ETS model",
       y = "thousabd of people")



#10 . Print the tsibble that shows the ARIMA bagged forecasts for the three years 
arima_forecasts <- sim_temp %>%
  model(arima = ARIMA(.sim ~ 0 + pdq(2,0,0) + PDQ(0,1,1))) %>%
  forecast(h = 36)
warnings(50)

arima_forecasts %>%
  update_tsibble(key = .rep) %>%
  autoplot(.mean) +
  autolayer(datanew,LogTEMPHELPN) +
  guides(colour = "none") +
  labs(title = "Bootstrapped forecasts",
       y="-LogTEMPHELPN")

bagged_arima <- arima_forecasts %>%
  summarise(bagged_mean = mean(.mean))

print(bagged_arima, n = 36)
















